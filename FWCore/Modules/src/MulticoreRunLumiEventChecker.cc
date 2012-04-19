// -*- C++ -*-
//
// Package:    Modules
// Class:      MulticoreRunLumiEventChecker
//
/**\class MulticoreRunLumiEventChecker MulticoreRunLumiEventChecker.cc FWCore/Modules/src/MulticoreRunLumiEventChecker.cc

 Description: Checks that the events passed to it come in the order specified in its configuration

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Chris Jones
//         Created:  Tue Jun 16 15:42:17 CDT 2009
//
//

// user include files
#include "DataFormats/Provenance/interface/EventID.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/UnixSignalHandlers.h"

// system include files
#include <boost/shared_ptr.hpp>

#include <algorithm>
#include <map>
#include <memory>
#include <vector>
#include <signal.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#include <sys/signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <errno.h>

#include <boost/thread/thread.hpp>
//
// class decleration
//

class MulticoreRunLumiEventChecker : public edm::EDAnalyzer {
public:
   explicit MulticoreRunLumiEventChecker(edm::ParameterSet const&);
   ~MulticoreRunLumiEventChecker();
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
   virtual void beginJob();
   virtual void analyze(edm::Event const&, edm::EventSetup const&);
   virtual void endJob();
   virtual void postForkReacquireResources(unsigned int iChildIndex, unsigned int iNumberOfChildren);
   virtual void preForkReleaseResources();

   virtual void beginRun(edm::Run const& run, edm::EventSetup const& es);
   virtual void endRun(edm::Run const& run, edm::EventSetup const& es);
   
   virtual void beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& es);
   virtual void endLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& es);

   void check(edm::EventID const& iID, bool isEvent);
   
   // ----------member data ---------------------------
   std::vector<edm::EventID> ids_;
   unsigned int index_;
   std::map<edm::EventID, unsigned int> seenIDs_;

   unsigned int multiProcessSequentialEvents_;
   unsigned int numberOfEventsLeftBeforeSearch_;
   bool mustSearch_;
   
   boost::shared_ptr<boost::thread> listenerThread_;
   int messageQueue_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
MulticoreRunLumiEventChecker::MulticoreRunLumiEventChecker(edm::ParameterSet const& iConfig) :
  ids_(iConfig.getUntrackedParameter<std::vector<edm::EventID> >("eventSequence")),
  index_(0),
  multiProcessSequentialEvents_(iConfig.getUntrackedParameter<unsigned int>("multiProcessSequentialEvents")),
  numberOfEventsLeftBeforeSearch_(0),
  mustSearch_(false),
  messageQueue_(-1)
{
   //now do what ever initialization is needed
}


MulticoreRunLumiEventChecker::~MulticoreRunLumiEventChecker() {
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

namespace {
   struct CompareWithoutLumi {
      CompareWithoutLumi(edm::EventID const& iThis):
      m_this(iThis) {}
      bool operator()(edm::EventID const& iOther) {
         return m_this.run() == iOther.run() && m_this.event() == iOther.event();
      }
      edm::EventID m_this;
   };
   
   struct MsgToListener {
      long mtype;
      edm::EventID id;
      MsgToListener():
      mtype(MsgToListener::messageType()) {}
      static size_t sizeForBuffer() {
         return sizeof(MsgToListener)-sizeof(long);
      }
      static long messageType() { return 10;}
   };
   
   class Listener {
   public:
      Listener(std::map<edm::EventID, unsigned int>* iToFill, int iQueueID, unsigned int iMaxChildren):
      fill_(iToFill),
      queueID_(iQueueID),
      maxChildren_(iMaxChildren),
      stoppedChildren_(0){}
      
      void operator()(){
         for(;;) {
            MsgToListener rcvmsg;
            if(msgrcv(queueID_, &rcvmsg, MsgToListener::sizeForBuffer(), MsgToListener::messageType(), 0) < 0) {
               perror("failed to receive message from controller");
               exit(EXIT_FAILURE);
            }
            if(rcvmsg.id.run() == 0) {
               ++stoppedChildren_;
               if(stoppedChildren_ == maxChildren_) {
                  return;
               }
               continue;
            }
            ++((*fill_)[rcvmsg.id]);
         }
      }
      
   private:
      std::map<edm::EventID, unsigned int>* fill_;
      int queueID_;
      unsigned int maxChildren_;
      unsigned int stoppedChildren_;
   };
}

void
MulticoreRunLumiEventChecker::check(edm::EventID const& iEventID, bool iIsEvent) {
   if(mustSearch_) { 
      if(0 == numberOfEventsLeftBeforeSearch_) {
         if(iIsEvent) {
            numberOfEventsLeftBeforeSearch_ = multiProcessSequentialEvents_;
         }
         //the event must be after the last event in our list since multicore doesn't go backwards
         //std::vector<edm::EventID>::iterator itFind= std::find_if(ids_.begin() + index_, ids_.end(), CompareWithoutLumi(iEventID));
         std::vector<edm::EventID>::iterator itFind= std::find(ids_.begin() + index_, ids_.end(), iEventID);
         if(itFind == ids_.end()) {
            throw cms::Exception("MissedEvent") << "The event " << iEventID << "is not in the list.\n";
         }
         index_ = itFind-ids_.begin();
      } 
      if(iIsEvent) {
         --numberOfEventsLeftBeforeSearch_;
      }
      MsgToListener sndmsg;
      sndmsg.id = iEventID;
      errno = 0;
      int value = msgsnd(messageQueue_, &sndmsg, MsgToListener::sizeForBuffer(), 0);
      if(value != 0) {
         throw cms::Exception("MessageFailure") << "Failed to send EventID message " << strerror(errno);
      }
   }
   
   if(index_ >= ids_.size()) {
      throw cms::Exception("TooManyEvents") << "Was passes " << ids_.size() << " EventIDs but have processed more events than that\n";
   }
   if(iEventID  != ids_[index_]) {
      throw cms::Exception("WrongEvent") << "Was expecting event " << ids_[index_] << " but was given " << iEventID << "\n";
   }
   ++index_;
}

// ------------ method called to for each event  ------------
void
MulticoreRunLumiEventChecker::analyze(edm::Event const& iEvent, edm::EventSetup const&) {
   check(iEvent.id(), true);
}

void 
MulticoreRunLumiEventChecker::beginRun(edm::Run const& run, edm::EventSetup const&) {
   check(edm::EventID(run.id().run(), 0, 0), false);   
}
void 
MulticoreRunLumiEventChecker::endRun(edm::Run const& run, edm::EventSetup const&) {
   check(edm::EventID(run.id().run(), 0, 0), false);   
}

void 
MulticoreRunLumiEventChecker::beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const&) {
   check(edm::EventID(lumi.id().run(), lumi.id().luminosityBlock(), 0), false);   
}

void 
MulticoreRunLumiEventChecker::endLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const&) {
   check(edm::EventID(lumi.id().run(), lumi.id().luminosityBlock(), 0), false);   
}


// ------------ method called once each job just before starting event loop  ------------
void
MulticoreRunLumiEventChecker::beginJob() {
}

// ------------ method called once each job just after ending the event loop  ------------
void
MulticoreRunLumiEventChecker::endJob() {
   if(mustSearch_) {
      MsgToListener sndmsg;
      sndmsg.id = edm::EventID();
      errno = 0;
      int value = msgsnd(messageQueue_,&sndmsg, MsgToListener::sizeForBuffer(),0);
      if(value != 0) {
        throw cms::Exception("MessageFailure")<<"Failed to send finished message "<<strerror(errno)<<"\n";
      }
   } else {
     if(index_ != ids_.size()) {
         throw cms::Exception("WrongNumberOfEvents")<<"Saw "<<index_<<" events but was supposed to see "<<ids_.size()<<"\n";
     }
   }
   
   if(listenerThread_) {
      listenerThread_->join();
      msgctl(messageQueue_, IPC_RMID, 0);
      
      std::set<edm::EventID> uniqueIDs(ids_.begin(), ids_.end());
      if(seenIDs_.size() != uniqueIDs.size()) {
         throw cms::Exception("WrongNumberOfEvents") << "Saw " << seenIDs_.size() << " events but was supposed to see " << ids_.size() << "\n";
      }
      
      std::set<edm::EventID> duplicates;
      for(std::map<edm::EventID, unsigned int>::iterator it = seenIDs_.begin(), itEnd = seenIDs_.end();
          it != itEnd;
          ++it) {
         if(it->second > 1 && it->first.event() != 0) {
            duplicates.insert(it->first);
         }
      }
      if(duplicates.size() != 0) {
         throw cms::Exception("DuplicateEvents") << "saw " << duplicates.size() << " events\n";
      }
   }
}

// ------------ method called once each job for validation
void
MulticoreRunLumiEventChecker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::vector<edm::EventID> >("eventSequence");
  desc.addUntracked<unsigned int>("multiProcessSequentialEvents", 0U);
  descriptions.add("eventIDChecker", desc);
}

void 
MulticoreRunLumiEventChecker::preForkReleaseResources() {
  //This queue is used to communicate between children
  messageQueue_ = msgget(IPC_PRIVATE, IPC_CREAT|0660);
  if(-1 == messageQueue_) {
    throw cms::Exception("FailedToCreateQueue")<<" call to 'msgget' failed to create a message queue. errno: "<<errno<<" "<<strerror(errno);
  }
}

void
MulticoreRunLumiEventChecker::postForkReacquireResources(unsigned int iChildIndex, unsigned int iNumberOfChildren) {
   mustSearch_ = true;
   
   if(0 == iChildIndex) {
      //NOTE: must temporarily disable signals so the new thread never tries to process a signal
      sigset_t oldset;
      edm::disableAllSigs(&oldset);
      
      Listener listener(&seenIDs_, messageQueue_, iNumberOfChildren);
      listenerThread_ = boost::shared_ptr<boost::thread>(new boost::thread(listener)) ;
      edm::reenableSigs(&oldset);
   }
}

//define this as a plug-in
DEFINE_FWK_MODULE(MulticoreRunLumiEventChecker);
