// -*- C++ -*-
//
// Package:     FWLite
// Class  :     ChainEvent
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Sat Jun 16 06:48:39 EDT 2007
//

// system include files

// user include files
#include "DataFormats/FWLite/interface/ChainEvent.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "TFile.h"
#include "TTree.h"
#include "TROOT.h"

namespace fwlite {
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
ChainEvent::ChainEvent(const std::vector<std::string>& iFileNames):
  fileNames_(),
  file_(),
  event_(),
  eventIndex_(0),
  accumulatedSize_()
{
  Long64_t summedSize=0;
  accumulatedSize_.reserve(iFileNames.size()+1);
  fileNames_.reserve(iFileNames.size());
    
  for (std::vector<std::string>::const_iterator it= iFileNames.begin(), itEnd = iFileNames.end();
      it!=itEnd;
      ++it) {
    TFile *tfilePtr = TFile::Open(it->c_str());
    file_ = boost::shared_ptr<TFile>(tfilePtr);
    gROOT->GetListOfFiles()->Remove(tfilePtr);
    TTree* tree = dynamic_cast<TTree*>(file_->Get(edm::poolNames::eventTreeName().c_str()));
    if (0 == tree) {
      throw cms::Exception("NotEdmFile")<<"The file "<<*it<<" has no 'Events' TTree and therefore is not an EDM ROOT file";
    }
    Long64_t nEvents = tree->GetEntries();
    if (nEvents > 0) { // skip empty files
      fileNames_.push_back(*it);
      // accumulatedSize_ is the entry # at the beginning of this file
      accumulatedSize_.push_back(summedSize);
      summedSize += nEvents;
    }
  }
  // total accumulated size (last enry + 1) at the end of last file
  accumulatedSize_.push_back(summedSize);

  if (fileNames_.size() > 0)
    switchToFile(0);
}

// ChainEvent::ChainEvent(const ChainEvent& rhs)
// {
//    // do actual copying here;
// }

ChainEvent::~ChainEvent()
{
}

//
// assignment operators
//
// const ChainEvent& ChainEvent::operator=(const ChainEvent& rhs)
// {
//   //An exception safe implementation is
//   ChainEvent temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

const ChainEvent&
ChainEvent::operator++()
{
   if(eventIndex_ != static_cast<Long64_t>(fileNames_.size())-1)
   {
      ++(*event_);
      if(event_->atEnd()) {
         switchToFile(++eventIndex_);
      }
   } else {
      if(*event_) {
         ++(*event_);
      }
   }
   return *this;
}

///Go to the event at index iIndex
bool
ChainEvent::to(Long64_t iIndex)
{
   if (iIndex >= accumulatedSize_.back())
   {
      // if we're here, then iIndex was not valid
      return false;
   }

   Long64_t offsetIndex = eventIndex_;

   // we're going backwards, so start from the beginning
   if (iIndex < accumulatedSize_[offsetIndex]) {
      offsetIndex = 0;
   }

   // is it past the end of this file?
   while (iIndex >= accumulatedSize_[offsetIndex+1]) {
      ++offsetIndex;
   }

   if(offsetIndex != eventIndex_) {
      switchToFile(eventIndex_ = offsetIndex);
   }

   // adjust to entry # in this file
   return event_->to( iIndex-accumulatedSize_[offsetIndex] );
}


///Go to event with event id "id"
bool
ChainEvent::to(const edm::EventID &id)
{
  return to(id.run(), id.luminosityBlock(), id.event());
}

///If lumi is non-zero, go to event with given run, lumi, and event number
///If lumi is zero, go to event with given run and event number
bool
ChainEvent::to(edm::RunNumber_t run, edm::LuminosityBlockNumber_t lumi, edm::EventNumber_t event)
{

   // First try this file
   if ( event_->to( run, lumi, event ) )
   {
      // found it, return
      return true;
   }
   else
   {
      // Did not find it, try the other files sequentially.
      // Someday I can make this smarter. For now... we get something working.
      Long64_t thisFile = eventIndex_;
      std::vector<std::string>::const_iterator filesBegin = fileNames_.begin(),
         filesEnd = fileNames_.end(), ifile = filesBegin;
      for ( ; ifile != filesEnd; ++ifile )
      {
         // skip the "first" file that we tried
         if ( ifile - filesBegin != thisFile )
         {
            // switch to the next file
            switchToFile( ifile - filesBegin );
            // check that tree for the desired event
            if ( event_->to( run, lumi, event ) )
            {
               // found it, return
               return true;
            }
         }// end ignore "first" file that we tried
      }// end loop over files

      // did not find the event with id "id".
      return false;
   }// end if we did not find event id in "first" file
}

bool
ChainEvent::to(edm::RunNumber_t run, edm::EventNumber_t event)
{
  return to(run, 0U, event);
}

/** Go to the very first Event*/

const ChainEvent&
ChainEvent::toBegin()
{
   if (eventIndex_ != 0)
   {
      switchToFile(0);
   }
   event_->toBegin();
   return *this;
}

void
ChainEvent::switchToFile(Long64_t iIndex)
{
  eventIndex_= iIndex;
  TFile *tfilePtr = TFile::Open(fileNames_[iIndex].c_str());
  file_ = boost::shared_ptr<TFile>(tfilePtr);
  gROOT->GetListOfFiles()->Remove(tfilePtr);
  event_ = boost::shared_ptr<Event>( new Event(file_.get()));
}

//
// const member functions
//
const std::string
ChainEvent::getBranchNameFor(const std::type_info& iType,
                             const char* iModule,
                             const char* iInstance,
                             const char* iProcess) const
{
  return event_->getBranchNameFor(iType,iModule,iInstance,iProcess);
}

const std::vector<edm::BranchDescription>&
ChainEvent::getBranchDescriptions() const
{
  return event_->getBranchDescriptions();
}

const std::vector<std::string>&
ChainEvent::getProcessHistory() const
{
  return event_->getProcessHistory();
}

edm::ProcessHistory const&
ChainEvent::processHistory() const
{
  return event_->processHistory();
}

edm::EventAuxiliary const&
ChainEvent::eventAuxiliary() const
{
   return event_->eventAuxiliary();
}

fwlite::LuminosityBlock const& ChainEvent::getLuminosityBlock()
{
   return event_->getLuminosityBlock();
}

fwlite::Run const& ChainEvent::getRun()
{
   return event_->getRun();
}


bool
ChainEvent::getByLabel(const std::type_info& iType,
                       const char* iModule,
                       const char* iInstance,
                       const char* iProcess,
                       void* iValue) const
{
  return event_->getByLabel(iType,iModule,iInstance,iProcess,iValue);
}

edm::EDProduct const* ChainEvent::getByProductID(edm::ProductID const& iID) const
{
  return event_->getByProductID( iID );
}

bool
ChainEvent::isValid() const
{
  return event_->isValid();
}
ChainEvent::operator bool() const
{
  return *event_;
}

bool
ChainEvent::atEnd() const
{
  if (eventIndex_ == static_cast<Long64_t>(fileNames_.size())-1) {
    return event_->atEnd();
  }
  return false;
}

Long64_t
ChainEvent::size() const
{
  return accumulatedSize_.back();
}

edm::TriggerNames const&
ChainEvent::triggerNames(edm::TriggerResults const& triggerResults) const
{
  return event_->triggerNames(triggerResults);
}

void
ChainEvent::fillParameterSetRegistry() const
{
  event_->fillParameterSetRegistry();
}

edm::TriggerResultsByName
ChainEvent::triggerResultsByName(std::string const& process) const {
  return event_->triggerResultsByName(process);
}

//
// static member functions
//
void
ChainEvent::throwProductNotFoundException(const std::type_info& iType,
                                          const char* iModule,
                                          const char* iInstance,
                                          const char* iProcess) {
  Event::throwProductNotFoundException(iType,iModule,iInstance,iProcess);
}
}
