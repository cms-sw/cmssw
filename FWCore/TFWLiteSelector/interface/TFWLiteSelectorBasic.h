#ifndef FWCore_TFWLiteSelector_TFWLiteSelectorBasic_h
#define FWCore_TFWLiteSelector_TFWLiteSelectorBasic_h
// -*- C++ -*-
//
// Package:     TFWLiteSelector
// Class  :     TFWLiteSelectorBasic
// 
/**\class TFWLiteSelectorBasic TFWLiteSelectorBasic.h FWCore/FWLite/interface/TFWLiteSelectorBasic.h

 Description: A ROOT TSelector which accesses data using an edm::Event

 Usage:
    By inheriting from this class one can make a TSelector for ROOT which works with PROOF and which 
allows you to access data using an edm::Event.

*/
//
// Original Author:  Chris Jones
//         Created:  Tue Jun 27 16:37:27 EDT 2006
//

// system include files
#include "TSelector.h"

// user include files
#include "boost/shared_ptr.hpp"

// forward declarations
class TFile;
class TList;
class TTree;

namespace edm {
  class Event;
  
  namespace root {
    struct TFWLiteSelectorMembers;
  }
}

class TFWLiteSelectorBasic : public TSelector
{

   public:
      TFWLiteSelectorBasic();
      virtual ~TFWLiteSelectorBasic();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      /**Called each time the 'client' begins processing (remote 'slaves' do not see this message)
        \param in an assignable pointer to a list of objects you want passed to 'preProcessing'. This
         list is used to communicate with remote slaves. NOTE: you are responsible for deleting this TList
         and its content once you are done with it.
        */
      virtual void begin(TList*& in) = 0;
      
      /**Called each time the 'slave' is about to start processing
        \param in a pointer to the list of objects created in 'begin()'.  The pointer can be 0
        \param out a list of objects that are the result of processing (e.g. histograms).
        You should call 'Add()' for each object you want sent to the 'terminate' method.
        */
      virtual void preProcessing(const TList* in, TList& out) = 0;
      
      /**Call each time the 'slave' gets a new Event
        \param event a standard edm::Event which works just like it does in cmsRun
        */
      virtual void process(const edm::Event& event) = 0;
      
      /**Called each time the 'slave' has seen all the events
        \param out the list of objects that will be sent to 'terminate'.
        You can Add() additional objects to 'out' at this point as well.
        */
      virtual void postProcessing(TList& out) =0;
      
      /**Called each time the 'client' has finished processing.
        \param out contains the accumulated output of all slaves.
        */
      virtual void terminate(TList& out) = 0;
      
   private:
      TFWLiteSelectorBasic(const TFWLiteSelectorBasic&); // stop default

      const TFWLiteSelectorBasic& operator=(const TFWLiteSelectorBasic&); // stop default

      virtual void        Begin(TTree *) ;
      virtual void        SlaveBegin(TTree *);
      virtual void        Init(TTree*);
      virtual Bool_t      Notify() ;
      virtual Bool_t      Process(Long64_t /*entry*/) ;
      virtual void        SlaveTerminate();
      virtual void        Terminate();
      virtual Int_t Version() const { return 1; }
      
      void setupNewFile(TFile&);
      // ---------- member data --------------------------------
      boost::shared_ptr<edm::root::TFWLiteSelectorMembers> m_;
      bool everythingOK_;
};


#endif
