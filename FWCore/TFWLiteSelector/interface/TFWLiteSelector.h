#ifndef FWLite_TFWLiteSelector_h
#define FWLite_TFWLiteSelector_h
// -*- C++ -*-
//
// Package:     FWLite
// Class  :     TFWLiteSelector
// 
/**\class TFWLiteSelector TFWLiteSelector.h FWCore/FWLite/interface/TFWLiteSelector.h

 Description: A ROOT TSelector which accesses data using an edm::Event

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Tue Jun 27 16:37:27 EDT 2006
// $Id$
//

// system include files
#include "TSelector.h"

// user include files

// forward declarations
namespace edm {
  class Event;
  
  namespace root {
    struct TFWLiteSelectorMembers;
  };
}

class TFWLiteSelector : public TSelector
{

   public:
      TFWLiteSelector();
      virtual ~TFWLiteSelector();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      virtual void begin() = 0;
      virtual void preProcessing(TList& out) = 0;
      virtual void process(const edm::Event&) = 0;
      virtual void postProcessing() =0;
      virtual void terminate(TList& in) = 0;
      
   private:
      TFWLiteSelector(const TFWLiteSelector&); // stop default

      const TFWLiteSelector& operator=(const TFWLiteSelector&); // stop default

      virtual void        Begin(TTree *) ;
      virtual void        SlaveBegin(TTree *);
      virtual void        Init(TTree*);
      virtual Bool_t      Notify() ;
      virtual Bool_t      Process(Long64_t /*entry*/) ;
      virtual void        SlaveTerminate();
      virtual void        Terminate();
      virtual Int_t Version() const { return 1; }
      
      // ---------- member data --------------------------------
      edm::root::TFWLiteSelectorMembers* m_;
      bool everythingOK_;
};


#endif
