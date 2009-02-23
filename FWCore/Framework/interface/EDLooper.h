#ifndef FWCore_Framework_EDLooper_h
#define FWCore_Framework_EDLooper_h
// -*- C++ -*-
//
// Package:     Framework
// Module:      EDLooper
// 
/**\class EDLooper EDLooper.h package/EDLooper.h

 Description: Base class for all looping components
*/
//
// Author:      Valentin Kuznetsov
// Created:     Wed Jul  5 11:42:17 EDT 2006
// $Id: EDLooper.h,v 1.9 2009/02/23 20:12:23 chrjones Exp $
//

#include "FWCore/Framework/interface/Frameworkfwd.h"

#include <set>

namespace edm {
  namespace eventsetup {
    class EventSetupRecordKey;
    class EventSetupProvider;
  }
  class ActionTable;

  class EDLooper
  {
    public:

      enum Status {kContinue, kStop};

      EDLooper();
      virtual ~EDLooper();

      void doStartingNewLoop();
      Status doDuringLoop(edm::EventPrincipal& eventPrincipal, const edm::EventSetup& es);
      Status doEndOfLoop(const edm::EventSetup& es);
      void prepareForNextLoop(eventsetup::EventSetupProvider* esp);

      //This interface is deprecated
      virtual void beginOfJob(const edm::EventSetup&); 
      virtual void beginOfJob();
      virtual void startingNewLoop(unsigned int ) = 0; 
      virtual Status duringLoop(const edm::Event&, const edm::EventSetup&) = 0; 
      virtual Status endOfLoop(const edm::EventSetup&, unsigned int iCounter) = 0; 
      virtual void endOfJob();

      void setActionTable(ActionTable* actionTable) { act_table_ = actionTable; }

      virtual std::set<eventsetup::EventSetupRecordKey> modifyingRecords() const;

    private:

      EDLooper( const EDLooper& ); // stop default
      const EDLooper& operator=( const EDLooper& ); // stop default

      unsigned int iCounter_;
      ActionTable* act_table_;
  };
}

#endif
