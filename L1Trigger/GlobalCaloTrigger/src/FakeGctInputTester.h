
// -*- C++ -*-
//
// Package:    FakeGctInputTester
// Class:      FakeGctInputTester
// 
/**\class FakeGctInputTester FakeGctInputTester.h L1Trigger/GlobalCalotrigger/src/FakeGctInputTester.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jim Brooke
//         Created:  Tue Nov 21 14:49:14 CET 2006
// $Id$
//
//



#ifndef FAKEGCTINPUTTESTER_H
#define FAKEGCTINPUTTESTER_H


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

class FakeGctInputTester : public edm::EDAnalyzer {
   public:
      explicit FakeGctInputTester(const edm::ParameterSet&);
      ~FakeGctInputTester();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------
};

#endif
