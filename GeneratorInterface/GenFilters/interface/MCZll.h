#ifndef MCZll_h
#define MCZll_h
// -*- C++ -*-
//
// Package:    MCZll
// Class:      MCZll
// 
/* 

 Description: filter events based on the Pythia ProcessID and the Pt_hat

 Implementation: inherits from generic EDFilter
     
*/
//
// Original Author:  Paolo Meridiani
// $Id: MCZll.h,v 1.2 2010/07/21 04:23:24 wmtan Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


//
// class decleration
//

class MCZll : public edm::EDFilter {
   public:
      explicit MCZll(const edm::ParameterSet&);
      ~MCZll();
      virtual void endJob() ;

      virtual bool filter(edm::Event&, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------
      std::string label_;
      int leptonFlavour_;
      double leptonPtMin_;
      double leptonPtMax_;
      double leptonEtaMin_;
      double leptonEtaMax_;
      std::pair<double,double> zMassRange_;
      unsigned int  nEvents_;
      unsigned int nAccepted_;
      bool filter_;
};
#endif
