#ifndef L1GCTVALIDATION_H
#define L1GCTVALIDATION_H
// -*- C++ -*-
//
// Package:    L1GlobalCaloTrigger
// Class:      L1GctValidation
// 
/**\class L1GctValidation L1GctValidation.cc L1Trigger/L1GlobalCaloTrigger/plugins/L1GctValidation.cc

 Description: produces standard plots of Gct output quantities to enable validation
              of global event quantities in particular

*/
//
// Author: Greg Heath
// Date:   February 2008
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"
#include "TH1.h"
#include "TH2.h"
//
// class declaration
//

class L1GctValidation : public edm::EDAnalyzer {
   public:
      explicit L1GctValidation(const edm::ParameterSet&);
      ~L1GctValidation();


   private:
      virtual void beginJob();
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------

      edm::InputTag m_gctinp_tag;
      edm::InputTag m_energy_tag;

      TH1F* theSumEtInLsb;
      TH1F* theSumHtInLsb;
      TH1F* theMissEtInLsb;
      TH1F* theMissHtInLsb;
      TH1F* theSumEtInGeV;
      TH1F* theSumHtInGeV;
      TH1F* theMissEtInGeV;
      TH1F* theMissEtAngle;
      TH2F* theMissEtVector;
      TH1F* theMissHtInGeV;
      TH1F* theMissHtAngle;
      TH2F* theMissHtVector;

      TH2F* theSumEtVsInputRegions;
      TH2F* theMissEtMagVsInputRegions;
      TH2F* theMissEtAngleVsInputRegions;
      TH2F* theMissHtMagVsInputRegions;

      TH2F* theMissEtVsMissHt;
      TH2F* theMissEtVsMissHtAngle;
      TH2F* theDPhiVsMissEt;
      TH2F* theDPhiVsMissHt;

      TH2F* theHtVsInternalJetsSum;
      TH2F* theMissHtVsInternalJetsSum;
      TH2F* theMissHtPhiVsInternalJetsSum;
      TH2F* theMissHxVsInternalJetsSum;
      TH2F* theMissHyVsInternalJetsSum;

      TH1F* theHfRing0EtSumPositiveEta;
      TH1F* theHfRing0EtSumNegativeEta;
      TH1F* theHfRing1EtSumPositiveEta;
      TH1F* theHfRing1EtSumNegativeEta;
      TH1F* theHfRing0CountPositiveEta;
      TH1F* theHfRing0CountNegativeEta;
      TH1F* theHfRing1CountPositiveEta;
      TH1F* theHfRing1CountNegativeEta;

};
#endif
