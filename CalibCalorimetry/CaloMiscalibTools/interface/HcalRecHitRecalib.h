#ifndef _HCALRECHITRECALIB_H
#define _HCALRECHITRECALIB_H

// -*- C++ -*-
//
// Package:    HcalRecHitRecalib
// Class:      HcalRecHitRecalib
// 
/**\class HcalRecHitRecalib HcalRecHitRecalib.cc CalibCalorimetry/CaloRecalibTools.src/HcalRecHitRecalib.cc

 Description: Producer to miscalibrate (calibrated) Hcal RecHit 

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Luca Malgeri
//         Created:  $Date: 2006/11/21 16:47:46 $
// $Id: HcalRecHitRecalib.h,v 1.3 2006/11/21 16:47:46 malgeri Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/CaloMiscalibMapHcal.h"

//
// class decleration
//

class HcalRecHitRecalib : public edm::EDProducer {
   public:
      explicit HcalRecHitRecalib(const edm::ParameterSet&);
      ~HcalRecHitRecalib();


      virtual void produce(edm::Event &, const edm::EventSetup&);

   private:
      // ----------member data ---------------------------

  //  edm::InputTag hbheLabel_,hoLabel_,hfLabel_;
  //  std::string HBHEHitsProducer_;
  //  std::string HFHitsProducer_;
  //  std::string HOHitsProducer_;
  //  std::string HBHEHits_;
  //  std::string HFHits_;
  //  std::string HOHits_;

  edm::InputTag hbheLabel_,hoLabel_,hfLabel_;
  std::string RecalibHBHEHits_;
  std::string RecalibHFHits_;
  std::string RecalibHOHits_;

  std::string hcalfile_;
  std::string hcalfileinpath_;

  CaloMiscalibMapHcal mapHcal_;
  double refactor_;
  double refactor_mean_;

};
#endif
