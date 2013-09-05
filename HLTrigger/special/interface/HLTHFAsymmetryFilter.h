#ifndef _HLTHFAsymmetryFilter_H
#define _HLTHFAsymetryFilter_H


///////////////////////////////////////////////////////
//
// HLTHFAsymetryFilter
//
// Filter definition
//
// We perform a selection on HF energy repartition 
//
// This filter is primarily used to select Beamgas (aka PKAM) events
// 
// An asymmetry parameter, based on the pixel clusters, is computed as follows
// 
//  asym1 = E_HF-/(E_HF- + E_HF+) for beam1
//  asym2 = E_HF+/(E_HF- + E_HF+) for beam2 
//
// where E_HF is the total energy of clusters passing a certain threshold (given by eCut_HF_)
//
//  Usually for PKAM events, asym1 is close to 1. for B1 BGas events, and close to 0 for B2 BGAS events  
//
//
// More details:
// http://sviret.web.cern.ch/sviret/Welcome.php?n=CMS.MIB
//
// S.Viret: 12/01/2011 (viret@in2p3.fr)
//
///////////////////////////////////////////////////////


// system include files
#include <memory>

// user include files
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

//
// class decleration
//

class HLTHFAsymmetryFilter : public edm::EDFilter {
   public:
      explicit HLTHFAsymmetryFilter(const edm::ParameterSet&);
      ~HLTHFAsymmetryFilter();

      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      virtual bool filter(edm::Event &, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------

 edm::EDGetTokenT<HFRecHitCollection> HFHitsToken_;
 edm::InputTag HFHits_;
 double eCut_HF_;
 double os_asym_;
 double ss_asym_;

};

#endif
