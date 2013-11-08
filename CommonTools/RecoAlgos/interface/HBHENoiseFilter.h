#ifndef __HBHE_NOISE_FILTER_H__
#define __HBHE_NOISE_FILTER_H__


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/METReco/interface/HcalNoiseSummary.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"

//
// class declaration
//

class HBHENoiseFilter : public edm::EDFilter {
   public:
      explicit HBHENoiseFilter(const edm::ParameterSet&);
      ~HBHENoiseFilter();

   private:
      virtual bool filter(edm::Event&, const edm::EventSetup&) override;

      // ----------member data ---------------------------

      // parameters
      edm::EDGetTokenT<HcalNoiseSummary> noisetoken_;
      double minRatio_, maxRatio_;
      int minHPDHits_, minRBXHits_, minHPDNoOtherHits_;
      int minZeros_;
      double minHighEHitTime_, maxHighEHitTime_;
      double maxRBXEMF_;
      int minNumIsolatedNoiseChannels_;
      double minIsolatedNoiseSumE_, minIsolatedNoiseSumEt_;
      bool useTS4TS5_;

      bool IgnoreTS4TS5ifJetInLowBVRegion_;
      edm::InputTag jetlabel_;
      edm::EDGetTokenT<reco::PFJetCollection> jettoken_;
      int maxjetindex_;
      double maxNHF_;
};

#endif
