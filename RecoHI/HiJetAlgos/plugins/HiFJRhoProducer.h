#ifndef HiJetBackground_HiFJRhoProducer_h
#define HiJetBackground_HiFJRhoProducer_h

// system include files
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// user include files
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/JetReco/interface/Jet.h"

//
// class declaration
//

class HiFJRhoProducer : public edm::EDProducer {
   public:
      explicit HiFJRhoProducer(const edm::ParameterSet&);
      ~HiFJRhoProducer();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void beginJob() override;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override;
      
      double calcMd(const reco::Jet *jet);
      bool   isPackedCandidate(const reco::Candidate* candidate);
      
      // ----------member data ---------------------------
      //input
      edm::EDGetTokenT<edm::View<reco::Jet>>    jetsToken_;
      
      //members
      unsigned int   nExcl_;              //Number of leading jets to exclude
      double         etaMaxExcl_;         //max eta for jets to exclude
      double         ptMinExcl_;          //min pt for excluded jets
      unsigned int   nExcl2_;             //Number of leading jets to exclude in 2nd eta region
      double         etaMaxExcl2_;        //max eta for jets to exclude in 2nd eta region
      double         ptMinExcl2_;         //min pt for excluded jets in 2nd eta region
      std::vector<double>         etaRanges;         //eta boundaries for rho calculation regions
      bool           checkJetCand, usingPackedCand;
};

#endif
