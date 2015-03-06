#ifndef HLTRHemisphere_h
#define HLTRHemisphere_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"

namespace edm {
   class ConfigurationDescriptions;
}

//
// class declaration
//

class HLTRHemisphere : public edm::stream::EDFilter<> {

   public:


      explicit HLTRHemisphere(const edm::ParameterSet&);
      ~HLTRHemisphere();
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      edm::EDGetTokenT<edm::View<reco::Jet>> m_theJetToken;
      edm::EDGetTokenT<std::vector<reco::RecoChargedCandidate>> m_theMuonToken;
      edm::InputTag inputTag_; // input tag identifying product
      edm::InputTag muonTag_;  // input tag for the muon objects 
      bool doMuonCorrection_;   // do the muon corrections
      double muonEta_;         // maximum muon eta
      double min_Jet_Pt_;      // minimum jet pT threshold for collection
      double max_Eta_;         // maximum eta
      int max_NJ_;             // don't calculate R if event has more than NJ jets
      bool accNJJets_;         // accept or reject events with high NJ

      void ComputeHemispheres(std::auto_ptr<std::vector<math::XYZTLorentzVector> >& hlist, const std::vector<math::XYZTLorentzVector>& JETS, std::vector<math::XYZTLorentzVector> *extraJets=0);
};

#endif //HLTRHemisphere_h
