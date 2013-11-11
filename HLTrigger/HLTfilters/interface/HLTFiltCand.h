#ifndef HLTFiltCand_h
#define HLTFiltCand_h

/** \class HLTFiltCand
 *
 *
 *  This class is an HLTFilter (-> EDFilter) implementing a very basic
 *  HLT trigger acting on candidates, requiring a g/e/m/j tuple above
 *  pt cuts
 *
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "DataFormats/Candidate/interface/CompositeCandidateFwd.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETFwd.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METFwd.h"

//
// class declaration
//

class HLTFiltCand : public HLTFilter {

   public:
      explicit HLTFiltCand(const edm::ParameterSet&);
      ~HLTFiltCand();
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;

   private:
      edm::InputTag photTag_;  // input tag identifying product containing photons
      edm::InputTag elecTag_;  // input tag identifying product containing electrons
      edm::InputTag muonTag_;  // input tag identifying product containing muons
      edm::InputTag tausTag_;  // input tag identifying product containing taus
      edm::InputTag jetsTag_;  // input tag identifying product containing jets
      edm::InputTag metsTag_;  // input tag identifying product containing METs
      edm::InputTag mhtsTag_;  // input tag identifying product containing HTs
      edm::InputTag trckTag_;  // input tag identifying product containing Tracks
      edm::InputTag ecalTag_;  // input tag identifying product containing SuperClusters

      edm::EDGetTokenT<reco::RecoEcalCandidateCollection>    photToken_;  // token identifying product containing photons
      edm::EDGetTokenT<reco::ElectronCollection>             elecToken_;  // token identifying product containing electrons
      edm::EDGetTokenT<reco::RecoChargedCandidateCollection> muonToken_;  // token identifying product containing muons
      edm::EDGetTokenT<reco::CaloJetCollection>              tausToken_;  // token identifying product containing taus
      edm::EDGetTokenT<reco::CaloJetCollection>              jetsToken_;  // token identifying product containing jets
      edm::EDGetTokenT<reco::CaloMETCollection>              metsToken_;  // token identifying product containing METs
      edm::EDGetTokenT<reco::METCollection>                  mhtsToken_;  // token identifying product containing HTs
      edm::EDGetTokenT<reco::RecoChargedCandidateCollection> trckToken_;  // token identifying product containing Tracks
      edm::EDGetTokenT<reco::RecoEcalCandidateCollection>    ecalToken_;  // token identifying product containing SuperClusters

      double min_Pt_;          // min pt cut
};

#endif //HLTFiltCand_h
