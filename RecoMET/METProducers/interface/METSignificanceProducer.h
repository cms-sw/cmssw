// -*- C++ -*-
//
// Package:    METProducers
// Class:      METSignificanceProducer
//
/**\class METSignificanceProducer

 Description: An EDProducer for CaloMET

 Implementation:

*/
//
//
//

//____________________________________________________________________________||
#ifndef METSignificanceProducer_h
#define METSignificanceProducer_h

//____________________________________________________________________________||
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METFwd.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETFwd.h"
#include "DataFormats/METReco/interface/CommonMETData.h"

#include "RecoMET/METAlgorithms/interface/METSignificance.h"

#include <string>

//____________________________________________________________________________||
namespace cms {
  class METSignificanceProducer : public edm::stream::EDProducer<> {
  public:
    explicit METSignificanceProducer(const edm::ParameterSet&);
    ~METSignificanceProducer() override;
    void produce(edm::Event&, const edm::EventSetup&) override;

  private:
    // ----------member data ---------------------------

    edm::EDGetTokenT<edm::View<reco::Jet>> pfjetsToken_;
    edm::EDGetTokenT<edm::View<reco::MET>> metToken_;
    edm::EDGetTokenT<edm::View<reco::Candidate>> pfCandidatesToken_;
    std::vector<edm::EDGetTokenT<edm::View<reco::Candidate>>> lepTokens_;
    edm::EDGetTokenT<double> rhoToken_;
    std::string jetSFType_;
    std::string jetResPtType_;
    std::string jetResPhiType_;
    edm::EDGetTokenT<edm::ValueMap<float>> weightsToken_;

    metsig::METSignificance* metSigAlgo_;
  };
}  // namespace cms

//____________________________________________________________________________||
#endif  // METSignificanceProducer_h
