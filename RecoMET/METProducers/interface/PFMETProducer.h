// -*- C++ -*-
//
// Package:    METProducers
// Class:      PFMETProducer
//
/**\class PFMETProducer

 Description: An EDProducer for CaloMET

 Implementation:

*/
//
//
//

//____________________________________________________________________________||
#ifndef PFMETProducer_h
#define PFMETProducer_h

//____________________________________________________________________________||
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/METReco/interface/METFwd.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETFwd.h"
#include "DataFormats/METReco/interface/CommonMETData.h"

#include "RecoMET/METAlgorithms/interface/METAlgo.h"
#include "RecoMET/METAlgorithms/interface/SignAlgoResolutions.h"
#include "RecoMET/METAlgorithms/interface/PFSpecificAlgo.h"
#include "RecoMET/METAlgorithms/interface/SignPFSpecificAlgo.h"
#include "RecoMET/METAlgorithms/interface/METSignificance.h"

#include "CondFormats/DataRecord/interface/JetResolutionRcd.h"
#include "CondFormats/DataRecord/interface/JetResolutionScaleFactorRcd.h"
#include "JetMETCorrections/Modules/interface/JetResolution.h"

#include <string>

//____________________________________________________________________________||
namespace metsig {
  class SignAlgoResolutions;
}

//____________________________________________________________________________||
namespace cms {
  class PFMETProducer : public edm::stream::EDProducer<> {
  public:
    explicit PFMETProducer(const edm::ParameterSet&);
    ~PFMETProducer() override {}
    void produce(edm::Event&, const edm::EventSetup&) override;
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    reco::METCovMatrix getMETCovMatrix(const edm::Event& event,
                                       const edm::EventSetup&,
                                       const edm::Handle<edm::View<reco::Candidate>>& input) const;
    edm::InputTag src_;
    edm::EDGetTokenT<edm::View<reco::Candidate>> inputToken_;

    bool calculateSignificance_;
    metsig::METSignificance* metSigAlgo_;

    double globalThreshold_;
    double jetThreshold_;

    edm::EDGetTokenT<edm::View<reco::Jet>> jetToken_;
    std::vector<edm::EDGetTokenT<edm::View<reco::Candidate>>> lepTokens_;

    edm::ESGetToken<JME::JetResolutionObject, JetResolutionScaleFactorRcd> jetSFToken_;
    edm::ESGetToken<JME::JetResolutionObject, JetResolutionRcd> jetResPtToken_;
    edm::ESGetToken<JME::JetResolutionObject, JetResolutionRcd> jetResPhiToken_;
    edm::EDGetTokenT<double> rhoToken_;
    bool applyWeight_;
    edm::EDGetTokenT<edm::ValueMap<float>> weightsToken_;
    edm::ValueMap<float> const* weights_;
  };
}  // namespace cms

//____________________________________________________________________________||
#endif  // PFMETProducer_h
