#ifndef JetProducers_QGTagger_h
#define JetProducers_QGTagger_h
#include <tuple>

#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"
#include "RecoJets/JetAlgorithms/interface/QGLikelihoodCalculator.h"
#include "CondFormats/DataRecord/interface/QGLikelihoodRcd.h"
#include "CondFormats/DataRecord/interface/QGLikelihoodSystematicsRcd.h"

class QGTagger : public edm::global::EDProducer<> {
public:
  explicit QGTagger(const edm::ParameterSet&);
  ~QGTagger() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  std::tuple<int, float, float> calcVariables(const reco::Jet*,
                                              edm::Handle<reco::VertexCollection>&,
                                              edm::ValueMap<float>&,
                                              bool) const;
  template <typename T>
  void putInEvent(const std::string&,
                  const edm::Handle<edm::View<reco::Jet>>&,
                  const std::vector<T>&,
                  edm::Event&) const;
  bool isPackedCandidate(const reco::Jet* jet) const;

  edm::EDGetTokenT<edm::View<reco::Jet>> jetsToken;
  edm::EDGetTokenT<reco::JetCorrector> jetCorrectorToken;
  edm::EDGetTokenT<reco::VertexCollection> vertexToken;
  edm::EDGetTokenT<double> rhoToken;
  const bool computeLikelihood;
  edm::ESGetToken<QGLikelihoodObject, QGLikelihoodRcd> paramsToken;
  edm::ESGetToken<QGLikelihoodSystematicsObject, QGLikelihoodSystematicsRcd> systToken;
  const bool useQC, useJetCorr, produceSyst;
  bool applyConstituentWeight;
  edm::EDGetTokenT<edm::ValueMap<float>> constituentWeightsToken;
  QGLikelihoodCalculator qgLikelihood;
};

#endif
