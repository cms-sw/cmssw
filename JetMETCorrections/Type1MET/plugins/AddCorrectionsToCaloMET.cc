// -*- C++ -*-

//____________________________________________________________________________||
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/METReco/interface/CaloMET.h"

#include "DataFormats/METReco/interface/CorrMETData.h"

#include <vector>

//____________________________________________________________________________||
class AddCorrectionsToCaloMET : public edm::EDProducer
{

public:

  explicit AddCorrectionsToCaloMET(const edm::ParameterSet& cfg)
    : token_(consumes<METCollection>(cfg.getParameter<edm::InputTag>("src")))
  {

  std::vector<edm::InputTag> corrInputTags = cfg.getParameter<std::vector<edm::InputTag> >("srcCorrections");
  for (std::vector<edm::InputTag>::const_iterator inputTag = corrInputTags.begin(); inputTag != corrInputTags.end(); ++inputTag)
      {
	corrTokens_.push_back(consumes<CorrMETData>(*inputTag));
      }

    produces<METCollection>("");
  }

  ~AddCorrectionsToCaloMET() { }

private:

  typedef std::vector<reco::CaloMET> METCollection;

  edm::EDGetTokenT<METCollection> token_;
  std::vector<edm::EDGetTokenT<CorrMETData> > corrTokens_;

  void produce(edm::Event& evt, const edm::EventSetup& es) override
  {
    edm::Handle<METCollection> srcMETCollection;
    evt.getByToken(token_, srcMETCollection);

    const reco::CaloMET& srcMET = (*srcMETCollection)[0];

    CorrMETData correction = readAndSumCorrections(evt, es);

    reco::CaloMET outMET = applyCorrection(srcMET, correction);

    std::auto_ptr<METCollection> product(new METCollection);
    product->push_back(outMET);
    evt.put(product);
  }

  CorrMETData readAndSumCorrections(edm::Event& evt, const edm::EventSetup& es)
  {
    CorrMETData ret;

    edm::Handle<CorrMETData> corr;
    for (std::vector<edm::EDGetTokenT<CorrMETData> >::const_iterator corrToken = corrTokens_.begin(); corrToken != corrTokens_.end(); ++corrToken)
      {
	evt.getByToken(*corrToken, corr);
	ret += (*corr);
      }


    return ret;
  }

  reco::CaloMET applyCorrection(const reco::CaloMET& srcMET, const CorrMETData& correction)
  {
    std::vector<CorrMETData> corrections = srcMET.mEtCorr();
    corrections.push_back(correction);
    return reco::CaloMET(srcMET.getSpecific(), srcMET.sumEt() + correction.sumet, corrections, constructP4From(srcMET, correction), srcMET.vertex());
  }

  reco::Candidate::LorentzVector constructP4From(const reco::CaloMET& met, const CorrMETData& correction)
  {
    double px = met.px() + correction.mex;
    double py = met.py() + correction.mey;
    double pt = sqrt(px*px + py*py);
    return reco::Candidate::LorentzVector(px, py, 0., pt);
  }
};

//____________________________________________________________________________||

DEFINE_FWK_MODULE(AddCorrectionsToCaloMET);
