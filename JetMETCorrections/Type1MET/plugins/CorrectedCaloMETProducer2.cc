// -*- C++ -*-
// $Id: CorrectedCaloMETProducer2.cc,v 1.1 2013/01/15 06:48:55 sakuma Exp $

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
class CorrectedCaloMETProducer2 : public edm::EDProducer  
{

public:

  explicit CorrectedCaloMETProducer2(const edm::ParameterSet& cfg)
    : src_(cfg.getParameter<edm::InputTag>("src")),
      srcCorrections_(cfg.getParameter<std::vector<edm::InputTag> >("srcCorrections"))
  {
    produces<METCollection>("");
  }

  ~CorrectedCaloMETProducer2() { }
    
private:

  typedef std::vector<reco::CaloMET> METCollection;

  edm::InputTag src_;
  std::vector<edm::InputTag> srcCorrections_;

  void produce(edm::Event& evt, const edm::EventSetup& es)
  {
    edm::Handle<METCollection> srcMETCollection;
    evt.getByLabel(src_, srcMETCollection);

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
    for (std::vector<edm::InputTag>::const_iterator inputTag = srcCorrections_.begin(); inputTag != srcCorrections_.end(); ++inputTag)
      {
	evt.getByLabel(*inputTag, corr);
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

DEFINE_FWK_MODULE(CorrectedCaloMETProducer2);
