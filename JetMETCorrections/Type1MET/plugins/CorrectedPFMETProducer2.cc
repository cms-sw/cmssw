// -*- C++ -*-
// $Id: CorrectedPFMETProducer2.cc,v 1.1 2013/01/15 06:49:03 sakuma Exp $

//____________________________________________________________________________||
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/METReco/interface/PFMET.h"

#include "DataFormats/METReco/interface/CorrMETData.h"

#include <vector>

//____________________________________________________________________________||
class CorrectedPFMETProducer2 : public edm::EDProducer  
{

public:

  explicit CorrectedPFMETProducer2(const edm::ParameterSet& cfg)
    : src_(cfg.getParameter<edm::InputTag>("src")),
      srcCorrections_(cfg.getParameter<std::vector<edm::InputTag> >("srcCorrections"))
  {
    produces<METCollection>("");
  }

  ~CorrectedPFMETProducer2() { }
    
private:

  typedef std::vector<reco::PFMET> METCollection;

  edm::InputTag src_;
  std::vector<edm::InputTag> srcCorrections_;

  void produce(edm::Event& evt, const edm::EventSetup& es)
  {
    edm::Handle<METCollection> srcMETCollection;
    evt.getByLabel(src_, srcMETCollection);

    const reco::PFMET& srcMET = (*srcMETCollection)[0];

    CorrMETData correction = readAndSumCorrections(evt, es);

    reco::PFMET outMET = applyCorrection(srcMET, correction);

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

  reco::PFMET applyCorrection(const reco::PFMET& srcMET, const CorrMETData& correction)
  {
    return reco::PFMET(srcMET.getSpecific(), srcMET.sumEt() + correction.sumet, constructP4From(srcMET, correction), srcMET.vertex());
  }

  reco::Candidate::LorentzVector constructP4From(const reco::PFMET& met, const CorrMETData& correction)
  {
    double px = met.px() + correction.mex;
    double py = met.py() + correction.mey;
    double pt = sqrt(px*px + py*py);
    return reco::Candidate::LorentzVector(px, py, 0., pt);
  }
};

//____________________________________________________________________________||

DEFINE_FWK_MODULE(CorrectedPFMETProducer2);

