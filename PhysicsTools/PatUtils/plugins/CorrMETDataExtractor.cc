// -*- C++ -*-

//____________________________________________________________________________||
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/METReco/interface/CorrMETData.h"

#include <vector>

//____________________________________________________________________________||
class CorrMETDataExtractor : public edm::stream::EDProducer<> {

public:

  explicit CorrMETDataExtractor(const edm::ParameterSet& cfg) {
    std::vector<edm::InputTag> corrInputTags = cfg.getParameter<std::vector<edm::InputTag> >("corrections");
    std::vector<edm::EDGetTokenT<CorrMETData> > corrTokens;
    for (std::vector<edm::InputTag>::const_iterator inputTag = corrInputTags.begin(); inputTag != corrInputTags.end(); ++inputTag) {
      corrTokens_.push_back(consumes<CorrMETData>(*inputTag));
    }

    produces<float>("corX");
    produces<float>("corY");
    produces<float>("corSumEt");
    
  }

  ~CorrMETDataExtractor() { }

private:

  std::vector<edm::EDGetTokenT<CorrMETData> > corrTokens_;
  

  void produce(edm::Event& evt, const edm::EventSetup& es) override
  {
  
    CorrMETData sumCor;
    edm::Handle<CorrMETData> corr;
    for (std::vector<edm::EDGetTokenT<CorrMETData> >::const_iterator corrToken = corrTokens_.begin(); corrToken != corrTokens_.end(); ++corrToken) {
      
      evt.getByToken(*corrToken, corr);
      sumCor += (*corr);
    }

    float cX=(float)sumCor.mex;
    float cY=(float)sumCor.mey;
    float cSEt=(float)sumCor.sumet;

    std::unique_ptr<float> corX(new float(0));
    std::unique_ptr<float> corY(new float(0));
    std::unique_ptr<float> corSumEt(new float(0));

    *corX = cX;
    *corY = cY;
    *corSumEt = cSEt;

    evt.put(std::move(corX),"corX"); 
    evt.put(std::move(corY),"corY");
    evt.put(std::move(corSumEt),"corSumEt");
  }

};

//____________________________________________________________________________||

DEFINE_FWK_MODULE(CorrMETDataExtractor);
