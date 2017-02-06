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

#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/CorrMETData.h"

#include "JetMETCorrections/Type1MET/interface/AddCorrectionsToGenericMET.h"
#include "RecoMET/METAlgorithms/interface/METSignificance.h"

#include <vector>

//____________________________________________________________________________||
class CorrectedPATMETProducer : public edm::stream::EDProducer<>
{

public:

  explicit CorrectedPATMETProducer(const edm::ParameterSet& cfg)
    : corrector(),
      token_(consumes<METCollection>(cfg.getParameter<edm::InputTag>("src")))
  {
    std::vector<edm::InputTag> corrInputTags = cfg.getParameter<std::vector<edm::InputTag> >("srcCorrections");
    std::vector<edm::EDGetTokenT<CorrMETData> > corrTokens;
    for (std::vector<edm::InputTag>::const_iterator inputTag = corrInputTags.begin(); inputTag != corrInputTags.end(); ++inputTag) {
      corrTokens.push_back(consumes<CorrMETData>(*inputTag));
    }
    
    corrector.setCorTokens(corrTokens);

    produces<METCollection>("");
  }

  ~CorrectedPATMETProducer() { }

private:

  AddCorrectionsToGenericMET corrector;

  typedef std::vector<pat::MET> METCollection;

  edm::EDGetTokenT<METCollection> token_;
 
  void produce(edm::Event& evt, const edm::EventSetup& es) override
  {
    edm::Handle<METCollection> srcMETCollection;
    evt.getByToken(token_, srcMETCollection);

    const pat::MET& srcMET = (*srcMETCollection)[0];
    
    //dispatching to be sure we retrieve all the informations
    reco::MET corrMET = corrector.getCorrectedMET(srcMET, evt, es);
    pat::MET outMET(corrMET, srcMET);

    reco::METCovMatrix cov=srcMET.getSignificanceMatrix();
    if( !(cov(0,0)==0 && cov(0,1)==0 && cov(1,0)==0 && cov(1,1)==0) ) {
      outMET.setSignificanceMatrix(cov);
      double metSig=metsig::METSignificance::getSignificance(cov, outMET);
      outMET.setMETSignificance(metSig);
    }  

    std::auto_ptr<METCollection> product(new METCollection);
    product->push_back(outMET);
    evt.put(product);
  }

};

//____________________________________________________________________________||

DEFINE_FWK_MODULE(CorrectedPATMETProducer);
