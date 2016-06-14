// -*- C++ -*-

//____________________________________________________________________________||
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/CorrMETData.h"

#include "JetMETCorrections/Type1MET/interface/AddCorrectionsToGenericMET.h"

#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"

#include <vector>

//____________________________________________________________________________||
class CorrectedPFMETProducer : public edm::stream::EDProducer<>
{

public:

  explicit CorrectedPFMETProducer(const edm::ParameterSet& cfg)
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

  ~CorrectedPFMETProducer() { }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("src",edm::InputTag("pfMet", ""));
    std::vector<edm::InputTag> tmpv;
    tmpv.push_back( edm::InputTag("corrPfMetType1", "type1") );
    desc.add<std::vector<edm::InputTag> >("srcCorrections",tmpv);
    descriptions.add(defaultModuleLabel<CorrectedPFMETProducer>(),desc);
  }

private:

  AddCorrectionsToGenericMET corrector;

  typedef std::vector<reco::PFMET> METCollection;

  edm::EDGetTokenT<METCollection> token_;
 

  void produce(edm::Event& evt, const edm::EventSetup& es) override
  {
    edm::Handle<METCollection> srcMETCollection;
    evt.getByToken(token_, srcMETCollection);

    const reco::PFMET& srcMET = (*srcMETCollection)[0];
        
    reco::PFMET outMET= corrector.getCorrectedPFMET(srcMET, evt, es);
    
    std::auto_ptr<METCollection> product(new METCollection);
    product->push_back(outMET);
    evt.put(product);
  }

};

//____________________________________________________________________________||

DEFINE_FWK_MODULE(CorrectedPFMETProducer);
