#include <string>
#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/Common/interface/RefToPtr.h" 
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "DataFormats/Candidate/interface/ShallowCloneCandidate.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/Math/interface/Point3D.h"

namespace pat {
  class BadPFCandidateJetsEEnoiseProducer : public edm::global::EDProducer<>{
  public:
    explicit BadPFCandidateJetsEEnoiseProducer(const edm::ParameterSet&);
    ~BadPFCandidateJetsEEnoiseProducer() override;
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
    void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  private:
    edm::EDGetTokenT<edm::View<pat::Jet> > jetsrc_;
    double ptThreshold_;
    double minEtaThreshold_;
    double maxEtaThreshold_;
    bool userawPt_;
  };
}


pat::BadPFCandidateJetsEEnoiseProducer::BadPFCandidateJetsEEnoiseProducer(const edm::ParameterSet& iConfig) :
  jetsrc_(consumes<edm::View<pat::Jet> >(iConfig.getParameter<edm::InputTag>("jetsrc") )),
  ptThreshold_(iConfig.getParameter<double>("ptThreshold")),
  minEtaThreshold_(iConfig.getParameter<double>("minEtaThreshold")),
  maxEtaThreshold_(iConfig.getParameter<double>("maxEtaThreshold")),
  userawPt_(iConfig.getParameter<bool>("userawPt"))
{
  
  produces<std::vector<pat::Jet>>("good");
  produces<std::vector<pat::Jet>>("bad");
  
}

pat::BadPFCandidateJetsEEnoiseProducer::~BadPFCandidateJetsEEnoiseProducer() {}

void pat::BadPFCandidateJetsEEnoiseProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  
  auto goodJets = std::make_unique<std::vector<pat::Jet>>();
  auto badJets = std::make_unique<std::vector<pat::Jet>>();
  
  edm::Handle<edm::View<pat::Jet> > jetcandidates;
  iEvent.getByToken(jetsrc_, jetcandidates);
  
  int njets   = jetcandidates->size();
 
  // find the bad jets
  for (int jetindex = 0; jetindex < njets; ++jetindex){
    edm::Ptr<pat::Jet> candjet = jetcandidates->ptrAt(jetindex);

    // Corrected Pt or Uncorrected Pt (It is defined from cfi file)
    double ptJet = userawPt_ ? candjet->correctedJet("Uncorrected").pt() : candjet->pt();
    double absEtaJet = std::abs(candjet->eta());
    
    if ( ptJet > ptThreshold_ || absEtaJet < minEtaThreshold_ || absEtaJet > maxEtaThreshold_) {
      // save good jets
      goodJets->emplace_back(candjet);
    }
    else {
      // save bad jets
      badJets->emplace_back(candjet);
    }
    
  }
  iEvent.put(std::move(goodJets),"good");
  iEvent.put(std::move(badJets),"bad");
  
}

void pat::BadPFCandidateJetsEEnoiseProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("jetsrc",edm::InputTag("slimmedJets"));
  desc.add<bool>("userawPt",true);
  desc.add<double>("ptThreshold",50.0);
  desc.add<double>("minEtaThreshold",2.65);
  desc.add<double>("maxEtaThreshold",3.139);

  descriptions.add("BadPFCandidateJetsEEnoiseProducer",desc);
}

using pat::BadPFCandidateJetsEEnoiseProducer;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(BadPFCandidateJetsEEnoiseProducer);
