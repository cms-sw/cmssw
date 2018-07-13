#include <string>
#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
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
    
    void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  private:
    edm::EDGetTokenT<edm::View<pat::PackedCandidate> > pfcandidatesrc_;
    edm::EDGetTokenT<edm::View<pat::Jet> > jetsrc_;
    double PtThreshold_;
    double MinEtaThreshold_;
    double MaxEtaThreshold_;
    bool userawPt_;
  };
}


pat::BadPFCandidateJetsEEnoiseProducer::BadPFCandidateJetsEEnoiseProducer(const edm::ParameterSet& iConfig) :
  
  pfcandidatesrc_(consumes<edm::View<pat::PackedCandidate> >(iConfig.getParameter<edm::InputTag>("pfcandidatesrc") )),
  jetsrc_(consumes<edm::View<pat::Jet> >(iConfig.getParameter<edm::InputTag>("jetsrc") )),
  PtThreshold_(iConfig.getParameter<double>("PtThreshold")),
  MinEtaThreshold_(iConfig.getParameter<double>("MinEtaThreshold")),
  MaxEtaThreshold_(iConfig.getParameter<double>("MaxEtaThreshold")),
  userawPt_(iConfig.getParameter<bool>("userawPt"))
{
  
  produces<edm::PtrVector<reco::Candidate> >();
  
}

pat::BadPFCandidateJetsEEnoiseProducer::~BadPFCandidateJetsEEnoiseProducer() {}

void pat::BadPFCandidateJetsEEnoiseProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {

  
  std::unique_ptr<edm::PtrVector<reco::Candidate> > badPFCandidates(new edm::PtrVector<reco::Candidate>);
  
  edm::Handle<edm::View<pat::PackedCandidate> > pfcandidates;
  iEvent.getByToken(pfcandidatesrc_, pfcandidates);
  edm::Handle<edm::View<pat::Jet> > jetcandidates;
  iEvent.getByToken(jetsrc_, jetcandidates);

  //if( userawPt_  ) std::cout << "Caution: Uncorrected Jet Pt is being used" << std::endl;
  //if( !userawPt_ ) std::cout << "Caution: Corrected Jet Pt is being used" << std::endl;
  
  int njets   = jetcandidates->size();
  
 
  for (int jetindex = 0; jetindex < njets; ++jetindex){
    edm::Ptr<pat::Jet> candjet = jetcandidates->ptrAt(jetindex);

    // find the bad jets
    double PtJet;
    // Corrected Pt or Uncorrected Pt (It is defined from cfi file)
    if( userawPt_  ) PtJet = candjet->correctedJet("Uncorrected").pt();
    if( !userawPt_ ) PtJet = candjet->pt();
    
    
    if ( PtJet > PtThreshold_ )continue;
    if (fabs(candjet->eta()) < MinEtaThreshold_ || fabs(candjet->eta()) > MaxEtaThreshold_)continue;
    
    
    // now get a list of the PF candidates used to build this jet
    for (unsigned int pfindex =0; pfindex < candjet->numberOfSourceCandidatePtrs(); pfindex++){
      badPFCandidates -> push_back(candjet->sourceCandidatePtr(pfindex));
    }
  }
  iEvent.put(std::move(badPFCandidates));
  
}
using pat::BadPFCandidateJetsEEnoiseProducer;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(BadPFCandidateJetsEEnoiseProducer);
