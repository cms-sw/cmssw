/**
  \class    pat::PATTauSlimmer PATTauSlimmer.h "PhysicsTools/PatAlgos/interface/PATTauSlimmer.h"
  \brief    Slimmer of PAT Taus 
*/


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include "DataFormats/PatCandidates/interface/Tau.h"

namespace pat {

  class PATTauSlimmer : public edm::EDProducer {
    public:
      explicit PATTauSlimmer(const edm::ParameterSet & iConfig);
      virtual ~PATTauSlimmer() { }

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

    private:
      edm::EDGetTokenT<edm::View<pat::Tau> > src_;
      bool linkToPackedPF_;
      edm::EDGetTokenT<edm::Association<pat::PackedCandidateCollection>> pf2pc_;
      bool dropPiZeroRefs_;
      bool dropTauChargedHadronRefs_;
      bool dropPFSpecific_;


  };

} // namespace

pat::PATTauSlimmer::PATTauSlimmer(const edm::ParameterSet & iConfig) :
    src_(consumes<edm::View<pat::Tau> >(iConfig.getParameter<edm::InputTag>("src"))),
    linkToPackedPF_(iConfig.getParameter<bool>("linkToPackedPFCandidates"))
{
    produces<std::vector<pat::Tau> >();
    if (linkToPackedPF_) pf2pc_ = consumes<edm::Association<pat::PackedCandidateCollection>>(iConfig.getParameter<edm::InputTag>("packedPFCandidates"));
    dropPiZeroRefs_ = iConfig.exists("dropPiZeroRefs") ? iConfig.getParameter<bool>("dropPiZeroRefs") : true;
    dropTauChargedHadronRefs_ = iConfig.exists("dropTauChargedHadronRefs") ? iConfig.getParameter<bool>("dropTauChargedHadronRefs") : true;
    dropPFSpecific_ = iConfig.exists("dropPFSpecific") ? iConfig.getParameter<bool>("dropPFSpecific"): true;

}

void 
pat::PATTauSlimmer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
    using namespace edm;
    using namespace std;

    Handle<View<pat::Tau> >      src;
    iEvent.getByToken(src_, src);

    Handle<edm::Association<pat::PackedCandidateCollection>> pf2pc;
    if (linkToPackedPF_) iEvent.getByToken(pf2pc_, pf2pc);

    auto_ptr<vector<pat::Tau> >  out(new vector<pat::Tau>());
    out->reserve(src->size());

    for (View<pat::Tau>::const_iterator it = src->begin(), ed = src->end(); it != ed; ++it) {
        out->push_back(*it);
	pat::Tau & tau = out->back();
	// clearing the pat isolation which is not used by taus
	tau.isolations_.clear();
	tau.isoDeposits_.clear();
	
        if (linkToPackedPF_) {

            reco::CandidatePtrVector signalChHPtrs, signalNHPtrs, signalGammaPtrs, isolationChHPtrs, isolationNHPtrs, isolationGammaPtrs;

	    for (const reco::PFCandidatePtr &p : tau.signalPFChargedHadrCands()) {
	      signalChHPtrs.push_back(edm::refToPtr((*pf2pc)[p]));
            }
            tau.setSignalChargedHadrCands(signalChHPtrs);

	    for (const reco::PFCandidatePtr &p : tau.signalPFNeutrHadrCands()) {
              signalNHPtrs.push_back(edm::refToPtr((*pf2pc)[p]));
            }
            tau.setSignalNeutralHadrCands(signalNHPtrs);

	    for (const reco::PFCandidatePtr &p : tau.signalPFGammaCands()) {
              signalGammaPtrs.push_back(edm::refToPtr((*pf2pc)[p]));
            }
            tau.setSignalGammaCands(signalGammaPtrs);

	    for (const reco::PFCandidatePtr &p : tau.isolationPFChargedHadrCands()) {
              isolationChHPtrs.push_back(edm::refToPtr((*pf2pc)[p]));
            }
            tau.setIsolationChargedHadrCands(isolationChHPtrs);

            for (const reco::PFCandidatePtr &p : tau.isolationPFNeutrHadrCands()) {
              isolationNHPtrs.push_back(edm::refToPtr((*pf2pc)[p]));
            }
            tau.setIsolationNeutralHadrCands(isolationNHPtrs);

            for (const reco::PFCandidatePtr &p : tau.isolationPFGammaCands()) {
              isolationGammaPtrs.push_back(edm::refToPtr((*pf2pc)[p]));
            }
            tau.setIsolationGammaCands(isolationGammaPtrs);

        }
     if(dropPiZeroRefs_){ 
         tau.pfSpecific_[0].signalPiZeroCandidates_.clear();
         tau.pfSpecific_[0].isolationPiZeroCandidates_.clear();
       }
     if(dropTauChargedHadronRefs_){ 
          tau.pfSpecific_[0].signalTauChargedHadronCandidates_.clear();
          tau.pfSpecific_[0].isolationTauChargedHadronCandidates_.clear();
        }
     if(dropPFSpecific_){ tau.pfSpecific_.clear();}

    }

    iEvent.put(out);
}

#include "FWCore/Framework/interface/MakerMacros.h"
using namespace pat;
DEFINE_FWK_MODULE(PATTauSlimmer);
