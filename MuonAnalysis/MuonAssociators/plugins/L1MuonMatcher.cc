//
//

/**
  \class    pat::L1MuonMatcher L1MuonMatcher.h "MuonAnalysis/MuonAssociators/interface/L1MuonMatcher.h"
  \brief    Matcher of reconstructed objects to L1 Muons

  \author   Giovanni Petrucciani
  \version  $Id: L1MuonMatcher.cc,v 1.4 2011/03/31 09:59:33 gpetrucc Exp $
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/View.h"

#include "MuonAnalysis/MuonAssociators/interface/L1MuonMatcherAlgo.h"

#include "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace pat {

  class L1MuonMatcher : public edm::EDProducer {
    public:
      explicit L1MuonMatcher(const edm::ParameterSet & iConfig);
      virtual ~L1MuonMatcher() { }

      virtual void produce(edm::Event & iEvent, const edm::EventSetup& iSetup) override;

      virtual void beginRun(const edm::Run & iRun, const edm::EventSetup& iSetup) override;
    private:
      typedef pat::TriggerObjectStandAlone           PATPrimitive;
      typedef pat::TriggerObjectStandAloneCollection PATPrimitiveCollection;
      typedef pat::TriggerObjectStandAloneMatch      PATTriggerAssociation;

      L1MuonMatcherAlgo matcher_;

      /// Tokens for input collections
      edm::EDGetTokenT<edm::View<reco::Candidate> > recoToken_;
      edm::EDGetTokenT<std::vector<l1extra::L1MuonParticle> > l1Token_;

      /// Labels to set as filter names in the output
      std::string labelL1_, labelProp_;

      /// Write out additional info as ValueMaps
      bool writeExtraInfo_;

      /// Store extra information in a ValueMap
      template<typename Hand, typename T>
      void storeExtraInfo(edm::Event &iEvent,
                     const Hand & handle,
                     const std::vector<T> & values,
                     const std::string    & label) const ;
  };

} // namespace

pat::L1MuonMatcher::L1MuonMatcher(const edm::ParameterSet & iConfig) :
    matcher_(iConfig),
    recoToken_(consumes<edm::View<reco::Candidate> >(iConfig.getParameter<edm::InputTag>("src"))),
    l1Token_(consumes<std::vector<l1extra::L1MuonParticle> >(iConfig.getParameter<edm::InputTag>("matched"))),
    labelL1_(iConfig.getParameter<std::string>(  "setL1Label")),
    labelProp_(iConfig.getParameter<std::string>("setPropLabel")),
    writeExtraInfo_(iConfig.getParameter<bool>("writeExtraInfo"))
{
    produces<PATPrimitiveCollection>("l1muons");        // l1 in PAT format
    produces<PATPrimitiveCollection>("propagatedReco"); // reco to muon station 2
    produces<PATTriggerAssociation>("propagatedReco");  // asso reco to propagated reco
    produces<PATTriggerAssociation>();                  // asso reco to l1
    if (writeExtraInfo_) {
        produces<edm::ValueMap<float> >("deltaR");
        produces<edm::ValueMap<float> >("deltaPhi");
        produces<edm::ValueMap<int>   >("quality");
        produces<edm::ValueMap<int>   >("bx");
        produces<edm::ValueMap<int>   >("isolated");
        produces<edm::ValueMap<reco::CandidatePtr> >();
        produces<edm::ValueMap<reco::CandidatePtr> >("l1ToReco");
    }
}

void
pat::L1MuonMatcher::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
    using namespace edm;
    using namespace std;

    Handle<View<reco::Candidate> > reco;
    Handle<vector<l1extra::L1MuonParticle> > l1s;

    iEvent.getByToken(recoToken_, reco);
    iEvent.getByToken(l1Token_, l1s);

    auto_ptr<PATPrimitiveCollection> propOut(new PATPrimitiveCollection());
    auto_ptr<PATPrimitiveCollection> l1Out(new PATPrimitiveCollection());
    std::vector<edm::Ptr<reco::Candidate> > l1rawMatches(reco->size());
    vector<int>   isSelected(l1s->size(), -1);
    std::vector<edm::Ptr<reco::Candidate> > whichRecoMatch(l1s->size());
    vector<int>   propMatches(reco->size(), -1);
    vector<int>   fullMatches(reco->size(), -1);
    vector<float> deltaRs(reco->size(), 999), deltaPhis(reco->size(), 999);
    vector<int>   quality(reco->size(),   0), bx(reco->size(), -999), isolated(reco->size(), -999);
    for (int i = 0, n = reco->size(); i < n; ++i) {
        TrajectoryStateOnSurface propagated;
        const reco::Candidate &mu = (*reco)[i];
        int match = matcher_.match(mu, *l1s, deltaRs[i], deltaPhis[i], propagated);
        if (propagated.isValid()) {
            GlobalPoint pos = propagated.globalPosition();
            propMatches[i] = propOut->size();
            propOut->push_back(PATPrimitive(math::PtEtaPhiMLorentzVector(mu.pt(), pos.eta(), pos.phi(), mu.mass())));
            propOut->back().addFilterLabel(labelProp_);
            propOut->back().setCharge(mu.charge());
        }
        if (match != -1) {
            const l1extra::L1MuonParticle & l1 = (*l1s)[match];
            whichRecoMatch[match] = reco->ptrAt(i);
            if (isSelected[match] == -1) { // copy to output if needed
                isSelected[match] = l1Out->size();
                l1Out->push_back(PATPrimitive(l1.polarP4()));
                l1Out->back().addFilterLabel(labelL1_);
                l1Out->back().setCharge(l1.charge());
            }
            fullMatches[i] = isSelected[match]; // index in the output collection
            const L1MuGMTCand & gmt = l1.gmtMuonCand();
            quality[i]  = gmt.quality();
            bx[i]       = gmt.bx();
            isolated[i] = gmt.isol();
            l1rawMatches[i] = edm::Ptr<reco::Candidate>(l1s, size_t(match));
        }
    }

    OrphanHandle<PATPrimitiveCollection> l1Done = iEvent.put(l1Out, "l1muons");
    OrphanHandle<PATPrimitiveCollection> propDone = iEvent.put(propOut, "propagatedReco");

    auto_ptr<PATTriggerAssociation> propAss(new PATTriggerAssociation(propDone));
    PATTriggerAssociation::Filler propFiller(*propAss);
    propFiller.insert(reco, propMatches.begin(), propMatches.end());
    propFiller.fill();
    iEvent.put(propAss, "propagatedReco");

    auto_ptr<PATTriggerAssociation> fullAss(new PATTriggerAssociation(  l1Done));
    PATTriggerAssociation::Filler fullFiller(*fullAss);
    fullFiller.insert(reco, fullMatches.begin(), fullMatches.end());
    fullFiller.fill();
    iEvent.put(fullAss);

    if (writeExtraInfo_) {
        storeExtraInfo(iEvent, reco, deltaRs,   "deltaR");
        storeExtraInfo(iEvent, reco, deltaPhis, "deltaPhi");
        storeExtraInfo(iEvent, reco, bx,        "bx");
        storeExtraInfo(iEvent, reco, isolated,  "isolated");
        storeExtraInfo(iEvent, reco, quality,   "quality");
        storeExtraInfo(iEvent, reco, l1rawMatches,   "");
        storeExtraInfo(iEvent, l1s,  whichRecoMatch, "l1ToReco");
    }
}

template<typename Hand, typename T>
void
pat::L1MuonMatcher::storeExtraInfo(edm::Event &iEvent,
                     const Hand & handle,
                     const std::vector<T> & values,
                     const std::string    & label) const {
    using namespace edm; using namespace std;
    auto_ptr<ValueMap<T> > valMap(new ValueMap<T>());
    typename edm::ValueMap<T>::Filler filler(*valMap);
    filler.insert(handle, values.begin(), values.end());
    filler.fill();
    iEvent.put(valMap, label);
}


void
pat::L1MuonMatcher::beginRun(const edm::Run & iRun, const edm::EventSetup & iSetup) {
    matcher_.init(iSetup);
}


#include "FWCore/Framework/interface/MakerMacros.h"
using namespace pat;
DEFINE_FWK_MODULE(L1MuonMatcher);
