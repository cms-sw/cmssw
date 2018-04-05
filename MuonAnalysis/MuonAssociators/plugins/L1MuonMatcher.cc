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
      ~L1MuonMatcher() override { }

      void produce(edm::Event & iEvent, const edm::EventSetup& iSetup) override;

      void beginRun(const edm::Run & iRun, const edm::EventSetup& iSetup) override;
    private:
      typedef pat::TriggerObjectStandAlone           PATPrimitive;
      typedef pat::TriggerObjectStandAloneCollection PATPrimitiveCollection;
      typedef pat::TriggerObjectStandAloneMatch      PATTriggerAssociation;

      L1MuonMatcherAlgo matcher_;

      /// Tokens for input collections
      edm::EDGetTokenT<edm::View<reco::Candidate> > recoToken_;
      edm::EDGetTokenT<std::vector<l1extra::L1MuonParticle> > l1Token_;
      edm::EDGetTokenT<l1t::MuonBxCollection> l1tToken_;
    
      /// Labels to set as filter names in the output
      std::string labelL1_, labelProp_;

      /// Write out additional info as ValueMaps
      bool writeExtraInfo_;

      /// Allow to run both on legacy or stage2 (2016) L1 Muon trigger output
      bool useStage2L1_;

      /// Skim stage2 BX vector
      int firstBX_;
      int lastBX_;
    
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
    labelL1_(iConfig.getParameter<std::string>("setL1Label")),
    labelProp_(iConfig.getParameter<std::string>("setPropLabel")),
    writeExtraInfo_(iConfig.getParameter<bool>("writeExtraInfo")),
    useStage2L1_(iConfig.getParameter<bool>("useStage2L1")),
    firstBX_(iConfig.getParameter<int>("firstBX")),
    lastBX_(iConfig.getParameter<int>("lastBX"))
{
    if (useStage2L1_) {
      l1tToken_ = consumes<l1t::MuonBxCollection>(iConfig.getParameter<edm::InputTag>("matched"));
    } else {
      l1Token_ = consumes<std::vector<l1extra::L1MuonParticle> >(iConfig.getParameter<edm::InputTag>("matched"));
    }
    produces<PATPrimitiveCollection>("l1muons");        // l1 in PAT format
    produces<PATPrimitiveCollection>("propagatedReco"); // reco to muon station 2
    produces<PATTriggerAssociation>("propagatedReco");  // asso reco to propagated reco
    produces<PATTriggerAssociation>();                  // asso reco to l1
    if (writeExtraInfo_) {
        produces<edm::ValueMap<float> >("deltaR");
        produces<edm::ValueMap<float> >("deltaPhi");
        produces<edm::ValueMap<int>   >("quality");
        produces<edm::ValueMap<int>   >("bx");
        if(useStage2L1_) { 
	  produces<edm::ValueMap<int> >("iPhi");
	  produces<edm::ValueMap<int> >("tfIndex");
	}
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
    Handle<l1t::MuonBxCollection> l1tBX;
  
    std::vector<l1t::Muon> l1ts;
    std::vector<size_t> bxIdxs;

    int minBxIdx = 0;
    size_t l1size = 0;
    
    iEvent.getByToken(recoToken_, reco);

    if (useStage2L1_) {
      iEvent.getByToken(l1tToken_, l1tBX);
      l1size = l1tBX->size();
	
      int minBX = max(firstBX_,l1tBX->getFirstBX());
      int maxBX = min(lastBX_,l1tBX->getLastBX());

      minBxIdx = l1tBX->begin(minBX) - l1tBX->begin();
      std::copy(l1tBX->begin(minBX), l1tBX->end(maxBX), std::back_inserter(l1ts));

      for (int ibx = l1tBX->getFirstBX(); ibx <= l1tBX->getLastBX(); ++ibx) {
	bxIdxs.push_back(l1tBX->end(ibx) - l1tBX->begin());
      } 
    } else {
      iEvent.getByToken(l1Token_, l1s);
      l1size = l1s->size();
    }

    unique_ptr<PATPrimitiveCollection> propOut(new PATPrimitiveCollection());
    unique_ptr<PATPrimitiveCollection> l1Out(new PATPrimitiveCollection());
    std::vector<edm::Ptr<reco::Candidate> > l1rawMatches(reco->size());
    vector<int>   isSelected(l1size, -1);
    std::vector<edm::Ptr<reco::Candidate> > whichRecoMatch(l1size);
    vector<int>   propMatches(reco->size(), -1);
    vector<int>   fullMatches(reco->size(), -1);
    vector<float> deltaRs(reco->size(), 999), deltaPhis(reco->size(), 999);
    vector<int>   quality(reco->size(),   0), bx(reco->size(), -999), isolated(reco->size(), -999);
    vector<int>   iPhi(reco->size(),   0), tfIndex(reco->size(), -999);
    for (int i = 0, n = reco->size(); i < n; ++i) {
        TrajectoryStateOnSurface propagated;
        const reco::Candidate &mu = (*reco)[i];
        int match = useStage2L1_ ? 
	  matcher_.match(mu, l1ts, deltaRs[i], deltaPhis[i], propagated) :
	  matcher_.match(mu, *l1s, deltaRs[i], deltaPhis[i], propagated);
        if (propagated.isValid()) {
            GlobalPoint pos = propagated.globalPosition();
            propMatches[i] = propOut->size();
            propOut->push_back(PATPrimitive(math::PtEtaPhiMLorentzVector(mu.pt(), pos.eta(), pos.phi(), mu.mass())));
            propOut->back().addFilterLabel(labelProp_);
            propOut->back().setCharge(mu.charge());
        }
        if (match != -1) {

	    if(useStage2L1_) {
	      match += minBxIdx;
	    }

	    whichRecoMatch[match] = reco->ptrAt(i);
  	 
	    int charge = 0;
	    math::PtEtaPhiMLorentzVector p4;
	    
	    if (useStage2L1_) {
	      const l1t::Muon & l1t = (*l1tBX)[match];
	      charge = l1t.charge();
	      p4     = l1t.polarP4();
	    }
	    else {
	      const l1extra::L1MuonParticle & l1 = (*l1s)[match];
	      charge = l1.charge();
	      p4     = l1.polarP4();
	    }

            if (isSelected[match] == -1) { // copy to output if needed
                isSelected[match] = l1Out->size();
                l1Out->push_back(PATPrimitive(p4));
                l1Out->back().addFilterLabel(labelL1_);
                l1Out->back().setCharge(charge);
            }

            fullMatches[i] = isSelected[match]; // index in the output collection
	    
	    if (useStage2L1_) {
	      const l1t::Muon & l1t = (*l1tBX)[match];
	      quality[i]  = l1t.hwQual();
	      bx[i] = l1tBX->getFirstBX() + (std::upper_bound(bxIdxs.begin(),bxIdxs.end(), match) - bxIdxs.begin()); 
	      isolated[i] = l1t.hwIso();
	      l1rawMatches[i] = edm::Ptr<reco::Candidate>(l1tBX, size_t(match));
	      iPhi[i]    = l1t.hwPhi();
	      tfIndex[i] = l1t.tfMuonIndex();      
	    }
	    else {
	      const L1MuGMTCand & gmt = (*l1s)[match].gmtMuonCand();
	      quality[i]  = gmt.quality();
	      bx[i]       = gmt.bx();
	      isolated[i] = gmt.isol();
	      l1rawMatches[i] = edm::Ptr<reco::Candidate>(l1s, size_t(match));
	    }
	}
    }

    OrphanHandle<PATPrimitiveCollection> l1Done = iEvent.put(std::move(l1Out), "l1muons");
    OrphanHandle<PATPrimitiveCollection> propDone = iEvent.put(std::move(propOut), "propagatedReco");

    unique_ptr<PATTriggerAssociation> propAss(new PATTriggerAssociation(propDone));
    PATTriggerAssociation::Filler propFiller(*propAss);
    propFiller.insert(reco, propMatches.begin(), propMatches.end());
    propFiller.fill();
    iEvent.put(std::move(propAss), "propagatedReco");

    unique_ptr<PATTriggerAssociation> fullAss(new PATTriggerAssociation(  l1Done));
    PATTriggerAssociation::Filler fullFiller(*fullAss);
    fullFiller.insert(reco, fullMatches.begin(), fullMatches.end());
    fullFiller.fill();
    iEvent.put(std::move(fullAss));

    if (writeExtraInfo_) {
        storeExtraInfo(iEvent, reco, deltaRs,   "deltaR");
        storeExtraInfo(iEvent, reco, deltaPhis, "deltaPhi");
        storeExtraInfo(iEvent, reco, bx,        "bx");
        storeExtraInfo(iEvent, reco, isolated,  "isolated");
        storeExtraInfo(iEvent, reco, quality,   "quality");
	storeExtraInfo(iEvent, reco, l1rawMatches,   "");
	if (useStage2L1_) {
	  storeExtraInfo(iEvent, l1tBX,  whichRecoMatch, "l1ToReco");
	  storeExtraInfo(iEvent, reco, tfIndex, "tfIndex");
	  storeExtraInfo(iEvent, reco, iPhi, "iPhi");
	}
	else {
	  storeExtraInfo(iEvent, l1s,  whichRecoMatch, "l1ToReco");
	}
    }

}

template<typename Hand, typename T>
void
pat::L1MuonMatcher::storeExtraInfo(edm::Event &iEvent,
                     const Hand & handle,
                     const std::vector<T> & values,
                     const std::string    & label) const {
    using namespace edm; using namespace std;
    unique_ptr<ValueMap<T> > valMap(new ValueMap<T>());
    typename edm::ValueMap<T>::Filler filler(*valMap);
    filler.insert(handle, values.begin(), values.end());
    filler.fill();
    iEvent.put(std::move(valMap), label);
}


void
pat::L1MuonMatcher::beginRun(const edm::Run & iRun, const edm::EventSetup & iSetup) {
    matcher_.init(iSetup);
}


#include "FWCore/Framework/interface/MakerMacros.h"
using namespace pat;
DEFINE_FWK_MODULE(L1MuonMatcher);
