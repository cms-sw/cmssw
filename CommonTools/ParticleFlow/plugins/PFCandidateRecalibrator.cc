/// Take:
//    - a PF candidate collection (which uses bugged HCAL respcorrs)
//    - respCorrs values from fixed GT and bugged GT
//  Produce:
//    - a new PFCandidate collection containing the recalibrated PFCandidates in HF and where the neutral had pointing to problematic cells are removed
//    - a second PFCandidate collection with just those discarded hadrons
//    - a ValueMap<reco::PFCandidateRef> that maps the old to the new, and vice-versa

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/Event.h"

#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Association.h"
#include <iostream>

class PFCandidateRecalibrator : public edm::global::EDProducer<> {
    public:
        PFCandidateRecalibrator(const edm::ParameterSet&);
        ~PFCandidateRecalibrator() override {};

        void produce(edm::StreamID iID, edm::Event&, const edm::EventSetup&) const override;

    private:
        edm::EDGetTokenT<std::vector<reco::PFCandidate> > pfcandidates_;
};

PFCandidateRecalibrator::PFCandidateRecalibrator(const edm::ParameterSet &iConfig) :
    pfcandidates_(consumes<std::vector<reco::PFCandidate>>(iConfig.getParameter<edm::InputTag>("pfcandidates")))
{
    produces<std::vector<reco::PFCandidate>>();
    produces<std::vector<reco::PFCandidate>>("discarded");
    produces<edm::ValueMap<reco::PFCandidateRef>>();
}

void PFCandidateRecalibrator::produce(edm::StreamID iID, edm::Event &iEvent, const edm::EventSetup &iSetup) const
{
    //Get Calib Constants from current GT
    edm::ESHandle<HcalDbService> GTCond;
    iSetup.get<HcalDbRecord>().get(GTCond);

    //Get Calib Constants from bugged tag
    edm::ESHandle<HcalTopology> htopo;
    iSetup.get<HcalRecNumberingRecord>().get(htopo);
    const HcalTopology* theHBHETopology = htopo.product();
    
    edm::ESHandle<HcalRespCorrs> buggedCond;
    iSetup.get<HcalRespCorrsRcd>().get("bugged", buggedCond);
    HcalRespCorrs* buggedRespCorrs = new HcalRespCorrs(*buggedCond.product());
    buggedRespCorrs->setTopo(theHBHETopology);

    //access calogeometry
    edm::ESHandle<CaloGeometry> calogeom;
    iSetup.get<CaloGeometryRecord>().get(calogeom);
    const CaloGeometry* cgeo = calogeom.product();
    HcalGeometry* hgeom = (HcalGeometry*)(cgeo->getSubdetectorGeometry(DetId::Hcal,HcalForward));
    
    //access PFCandidates
    edm::Handle<std::vector<reco::PFCandidate>> pfcandidates;
    iEvent.getByToken(pfcandidates_, pfcandidates);

    int n = pfcandidates->size();
    std::unique_ptr<std::vector<reco::PFCandidate>> copy(new std::vector<reco::PFCandidate>());
    std::unique_ptr<std::vector<reco::PFCandidate>> discarded(new std::vector<reco::PFCandidate>());
    copy->reserve(n);
    std::vector<int> oldToNew(n), newToOld, badToOld; 
    newToOld.reserve(n);


    int i = -1;
    for (const reco::PFCandidate &pf : *pfcandidates) 
      {
	++i;
	math::XYZPointF ecalPoint = pf.positionAtECALEntrance();
	GlobalPoint ecalGPoint(ecalPoint.X(),ecalPoint.Y(),ecalPoint.Z());
	DetId closestDetId(hgeom->getClosestCell(ecalGPoint));
	
	//make sure we are in HE or HF
	HcalDetId hDetId(closestDetId.rawId());
	if(hDetId.subdet() == 2 || hDetId.subdet() == 4)
	  {
	    //access the calib values
	    const HcalRespCorr* GTrespcorr = GTCond->getHcalRespCorr(hDetId);
	    float currentRespCorr = GTrespcorr->getValue();
	    float buggedRespCorr = buggedRespCorrs->getValues(hDetId)->getValue();
	    float scalingFactor = currentRespCorr/buggedRespCorr;
	    
	    //if same calib then don't do anything
	    if (scalingFactor != 1)
	      {	    
		//kill pfCandidate if neutral and HE
		if (hDetId.subdet() == 2 && pf.particleId() == 5)
		  {
		    discarded->push_back(pf);
		    oldToNew[i] = (-discarded->size());
		    badToOld.push_back(i);
		  }
		//recalibrate pfCandidate if HF
		else if (hDetId.subdet() == 4)
		  {
		    copy->push_back(pf);
		    oldToNew[i] = (copy->size());
		    newToOld.push_back(i);
		    copy->back().setHcalEnergy(pf.rawHcalEnergy() * scalingFactor,
					       pf.hcalEnergy() * scalingFactor);
		    copy->back().setEcalEnergy(pf.rawEcalEnergy() * scalingFactor,
					       pf.ecalEnergy() * scalingFactor);
		  }
		else
		  {
		    copy->push_back(pf);
		    oldToNew[i] = (copy->size());
		    newToOld.push_back(i);
		  }
	      }
	    else
	      {
		copy->push_back(pf);
		oldToNew[i] = (copy->size());
		newToOld.push_back(i);
	      }
	  }
	else
	{
	    copy->push_back(pf);
	    oldToNew[i] = (copy->size());
	    newToOld.push_back(i);
	}
      }
	
	
    // Now we put things in the event
    edm::OrphanHandle<std::vector<reco::PFCandidate>> newpf = iEvent.put(std::move(copy));
    edm::OrphanHandle<std::vector<reco::PFCandidate>> badpf = iEvent.put(std::move(discarded), "discarded");

    std::unique_ptr<edm::ValueMap<reco::PFCandidateRef>> pf2pf(new edm::ValueMap<reco::PFCandidateRef>());
    edm::ValueMap<reco::PFCandidateRef>::Filler filler(*pf2pf);
    std::vector<reco::PFCandidateRef> refs; refs.reserve(n);

    // old to new
    for (i = 0; i < n; ++i) {
      if (oldToNew[i] > 0) {
	refs.push_back(reco::PFCandidateRef(newpf, oldToNew[i]-1));
      } else {
	refs.push_back(reco::PFCandidateRef(badpf,-oldToNew[i]-1));
      }
    }
    filler.insert(pfcandidates, refs.begin(), refs.end());
    // new good to old
    refs.clear();
    for (int i : newToOld) {
      refs.push_back(reco::PFCandidateRef(pfcandidates,i));
    }
    filler.insert(newpf, refs.begin(), refs.end());
    // new bad to old
    refs.clear();
    for (int i : badToOld) {
      refs.push_back(reco::PFCandidateRef(pfcandidates,i));
    }
    filler.insert(badpf, refs.begin(), refs.end());
    // done
    filler.fill();
    iEvent.put(std::move(pf2pf));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFCandidateRecalibrator);
