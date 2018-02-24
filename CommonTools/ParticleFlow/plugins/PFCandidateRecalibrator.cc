/// Take:
//    - a PF candidate collection (which uses bugged HCAL respcorrs)
//    - respCorrs values from fixed GT and bugged GT
//  Produce:
//    - a new PFCandidate collection containing the recalibrated PFCandidates in HF and where the neutral had pointing to problematic cells are removed
//    - a second PFCandidate collection with just those discarded hadrons
//    - a ValueMap<reco::PFCandidateRef> that maps the old to the new, and vice-versa

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
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

class PFCandidateRecalibrator : public edm::EDProducer {
    public:
        PFCandidateRecalibrator(const edm::ParameterSet&);
        ~PFCandidateRecalibrator() override {};

    private:
        void beginRun(const edm::Run& iRun, edm::EventSetup const& iSetup) override;
        void produce(edm::Event&, const edm::EventSetup&) override;

        edm::EDGetTokenT<std::vector<reco::PFCandidate> > pfcandidates_;

        std::vector<std::tuple<float,float,float>> badChHE_;
        std::vector<std::tuple<int,int,int,float>> badChHF_;
};

PFCandidateRecalibrator::PFCandidateRecalibrator(const edm::ParameterSet &iConfig) :
    pfcandidates_(consumes<std::vector<reco::PFCandidate>>(iConfig.getParameter<edm::InputTag>("pfcandidates")))
{
    produces<std::vector<reco::PFCandidate>>();
    produces<std::vector<reco::PFCandidate>>("discarded");
    produces<edm::ValueMap<reco::PFCandidateRef>>();
}

void PFCandidateRecalibrator::beginRun(const edm::Run &iRun, const edm::EventSetup &iSetup)
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
    HcalRespCorrs buggedRespCorrs(*buggedCond.product());
    buggedRespCorrs.setTopo(theHBHETopology);

    //access calogeometry
    edm::ESHandle<CaloGeometry> calogeom;
    iSetup.get<CaloGeometryRecord>().get(calogeom);
    const CaloGeometry* cgeo = calogeom.product();
    HcalGeometry* hgeom = (HcalGeometry*)(cgeo->getSubdetectorGeometry(DetId::Hcal,HcalForward));
    
    //fill bad cells HE (use eta, phi)
    std::vector<DetId> cellsHE = hgeom->getValidDetIds(DetId::Detector::Hcal, HcalEndcap);
    for(std::vector<DetId>::const_iterator ii=cellsHE.begin(); ii!=cellsHE.end();++ii)
      {
	float currentRespCorr = GTCond->getHcalRespCorr(*ii)->getValue();
	float buggedRespCorr = buggedRespCorrs.getValues(*ii)->getValue();
	float ratio = currentRespCorr/buggedRespCorr;

	if(ratio != 1.)
	  {
	    GlobalPoint pos = hgeom->getPosition(*ii);
	    badChHE_.push_back(std::make_tuple(pos.eta(),pos.phi(),ratio));
	  }
      }

    //fill bad cells HF (use ieta, iphi)
    std::vector<DetId> cellsHF = hgeom->getValidDetIds(DetId::Detector::Hcal, HcalForward);
    for(std::vector<DetId>::const_iterator ii=cellsHF.begin(); ii!=cellsHF.end();++ii)
      {
	float currentRespCorr = GTCond->getHcalRespCorr(*ii)->getValue();
	float buggedRespCorr = buggedRespCorrs.getValues(*ii)->getValue();
	float ratio = currentRespCorr/buggedRespCorr;

	if(ratio != 1.)
	  {
	    HcalDetId dummyId(*ii);
	    badChHF_.push_back(std::make_tuple(dummyId.ieta(), dummyId.iphi(), dummyId.depth(), ratio));
	  }
      }
}

void PFCandidateRecalibrator::produce(edm::Event &iEvent, const edm::EventSetup &iSetup)
{
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

    //std::cout << "NEW EV:" << std::endl;

    //loop over PFCandidates
    int i = -1;
    for (const reco::PFCandidate &pf : *pfcandidates) 
      {
	++i;

	//deal with HE
	if( pf.particleId() == 5 && 
	    fabs(pf.eta()) > 1.4  && fabs(pf.eta()) < 3.)
	  {
	    bool toKill = false;
	    for(auto badIt: badChHE_)
	      if( reco::deltaR2(pf.eta(), pf.phi(), std::get<0>(badIt), std::get<1>(badIt)) < 0.07 )
		toKill = true;
	    
	    
	    if(toKill)
	      {
		discarded->push_back(pf);
		oldToNew[i] = (-discarded->size());
		badToOld.push_back(i);
		continue;
	      }
	    else
	      {
		copy->push_back(pf);
		oldToNew[i] = (copy->size());
		newToOld.push_back(i);
	      }
	  }
	//deal with HF
	else if(fabs(pf.eta()) > 3.)
	  {
	    math::XYZPointF ecalPoint = pf.positionAtECALEntrance();
	    GlobalPoint ecalGPoint(ecalPoint.X(),ecalPoint.Y(),ecalPoint.Z());
	    HcalDetId closestDetId(hgeom->getClosestCell(ecalGPoint));
	    
	    if(closestDetId.subdet() == 4)
	      {
		HcalDetId hDetId(closestDetId.subdet(),closestDetId.ieta(),closestDetId.iphi(),1); //depth1
		
		//raw*calEnergy() is the same as *calEnergy() - no corrections are done for HF
		float longE  = pf.rawEcalEnergy() + pf.rawHcalEnergy()/2.;  //depth1
		float shortE = pf.rawHcalEnergy()/2.;                       //depth2
		
		float ecalEnergy = pf.rawEcalEnergy();
		float hcalEnergy = pf.rawHcalEnergy();
		float totEnergy = ecalEnergy + hcalEnergy;
		
		bool toKill = false;
			
		for(auto badIt: badChHF_)
		  {
		    if ( hDetId.ieta() == std::get<0>(badIt) &&
			 hDetId.iphi() == std::get<1>(badIt) )
		      {
			//std::cout << "==> orig en (tot,H,E): " << pf.energy() << " " << pf.rawHcalEnergy() << " " << pf.rawEcalEnergy() << std::endl;
			if(std::get<2>(badIt) == 1) //depth1
			  {
			    longE *= std::get<3>(badIt);
			    ecalEnergy = longE - shortE;
			    totEnergy = ecalEnergy + hcalEnergy;
			  }
			else //depth2
			  {
			    shortE *= std::get<3>(badIt);
			    hcalEnergy = 2*shortE;
			    if(ecalEnergy > 0)
			      ecalEnergy = longE - shortE;
			    totEnergy = ecalEnergy + hcalEnergy;
			  }
			//kill candidate if goes below thr
			if((pf.pdgId()==1 && shortE < 1.4) || 
			   (pf.pdgId()==2 && longE < 1.4))
			  toKill = true;

			//std::cout << "====> ieta,iphi,depth: " <<std::get<0>(badIt) << " " << std::get<1>(badIt) << " " << std::get<2>(badIt) << " corr: " << std::get<3>(badIt) << std::endl;
			//std::cout << "====> recal en (tot,H,E): " << totEnergy << " " << hcalEnergy << " " << ecalEnergy << std::endl;

		      }
		  }
		
		if(toKill == true)
		  {
		    discarded->push_back(pf);
		    oldToNew[i] = (-discarded->size());
		    badToOld.push_back(i);

		    //std::cout << "==> KILLED " << std::endl;
		  }
		else
		  {
		    copy->push_back(pf);
		    oldToNew[i] = (copy->size());
		    newToOld.push_back(i);
		    
		    copy->back().setHcalEnergy(hcalEnergy, hcalEnergy);
		    copy->back().setEcalEnergy(ecalEnergy, ecalEnergy);
		    math::XYZTLorentzVector recalibP4(pf.px(), pf.py(), pf.pz(), totEnergy);
		    copy->back().setP4( recalibP4 );

		    //std::cout << "====> stored en (tot,H,E): " << copy->back().energy() << " " << copy->back().hcalEnergy() << " " << copy->back().ecalEnergy() << std::endl;
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
