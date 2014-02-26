//
// Producer to converts ECAL and HCAL hits into Candidates
// Author: F. Ratnikov (Maryland)
// Jan. 7, 2007
//

#include "CaloRecHitCandidateProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Math/interface/Vector3D.h"

#include "DataFormats/RecoCandidate/interface/CaloRecHitCandidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

using namespace edm;
using namespace reco;
using namespace std;

namespace {
  template <class HitCollection>
  void processHits (const edm::Handle<HitCollection>& fInput, 
		    const CaloRecHitCandidateProducer& fProducer,
		    const CaloGeometry& fGeometry,
		    const HcalTopology& fTopology,
		    CandidateCollection* fOutput) { 
    const CaloSubdetectorGeometry* geometry = 0; // cache
    DetId geometryId (0);
    
    size_t size = fInput->size();
    for (unsigned ihit = 0; ihit < size; ihit++) {
      const CaloRecHit* hit = &((*fInput)[ihit]);
      double weight = fProducer.cellTresholdAndWeight (*hit, fTopology);
      if (weight > 0) { // accept hit
	DetId cell = hit->detid ();
	// get geometry
	if (cell.det() != geometryId.det() || cell.subdetId() != geometryId.subdetId()) {
	  geometry = fGeometry.getSubdetectorGeometry (cell);
	  geometryId = cell;
	}
	const CaloCellGeometry* cellGeometry = geometry->getGeometry (cell);
	double eta = cellGeometry->getPosition().eta ();
	double phi =  cellGeometry->getPosition().phi ();
	double energy = hit->energy() * weight;
	math::RhoEtaPhiVector p( 1, eta, phi );
	p *= ( energy / p.r() );
	CaloRecHitCandidate * c = new CaloRecHitCandidate( Candidate::LorentzVector( p.x(), p.y(), p.z(), energy ) );
	c->setCaloRecHit( RefToBase<CaloRecHit>( Ref<HitCollection>( fInput, ihit ) ) );
	fOutput->push_back( c );
      }
    }
  } 
}


CaloRecHitCandidateProducer::CaloRecHitCandidateProducer ( const edm::ParameterSet & fConfig ) 
  :  mHBHELabel (fConfig.getParameter<edm::InputTag>("hbheInput")),
     mHOLabel (fConfig.getParameter<edm::InputTag>("hoInput")),
     mHFLabel (fConfig.getParameter<edm::InputTag>("hfInput")),
     mEcalLabels (fConfig.getParameter<std::vector<edm::InputTag> >("ecalInputs")),
     mAllowMissingInputs (fConfig.getUntrackedParameter<bool>("AllowMissingInputs",false)),
     mUseHO (fConfig.getParameter<bool>("UseHO")),

     mEBthreshold (fConfig.getParameter<double>("EBThreshold")),
     mEEthreshold  (fConfig.getParameter<double>("EEThreshold")),
     mHBthreshold  (fConfig.getParameter<double>("HBThreshold")),
     mHESthreshold  (fConfig.getParameter<double>("HESThreshold")),
     mHEDthreshold  (fConfig.getParameter<double>("HEDThreshold")),
     mHOthreshold (fConfig.getParameter<double>("HOThreshold")),
     mHF1threshold (fConfig.getParameter<double>("HF1Threshold")),
     mHF2threshold (fConfig.getParameter<double>("HF2Threshold")),
     mEBweight (fConfig.getParameter<double>("EBWeight")),
     mEEweight (fConfig.getParameter<double>("EEWeight")),
     mHBweight (fConfig.getParameter<double>("HBWeight")),
     mHESweight (fConfig.getParameter<double>("HESWeight")),
     mHEDweight (fConfig.getParameter<double>("HEDWeight")),
     mHOweight (fConfig.getParameter<double>("HOWeight")),
     mHF1weight (fConfig.getParameter<double>("HF1Weight")),
     mHF2weight (fConfig.getParameter<double>("HF2Weight"))
{
  produces<CandidateCollection>();
}

void CaloRecHitCandidateProducer::produce( edm::Event & fEvent, const edm::EventSetup & fSetup) {
  // get geometry

  ESHandle<CaloGeometry> geometry;
  fSetup.get<CaloGeometryRecord>().get (geometry);

  ESHandle<HcalTopology> topology;
  fSetup.get<HcalRecNumberingRecord>().get(topology);     

  // set Output
  auto_ptr<CandidateCollection> output ( new CandidateCollection );
  // get and process Inputs
  edm::Handle<HBHERecHitCollection> hbhe;
  fEvent.getByLabel(mHBHELabel,hbhe);
  if (!hbhe.isValid()) {
    // can't find it!
    if (!mAllowMissingInputs) {
      *hbhe;  // will throw the proper exception
    }
  } else {
    processHits (hbhe, *this, *geometry, *topology, &*output);
  }

  if (mUseHO) {
    edm::Handle<HORecHitCollection> ho;
    fEvent.getByLabel(mHOLabel,ho);
    if (!ho.isValid()) {
      // can't find it!
      if (!mAllowMissingInputs) {
	*ho;  // will throw the proper exception	
      }
    } else {
      processHits (ho, *this, *geometry, *topology, &*output);
    }
  }

  edm::Handle<HFRecHitCollection> hf;
  fEvent.getByLabel(mHFLabel,hf);
  if (!hf.isValid()) {
    // can't find it!
    if (!mAllowMissingInputs) {
      *hf;  // will throw the proper exception
    }
  } else {
    processHits (hf, *this, *geometry, *topology, &*output);
  }

  std::vector<edm::InputTag>::const_iterator i;
  for (i=mEcalLabels.begin(); i!=mEcalLabels.end(); i++) {
    edm::Handle<EcalRecHitCollection> ec;
    fEvent.getByLabel(*i,ec);
    if (!ec.isValid()) {
      // can't find it!
      if (!mAllowMissingInputs) {
	*ec;  // will throw the proper exception
      }
    } else {
      processHits (ec, *this, *geometry, *topology, &*output);
    }
  }
  fEvent.put(output);
}

double CaloRecHitCandidateProducer::cellTresholdAndWeight (const CaloRecHit& fHit, const HcalTopology& fTopology) const {
  double weight = 0;
  double threshold = 0;
  DetId detId = fHit.detid ();
  DetId::Detector det = detId.det ();
  if(det == DetId::Ecal) {
    // may or may not be EB.  We'll find out.
    
    EcalSubdetector subdet = (EcalSubdetector)(detId.subdetId());
    if(subdet == EcalBarrel) {
      threshold = mEBthreshold;
      weight = mEBweight;
    }
    else if(subdet == EcalEndcap) {
      threshold = mEEthreshold;
      weight = mEEweight;
    }
  }
  else if(det == DetId::Hcal) {
    HcalDetId hcalDetId(detId);
    HcalSubdetector subdet = hcalDetId.subdet();
    
    if(subdet == HcalBarrel) {
      threshold = mHBthreshold;
      weight = mHBweight;
    }
    
    else if(subdet == HcalEndcap) {
      // check if it's single or double tower
      if(hcalDetId.ietaAbs() < fTopology.firstHEDoublePhiRing()) {
        threshold = mHESthreshold;
        weight = mHESweight;
      }
      else {
        threshold = mHEDthreshold;
        weight = mHEDweight;
      }
    } else if(subdet == HcalOuter) {
      threshold = mHOthreshold;
      weight = mHOweight;
    } else if(subdet == HcalForward) {
      if(hcalDetId.depth() == 1) {
        threshold = mHF1threshold;
        weight = mHF1weight;
      } else {
        threshold = mHF2threshold;
        weight = mHF2weight;
      }
    }
  }
  return fHit.energy () >= threshold ? weight : 0; 
}


