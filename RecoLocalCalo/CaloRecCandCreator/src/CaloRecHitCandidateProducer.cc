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
  :  mEcalLabels (fConfig.getParameter<std::vector<edm::InputTag> >("ecalInputs")),
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

  tok_hbhe_ = consumes<HBHERecHitCollection>(fConfig.getParameter<edm::InputTag>("hbheInput"));
     tok_ho_ = consumes<HORecHitCollection> (fConfig.getParameter<edm::InputTag>("hoInput"));
     tok_hf_ = consumes<HFRecHitCollection> (fConfig.getParameter<edm::InputTag>("hfInput"));

  const unsigned nLabels = mEcalLabels.size();
  for ( unsigned i=0; i != nLabels; i++ )
    toks_ecal_.push_back(consumes<EcalRecHitCollection>(mEcalLabels[i]));

  produces<CandidateCollection>();
}

void CaloRecHitCandidateProducer::produce( edm::Event & fEvent, const edm::EventSetup & fSetup) {
  // get geometry
  //  const IdealGeometryRecord& record = fSetup.template get<IdealGeometryRecord>();
  const CaloGeometryRecord& caloRecord = fSetup.get<CaloGeometryRecord>();
  ESHandle<CaloGeometry> geometry;
  caloRecord.get (geometry);
  const IdealGeometryRecord& record = fSetup.get<IdealGeometryRecord>();
  ESHandle<HcalTopology> topology;
  record.get (topology);
  // set Output
  auto_ptr<CandidateCollection> output ( new CandidateCollection );
  // get and process Inputs
  edm::Handle<HBHERecHitCollection> hbhe;
  fEvent.getByToken(tok_hbhe_,hbhe);
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
    fEvent.getByToken(tok_ho_,ho);
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
  fEvent.getByToken(tok_hf_,hf);
  if (!hf.isValid()) {
    // can't find it!
    if (!mAllowMissingInputs) {
      *hf;  // will throw the proper exception
    }
  } else {
    processHits (hf, *this, *geometry, *topology, &*output);
  }

  std::vector<edm::EDGetTokenT<EcalRecHitCollection> >::const_iterator i;
  for (i=toks_ecal_.begin(); i!=toks_ecal_.end(); i++) {
    edm::Handle<EcalRecHitCollection> ec;
    fEvent.getByToken(*i,ec);
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


