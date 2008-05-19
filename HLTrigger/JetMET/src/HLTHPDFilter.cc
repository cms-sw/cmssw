/** \class HLTHPDFilter
 *
 * $Id: HLTHPDFilter.cc,v 1.4.2.1 2007/08/19 03:15:51 apana Exp $
 *
 * Fedor Ratnikov (UMd) May 19, 2008
 */

#include "HLTrigger/JetMET/interface/HLTHPDFilter.h"

#include <math.h>

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

namespace {
  float shoulderEnergy (const HBHERecHitCollection& fHits, const std::vector<DetId>& fNeighbors) {
    float result = 0;
    for (unsigned i = 0; i < fNeighbors.size(); ++i) {
      if (fNeighbors[i].det() == DetId::Hcal) {
	HcalDetId hcalId = fNeighbors[i];
	if (hcalId.subdet() == HcalBarrel || hcalId.subdet() == HcalEndcap) {
	  HBHERecHitCollection::const_iterator hit = fHits.find (hcalId);
	  if (hit != fHits.end()) {
	    result += hit->energy();
	  }
	}
      }
    }
    return result;
  }
}

HLTHPDFilter::HLTHPDFilter(const edm::ParameterSet& iConfig)
  :  mInputTag (iConfig.getParameter <edm::InputTag> ("inputTag")),
     mSeedThresholdEnergy (iConfig.getParameter <double> ("seedEnergy")),
     mShoulderThresholdEnergy (iConfig.getParameter <double> ("shoulderEnergy")),
     mShoulderToSeedRatio (iConfig.getParameter <double> ("shoulderToSeedRatio"))
{}

HLTHPDFilter::~HLTHPDFilter(){}

bool HLTHPDFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // get hits
  edm::Handle<HBHERecHitCollection> hbhe;
  iEvent.getByLabel(mInputTag,hbhe);
  // get topology
  edm::ESHandle<HcalTopology> topology;
  iSetup.get<IdealGeometryRecord>().get(topology);

  // look for maximum energy cell in the event
  const HBHERecHit* maxEnergyHit = 0;
  float maxEnergy = 0;
  for (size_t i = 0; i < hbhe->size (); ++i) {
    if (!maxEnergyHit || (*hbhe)[i].energy () > maxEnergy) {
      maxEnergyHit = &((*hbhe)[i]);
      maxEnergy = maxEnergyHit->energy();
    }
  }
  if (maxEnergy < mSeedThresholdEnergy && maxEnergy <= 0) return true;  // no signal - no suspition at all
  DetId id = maxEnergyHit->id();
  float energy = 0;
  energy = fmaxf (energy, shoulderEnergy (*hbhe, topology->east (id)));
  energy = fmaxf (energy, shoulderEnergy (*hbhe, topology->west (id)));
  energy = fmaxf (energy, shoulderEnergy (*hbhe, topology->north (id)));
  energy = fmaxf (energy, shoulderEnergy (*hbhe, topology->south (id)));
  energy = fmaxf (energy, shoulderEnergy (*hbhe, topology->up (id)));
  energy = fmaxf (energy, shoulderEnergy (*hbhe, topology->down (id)));
  if ((energy > mShoulderThresholdEnergy) || (energy / maxEnergy > mShoulderToSeedRatio)) return true;
  return false; // spike-like energy deposition
}
 
