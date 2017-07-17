#ifndef HiEgammaAlgos_HcalRechitIsoCalculator_h
#define HiEgammaAlgos_HcalRechitIsoCalculator_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

class HcalRechitIsoCalculator
{
public:

  HcalRechitIsoCalculator(const edm::Event &iEvent, const edm::EventSetup &iSetup,
	       const edm::Handle<HBHERecHitCollection> hbhe,
	       const edm::Handle<HFRecHitCollection> hfLabel,
	       const edm::Handle<HORecHitCollection> hoLabel) ;

  /// Return the hcal rechit energy in a cone around the SC
  double getHcalRechitIso (const reco::SuperClusterRef clus, const double i, const double threshold, const double innerR=0.0);
  /// Return the background-subtracted hcal rechit energy in a cone around the SC
  double getBkgSubHcalRechitIso(const reco::SuperClusterRef clus, const double i, const double threshold, const double innerR=0.0);

private:

  const HBHERecHitCollection         *fHBHERecHits_;
  const HORecHitCollection           *fHORecHits_;
  const HFRecHitCollection           *fHFRecHits_;
  const CaloGeometry                 *geometry_;
};

#endif
