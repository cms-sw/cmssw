#ifndef CxCalculator_h
#define CxCalculator_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"


class CxCalculator
{
public:

  CxCalculator(const edm::Event &iEvent, const edm::EventSetup &iSetup, const edm::Handle<reco::BasicClusterCollection> barrel, const edm::Handle<reco::BasicClusterCollection> endcap);

  double getCx(const reco::SuperClusterRef clus, const double i, const double threshold);
  double getCCx(const reco::SuperClusterRef clus, const double i, const double threshold); // background subtracted Cx

private:

  const reco::BasicClusterCollection *fEBclusters_;
  const reco::BasicClusterCollection *fEEclusters_;
  const CaloGeometry                 *geometry_;

};

#endif
