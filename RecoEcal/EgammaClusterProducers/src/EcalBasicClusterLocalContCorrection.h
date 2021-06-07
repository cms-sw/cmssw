#ifndef RecoEcal_EgammaClusterProducers_EcalBasicClusterLocalContCorrection_h_
#define RecoEcal_EgammaClusterProducers_EcalBasicClusterLocalContCorrection_h_

/** \class EcalBasicClusterLocalContCorrection
  *  Function to correct em object energy for energy not contained in a 5x5 crystal area in the calorimeter
  *
  *  $Id: EcalBasicClusterLocalContCorrection.h
  *  $Date:
  *  $Revision:
  *  \author Federico Ferri, CEA Saclay, November 2008
  */

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "CondFormats/DataRecord/interface/EcalClusterLocalContCorrParametersRcd.h"
#include "CondFormats/EcalObjects/interface/EcalClusterLocalContCorrParameters.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EventSetup.h"

class EcalBasicClusterLocalContCorrection {
public:
  EcalBasicClusterLocalContCorrection(edm::ConsumesCollector &&cc);

  // check initialization
  void checkInit() const;

  // compute the correction
  float operator()(const reco::BasicCluster &, const EcalRecHitCollection &) const;

  // set parameters
  void init(const edm::EventSetup &es);

private:
  int getEcalModule(DetId id) const;

  const edm::ESGetToken<EcalClusterLocalContCorrParameters, EcalClusterLocalContCorrParametersRcd> paramsToken_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeometryToken_;

  EcalClusterLocalContCorrParameters const *params_;
  CaloGeometry const *caloGeometry_;
};

#endif
