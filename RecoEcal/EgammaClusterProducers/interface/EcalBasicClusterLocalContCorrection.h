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

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "CondFormats/DataRecord/interface/EcalClusterLocalContCorrParametersRcd.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionBaseClass.h"
#include "CondFormats/EcalObjects/interface/EcalClusterLocalContCorrParameters.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "TVector2.h"

class EcalBasicClusterLocalContCorrection {
public:
  // get/set explicit methods for parameters
  const EcalClusterLocalContCorrParameters *getParameters() const { return params_; }
  // check initialization
  void checkInit() const;

  // compute the correction
  float operator()(const reco::BasicCluster &, const EcalRecHitCollection &) const;

  // set parameters
  void init(const edm::EventSetup &es);

private:
  int getEcalModule(DetId id) const;

  edm::ESHandle<EcalClusterLocalContCorrParameters> esParams_;
  const EcalClusterLocalContCorrParameters *params_;
  const edm::EventSetup *es_;  //needed to access the ECAL geometry
};

#endif
