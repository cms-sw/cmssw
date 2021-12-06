#ifndef CalibCalorimetry_EcalTrivialCondModules_EcalTrivialObjectAnalyzer_h
#define CalibCalorimetry_EcalTrivialCondModules_EcalTrivialObjectAnalyzer_h
//
// Created: 2 Mar 2006
//          Shahram Rahatlou, University of Rome & INFN
//
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalWeightXtalGroups.h"
#include "CondFormats/DataRecord/interface/EcalWeightXtalGroupsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalTBWeights.h"
#include "CondFormats/DataRecord/interface/EcalTBWeightsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalIntercalibErrors.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibErrorsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalTimeCalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalTimeCalibConstantsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalTimeCalibErrors.h"
#include "CondFormats/DataRecord/interface/EcalTimeCalibErrorsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalTimeOffsetConstant.h"
#include "CondFormats/DataRecord/interface/EcalTimeOffsetConstantRcd.h"

#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"

#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"

#include "CondFormats/EcalObjects/interface/EcalLaserAlphas.h"
#include "CondFormats/DataRecord/interface/EcalLaserAlphasRcd.h"

#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatiosRef.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRefRcd.h"

#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatios.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRcd.h"

#include "CondFormats/EcalObjects/interface/EcalClusterLocalContCorrParameters.h"
#include "CondFormats/DataRecord/interface/EcalClusterLocalContCorrParametersRcd.h"
#include "CondFormats/EcalObjects/interface/EcalClusterCrackCorrParameters.h"
#include "CondFormats/DataRecord/interface/EcalClusterCrackCorrParametersRcd.h"
#include "CondFormats/EcalObjects/interface/EcalClusterEnergyCorrectionParameters.h"
#include "CondFormats/DataRecord/interface/EcalClusterEnergyCorrectionParametersRcd.h"
#include "CondFormats/EcalObjects/interface/EcalClusterEnergyUncertaintyParameters.h"
#include "CondFormats/DataRecord/interface/EcalClusterEnergyUncertaintyParametersRcd.h"
#include "CondFormats/EcalObjects/interface/EcalClusterEnergyCorrectionObjectSpecificParameters.h"
#include "CondFormats/DataRecord/interface/EcalClusterEnergyCorrectionObjectSpecificParametersRcd.h"

#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"

#include "CondFormats/EcalObjects/interface/EcalSampleMask.h"
#include "CondFormats/DataRecord/interface/EcalSampleMaskRcd.h"

class EcalTrivialObjectAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit EcalTrivialObjectAnalyzer(edm::ParameterSet const& p);
  ~EcalTrivialObjectAnalyzer() override {}
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

private:
  const edm::ESGetToken<EcalPedestals, EcalPedestalsRcd> pedestalsToken_;
  const edm::ESGetToken<EcalADCToGeVConstant, EcalADCToGeVConstantRcd> adcToGevConstantToken_;
  const edm::ESGetToken<EcalWeightXtalGroups, EcalWeightXtalGroupsRcd> weightXtalGroupsToken_;
  const edm::ESGetToken<EcalGainRatios, EcalGainRatiosRcd> gainRatiosToken_;
  const edm::ESGetToken<EcalIntercalibConstants, EcalIntercalibConstantsRcd> intercalibConstantsToken_;
  const edm::ESGetToken<EcalIntercalibErrors, EcalIntercalibErrorsRcd> intercalibErrorsToken_;
  const edm::ESGetToken<EcalTimeCalibConstants, EcalTimeCalibConstantsRcd> timeCalibConstantsToken_;
  const edm::ESGetToken<EcalTimeCalibErrors, EcalTimeCalibErrorsRcd> timeCalibErrorsToken_;
  const edm::ESGetToken<EcalTimeOffsetConstant, EcalTimeOffsetConstantRcd> timeOffsetConstantToken_;
  const edm::ESGetToken<EcalTBWeights, EcalTBWeightsRcd> tbWeightsToken_;
  const edm::ESGetToken<EcalClusterLocalContCorrParameters, EcalClusterLocalContCorrParametersRcd>
      clusterLocalContCorrToken_;
  const edm::ESGetToken<EcalClusterCrackCorrParameters, EcalClusterCrackCorrParametersRcd> clusterCrackCorrToken_;
  const edm::ESGetToken<EcalClusterEnergyCorrectionParameters, EcalClusterEnergyCorrectionParametersRcd>
      clusterEnergyCorrectionToken_;
  const edm::ESGetToken<EcalClusterEnergyUncertaintyParameters, EcalClusterEnergyUncertaintyParametersRcd>
      clusterEnergyUncertaintyToken_;
  const edm::ESGetToken<EcalClusterEnergyCorrectionObjectSpecificParameters,
                        EcalClusterEnergyCorrectionObjectSpecificParametersRcd>
      clusterEnergyCorrectionObjectSpecificToken_;
  const edm::ESGetToken<EcalLaserAlphas, EcalLaserAlphasRcd> laserAlphasToken_;
  const edm::ESGetToken<EcalLaserAPDPNRatiosRef, EcalLaserAPDPNRatiosRefRcd> laserAPDPNRatiosRefToken_;
  const edm::ESGetToken<EcalLaserAPDPNRatios, EcalLaserAPDPNRatiosRcd> laserAPDPNRatiosToken_;
  const edm::ESGetToken<EcalChannelStatus, EcalChannelStatusRcd> channelStatusToken_;
  const edm::ESGetToken<EcalSampleMask, EcalSampleMaskRcd> sampleMaskToken_;
};
#endif
