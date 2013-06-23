#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"


#include "CondTools/Ecal/interface/EcalDBCopy.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"
#include "CondFormats/EcalObjects/interface/EcalTimeCalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalTimeCalibConstantsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalLinearCorrections.h"
#include "CondFormats/DataRecord/interface/EcalLinearCorrectionsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstantsMC.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsMCRcd.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibErrors.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibErrorsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"
#include "CondFormats/EcalObjects/interface/EcalWeightXtalGroups.h"
#include "CondFormats/DataRecord/interface/EcalWeightXtalGroupsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalTBWeights.h"
#include "CondFormats/DataRecord/interface/EcalTBWeightsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAlphas.h"
#include "CondFormats/DataRecord/interface/EcalLaserAlphasRcd.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatios.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRcd.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatiosRef.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRefRcd.h"
#include "CondFormats/EcalObjects/interface/EcalTPGCrystalStatus.h"
#include "CondFormats/DataRecord/interface/EcalTPGCrystalStatusRcd.h"

#include "CondFormats/EcalObjects/interface/EcalDCSTowerStatus.h"
#include "CondFormats/DataRecord/interface/EcalDCSTowerStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalDAQTowerStatus.h"
#include "CondFormats/DataRecord/interface/EcalDAQTowerStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalDQMTowerStatus.h"
#include "CondFormats/DataRecord/interface/EcalDQMTowerStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalDQMChannelStatus.h"
#include "CondFormats/DataRecord/interface/EcalDQMChannelStatusRcd.h"

#include "CondFormats/EcalObjects/interface/EcalClusterCrackCorrParameters.h"
#include "CondFormats/DataRecord/interface/EcalClusterCrackCorrParametersRcd.h"
#include "CondFormats/EcalObjects/interface/EcalClusterEnergyCorrectionParameters.h"
#include "CondFormats/DataRecord/interface/EcalClusterEnergyCorrectionParametersRcd.h"
#include "CondFormats/EcalObjects/interface/EcalClusterEnergyUncertaintyParameters.h"
#include "CondFormats/DataRecord/interface/EcalClusterEnergyUncertaintyParametersRcd.h"
#include "CondFormats/EcalObjects/interface/EcalClusterEnergyCorrectionObjectSpecificParameters.h"
#include "CondFormats/DataRecord/interface/EcalClusterEnergyCorrectionObjectSpecificParametersRcd.h"
#include "CondFormats/EcalObjects/interface/EcalClusterLocalContCorrParameters.h"
#include "CondFormats/DataRecord/interface/EcalClusterLocalContCorrParametersRcd.h"

#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/AlignmentRecord/interface/EBAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/EEAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/ESAlignmentRcd.h"
#include "CondFormats/EcalObjects/interface/EcalTimeOffsetConstant.h"
#include "CondFormats/DataRecord/interface/EcalTimeOffsetConstantRcd.h"

#include "CondFormats/EcalObjects/interface/EcalSampleMask.h"
#include "CondFormats/DataRecord/interface/EcalSampleMaskRcd.h"

#include <vector>

EcalDBCopy::EcalDBCopy(const edm::ParameterSet& iConfig) :
  m_timetype(iConfig.getParameter<std::string>("timetype")),
  m_cacheIDs(),
  m_records()
{

  std::string container;
  std::string tag;
  std::string record;
  typedef std::vector< edm::ParameterSet > Parameters;
  Parameters toCopy = iConfig.getParameter<Parameters>("toCopy");
  for(Parameters::iterator i = toCopy.begin(); i != toCopy.end(); ++i) {
    container = i->getParameter<std::string>("container");
    record = i->getParameter<std::string>("record");
    m_cacheIDs.insert( std::make_pair(container, 0) );
    m_records.insert( std::make_pair(container, record) );
  }
  
}


EcalDBCopy::~EcalDBCopy()
{
  
}

void EcalDBCopy::analyze( const edm::Event& evt, const edm::EventSetup& evtSetup)
{

  std::string container;
  std::string record;
  typedef std::map<std::string, std::string>::const_iterator recordIter;
  for (recordIter i = m_records.begin(); i != m_records.end(); ++i) {
    container = (*i).first;
    record = (*i).second;
    if ( shouldCopy(evtSetup, container) ) {
      copyToDB(evtSetup, container);
    }
  }
  
}



bool EcalDBCopy::shouldCopy(const edm::EventSetup& evtSetup, std::string container)
{

  unsigned long long cacheID = 0;
  if (container == "EcalPedestals") {
    cacheID = evtSetup.get<EcalPedestalsRcd>().cacheIdentifier();
  } else if (container == "EcalADCToGeVConstant") {
    cacheID = evtSetup.get<EcalADCToGeVConstantRcd>().cacheIdentifier();
  } else if (container == "EcalLinearCorrections") {
    cacheID = evtSetup.get<EcalLinearCorrectionsRcd>().cacheIdentifier();
  } else if (container == "EcalIntercalibConstants") {
    cacheID = evtSetup.get<EcalIntercalibConstantsRcd>().cacheIdentifier();
  } else if (container == "EcalIntercalibConstantsMC") {
    cacheID = evtSetup.get<EcalIntercalibConstantsMCRcd>().cacheIdentifier();
  } else if (container == "EcalIntercalibErrors") {
    cacheID = evtSetup.get<EcalIntercalibErrorsRcd>().cacheIdentifier();
  } else if (container == "EcalGainRatios") {
    cacheID = evtSetup.get<EcalGainRatiosRcd>().cacheIdentifier();
  } else if (container == "EcalWeightXtalGroups") {
    cacheID = evtSetup.get<EcalWeightXtalGroupsRcd>().cacheIdentifier();
  } else if (container == "EcalTBWeights") {
    cacheID = evtSetup.get<EcalTBWeightsRcd>().cacheIdentifier();
  } else if (container == "EcalLaserAPDPNRatios") {
    cacheID = evtSetup.get<EcalLaserAPDPNRatiosRcd>().cacheIdentifier();
  } else if (container == "EcalLaserAPDPNRatiosRef") {
    cacheID = evtSetup.get<EcalTBWeightsRcd>().cacheIdentifier();
  } else if (container == "EcalLaserAlphas") {
    cacheID = evtSetup.get<EcalTBWeightsRcd>().cacheIdentifier();
  } else if (container == "EcalChannelStatus") {
    cacheID = evtSetup.get<EcalChannelStatusRcd>().cacheIdentifier();
  } else if (container == "EcalDQMTowerStatus") {
    cacheID = evtSetup.get<EcalDQMTowerStatusRcd>().cacheIdentifier();
  } else if (container == "EcalDQMChannelStatus") {
    cacheID = evtSetup.get<EcalDQMChannelStatusRcd>().cacheIdentifier();
  } else if (container == "EcalDCSTowerStatus") {
    cacheID = evtSetup.get<EcalDCSTowerStatusRcd>().cacheIdentifier();
  } else if (container == "EcalDAQTowerStatus") {
    cacheID = evtSetup.get<EcalDAQTowerStatusRcd>().cacheIdentifier();
  } else if (container == "EcalTimeCalibConstants") {
    cacheID = evtSetup.get<EcalTimeCalibConstantsRcd>().cacheIdentifier();
  } else if (container == "EcalClusterCrackCorrParameters") {
    cacheID = evtSetup.get<EcalClusterCrackCorrParametersRcd>().cacheIdentifier();
  } else if (container == "EcalClusterEnergyUncertaintyParameters") {
    cacheID = evtSetup.get<EcalClusterEnergyUncertaintyParametersRcd>().cacheIdentifier();
  } else if (container == "EcalClusterEnergyCorrectionParameters") {
    cacheID = evtSetup.get<EcalClusterEnergyCorrectionParametersRcd>().cacheIdentifier();
  } else if (container == "EcalClusterEnergyCorrectionObjectSpecificParameters") {
    cacheID = evtSetup.get<EcalClusterEnergyCorrectionObjectSpecificParametersRcd>().cacheIdentifier();
  } else if (container == "EcalClusterLocalContCorrParameters") {
    cacheID = evtSetup.get<EcalClusterLocalContCorrParametersRcd>().cacheIdentifier();
  } else if (container == "EcalTPGCrystalStatus") {
    cacheID = evtSetup.get<EcalTPGCrystalStatusRcd>().cacheIdentifier();
  } else if (container == "EBAlignment") {
    cacheID = evtSetup.get<EBAlignmentRcd>().cacheIdentifier();
  } else if (container == "EEAlignment") {
    cacheID = evtSetup.get<EEAlignmentRcd>().cacheIdentifier();
  } else if (container == "ESAlignment") {
    cacheID = evtSetup.get<ESAlignmentRcd>().cacheIdentifier();
  } else if (container == "EcalTimeOffsetConstant") {
    cacheID = evtSetup.get<EcalTimeOffsetConstantRcd>().cacheIdentifier();
  } else if (container == "EcalSampleMask") {
    cacheID = evtSetup.get<EcalSampleMaskRcd>().cacheIdentifier();
  }

  else {
    throw cms::Exception("Unknown container");
  }
  
  if (m_cacheIDs[container] == cacheID) {
    return 0;
  } else {
    m_cacheIDs[container] = cacheID;
    return 1;
  }

}



void EcalDBCopy::copyToDB(const edm::EventSetup& evtSetup, std::string container)
{
  edm::Service<cond::service::PoolDBOutputService> dbOutput;
  if ( !dbOutput.isAvailable() ) {
    throw cms::Exception("PoolDBOutputService is not available");
  }

  std::string recordName = m_records[container];

  if (container == "EcalPedestals") {
    edm::ESHandle<EcalPedestals> handle;
    evtSetup.get<EcalPedestalsRcd>().get(handle);
    const EcalPedestals* obj = handle.product();
    std::cout << "ped pointer is: "<< obj<< std::endl;
    dbOutput->createNewIOV<const EcalPedestals>( new EcalPedestals(*obj), dbOutput->beginOfTime(),dbOutput->endOfTime(),recordName);

  }  else if (container == "EcalADCToGeVConstant") {
    edm::ESHandle<EcalADCToGeVConstant> handle;
    evtSetup.get<EcalADCToGeVConstantRcd>().get(handle);
    const EcalADCToGeVConstant* obj = handle.product();
    std::cout << "adc pointer is: "<< obj<< std::endl;

   dbOutput->createNewIOV<const EcalADCToGeVConstant>( new EcalADCToGeVConstant(*obj),dbOutput->beginOfTime(), dbOutput->endOfTime(),recordName);


  }  else if (container == "EcalTimeCalibConstants") {
    edm::ESHandle<EcalTimeCalibConstants> handle;
    evtSetup.get<EcalTimeCalibConstantsRcd>().get(handle);
    const EcalTimeCalibConstants* obj = handle.product();
    std::cout << "adc pointer is: "<< obj<< std::endl;

   dbOutput->createNewIOV<const EcalTimeCalibConstants>( new EcalTimeCalibConstants(*obj),dbOutput->beginOfTime(), dbOutput->endOfTime(),recordName);

  }  else if (container == "EcalChannelStatus") {
    edm::ESHandle<EcalChannelStatus> handle;
    evtSetup.get<EcalChannelStatusRcd>().get(handle);
    const EcalChannelStatus* obj = handle.product();
    std::cout << "channel status pointer is: "<< obj<< std::endl;

   dbOutput->createNewIOV<const EcalChannelStatus>( new EcalChannelStatus(*obj),dbOutput->beginOfTime(), dbOutput->endOfTime(),recordName);

  }  else if (container == "EcalDQMChannelStatus") {
    edm::ESHandle<EcalDQMChannelStatus> handle;
    evtSetup.get<EcalDQMChannelStatusRcd>().get(handle);
    const EcalDQMChannelStatus* obj = handle.product();
    std::cout << "DQM channel status pointer is: "<< obj<< std::endl;
    dbOutput->createNewIOV<const EcalDQMChannelStatus>( new EcalDQMChannelStatus(*obj),dbOutput->beginOfTime(), dbOutput->endOfTime(),recordName);

  }  else if (container == "EcalDQMTowerStatus") {
    edm::ESHandle<EcalDQMTowerStatus> handle;
    evtSetup.get<EcalDQMTowerStatusRcd>().get(handle);
    const EcalDQMTowerStatus* obj = handle.product();
    std::cout << "DQM Tower status pointer is: "<< obj<< std::endl;
    dbOutput->createNewIOV<const EcalDQMTowerStatus>( new EcalDQMTowerStatus(*obj),dbOutput->beginOfTime(), dbOutput->endOfTime(),recordName);

  }  else if (container == "EcalDCSTowerStatus") {
    edm::ESHandle<EcalDCSTowerStatus> handle;
    evtSetup.get<EcalDCSTowerStatusRcd>().get(handle);
    const EcalDCSTowerStatus* obj = handle.product();
    std::cout << "channel status pointer is: "<< obj<< std::endl;

   dbOutput->createNewIOV<const EcalDCSTowerStatus>( new EcalDCSTowerStatus(*obj),dbOutput->beginOfTime(), dbOutput->endOfTime(),recordName);

  }  else if (container == "EcalDAQTowerStatus") {
    edm::ESHandle<EcalDAQTowerStatus> handle;
    evtSetup.get<EcalDAQTowerStatusRcd>().get(handle);
    const EcalDAQTowerStatus* obj = handle.product();
    std::cout << "DAQ channel status pointer is: "<< obj<< std::endl;

   dbOutput->createNewIOV<const EcalDAQTowerStatus>( new EcalDAQTowerStatus(*obj),dbOutput->beginOfTime(), dbOutput->endOfTime(),recordName);


  }  else if (container == "EcalTPGCrystalStatus") {
    edm::ESHandle<EcalTPGCrystalStatus> handle;
    evtSetup.get<EcalTPGCrystalStatusRcd>().get(handle);
    const EcalTPGCrystalStatus* obj = handle.product();
    std::cout << "TPG channel status pointer is: "<< obj<< std::endl;

   dbOutput->createNewIOV<const EcalTPGCrystalStatus>( new EcalTPGCrystalStatus(*obj),dbOutput->beginOfTime(), dbOutput->endOfTime(),recordName);


  } else if (container == "EcalIntercalibConstants") {
    edm::ESHandle<EcalIntercalibConstants> handle;
    evtSetup.get<EcalIntercalibConstantsRcd>().get(handle);
    const EcalIntercalibConstants* obj = handle.product();
    std::cout << "inter pointer is: "<< obj<< std::endl;
   dbOutput->createNewIOV<const EcalIntercalibConstants>( new EcalIntercalibConstants(*obj),dbOutput->beginOfTime(), dbOutput->endOfTime(),recordName);

  } else if (container == "EcalLinearCorrections") {
    edm::ESHandle<EcalLinearCorrections> handle;
    evtSetup.get<EcalLinearCorrectionsRcd>().get(handle);
    const EcalLinearCorrections* obj = handle.product();
    std::cout << "inter pointer is: "<< obj<< std::endl;
   dbOutput->createNewIOV<const EcalLinearCorrections>( new EcalLinearCorrections(*obj),dbOutput->beginOfTime(), dbOutput->endOfTime(),recordName);

  }
else if (container == "EcalIntercalibConstantsMC") {
    edm::ESHandle<EcalIntercalibConstantsMC> handle;
    evtSetup.get<EcalIntercalibConstantsMCRcd>().get(handle);
    const EcalIntercalibConstantsMC* obj = handle.product();
    std::cout << "intercalib MC pointer is: "<< obj<< std::endl;
   dbOutput->createNewIOV<const EcalIntercalibConstantsMC>( new EcalIntercalibConstantsMC(*obj),dbOutput->beginOfTime(), dbOutput->endOfTime(),recordName);

  } else if (container == "EcalIntercalibErrors") {
    edm::ESHandle<EcalIntercalibErrors> handle;
    evtSetup.get<EcalIntercalibErrorsRcd>().get(handle);
    const EcalIntercalibErrors* obj = handle.product();
    std::cout << "inter pointer is: "<< obj<< std::endl;
   dbOutput->createNewIOV<const EcalIntercalibErrors>( new EcalIntercalibErrors(*obj),dbOutput->beginOfTime(), dbOutput->endOfTime(),recordName);

  } else if (container == "EcalGainRatios") {
    edm::ESHandle<EcalGainRatios> handle;
    evtSetup.get<EcalGainRatiosRcd>().get(handle);
    const EcalGainRatios* obj = handle.product();
    std::cout << "gain pointer is: "<< obj<< std::endl;
   dbOutput->createNewIOV<const EcalGainRatios>( new EcalGainRatios(*obj),dbOutput->beginOfTime(), dbOutput->endOfTime(),recordName);

  } else if (container == "EcalWeightXtalGroups") {
    edm::ESHandle<EcalWeightXtalGroups> handle;
    evtSetup.get<EcalWeightXtalGroupsRcd>().get(handle);
    const EcalWeightXtalGroups* obj = handle.product();
    std::cout << "weight pointer is: "<< obj<< std::endl;
   dbOutput->createNewIOV<const EcalWeightXtalGroups>( new EcalWeightXtalGroups(*obj),dbOutput->beginOfTime(), dbOutput->endOfTime(),recordName);

  } else if (container == "EcalTBWeights") {
    edm::ESHandle<EcalTBWeights> handle;
    evtSetup.get<EcalTBWeightsRcd>().get(handle);
    const EcalTBWeights* obj = handle.product();
    std::cout << "tbweight pointer is: "<< obj<< std::endl;
   dbOutput->createNewIOV<const EcalTBWeights>( new EcalTBWeights(*obj), dbOutput->beginOfTime(),dbOutput->endOfTime(),recordName);

  } else if (container == "EcalLaserAlphas") {
    edm::ESHandle<EcalLaserAlphas> handle;
    evtSetup.get<EcalLaserAlphasRcd>().get(handle);
    const EcalLaserAlphas* obj = handle.product();
    std::cout << "ecalLaserAlpha pointer is: "<< obj<< std::endl;
   dbOutput->createNewIOV<const EcalLaserAlphas>( new EcalLaserAlphas(*obj),dbOutput->beginOfTime(), dbOutput->endOfTime(),recordName);

  } else if (container == "EcalLaserAPDPNRatios") {
    edm::ESHandle<EcalLaserAPDPNRatios> handle;
    evtSetup.get<EcalLaserAPDPNRatiosRcd>().get(handle);
    const EcalLaserAPDPNRatios* obj = handle.product();
    std::cout << "tbweight pointer is: "<< obj<< std::endl;
   dbOutput->createNewIOV<const EcalLaserAPDPNRatios>( new EcalLaserAPDPNRatios(*obj),dbOutput->beginOfTime(), dbOutput->endOfTime(),recordName);

  } else if (container == "EcalLaserAPDPNRatiosRef") {
    edm::ESHandle<EcalLaserAPDPNRatiosRef> handle;
    evtSetup.get<EcalLaserAPDPNRatiosRefRcd>().get(handle);
    const EcalLaserAPDPNRatiosRef* obj = handle.product();
    std::cout << "tbweight pointer is: "<< obj<< std::endl;
   dbOutput->createNewIOV<const EcalLaserAPDPNRatiosRef>( new EcalLaserAPDPNRatiosRef(*obj),dbOutput->beginOfTime(), dbOutput->endOfTime(),recordName);


  } else if (container == "EcalClusterCrackCorrParameters") {
    edm::ESHandle<EcalClusterCrackCorrParameters> handle;
    evtSetup.get<EcalClusterCrackCorrParametersRcd>().get(handle);
    const EcalClusterCrackCorrParameters* obj = handle.product();
    std::cout << "tbweight pointer is: "<< obj<< std::endl;
   dbOutput->createNewIOV<const EcalClusterCrackCorrParameters>( new EcalClusterCrackCorrParameters(*obj),dbOutput->beginOfTime(), dbOutput->endOfTime(),recordName);


  } else if (container == "EcalClusterEnergyUncertaintyParameters") {
    edm::ESHandle<EcalClusterEnergyUncertaintyParameters> handle;
    evtSetup.get<EcalClusterEnergyUncertaintyParametersRcd>().get(handle);
    const EcalClusterEnergyUncertaintyParameters* obj = handle.product();
    std::cout << "tbweight pointer is: "<< obj<< std::endl;
   dbOutput->createNewIOV<const EcalClusterEnergyUncertaintyParameters>( new EcalClusterEnergyUncertaintyParameters(*obj),dbOutput->beginOfTime(), dbOutput->endOfTime(),recordName);


  } else if (container == "EcalClusterEnergyCorrectionParameters") {
    edm::ESHandle<EcalClusterEnergyCorrectionParameters> handle;
    evtSetup.get<EcalClusterEnergyCorrectionParametersRcd>().get(handle);
    const EcalClusterEnergyCorrectionParameters* obj = handle.product();
    std::cout << "tbweight pointer is: "<< obj<< std::endl;
   dbOutput->createNewIOV<const EcalClusterEnergyCorrectionParameters>( new EcalClusterEnergyCorrectionParameters(*obj),dbOutput->beginOfTime(), dbOutput->endOfTime(),recordName);

  } else if (container == "EcalClusterEnergyCorrectionObjectSpecificParameters") {
    edm::ESHandle<EcalClusterEnergyCorrectionObjectSpecificParameters> handle;
    evtSetup.get<EcalClusterEnergyCorrectionObjectSpecificParametersRcd>().get(handle);
    const EcalClusterEnergyCorrectionObjectSpecificParameters* obj = handle.product();
    std::cout << "tbweight pointer is: "<< obj<< std::endl;
    dbOutput->createNewIOV<const EcalClusterEnergyCorrectionObjectSpecificParameters>( new EcalClusterEnergyCorrectionObjectSpecificParameters(*obj),dbOutput->beginOfTime(), dbOutput->endOfTime(),recordName);


  } else if (container == "EcalClusterLocalContCorrParameters") {
    edm::ESHandle<EcalClusterLocalContCorrParameters> handle;
    evtSetup.get<EcalClusterLocalContCorrParametersRcd>().get(handle);
    const EcalClusterLocalContCorrParameters* obj = handle.product();
    std::cout << "tbweight pointer is: "<< obj<< std::endl;
   dbOutput->createNewIOV<const EcalClusterLocalContCorrParameters>( new EcalClusterLocalContCorrParameters(*obj),dbOutput->beginOfTime(), dbOutput->endOfTime(),recordName);

  } else if (container == "EBAlignment") {
    edm::ESHandle<Alignments> handle;
    evtSetup.get<EBAlignmentRcd>().get(handle);
    const Alignments* obj = handle.product();
    std::cout << "EB alignment pointer is: "<< obj<< std::endl;
    dbOutput->createNewIOV<const Alignments>( new Alignments(*obj),dbOutput->beginOfTime(), dbOutput->endOfTime(),recordName);

  } else if (container == "EEAlignment") {
    edm::ESHandle<Alignments> handle;
    evtSetup.get<EEAlignmentRcd>().get(handle);
    const Alignments* obj = handle.product();
    std::cout << "EE alignment pointer is: "<< obj<< std::endl;
    dbOutput->createNewIOV<const Alignments>( new Alignments(*obj),dbOutput->beginOfTime(), dbOutput->endOfTime(),recordName);

  } else if (container == "ESAlignment") {
    edm::ESHandle<Alignments> handle;
    evtSetup.get<ESAlignmentRcd>().get(handle);
    const Alignments* obj = handle.product();
    std::cout << "ES alignment pointer is: "<< obj<< std::endl;
    dbOutput->createNewIOV<const Alignments>( new Alignments(*obj),dbOutput->beginOfTime(), dbOutput->endOfTime(),recordName);

  } else if (container == "EcalTimeOffsetConstant") {
    edm::ESHandle<EcalTimeOffsetConstant> handle;
    evtSetup.get<EcalTimeOffsetConstantRcd>().get(handle);
    const EcalTimeOffsetConstant* obj = handle.product();
    std::cout << "TimeOffset pointer is: "<< obj<< std::endl;
    dbOutput->createNewIOV<const EcalTimeOffsetConstant>( new EcalTimeOffsetConstant(*obj),dbOutput->beginOfTime(), dbOutput->endOfTime(),recordName);

 } else if (container == "EcalSampleMask") {
   edm::ESHandle<EcalSampleMask> handle;
   evtSetup.get<EcalSampleMaskRcd>().get(handle);
   const EcalSampleMask* obj = handle.product();
   std::cout << "sample mask pointer is: "<< obj<< std::endl;
   dbOutput->createNewIOV<const EcalSampleMask>( new EcalSampleMask(*obj),dbOutput->beginOfTime(), dbOutput->endOfTime(),recordName);

 } else {
    throw cms::Exception("Unknown container");
  }

  std::cout<< "EcalDBCopy wrote " << recordName << std::endl;
}
