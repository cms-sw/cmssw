#include "MTDTopologyEP.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/PMTDParametersRcd.h"

//#define EDM_ML_DEBUG

MTDTopologyEP::MTDTopologyEP( const edm::ParameterSet& conf )
{
  edm::LogInfo("MTD") << "MTDTopologyEP::MTDTopologyEP";

  setWhatProduced(this);
}

MTDTopologyEP::~MTDTopologyEP()
{ 
}

void
MTDTopologyEP::fillDescriptions( edm::ConfigurationDescriptions & descriptions ) 
{
  edm::ParameterSetDescription ttc;
  descriptions.add( "mtdTopology", ttc );
}

MTDTopologyEP::ReturnType
MTDTopologyEP::produce( const MTDTopologyRcd& iRecord )
{
  edm::LogInfo("MTDTopologyEP") <<  "MTDTopologyEP::produce(const MTDTopologyRcd& iRecord)";
  edm::ESHandle<PMTDParameters> ptp;
  iRecord.getRecord<PMTDParametersRcd>().get( ptp );
  fillParameters( *ptp );
  
  return std::make_unique<MTDTopology>( mtdTopologyMode_, btlVals_, etlVals_ );
}

void
MTDTopologyEP::fillParameters( const PMTDParameters& ptp )
{  
  mtdTopologyMode_ = ptp.topologyMode_; 

  btlVals_.sideStartBit_ = ptp.vitems_[0].vpars_[0]; // 16
  btlVals_.layerStartBit_ = ptp.vitems_[0].vpars_[1]; // 16
  btlVals_.trayStartBit_ = ptp.vitems_[0].vpars_[2]; // 8
  btlVals_.moduleStartBit_ = ptp.vitems_[0].vpars_[3]; // 2
  btlVals_.sideMask_ = ptp.vitems_[0].vpars_[4]; // 0xF
  btlVals_.layerMask_ = ptp.vitems_[0].vpars_[5]; // 0xF
  btlVals_.trayMask_ = ptp.vitems_[0].vpars_[6]; // 0xFF
  btlVals_.moduleMask_ = ptp.vitems_[0].vpars_[7]; // 0x3F
  
  etlVals_.sideStartBit_ = ptp.vitems_[1].vpars_[0];
  etlVals_.layerStartBit_ = ptp.vitems_[1].vpars_[1];
  etlVals_.ringStartBit_ = ptp.vitems_[1].vpars_[2];
  etlVals_.moduleStartBit_ = ptp.vitems_[1].vpars_[3];
  etlVals_.sideMask_ = ptp.vitems_[1].vpars_[4];
  etlVals_.layerMask_ = ptp.vitems_[1].vpars_[5];
  etlVals_.ringMask_ = ptp.vitems_[1].vpars_[6];
  etlVals_.moduleMask_ = ptp.vitems_[1].vpars_[7];   

#ifdef EDM_ML_DEBUG
  
  edm::LogInfo("MTDTopologyEP") <<  "BTL values = " 
                                << btlVals_.sideStartBit_ << " " 
                                << btlVals_.layerStartBit_ << " " 
                                << btlVals_.trayStartBit_ << " " 
                                << btlVals_.moduleStartBit_ << " " 
                                << std::hex << btlVals_.sideMask_ << " " 
                                << std::hex << btlVals_.layerMask_ << " " 
                                << std::hex << btlVals_.trayMask_ << " " 
                                << std::hex << btlVals_.moduleMask_ << " " ;
  edm::LogInfo("MTDTopologyEP") << "ETL values = " 
                                << etlVals_.sideStartBit_ << " " 
                                << etlVals_.layerStartBit_ << " " 
                                << etlVals_.ringStartBit_ << " " 
                                << etlVals_.moduleStartBit_ << " " 
                                << std::hex << etlVals_.sideMask_ << " " 
                                << std::hex << etlVals_.layerMask_ << " " 
                                << std::hex << etlVals_.ringMask_ << " " 
                                << std::hex << etlVals_.moduleMask_ << " " ;

#endif

}

DEFINE_FWK_EVENTSETUP_MODULE( MTDTopologyEP);

