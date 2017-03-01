
#include "OnlineDB/SiStripESSources/interface/SiStripCondObjBuilderFromDb.h"
#include "OnlineDB/SiStripESSources/interface/SiStripFedCablingBuilderFromDb.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/SiStripObjects/interface/SiStripThreshold.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "CondFormats/SiStripObjects/interface/SiStripLatency.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <iomanip>
#include "Fed9UUtils.hh"

using namespace std;
using namespace sistrip;

                                   
// -----------------------------------------------------------------------------
/** */
SiStripCondObjBuilderFromDb::SiStripCondObjBuilderFromDb(const edm::ParameterSet& pset,
							 const edm::ActivityRegistry&):
  m_gaincalibrationfactor(static_cast<float>(pset.getUntrackedParameter<double>("GainNormalizationFactor",690.))), 
  m_defaultpedestalvalue(static_cast<float>(pset.getUntrackedParameter<double>("DefaultPedestal",0.))), 
  m_defaultnoisevalue(static_cast<float>(pset.getUntrackedParameter<double>("DefaultNoise",0.))), 
  m_defaultthresholdhighvalue(static_cast<float>(pset.getUntrackedParameter<double>("DefaultThresholdHigh",0.))), 
  m_defaultthresholdlowvalue(static_cast<float>(pset.getUntrackedParameter<double>("DefaultThresholdLow",0.))), 
  m_defaultapvmodevalue(static_cast<uint16_t>(pset.getUntrackedParameter<uint32_t>("DefaultAPVMode",37))),
  m_defaultapvlatencyvalue(static_cast<uint16_t>(pset.getUntrackedParameter<uint32_t>("DefaultAPVLatency",142))),
  m_defaulttickheightvalue(static_cast<float>(pset.getUntrackedParameter<double>("DefaultTickHeight",690.))),
  m_useanalysis(static_cast<bool>(pset.getUntrackedParameter<bool>("UseAnalysis",false))),
  m_usefed(static_cast<bool>(pset.getUntrackedParameter<bool>("UseFED",false))),
  m_usefec(static_cast<bool>(pset.getUntrackedParameter<bool>("UseFEC",false))),
  m_debug(static_cast<bool>(pset.getUntrackedParameter<bool>("DebugMode",false))),
  tTopo(buildTrackerTopology())
{
  LogTrace(mlESSources_) 
    << "[SiStripCondObjBuilderFromDb::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
SiStripCondObjBuilderFromDb::SiStripCondObjBuilderFromDb(): tTopo(buildTrackerTopology())
{
  LogTrace(mlESSources_) 
    << "[SiStripCondObjBuilderFromDb::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
SiStripCondObjBuilderFromDb::~SiStripCondObjBuilderFromDb() {
  LogTrace(mlESSources_)
    << "[SiStripCondObjBuilderFromDb::" << __func__ << "]"
    << " Destructing object...";
  delete tTopo;
}

// -----------------------------------------------------------------------------

/** */
void SiStripCondObjBuilderFromDb::checkUpdate() {
  if (!(dbParams_==dbParams())){
    dbParams_=dbParams();
    buildCondObj();
  }  
}
// -----------------------------------------------------------------------------
TrackerTopology * SiStripCondObjBuilderFromDb::buildTrackerTopology() {
  TrackerTopology::PixelBarrelValues pxbVals_;
  TrackerTopology::PixelEndcapValues pxfVals_;
  TrackerTopology::TECValues tecVals_;
  TrackerTopology::TIBValues tibVals_;
  TrackerTopology::TIDValues tidVals_;
  TrackerTopology::TOBValues tobVals_;

  pxbVals_.layerStartBit_        =   16;
  pxbVals_.ladderStartBit_       =   8;
  pxbVals_.moduleStartBit_       =   2;
  pxbVals_.layerMask_            =   0xF;
  pxbVals_.ladderMask_           =   0xFF;
  pxbVals_.moduleMask_           =   0x3F;
  pxfVals_.sideStartBit_         =   23;
  pxfVals_.diskStartBit_         =   16;
  pxfVals_.bladeStartBit_        =   10;
  pxfVals_.panelStartBit_        =   8;
  pxfVals_.moduleStartBit_       =   2;
  pxfVals_.sideMask_             =   0x3;
  pxfVals_.diskMask_             =   0xF;
  pxfVals_.bladeMask_            =   0x3F;
  pxfVals_.panelMask_            =   0x3;
  pxfVals_.moduleMask_           =   0x3F;
  tecVals_.sideStartBit_         =   18;
  tecVals_.wheelStartBit_        =   14;
  tecVals_.petal_fw_bwStartBit_  =   12;
  tecVals_.petalStartBit_        =   8;
  tecVals_.ringStartBit_         =   5;
  tecVals_.moduleStartBit_       =   2;
  tecVals_.sterStartBit_         =   0;
  tecVals_.sideMask_             =   0x3;
  tecVals_.wheelMask_            =   0xF;
  tecVals_.petal_fw_bwMask_      =   0x3;
  tecVals_.petalMask_            =   0xF;
  tecVals_.ringMask_             =   0x7;
  tecVals_.moduleMask_           =   0x7;
  tecVals_.sterMask_             =   0x3;
  tibVals_.layerStartBit_        =   14;
  tibVals_.str_fw_bwStartBit_    =   12;
  tibVals_.str_int_extStartBit_  =   10;
  tibVals_.strStartBit_          =   4;
  tibVals_.moduleStartBit_       =   2;
  tibVals_.sterStartBit_         =   0;
  tibVals_.layerMask_            =   0x7;
  tibVals_.str_fw_bwMask_        =   0x3;
  tibVals_.str_int_extMask_      =   0x3;
  tibVals_.strMask_              =   0x3F;
  tibVals_.moduleMask_           =   0x3;
  tibVals_.sterMask_             =   0x3;
  tidVals_.sideStartBit_         =   13;
  tidVals_.wheelStartBit_        =   11;
  tidVals_.ringStartBit_         =   9;
  tidVals_.module_fw_bwStartBit_ =   7;
  tidVals_.moduleStartBit_       =   2;
  tidVals_.sterStartBit_         =   0;
  tidVals_.sideMask_             =   0x3;
  tidVals_.wheelMask_            =   0x3;
  tidVals_.ringMask_             =   0x3;
  tidVals_.module_fw_bwMask_     =   0x3;
  tidVals_.moduleMask_           =   0x1F;
  tidVals_.sterMask_             =   0x3;
  tobVals_.layerStartBit_        =   14;
  tobVals_.rod_fw_bwStartBit_    =   12;
  tobVals_.rodStartBit_          =   5;
  tobVals_.moduleStartBit_       =   2;
  tobVals_.sterStartBit_         =   0;
  tobVals_.layerMask_            =   0x7;
  tobVals_.rod_fw_bwMask_        =   0x3;
  tobVals_.rodMask_              =   0x7F;
  tobVals_.moduleMask_           =   0x7;
  tobVals_.sterMask_             =   0x3;

  return new TrackerTopology(pxbVals_, pxfVals_, tecVals_, tibVals_, tidVals_, tobVals_);
}
// -----------------------------------------------------------------------------
/** */
bool SiStripCondObjBuilderFromDb::checkForCompatibility(std::stringstream& input,std::stringstream& output,std::string& label){
  // DEPRECATED. Superseded by SiStripCondObjBuilderFromDb::getConfigString(const std::type_info& typeInfo).

  //get current config DB parameter
      
  SiStripDbParams::const_iterator_range partitionsRange = dbParams().partitions(); 

  SiStripDbParams::SiStripPartitions::const_iterator ipart = partitionsRange.begin();
  SiStripDbParams::SiStripPartitions::const_iterator ipartEnd = partitionsRange.end();
  for ( ; ipart != ipartEnd; ++ipart ) { 
    SiStripPartition partition=ipart->second;
    output  << "@ "
	    << " Partition " << partition.partitionName() ;
    if (label!="Cabling" && label !="ApvLatency")
	output << " FedVer "    << partition.fedVersion().first << "." << partition.fedVersion().second;       
    if(label=="Cabling")
      output << " CabVer "    << partition.cabVersion().first << "." << partition.cabVersion().second
	     << " MaskVer "   << partition.maskVersion().first << "." << partition.maskVersion().second;
    if(label=="ApvTiming")
      output<< " ApvTimingVer " << partition.apvTimingVersion().first << "." << partition.apvTimingVersion().second;
    if(label=="ApvLatency")
      output<< " FecVersion " << partition.fecVersion().first << "." << partition.fecVersion().second;
  }
  
  if (!strcmp(output.str().c_str(),input.str().c_str()))
    return false;

  return true;
}
// -----------------------------------------------------------------------------
/** */
std::string SiStripCondObjBuilderFromDb::getConfigString(const std::type_info& typeInfo){
  // create config line used by fast O2O

  std::stringstream output;

  SiStripDbParams::const_iterator_range partitionsRange = dbParams().partitions();
  SiStripDbParams::SiStripPartitions::const_iterator ipart = partitionsRange.begin();
  SiStripDbParams::SiStripPartitions::const_iterator ipartEnd = partitionsRange.end();
  for ( ; ipart != ipartEnd; ++ipart ) {
    SiStripPartition partition=ipart->second;
    output << "%%" << "Partition: " << partition.partitionName();

    // Make everything depend on cabVersion and maskVersion!
    output << " CablingVersion: " << partition.cabVersion().first << "." << partition.cabVersion().second;
    output << " MaskVersion: " << partition.maskVersion().first << "." << partition.maskVersion().second;

    if(typeInfo==typeid(SiStripFedCabling)){
      // Do nothing. FedCabling only depends on cabVersion and maskVersion.
    }
    else if(typeInfo==typeid(SiStripLatency)){
      // Latency is FEC related, add fecVersion.
      output << " FecVersion: " << partition.fecVersion().first << "." << partition.fecVersion().second;
    }else{
      // BadStrip, Noises, Pedestals and Thresholds are FED related, add fecVersion.
      output << " FedVersion: " << partition.fedVersion().first << "." << partition.fedVersion().second;
      if(typeInfo==typeid(SiStripApvGain)){
        // Not used in O2O.
        output << " ApvTimingVersion: " << partition.apvTimingVersion().first << "." << partition.apvTimingVersion().second;
      }
    }
  }

  return output.str();

}
// -----------------------------------------------------------------------------
/** */
void SiStripCondObjBuilderFromDb::buildCondObj() {
  LogTrace(mlESSources_) 
    << "[SiStripCondObjBuilderFromDb::" << __func__ << "]";

  // Check if DB connection is made 
  if ( db_ ) { 
    
    // Check if DB connection is made 
    if ( db_->deviceFactory() || 
	 db_->databaseCache() ) { 
      
      // Build FEC cabling object
      SiStripFecCabling fec_cabling;
      
      SiStripFedCablingBuilderFromDb::buildFecCabling( &*db_, 
						       fec_cabling, 
						       sistrip::CABLING_FROM_CONNS );
      fed_cabling_=new SiStripFedCabling;

      SiStripFedCablingBuilderFromDb::getFedCabling( fec_cabling, *fed_cabling_ );
      SiStripDetCabling det_cabling( *fed_cabling_, tTopo );
      buildStripRelatedObjects( &*db_, det_cabling );
     
     
      if(m_useanalysis)buildAnalysisRelatedObjects(&*db_, v_trackercon);
      if(m_usefed) buildFEDRelatedObjects(&*db_, v_trackercon);
      if(m_usefec) buildFECRelatedObjects(&*db_, v_trackercon);
         
   
      
    } else {
      edm::LogWarning(mlESSources_)
	<< "[SiStripCondObjBuilderFromDb::" << __func__ << "]"
	<< " NULL pointer to DeviceFactory returned by SiStripConfigDb!"
	<< " Cannot build Pedestals object!";
    }
  } else {
    edm::LogWarning(mlESSources_)
      << "[SiStripCondObjBuilderFromDb::" << __func__ << "]"
      << " NULL pointer to SiStripConfigDb returned by DB \"service\"!"
      << " Cannot build Pedestals object!";
  }
}

// -----------------------------------------------------------------------------
/** */
//Retrieve FedDescriptions from configuration database
bool SiStripCondObjBuilderFromDb::retrieveFedDescriptions(SiStripConfigDb* const db){
  SiStripConfigDb::FedDescriptionsRange descriptions = db->getFedDescriptions();
  if ( descriptions.empty() ) {
    edm::LogWarning(mlESSources_)
      << "SiStripCondObjBuilderFromDb::" << __func__ << "]"
      << " No FED descriptions found!";
    
    return false;
  }
  return true;
}

// -----------------------------------------------------------------------------
/** */
  // Retrieve gain from configuration database
bool SiStripCondObjBuilderFromDb::retrieveTimingAnalysisDescriptions( SiStripConfigDb* const db){
  SiStripConfigDb::AnalysisDescriptionsRange anal_descriptions = 
    db->getAnalysisDescriptions( CommissioningAnalysisDescription::T_ANALYSIS_TIMING );
  if ( anal_descriptions.empty() ) {
    edm::LogWarning(mlESSources_)
      << "SiStripCondObjBuilderFromDb::" << __func__ << "]"
      << " Unable to build SiStripApvGain object!"
      << " No timing-scan analysis descriptions found!";
    return false;
  }
  return true;
}

// -----------------------------------------------------------------------------
/** */
  // Retrieve list of active DetIds
vector<uint32_t> SiStripCondObjBuilderFromDb::retrieveActiveDetIds(const SiStripDetCabling& det_cabling){
  vector<uint32_t> det_ids;
  det_cabling.addActiveDetectorsRawIds(det_ids);
  if ( det_ids.empty() ) {
    edm::LogWarning(mlESSources_)
      << "SiStripCondObjBuilderFromDb::" << __func__ << "]"
      << " Unable to build Pedestals object!"
      << " No DetIds found!";
    return det_ids;
  }  
  LogTrace(mlESSources_)
    << "\n\nSiStripCondObjBuilderFromDb::" << __func__ << "]"
    << " Found " << det_ids.size() << " active DetIds";
  return det_ids;
}

// -----------------------------------------------------------------------------
/** */
 //build connections per DetId
vector<const FedChannelConnection *> SiStripCondObjBuilderFromDb::buildConnections(const SiStripDetCabling& det_cabling, uint32_t det_id ){
  vector<const FedChannelConnection *> conns = det_cabling.getConnections(det_id);
  if (conns.size()==0){
    edm::LogWarning(mlESSources_)
	<< "SiStripCondObjBuilderFromDb::" << __func__ << "]"
	<< " Unable to build condition object!"
	<< " No FED channel connections found for detid "<< det_id;
  }
  return conns;
}

// -----------------------------------------------------------------------------
/** */
//retrieve number of APV pairs per detid
uint16_t SiStripCondObjBuilderFromDb::retrieveNumberAPVPairs(uint32_t det_id){
  uint16_t nApvPairs;
  SiStripDetInfoFileReader * fr=edm::Service<SiStripDetInfoFileReader>().operator->();
  nApvPairs=fr->getNumberOfApvsAndStripLength(det_id).first/2;
  return nApvPairs;
}

// -----------------------------------------------------------------------------
/** */
//set default values for Cabling Objects Peds, Noise, thresh, Quality
void SiStripCondObjBuilderFromDb::setDefaultValuesCabling(uint16_t apvPair){
  uint16_t istrip = apvPair*sistrip::STRIPS_PER_FEDCH;  
  std::cout << "Found disabled FedConnection!  APVPair: " << apvPair << " Strips: " << sistrip::STRIPS_PER_FEDCH << std::endl;
  inputQuality.push_back(quality_->encode(istrip,sistrip::STRIPS_PER_FEDCH));
  for ( ;istrip < (apvPair+1)*sistrip::STRIPS_PER_FEDCH; ++istrip ){
    pedestals_->setData(m_defaultpedestalvalue,inputPedestals );
    noises_->setData(m_defaultnoisevalue ,inputNoises );
    threshold_->setData( istrip, m_defaultthresholdlowvalue, m_defaultthresholdhighvalue, inputThreshold );
   }
}

// -----------------------------------------------------------------------------
/** */
void SiStripCondObjBuilderFromDb::setDefaultValuesApvTiming(){
  inputApvGain.push_back(m_defaulttickheightvalue/m_gaincalibrationfactor); // APV0
  inputApvGain.push_back(m_defaulttickheightvalue/m_gaincalibrationfactor); // APV1
}

// -----------------------------------------------------------------------------
/** */
void SiStripCondObjBuilderFromDb::setDefaultValuesApvLatency(SiStripLatency & latency_, const FedChannelConnection& ipair, uint32_t detid, uint16_t apvnr){
  std::cout << "[SiStripCondObjBuilderFromDb::"<<__func__<<"]: Set Default Latency for Detid: " << detid << " ApvNr: " << apvnr << std::endl;
  if(!latency_.put( detid, apvnr, m_defaultapvmodevalue, m_defaultapvlatencyvalue))
    {
      std::cout << "[SiStripCondObjBuilderFromDb::"<<__func__<<"]: Unable to fill Latency for Detid: " << detid << " ApvNr: " << apvnr << std::endl;
    }
     if(!latency_.put( detid, ++apvnr, m_defaultapvmodevalue, m_defaultapvlatencyvalue))
       {
	 std::cout << "[SiStripCondObjBuilderFromDb::"<<__func__<<"]: Unable to fill Latency for Detid: " << detid << " ApvNr: " << apvnr << std::endl;
       }

}



// -----------------------------------------------------------------------------
/** */
bool SiStripCondObjBuilderFromDb::setValuesApvTiming(SiStripConfigDb* const db, FedChannelConnection &ipair){
  SiStripConfigDb::AnalysisDescriptionsRange anal_descriptions = db->getAnalysisDescriptions( CommissioningAnalysisDescription::T_ANALYSIS_TIMING );
   SiStripConfigDb::AnalysisDescriptionsV::const_iterator iii = anal_descriptions.begin();
  SiStripConfigDb::AnalysisDescriptionsV::const_iterator jjj = anal_descriptions.end();
 
  while ( iii != jjj ) {
    CommissioningAnalysisDescription* tmp = *iii;
    uint16_t fed_id = tmp->getFedId();
    uint16_t fed_ch = SiStripFedKey::fedCh( tmp->getFeUnit(), tmp->getFeChan() );
    if ( fed_id == ipair.fedId() && fed_ch == ipair.fedCh() ) { break; }
    iii++;
  }
  
  TimingAnalysisDescription *anal=0;
  if ( iii != jjj ) { anal = dynamic_cast<TimingAnalysisDescription*>(*iii); }
  if ( anal ) {
    float tick_height = (anal->getHeight() / m_gaincalibrationfactor);
    inputApvGain.push_back( tick_height ); // APV0
    inputApvGain.push_back( tick_height); // APV1
  } else {
    inputApvGain.push_back(m_defaulttickheightvalue/m_gaincalibrationfactor); // APV0
    inputApvGain.push_back(m_defaulttickheightvalue/m_gaincalibrationfactor); // APV1
    return false;
  }

  return true;
}

// -----------------------------------------------------------------------------
/** */
bool SiStripCondObjBuilderFromDb::setValuesApvLatency(SiStripLatency & latency_, SiStripConfigDb* const db, FedChannelConnection &ipair, uint32_t detid, uint16_t apvnr, SiStripConfigDb::DeviceDescriptionsRange apvs  ){
SiStripDetInfoFileReader * fr=edm::Service<SiStripDetInfoFileReader>().operator->();
 fr->getNumberOfApvsAndStripLength(detid);
 
 SiStripConfigDb::DeviceDescriptionsV::const_iterator iapv = apvs.begin();
 SiStripConfigDb::DeviceDescriptionsV::const_iterator japv = apvs.end();
 if(iapv==japv) return false;
 for ( ; iapv != japv; ++iapv ) {
   apvDescription* apv = dynamic_cast<apvDescription*>( *iapv );
   if ( !apv ) { continue; }
   if((apv->getCrateId()) != (ipair.fecCrate())) continue;
   if((apv->getFecSlot()) != (ipair.fecSlot())) continue;
   if((apv->getRingSlot()) != (ipair.fecRing())) continue;
   if((apv->getCcuAddress()) != (ipair.ccuAddr())) continue;
   if((apv->getChannel()) != (ipair.ccuChan())) continue;
     // Insert latency values into latency object
   if((apv->getAddress()) == (ipair.i2cAddr(0))) {
     if(!latency_.put( detid, apvnr, static_cast<uint16_t>(apv->getLatency()), static_cast<uint16_t>(apv->getApvMode()))){
       std::cout << "UNABLE APVLatency Put: Detid "<< dec<<detid 
                 << " APVNr.: " << apvnr 
                 << " Latency Value: " << dec <<static_cast<uint16_t>(apv->getLatency()) 
                 << " APV Mode: " << dec<< static_cast<uint16_t>(apv->getApvMode())
                 << std::endl;
       return false;
     }else{++apvnr;}
   }
   if((apv->getAddress()) == (ipair.i2cAddr(1))) {
     if(!latency_.put( detid, apvnr, static_cast<uint16_t>(apv->getLatency()), static_cast<uint16_t>(apv->getApvMode()))){
       std::cout << "UNABLE APVLatency Put: Detid "<<dec<< detid
                 << " APVNr.: " << apvnr
                 << " Latency Value: " << dec <<static_cast<uint16_t>(apv->getLatency())
                 << " APV Mode: " << dec <<static_cast<uint16_t>(apv->getApvMode())
                 << std::endl;
       continue;
       return false;
     }else{++apvnr;}
   }
  }
 return true;
}

// -----------------------------------------------------------------------------
/** */
//bool SiStripCondObjBuilderFromDb::setValuesCabling(SiStripConfigDb* const db, FedChannelConnection &ipair, uint32_t detid){ 
bool SiStripCondObjBuilderFromDb::setValuesCabling(SiStripConfigDb::FedDescriptionsRange &descriptions, FedChannelConnection &ipair, uint32_t detid){
  //SiStripConfigDb::FedDescriptionsRange descriptions = db->getFedDescriptions();
  SiStripConfigDb::FedDescriptionsV::const_iterator description = descriptions.begin();
  while ( description != descriptions.end() ) {
    if ( (*description)->getFedId() ==ipair.fedId() ) { break; }
    description++;
  }
  if ( description == descriptions.end() ) {return false;}
  // Retrieve Fed9UStrips object from FED description
  const Fed9U::Fed9UStrips& strips = (*description)->getFedStrips();
      
      
  // Retrieve StripDescriptions for each APV
  uint16_t jstrip = ipair.apvPairNumber()*sistrip::STRIPS_PER_FEDCH;
  for ( uint16_t iapv = 2*ipair.fedCh(); iapv < 2*ipair.fedCh()+2; iapv++ ) {
	
    // Get StripDescriptions for the given APV
    Fed9U::Fed9UAddress addr;
    addr.setFedApv(iapv);
    vector<Fed9U::Fed9UStripDescription> strip = strips.getApvStrips(addr);
	
    vector<Fed9U::Fed9UStripDescription>::const_iterator istrip = strip.begin();

    for ( ; istrip != strip.end(); istrip++ ) {
      pedestals_->setData( istrip->getPedestal() , inputPedestals);
      noises_   ->setData( istrip->getNoise()    , inputNoises );
      threshold_->setData( jstrip, istrip->getLowThresholdFactor(),
			   istrip->getHighThresholdFactor(), inputThreshold );
      if(istrip->getDisable()){
	std::cout << "Found disabled strip! Detid: " << detid << " APVNr: " << iapv << " Strips: " << jstrip << std::endl;

	inputQuality.push_back(quality_->encode(jstrip,1));
      }
      jstrip++;
    }
  }
  return true;
}


// -----------------------------------------------------------------------------
/** */
//store objects
void SiStripCondObjBuilderFromDb::storePedestals(uint32_t det_id){
  if ( !pedestals_->put(det_id, inputPedestals ) ) {
    std::cout
      << "[SiStripCondObjBuilderFromDb::" << __func__ << "]"
      << " Unable to insert values into SiStripPedestals object!"
      << " DetId already exists!" << std::endl;
  }
  inputPedestals.clear();

  }



// -----------------------------------------------------------------------------
/** */
void SiStripCondObjBuilderFromDb::storeNoise(uint32_t det_id){
  // Insert noise values into Noises object

    if ( !noises_->put(det_id, inputNoises ) ) {
      std::cout
	<< "[SiStripCondObjBuilderFromDb::" << __func__ << "]"
	<< " Unable to insert values into SiStripNoises object!"
	<< " DetId already exists!" << std::endl;
    }
    inputNoises.clear();

 }

// -----------------------------------------------------------------------------
/** */
void SiStripCondObjBuilderFromDb::storeThreshold(uint32_t det_id){
  // Insert threshold values into Threshold object
    if ( !threshold_->put(det_id, inputThreshold ) ) {
      std::cout
	<< "[SiStripCondObjBuilderFromDb::" << __func__ << "]"
	<< " Unable to insert values into SiStripThreshold object!"
	<< " DetId already exists!" << std::endl;
    }
    inputThreshold.clear();
}

// -----------------------------------------------------------------------------
/** */
void SiStripCondObjBuilderFromDb::storeQuality(uint32_t det_id){
  // Insert quality values into Quality object
    if (inputQuality.size()){
      quality_->compact(det_id,inputQuality);
      if ( !quality_->put(det_id, inputQuality ) ) {
	std::cout
	  << "[SiStripCondObjBuilderFromDb::" << __func__ << "]"
	  << " Unable to insert values into SiStripQuality object!"
	  << " DetId already exists!" << std::endl;
      }
    }
    inputQuality.clear();
}

// -----------------------------------------------------------------------------
/** */
void SiStripCondObjBuilderFromDb::storeTiming(uint32_t det_id){
  // Insert tick height values into Gain object
      SiStripApvGain::Range range( inputApvGain.begin(), inputApvGain.end() );
      if ( !gain_->put( det_id, range ) ) {
	edm::LogWarning(mlESSources_)
	  << "[SiStripCondObjBuilderFromDb::" << __func__ << "]"
	  << " Unable to insert values into SiStripApvGain object!"
	  << " DetId already exists!";
      }
      inputApvGain.clear();
}

// -----------------------------------------------------------------------------
/** */
void SiStripCondObjBuilderFromDb::buildStripRelatedObjects( SiStripConfigDb* const db,
							    const SiStripDetCabling& det_cabling){
  //variables needed in this function
  uint16_t nApvPairs;
  vector<uint32_t>::const_iterator det_id;
  vector<uint32_t> det_ids;
 
  edm::LogInfo(mlESSources_)
    << "\n[SiStripCondObjBuilderFromDb::" << __func__ << "] first call to this method";

  //Check if FedDescriptions exist, if not return
  if (!retrieveFedDescriptions(db)) {std::cout<< "Found no FedDescription!" << std::endl;return;}
  // Retrieve list of active DetIds
  det_cabling.addActiveDetectorsRawIds(det_ids);
  if ( det_ids.empty() ) {
    std::cout
      << "SiStripCondObjBuilderFromDb::" << __func__ << "]"
      << " Unable to build Pedestals object!"
      << " No DetIds found!" << std::endl;
    return;
  }  
  std::cout << "\n\nSiStripCondObjBuilderFromDb::" << __func__ << "]"
	    << " Found " << det_ids.size() << " active DetIds";

  // Loop Active DetIds
  det_id = det_ids.begin();
  for ( ; det_id != det_ids.end(); det_id++ ) {
    std::stringstream ssMessage;
              
    // Ignore NULL DetIds
    if ( !(*det_id) ) { continue; }
    if ( *det_id == sistrip::invalid32_ ) { continue; }
       
       
    //build connections per DetId
    const vector<const FedChannelConnection *>& conns=buildConnections(det_cabling, *det_id);
       
    vector<const FedChannelConnection *>::const_iterator ipair = conns.begin();
    if(conns.size() ==0 ) continue;
       
    //retrieve number of APV pairs per detid
    nApvPairs=retrieveNumberAPVPairs(*det_id);
       

    //loop connections and check if APVPair is connected
    vector< vector<const FedChannelConnection *>::const_iterator > listConns(nApvPairs,conns.end());
              
    for ( ; ipair != conns.end(); ++ipair ){
      // Check if the ApvPair is connected
      if ( !(*ipair) ) continue;
      if ((*ipair)->fedId()!=sistrip::invalid_ && (*ipair)->apvPairNumber()<3){
        // (*ipair)->print(ssMessage);
	// ssMessage<< std::endl;
	listConns[ipair-conns.begin()]=ipair;
      } else {
	std::cout
	  << "\n impossible to assign connection position in listConns " << std::endl;
        // (*ipair)->print(ssMessage);
	// ssMessage << std::endl;
      }
    }
    
    // get data
    // vector< vector<const FedChannelConnection *>::const_iterator >::const_iterator ilistConns=listConns.begin();
    for (uint16_t apvPair=0;apvPair<listConns.size();++apvPair){
      ipair=listConns[apvPair];
          if ( ipair == conns.end() ) {
	// Fill object with default values
	std::cout
	  << "\n "
	  << " Unable to find FED connection for detid : " << std::dec << *det_id << " APV pair number " << apvPair
	  << " Writing default values" << std::endl;
//	(*ipair)->print(ssMessage); // this will crash!
	//If no connection was found, add 100 to apvpair
	apvPair+=100;
	std::cout << " Put apvPair+100:" << apvPair << " into vector!" << std::endl;
	// use dummy FedChannelConnection since it's not used in this case
	FedChannelConnection dummy;
	p_apvpcon=std::make_pair(apvPair,dummy);
	v_apvpcon.push_back(p_apvpcon);
	apvPair=apvPair-100;
	continue;
      }
      p_apvpcon=std::make_pair(apvPair,**ipair);
      v_apvpcon.push_back(p_apvpcon);
    } //conns loop 
    p_detcon=std::make_pair(*det_id,v_apvpcon);
    v_trackercon.push_back(p_detcon);
    v_apvpcon.clear();
  } // det id loop
}

// -----------------------------------------------------------------------------
/** */
void SiStripCondObjBuilderFromDb::buildAnalysisRelatedObjects( SiStripConfigDb* const db, const trackercon& _tc){
  trackercon tc = _tc;
  std::cout << "Entering [SiStripCondObjBuilderFromDb::"<<__func__ <<"]"<<std::endl;
  //data container
  gain_= new SiStripApvGain();

  //check if Timing analysis description is found, otherwise quit
  if(!retrieveTimingAnalysisDescriptions(&*db_)){
    edm::LogWarning(mlESSources_)
      << "[SiStripCondObjBuilderFromDb::" << __func__ << "]"
      << " NULL pointer to AnalysisDescriptions returned by SiStripConfigDb!"
      << " Cannot build Analysis object! QUIT";
    // some values have to be set, otherwise PopCon will crash
    setDefaultValuesApvTiming();
    storeTiming(4711);
    return;
  }

  i_trackercon detids_end=tc.end();

  //loop detids
  for(i_trackercon detids=tc.begin();detids!=detids_end;detids++){
    uint32_t detid = (*detids).first;
    i_apvpairconn connections_end=((*detids).second).end();
    
    //loop connections
    for(i_apvpairconn connections=((*detids).second).begin();connections!=connections_end;connections++){
      uint32_t apvPair =(*connections).first;
      FedChannelConnection ipair =(*connections).second;

          
      //no connection for apvPair found
      if(apvPair>=100){
	setDefaultValuesApvTiming();  
	continue;
      }
      
      //fill data
      if(!setValuesApvTiming(db, ipair)){
 	std::cout
 	  << "\n "
 	  << " Unable to find Timing Analysis Description"
 	  << " Writing default values for DetId: " << detid
 	  << " Value: " << m_defaulttickheightvalue/m_gaincalibrationfactor << std::endl;
 	setDefaultValuesApvTiming();
      }
    }//connections
    storeTiming(detid);
  }//detids

}
 
// -----------------------------------------------------------------------------
/** */
void SiStripCondObjBuilderFromDb::buildFECRelatedObjects( SiStripConfigDb* const db, const trackercon& _tc){
  trackercon tc = _tc;
  std::cout << "Entering [SiStripCondObjBuilderFromDb::"<<__func__ <<"]"<<std::endl;
  //data container
  latency_ = new SiStripLatency();

  i_trackercon detids_end=tc.end();
 
  // get APV DeviceDescriptions
  SiStripConfigDb::DeviceDescriptionsRange apvs= db->getDeviceDescriptions( APV25 );;


  //loop detids
  for(i_trackercon detids=tc.begin();detids!=detids_end;detids++){
    uint32_t detid = (*detids).first;
    uint16_t apvnr=1;
    i_apvpairconn connections_end=((*detids).second).end();
    

    //loop connections
    for(i_apvpairconn connections=((*detids).second).begin();connections!=connections_end;connections++){
      uint32_t apvPair =(*connections).first;
      FedChannelConnection ipair =(*connections).second;
      
      //no connection for apvPair found
      if(apvPair>=100){
	//setDefaultValuesApvLatency((*latency_),ipair, detid, apvnr);  
	std::cout << "[SiStripCondObjBuilderFromDb::" << __func__ << "] No FEDConnection for DetId " << detid << " ApvPair " << apvPair-100 << " found, skipping Latency Insertion!" << std::endl;
	continue;
      }
      
      //fill data
           if(!setValuesApvLatency((*latency_),db, ipair, detid, apvnr, apvs)){
 	std::cout
 	  << "\n "
 	  << " Unable to find FEC Description"
 	  << " Skipping Insertion for DetId: " << detid << std::endl;
	  //setDefaultValuesApvLatency((*latency_),ipair, detid, apvnr);
       }
       apvnr+=2;
    }//connections
     // compact Latency Object
  
  }//detids
  latency_->compress();
  std::stringstream ss;
  // latency debug output
  latency_->printSummary(ss);
  latency_->printDebug(ss);
  std::cout << ss.str() << std::endl;
}

// -----------------------------------------------------------------------------
/** */
void SiStripCondObjBuilderFromDb::buildFEDRelatedObjects( SiStripConfigDb* const db, const trackercon& _tc){
  trackercon tc = _tc;
  std::cout << "Entering [SiStripCondObjBuilderFromDb::"<<__func__ <<"]"<<std::endl;

  //data containers
  pedestals_= new SiStripPedestals();  
  noises_ = new SiStripNoises();  
  threshold_= new SiStripThreshold();  
  quality_ = new SiStripQuality();  
 
  i_trackercon detids_end=tc.end();

  //Build FED Descriptions out of db object
  SiStripConfigDb::FedDescriptionsRange descriptions = db->getFedDescriptions();

  //loop detids
  for(i_trackercon detids=tc.begin();detids!=detids_end;detids++){
    uint32_t detid = (*detids).first;
    i_apvpairconn connections_end=((*detids).second).end();
  
    //loop connections
    for(i_apvpairconn connections=((*detids).second).begin();connections!=connections_end;connections++){
      uint32_t apvPair =(*connections).first;
      FedChannelConnection ipair =(*connections).second;
            
      //no connection for apvPair found
      if(apvPair>=100){
	std::cout
	  << "[SiStripCondObjBuilderFromDb::" << __func__ << "]"
	  << " Unable to find FED description for FED id: " << ipair.fedId()
	  << " detid : " << detid << " APV pair number " << apvPair
	  << " Writing default values"<< std::endl; 
	setDefaultValuesCabling((apvPair-100)); 
	continue;
      }
      //  if(!setValuesCabling(db, ipair, detid)){
      if(!setValuesCabling(descriptions, ipair, detid)){
	std::cout
	  << "[SiStripCondObjBuilderFromDb::" << __func__ << "]"
	  << " Unable to find FED description for FED id: " << ipair.fedId()
	  << " detid : " << detid << " APV pair number " << apvPair
	  << " Writing default values"<< std::endl; 
	setDefaultValuesCabling(apvPair); 
      }
    }//connections
    storePedestals(detid);
    storeNoise(detid);
    storeThreshold(detid);
    storeQuality(detid);
  }//detids
}


