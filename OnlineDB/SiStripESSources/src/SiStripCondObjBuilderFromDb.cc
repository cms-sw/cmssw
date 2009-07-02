// Last commit: $Id: SiStripCondObjBuilderFromDb.cc,v 1.14 2009/04/16 12:15:30 alinn Exp $
// Latest tag:  $Name: V03-02-05 $
// Location:    $Source: /cvs_server/repositories/CMSSW/CMSSW/OnlineDB/SiStripESSources/src/SiStripCondObjBuilderFromDb.cc,v $

#include "OnlineDB/SiStripESSources/interface/SiStripCondObjBuilderFromDb.h"
#include "OnlineDB/SiStripESSources/interface/SiStripFedCablingBuilderFromDb.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/SiStripObjects/interface/SiStripThreshold.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
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
SiStripCondObjBuilderFromDb::SiStripCondObjBuilderFromDb(const edm::ParameterSet&,
							 const edm::ActivityRegistry&) 
{
  LogTrace(mlESSources_) 
    << "[SiStripCondObjBuilderFromDb::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
SiStripCondObjBuilderFromDb::SiStripCondObjBuilderFromDb() 
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
/** */
bool SiStripCondObjBuilderFromDb::checkForCompatibility(std::stringstream& input,std::stringstream& output,std::string& label){


  //get current config DB parameter
      
  SiStripDbParams::const_iterator_range partitionsRange = dbParams().partitions(); 

  SiStripDbParams::SiStripPartitions::const_iterator ipart = partitionsRange.begin();
  SiStripDbParams::SiStripPartitions::const_iterator ipartEnd = partitionsRange.end();
  for ( ; ipart != ipartEnd; ++ipart ) { 
    SiStripPartition partition=ipart->second;
    output  << "@ "
	<< " Partition " << partition.partitionName() 
	<< " CabVer "    << partition.cabVersion().first << "." << partition.cabVersion().second
        << " MaskVer "   << partition.maskVersion().first << "." << partition.maskVersion().second
      ;
    if (label!="Cabling")
      output << " FedVer "    << partition.fedVersion().first << "." << partition.fedVersion().second;
  }
  
  if (!strcmp(output.str().c_str(),input.str().c_str()))
    return false;

  return true;
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
      
      // Retrieve DET cabling (should be improved)
      fed_cabling_=new SiStripFedCabling;
      SiStripFedCablingBuilderFromDb::getFedCabling( fec_cabling, *fed_cabling_ );
      SiStripDetCabling det_cabling( *fed_cabling_ );
      
      buildStripRelatedObjects( &*db_, det_cabling );
      
      // Call virtual method that writes FED cabling object to conditions DB
      //writePedestalsToCondDb( *pedestals );
      
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
void SiStripCondObjBuilderFromDb::buildStripRelatedObjects( SiStripConfigDb* const db,
							    const SiStripDetCabling& det_cabling){

  edm::LogInfo(mlESSources_)
    << "\nSiStripCondObjBuilderFromDb::" << __func__ << "] first call to this method";


  
  // Retrieve FedDescriptions from configuration database
  SiStripConfigDb::FedDescriptionsRange descriptions = db->getFedDescriptions();
  if ( descriptions.empty() ) {
    edm::LogWarning(mlESSources_)
      << "SiStripCondObjBuilderFromDb::" << __func__ << "]"
      << " Unable to build Pedestals object!"
      << " No FED descriptions found!";

    return;
  }

  

  // Retrieve gain from configuration database
  bool IsTiming = false; 
  SiStripConfigDb::AnalysisDescriptionsRange anal_descriptions = 
    db->getAnalysisDescriptions( CommissioningAnalysisDescription::T_ANALYSIS_TIMING );
  if ( anal_descriptions.empty() ) {
        edm::LogWarning(mlESSources_)
      << "SiStripCondObjBuilderFromDb::" << __func__ << "]"
      << " Unable to build SiStripApvGain object!"
      << " No timing-scan analysis descriptions found!";
  }else {IsTiming=true;}

    
  // Retrieve list of active DetIds
  vector<uint32_t> det_ids;
  det_cabling.addActiveDetectorsRawIds(det_ids);
  if ( det_ids.empty() ) {
    edm::LogWarning(mlESSources_)
      << "SiStripCondObjBuilderFromDb::" << __func__ << "]"
      << " Unable to build Pedestals object!"
      << " No DetIds found!";
    return;
  }  
  LogTrace(mlESSources_)
    << "\n\nSiStripCondObjBuilderFromDb::" << __func__ << "]"
    << " Found " << det_ids.size() << " active DetIds";

  pedestals_=new SiStripPedestals();
  noises_=new SiStripNoises();
  threshold_= new SiStripThreshold();
  quality_=new SiStripQuality();
  gain_ = new SiStripApvGain();


  SiStripDetInfoFileReader * fr=edm::Service<SiStripDetInfoFileReader>().operator->();
  
  uint16_t nApvPairs;

  // Iterate through active DetIds
  vector<uint32_t>::const_iterator det_id = det_ids.begin();
  for ( ; det_id != det_ids.end(); det_id++ ) {

  std::stringstream ssMessage;

    
    // Ignore NULL DetIds
    if ( !(*det_id) ) { continue; }
    if ( *det_id == sistrip::invalid32_ ) { continue; }
    
    const vector<FedChannelConnection>& conns = det_cabling.getConnections(*det_id);

    if (conns.size()==0){
      edm::LogWarning(mlESSources_)
	<< "SiStripCondObjBuilderFromDb::" << __func__ << "]"
	<< " Unable to build condition object!"
	<< " No FED channel connections found for detid "<< *det_id;
      continue;
    }

    nApvPairs=fr->getNumberOfApvsAndStripLength(*det_id).first/2;
    
    ssMessage
      << "SiStripCondObjBuilderFromDb::" << __func__ << "]\n"
      << " Looking for FED channel connections for detid " << *det_id << " number of apvPairs " << nApvPairs << std::endl;


    vector<FedChannelConnection>::const_iterator ipair = conns.begin();
    vector< vector<FedChannelConnection>::const_iterator > listConns(nApvPairs,conns.end());
    for ( ; ipair != conns.end(); ipair++ ){ 
      // Check if the ApvPair is connected
      if (ipair->fedId()!=sistrip::invalid_ && ipair->apvPairNumber()<3){
	//ssMessage << "\nTEST filling listConns for detid " << *det_id << "  position " << ipair-conns.begin() << " ipair->ApvPairNumber " << ipair->apvPairNumber() << std::endl;
	ipair->print(ssMessage);
	ssMessage<< std::endl;
	listConns[ipair-conns.begin()]=ipair;
      } else {
	ssMessage
	  << "\n impossible to assign connection position in listConns " << std::endl;
	ipair->print(ssMessage);
	ssMessage << std::endl;
      } 
    }


    // Iterate through connections for given DetId and fill peds container
    SiStripPedestals::InputVector inputPedestals;
    SiStripNoises::InputVector inputNoises;
    SiStripThreshold::InputVector inputThreshold;
    SiStripQuality::InputVector inputQuality;
    SiStripApvGain::InputVector inputApvGain;


    vector< vector<FedChannelConnection>::const_iterator >::const_iterator ilistConns=listConns.begin();
  for (uint16_t apvPair=0;apvPair<listConns.size();apvPair++){
      ipair=listConns[apvPair];
      
      ssMessage << "\n Looping on listConns: connection idx " << apvPair << std::endl;
      if ( ipair == conns.end() ) {
	// Fill object with default values
	ssMessage
	  << "\n "
	  << " Unable to find FED connection for detid : " << *det_id << " APV pair number " << apvPair
	  << " Writing default values" << std::endl;
	uint16_t istrip = apvPair*sistrip::STRIPS_PER_FEDCH;  
	inputQuality.push_back(quality_->encode(istrip,sistrip::STRIPS_PER_FEDCH));
	threshold_->setData( istrip, 0., 0., inputThreshold );
	for ( ;istrip < (apvPair+1)*sistrip::STRIPS_PER_FEDCH; ++istrip ){
	  pedestals_->setData( 0.,inputPedestals );
	  noises_->setData( 0., inputNoises );
	}
	
	if(IsTiming){
	  float defaultTickHeight_ = 666.6;
	  inputApvGain.push_back(defaultTickHeight_); // APV0
	  inputApvGain.push_back(defaultTickHeight_); // APV1
	}
	continue;
      }



    
      SiStripConfigDb::AnalysisDescriptionsV::const_iterator iii = anal_descriptions.begin();
      SiStripConfigDb::AnalysisDescriptionsV::const_iterator jjj = anal_descriptions.end();
      while ( iii != jjj ) {
	CommissioningAnalysisDescription* tmp = *iii;
	uint16_t fed_id = tmp->getFedId();
	uint16_t fed_ch = SiStripFedKey::fedCh( tmp->getFeUnit(), tmp->getFeChan() );
	if ( fed_id == ipair->fedId() && fed_ch == ipair->fedCh() ) { break; }
	iii++;
      }

      if(IsTiming){
	TimingAnalysisDescription *anal=0;
	float defaultTickHeight=888.8;
	if ( iii != jjj ) { anal = dynamic_cast<TimingAnalysisDescription*>(*iii); }
	if ( anal ) {
	  float tick_height = anal->getHeight();
	  //float tick_base = anal->getBase();
	  //float tick_top = anal->getPeak();
	  inputApvGain.push_back( tick_height ); // APV0
	  inputApvGain.push_back( tick_height); // APV1
	} else {
	  inputApvGain.push_back(defaultTickHeight); // APV0
	  inputApvGain.push_back(defaultTickHeight); // APV1
	  ssMessage
	    << "\n "
	    << " Unable to find Timing Analysis Description"
	    << " Writing default values for DetId: " << *det_id
	    << " Value: " << defaultTickHeight << std::endl;
	}
      }
          
      // Check if description exists for given FED id 
      SiStripConfigDb::FedDescriptionsV::const_iterator description = descriptions.begin();
      while ( description != descriptions.end() ) {
	if ( (*description)->getFedId() ==ipair->fedId() ) { break; }
	description++;
      }
      if ( description == descriptions.end() ) { 
	ssMessage
	  << "SiStripCondObjBuilderFromDb::" << __func__ << "]"
	  << " Unable to find FED description for FED id: " << ipair->fedId() << " detid : " << *det_id << " APV pair number " << apvPair
	  << " Writing default values"<< std::endl;
	ipair->print(ssMessage);
	ssMessage<< std::endl;
	uint16_t istrip = apvPair*sistrip::STRIPS_PER_FEDCH;  
	inputQuality.push_back(quality_->encode(istrip,sistrip::STRIPS_PER_FEDCH));
	threshold_->setData( istrip, 0., 0., inputThreshold );
	for ( ;istrip < (apvPair+1)*sistrip::STRIPS_PER_FEDCH; ++istrip ){
	  pedestals_->setData( 0.,inputPedestals );
	  noises_->setData( 0., inputNoises );
	}
	continue; 
      }
      
      // Retrieve Fed9UStrips object from FED description
      const Fed9U::Fed9UStrips& strips = (*description)->getFedStrips();


      // Retrieve StripDescriptions for each APV
      uint16_t jstrip = ipair->apvPairNumber()*sistrip::STRIPS_PER_FEDCH;
      for ( uint16_t iapv = 2*ipair->fedCh(); iapv < 2*ipair->fedCh()+2; iapv++ ) {
	
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
	  if(istrip->getDisable())
	    inputQuality.push_back(quality_->encode(jstrip,1));
	  jstrip++;
	} // strip loop
      } // apv loop
    } // connection loop
    
    // Insert pedestal values into Pedestals object
    if ( !pedestals_->put( *det_id, inputPedestals ) ) {
      ssMessage
	<< "[SiStripCondObjBuilderFromDb::" << __func__ << "]"
	<< " Unable to insert values into SiStripPedestals object!"
	<< " DetId already exists!" << std::endl;
    }

    // Insert noise values into Noises object
    if ( !noises_->put( *det_id, inputNoises ) ) {
      ssMessage
	<< "[SiStripCondObjBuilderFromDb::" << __func__ << "]"
	<< " Unable to insert values into SiStripNoises object!"
	<< " DetId already exists!" << std::endl;
    }

    // Insert threshold values into Threshold object
    if ( !threshold_->put( *det_id, inputThreshold ) ) {
      ssMessage
	<< "[SiStripCondObjBuilderFromDb::" << __func__ << "]"
	<< " Unable to insert values into SiStripThreshold object!"
	<< " DetId already exists!" << std::endl;
    }

    // Insert quality values into Quality object
    uint32_t detid=*det_id;
    if (inputQuality.size()){
      quality_->compact(detid,inputQuality);
      if ( !quality_->put( *det_id, inputQuality ) ) {
	ssMessage
	  << "[SiStripCondObjBuilderFromDb::" << __func__ << "]"
	  << " Unable to insert values into SiStripThreshold object!"
	  << " DetId already exists!" << std::endl;
      }
    }

    if(IsTiming){
      // Insert tick height values into Gain object
      SiStripApvGain::Range range( inputApvGain.begin(), inputApvGain.end() );
      if ( !gain_->put( *det_id, range ) ) {
	edm::LogWarning(mlESSources_)
	  << "[SiStripCondObjBuilderFromDb::" << __func__ << "]"
	  << " Unable to insert values into SiStripApvGain object!"
	  << " DetId already exists!";
      }
    }
    
    edm::LogInfo(mlESSources_) << "\n\n--------------\n" << ssMessage.str();
  } // det id loop

}
