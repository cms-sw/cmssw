#include "CondTools/SiStrip/plugins/SiStripFedCablingReader.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripRegionCabling.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CalibTracker/Records/interface/SiStripFecCablingRcd.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibTracker/Records/interface/SiStripRegionCablingRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>

// -----------------------------------------------------------------------------
// 
SiStripFedCablingReader::SiStripFedCablingReader( const edm::ParameterSet& pset ) :
  printFecCabling_( pset.getUntrackedParameter<bool>("PrintFecCabling",false) ),
  printDetCabling_( pset.getUntrackedParameter<bool>("PrintDetCabling",false) ),
  printRegionCabling_( pset.getUntrackedParameter<bool>("PrintRegionCabling",false) )
{;}

// -----------------------------------------------------------------------------
// 
void SiStripFedCablingReader::beginRun( const edm::Run& run, 
					const edm::EventSetup& setup ) {

  edm::eventsetup::EventSetupRecordKey FedRecordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType("SiStripFedCablingRcd"));
  edm::eventsetup::EventSetupRecordKey FecRecordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType("SiStripFecCablingRcd"));
  edm::eventsetup::EventSetupRecordKey DetRecordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType("SiStripDetCablingRcd"));
  edm::eventsetup::EventSetupRecordKey RegRecordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType("SiStripRegionCablingRcd"));

  bool FedRcdfound=setup.find(FedRecordKey) != nullptr?true:false;  
  bool FecRcdfound=setup.find(FecRecordKey) != nullptr?true:false;  
  bool DetRcdfound=setup.find(DetRecordKey) != nullptr?true:false;  
  bool RegRcdfound=setup.find(RegRecordKey) != nullptr?true:false;  

  edm::ESHandle<SiStripFedCabling> fed;
  if(FedRcdfound){
    edm::LogVerbatim("SiStripFedCablingReader") 
      << "[SiStripFedCablingReader::" << __func__ << "]"
      << " Retrieving FED cabling...";
    setup.get<SiStripFedCablingRcd>().get( fed ); 
  }

  edm::ESHandle<SiStripFecCabling> fec;
  if(FecRcdfound){
    edm::LogVerbatim("SiStripFedCablingReader") 
      << "[SiStripFedCablingReader::" << __func__ << "]"
      << " Retrieving FEC cabling...";
      setup.get<SiStripFecCablingRcd>().get( fec ); 
  }

  edm::ESHandle<SiStripDetCabling> det;
  if(DetRcdfound){
    edm::LogVerbatim("SiStripFedCablingReader") 
      << "[SiStripFedCablingReader::" << __func__ << "]"
      << " Retrieving DET cabling...";
    setup.get<SiStripDetCablingRcd>().get( det ); 
  }

  edm::ESHandle<SiStripRegionCabling> region;
  if(RegRcdfound){
    edm::LogVerbatim("SiStripFedCablingReader") 
      << "[SiStripFedCablingReader::" << __func__ << "]"
      << " Retrieving REGION cabling...";
    setup.get<SiStripRegionCablingRcd>().get( region ); 
  }

  if ( !fed.isValid() ) {
    edm::LogError("SiStripFedCablingReader") 
      << " Invalid handle to FED cabling object: ";
    return;
  }

  {
    std::stringstream ss;
    ss << "[SiStripFedCablingReader::" << __func__ << "]"
       << " VERBOSE DEBUG" << std::endl;
    if(FedRcdfound) {
      edm::ESHandle<TrackerTopology> tTopo;
      setup.get<TrackerTopologyRcd>().get(tTopo);
      fed->print(ss, tTopo.product());
    }
    ss << std::endl;
    if ( FecRcdfound && printFecCabling_ && fec.isValid() ) { fec->print( ss ); }
    ss << std::endl;
    if ( DetRcdfound && printDetCabling_ && det.isValid() ) { det->print( ss ); }
    ss << std::endl;
    if ( RegRcdfound && printRegionCabling_ && region.isValid() ) { region->print( ss ); }
    ss << std::endl;
    edm::LogVerbatim("SiStripFedCablingReader") << ss.str();
  }
  
  if(FedRcdfound){
    std::stringstream ss;
    ss << "[SiStripFedCablingReader::" << __func__ << "]"
       << " TERSE DEBUG" << std::endl;
    fed->terse( ss );
    ss << std::endl;
    edm::LogVerbatim("SiStripFedCablingReader") << ss.str();
  }
  
  if(FedRcdfound){
    std::stringstream ss;
    ss << "[SiStripFedCablingReader::" << __func__ << "]"
       << " SUMMARY DEBUG" << std::endl;
    edm::ESHandle<TrackerTopology> tTopo;
    setup.get<TrackerTopologyRcd>().get(tTopo);
    fed->summary(ss, tTopo.product());
    ss << std::endl;
    edm::LogVerbatim("SiStripFedCablingReader") << ss.str();
  }
  
}
