#include "OnlineDB/SiStripESSources/interface/SiStripPopulateConfigDb.h"
// 
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
//
#include "DataFormats/SiStripDetId/interface/SiStripControlKey.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
//
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
//
//#include "OnlineDB/SiStripESSources/interface/SiStripFedCablingBuilderFromDb.h"
// 
#include <iostream>
#include <sstream>
#include <fstream>

using namespace std;

// -----------------------------------------------------------------------------
/** */
SiStripPopulateConfigDb::SiStripPopulateConfigDb( const edm::ParameterSet& pset ) 
  : db_(0),
    maxNumberOfDets_( pset.getUntrackedParameter<int>("MaxNumberOfDets",10) )
{
  edm::LogInfo("SiStripConfigDb") << "[SiStripPopulateConfigDb::SiStripPopulateConfigDb]"
				  << " Constructing object...";
  if ( pset.getUntrackedParameter<bool>( "UsingDb", false ) ) {
    // Using database 
    db_ = new SiStripConfigDb( pset.getUntrackedParameter<string>("User",""),
			       pset.getUntrackedParameter<string>("Passwd",""),
			       pset.getUntrackedParameter<string>("Path",""),
			       pset.getUntrackedParameter<string>("Partition","") );
  } else {
    // Using xml files
    db_ = new SiStripConfigDb( pset.getUntrackedParameter<string>("InputModuleXml",""),
			       pset.getUntrackedParameter<string>("InputDcuInfoXml",""),
			       pset.getUntrackedParameter< vector<string> >( "InputFecXml", vector<string>() ),
			       pset.getUntrackedParameter< vector<string> >( "InputFedXml", vector<string>() ) );
  }

  // Establish connection
  db_->openDbConnection();

}

// -----------------------------------------------------------------------------
/** */
SiStripPopulateConfigDb::~SiStripPopulateConfigDb() {
  edm::LogInfo("SiStripCabling") << "[SiStripPopulateConfigDb::~SiStripPopulateConfigDb]"
				 << " Destructing object...";
  if ( db_ ) { 
    db_->closeDbConnection();
    delete db_; 
  } 
}

// -----------------------------------------------------------------------------
/** */
void SiStripPopulateConfigDb::beginJob( const edm::EventSetup& iSetup ) {
  string method = "SiStripPopulateConfigDb::beginJob";
  edm::LogInfo("SiStripConfigDb") << "[SiStripPopulateConfigDb::beginJob]"
				  << " Creating TK partitions based on DetIds...";
  
  // Retrieve and organise DetIds according to partition name
  TkPartitions partitions;
  retrieveDetIds( iSetup, maxNumberOfDets_, partitions );
  
  // Create FEC cabling and descriptions, populate DCU-DetId map 
  SiStripConfigDb::DcuDetIdMap dcu_detid_map;
  for ( uint16_t ip = 0; ip < partitions.size(); ip++ ) {
    edm::LogInfo("SiStripConfigDb") 
      << "[SiStripPopulateConfigDb::beginJob]"
      << " Creating partition '" << partitions[ip].first 
      << "' with " << partitions[ip].second.size() << " dets";
    SiStripFecCabling fec_cabling;
    createFecCabling( ip, partitions, fec_cabling, dcu_detid_map );
    db_->createPartition( partitions[ip].first, fec_cabling );
  }

  // Upload DCU-DetId map to database
  if ( dcu_detid_map.empty() ) {
    stringstream ss;
    ss << "["<<method<<"] Empty DCU-DetId map!";
    edm::LogError("SiStripConfigDb") << ss.str() << "\n";
    //throw cms::Exception(errorCategory_) << ss.str() << "\n";
  } else {
    db_->resetDcuDetIdMap();
    db_->setDcuDetIdMap( dcu_detid_map );
    db_->uploadDcuDetIdMap();
  }
  
  // Refresh local caches with newly created descriptions
  //refreshLocalCaches();
  
  edm::LogInfo("SiStripConfigDb") << "[SiStripPopulateConfigDb::beginJob] Finished!";
  
}

// -----------------------------------------------------------------------------
/** */ 
void SiStripPopulateConfigDb::retrieveDetIds( const edm::EventSetup& iSetup,
					      const uint32_t& number_of_dets,
					      TkPartitions& partitions ) {
  edm::LogInfo("SiStripConfigDb") << "[SiStripPopulateConfigDb::retrieveDetIds]";

  // Define the partitions
  partitions.clear();
  partitions.resize(4);
  partitions[0].first = "TIB";
  partitions[1].first = "TOB";
  partitions[2].first = "TEC+";
  partitions[3].first = "TEC-";

  // Retrieve geometry
  edm::ESHandle<TrackerGeometry> pDD;
  iSetup.get<TrackerDigiGeometryRecord>().get( pDD );
  edm::LogInfo("SiStripPopulateConfigDb") 
    << "[SiStripPopulateConfigDb::retrieveDetIds]"
    << " Iterating through "<< number_of_dets 
    << " of " << pDD->detIds().size()
    << " detectors found in geometry";

  // Iterate through strip dets
  uint32_t ndets = 0;
  TrackerGeometry::DetContainer::const_iterator idet = pDD->dets().begin();
  for( ; idet != pDD->dets().end(); idet++ ) {

    // Some checks
    if ( ndets >= number_of_dets && number_of_dets ) { break; }
    StripGeomDetUnit* strip_det = dynamic_cast<StripGeomDetUnit*>( *idet );
    if( strip_det ) {
      const StripTopology& topol = strip_det->specificTopology();
      uint32_t det_id = strip_det->geographicalId().rawId();
      uint32_t napvs = topol.nstrips() / 128;
      if( napvs!=4 && napvs!=6 ) { continue; }
      pair<uint32_t,uint16_t> data = pair<uint32_t,uint16_t>(det_id,napvs);

      // Push back into appropriate vector
      string name = "";
      DetId det( det_id );
      if ( det.det() == 1 &&               // Tracker detector
	   ( det.subdetId() == 3 ||        // TIB sub-detector
	     det.subdetId() == 4 ) ) {     // TID sub-detector
	partitions[0].second.push_back( data );
	name = partitions[0].first;
      } else if ( det.det() == 1 &&        // Tracker detector
		  det.subdetId() == 5 ) {  // TOB sub-detector
	partitions[1].second.push_back( data );
	name = partitions[1].first;
      } else if ( det.det() == 1 &&        // Tracker detector
		  det.subdetId() == 6 ) {  // TEC sub-detector
	TECDetId tec( det_id );
	if ( tec.petal()[0] == 1 ) {       // TEC forward petal
	  partitions[2].second.push_back( data );
	  name = partitions[2].first;
	} else if ( tec.petal()[0] == 0 ) { // TEC backward petal
	  partitions[3].second.push_back( data );
	  name = partitions[3].first;
	}
      } else {;} //@@ anything here?

      if ( name != "" ) {
	stringstream ss;
	ss << "[SiStripPopulateConfigDb::retrieveDetIds]"
	   << " Found strip det in partition " << name
	   << " with number " << ndets
	   << " and DetId 0x" << hex << setw(8) << setfill('0') << det_id << dec 
	   << " and " << napvs << " APVs";
	edm::LogInfo("SiStripConfigDb") << ss.str();
	ndets++;
      }
      
    }
  }
  
  // Some debug
  stringstream ss;
  ss << "[SiStripPopulateConfigDb::retrieveDetIds]"
     << " Found " << partitions.size() << " partitions with the following name/nDets:";
  TkPartitions::iterator iter = partitions.begin();
  for ( ; iter != partitions.end(); iter++ ) {
    ss << "  " << iter->first << "/" << iter->second.size();
  }
  edm::LogInfo("SiStripConfigDb") << ss.str();
  
}  

// -----------------------------------------------------------------------------
/** */
void SiStripPopulateConfigDb::createFecCabling( const uint16_t& partition_number,
						const TkPartitions& partitions,
						SiStripFecCabling& fec_cabling,
						SiStripConfigDb::DcuDetIdMap& dcu_detid_map ) {
  
  

  edm::LogInfo("SiStripConfigDb")
    << "[SiStripPopulateConfigDb::createFecCabling]"
    << " Creating FEC cabling for partition " << partitions[partition_number].first << "...";
  
  // Some fixed constants
  uint32_t fecs_per_crate = 19;
  uint32_t rings_per_fec = 8;
  uint32_t ccus_per_ring = 6;
  uint32_t modules_per_ccu = 6;
  
  // Create front-end devices
  uint32_t imod = 0;
  uint32_t idevice = 0;
  TkPartition::const_iterator iter = partitions[partition_number].second.begin(); 
  for ( ; iter != partitions[partition_number].second.end(); iter++ ) {
    uint16_t npairs = iter->second/2;
    for ( uint16_t ipair = 0; ipair < npairs; ipair++ ) {
      uint16_t apv_addr = 0;
      if      ( npairs == 2 && ipair == 0 ) { apv_addr = 32; }
      else if ( npairs == 2 && ipair == 1 ) { apv_addr = 36; }
      else if ( npairs == 3 && ipair == 0 ) { apv_addr = 32; }
      else if ( npairs == 3 && ipair == 1 ) { apv_addr = 34; }
      else if ( npairs == 3 && ipair == 2 ) { apv_addr = 36; }
      else {
	edm::LogError("SiStripConfigDb")
	  << "[SiStripPopulateConfigDb::createFecCabling]"
	  << " Unexpected values: nPairs/ipair = "
	  << npairs << "/" << ipair;
      }
      
      // Control path 
      uint16_t fec_crate = partition_number;
      uint16_t fec_slot  = (imod/(modules_per_ccu*ccus_per_ring*rings_per_fec)) % fecs_per_crate + 1;
      uint16_t fec_ring  = (imod/(modules_per_ccu*ccus_per_ring)) % rings_per_fec + 1;
      uint16_t ccu_addr  = (imod/(modules_per_ccu)) % ccus_per_ring + 1;
      uint16_t ccu_chan  = (imod) % modules_per_ccu + 26;
      uint32_t dcu_id = SiStripControlKey::key( fec_crate,
						fec_slot,
						fec_ring,
						ccu_addr,
						ccu_chan );
      
      FedChannelConnection conn( fec_crate, fec_slot, fec_ring, ccu_addr, ccu_chan,
				 apv_addr, apv_addr+1,
				 dcu_id, iter->first, npairs,
				 0, 0, // fed id/ch 
				 0, // fibre length
				 true, true, true, true ); // dcu, pll, mux, lld
      fec_cabling.addDevices( conn );

      LogDebug("SiStripConfigDb")
	<< "[SiStripPopulateConfigDb::createFecCabling]"
	<< " Adding device numbers " << idevice << " and " << idevice+1
	<< " with crate/FEC/Ring/CCU/Module/APV0/APV1/DcuId/DetId/nApvPairs/FedId/FedCh/Length/DCU/PLL/MUX/LLD: " 
	<< fec_crate << "/" << fec_slot << "/" << fec_ring << "/" << ccu_addr << "/" << ccu_chan << "/" 
	<< apv_addr << "/" << apv_addr+1<< "/" 
	<< dcu_id << "/" << iter->first << "/" << npairs << "/"
	<< 0 << "/" << 0 << "/" << 0 << "/"
	<< true << "/" << true << "/" << true << "/" << true;
      
      idevice += 2;
    }
    imod++;
  }
  
  uint16_t fed_id = 50;
  uint16_t fed_ch = 0;
  for ( vector<SiStripFecCrate>::const_iterator icrate = fec_cabling.crates().begin(); icrate != fec_cabling.crates().end(); icrate++ ) {
    for ( vector<SiStripFec>::const_iterator ifec = icrate->fecs().begin(); ifec != icrate->fecs().end(); ifec++ ) {
      for ( vector<SiStripRing>::const_iterator iring = ifec->rings().begin(); iring != ifec->rings().end(); iring++ ) {
	for ( vector<SiStripCcu>::const_iterator iccu = iring->ccus().begin(); iccu != iring->ccus().end(); iccu++ ) {
	  for ( vector<SiStripModule>::const_iterator imod = iccu->modules().begin(); imod != iccu->modules().end(); imod++ ) {

	    // Set "dummy" FED id / channel
	    if ( 96-fed_ch < imod->nApvPairs() ) { fed_id++; fed_ch = 0; } // move to next FED
	    for ( uint16_t ipair = 0; ipair < imod->nApvPairs(); ipair++ ) {
	      pair<uint16_t,uint16_t> addr = imod->activeApvPair( (*imod).lldChannel(ipair) );
	      pair<uint16_t,uint16_t> fed_channel = pair<uint16_t,uint16_t>( fed_id, fed_ch );
	      const_cast<SiStripModule&>(*imod).fedCh( addr.first, fed_channel );
	      LogDebug("SiStripConfigDb")
		<< "[SiStripPopulateConfigDb::createFecCabling]"
		<< " Setting FED id/channel = " 
		<< fed_id << "/" << fed_ch
		<<" for FEC/Ring/CCU/Module/Pair/APVaddr " 
		<< ifec->fecSlot() << "/" 
		<< iring->fecRing() << "/" 
		<< iccu->ccuAddr() << "/" 
		<< imod->ccuChan() << "/"
		<< ipair << "/"
		<< addr.first; 
	      fed_ch++;
	    }

	    // Create TkDcuInfo object
	    TkDcuInfo* dcu = new TkDcuInfo( imod->dcuId(), 
					    imod->detId(), 
					    0., // fibre length
					    2*imod->nApvPairs() );
	    if ( dcu_detid_map.find( imod->dcuId() ) == dcu_detid_map.end() ) {
	      dcu_detid_map[imod->dcuId()] = dcu;
	      stringstream ss;
	      ss << "[SiStripPopulateConfigDb::retrieveDetIds]"
		 << " Created TkDcuInfo object with number " 
		 << dcu_detid_map.size()
		 << " and DetId 0x" 
		 << hex << setw(8) << setfill('0') << dcu->getDetId() << dec
		 << " and " << dcu->getApvNumber() 
		 << " APVs in partition " 
		 << partitions[partition_number].first;
	      edm::LogInfo("SiStripConfigDb") << ss.str();
	    } else { 
	      edm::LogWarning("SiStripConfigDb") 
		<< "[SiStripPopulateConfigDb::retrieveDetIds]"
		<< " DCU id already exists in DCU-DetId map!";
	    }
	    
	  }
	}
      }
    }
  }

  // Debug
  fec_cabling.countDevices();
  
} 










// // -----------------------------------------------------------------------------
// //
// void SiStripPopulateConfigDb::uploadFrontEndDevicesToDb( const TkPartition& ip ) {
//   edm::LogInfo("FedCabling") << "[SiStripPopulateConfigDb::uploadFrontEndDevicesToDb]"; 
  
//   // need crate number...
//   uint32_t crate_number = 1;
  
//   tkDcuInfoVector dcus;
//   for ( uint32_t idcu = 0; idcu < (ip->second).size(); idcu++ ) { 
//     uint32_t dcu_id = (ip->second)[idcu].dcuId_ + ((crate_number&0xF)<<28);
//     TkDcuInfo* dcu = new TkDcuInfo( dcu_id,
// 				    (ip->second)[idcu].detId_, 
// 				    0.,
// 				    (ip->second)[idcu].nApvPairs_*2 ); 
//     dcus.push_back( dcu ); 
//   }
  
  
//   try {
//     db_->deviceFactory()->setTkDcuInfo( dcus );
//     setConversionFactors( dcus, *(db_->deviceFactory()), ip->first );
//     Sgi::hash_map< unsigned long, keyType > det_ids;
//     fillIndex( dcus, det_ids, ip->first );
//     uploadDevices( dcus, det_ids, ip->first, *(db_->deviceFactory()), crate_number );
//   }
//   catch (FecExceptionHandler e) {
//     std::cerr << "Unable to upload information to the DB: " << e.getMessage() << std::endl ;
//     exit (EXIT_FAILURE) ;
//   }
  
//   tkDcuInfoVector::iterator it = dcus.begin(); 
//   for ( ; it != dcus.end() ; it++ ) { delete *it; }
  
// }

// // -----------------------------------------------------------------------------
// //
// void SiStripPopulateConfigDb::uploadFedCablingToDb( const TkPartition& partition,
// 						    const SiStripFedCabling& fed_cabling ) {
//   edm::LogInfo("FedCabling") << "[SiStripPopulateConfigDb::uploadFedDescriptionsToDb]"; 
//   //   db_->deviceFactory()->setInputDBVersion();
// }

// // -----------------------------------------------------------------------------
// //
// void SiStripPopulateConfigDb::uploadFedDescriptionsToDb( const TkPartition& ip, 
// 							 const vector<uint16_t>& feds ) {
//   edm::LogInfo("FedCabling") << "[SiStripPopulateConfigDb::uploadFedDescriptionsToDb]"; 
  
//   db_->deviceFactory()->setUsingStrips(true);
  
//   // Iterate through FEDs
//   vector<uint16_t>::const_iterator ifed = feds.begin();
//   for ( ; ifed != feds.end(); ifed++ ) {
    
//     // create description
//     try {
//       Fed9U::Fed9UAddress addr;
//       Fed9U::Fed9UDescription f;
//       f.setFedId( *ifed );
//       f.setFedHardwareId( *ifed );
//       for ( uint32_t i = 0; i < Fed9U::APVS_PER_FED; i++ ) {
// 	addr.setFedApv(i);
// 	vector<Fed9U::u32> pedestals(128,100);
// 	vector<bool> disableStrips(128,false);
// 	vector<Fed9U::u32> highThresholds(128,50);
// 	vector<Fed9U::u32> lowThresholds(128,20);
// 	vector<Fed9U::Fed9UStripDescription> apvStripDescription(128);
// 	for ( uint32_t j = 0; j < Fed9U::STRIPS_PER_APV; j++) {
// 	  apvStripDescription[j].setPedestal(pedestals[j]);
// 	  apvStripDescription[j].setDisable(disableStrips[j]);
// 	  apvStripDescription[j].setLowThreshold(lowThresholds[j]);
// 	  apvStripDescription[j].setHighThreshold(highThresholds[j]);
// 	}
// 	f.getFedStrips().setApvStrips (addr, apvStripDescription);
//       }
      
//       db_->deviceFactory()->setFed9UDescription( f,
// 						 ip->first,
// 						 0,
// 						 0,
// 						 true );
      
//     }
//     catch(exception& e) {
//       cout << "Caught exception:\n" << e.what() << endl;
//     }
//     catch( ... ) {
//       cout << "Caught an unknown exception." << endl;
//     }
    
//   } // feds

// }


// // -----------------------------------------------------------------------------
// // -----------------------------------------------------------------------------
// // -----------------------------------------------------------------------------
// // fred's methods...

// /** Set the complete information about det id: Det ID, DCU Hard ID, Fiber length, APV number
//  * \param vDcuInfo - vector of TkDcuInfo
//  * \param deviceFactory - database access
//  * \param partitionName - partition name
//  */
// void SiStripPopulateConfigDb::setConversionFactors ( tkDcuInfoVector vDcuInfoPartition, 
// 						     DeviceFactory &deviceFactory, 
// 						     std::string partitionName ) throw (FecExceptionHandler) {
  
//   // Create the conversion factors
//   dcuConversionVector vConversionFactors ;

//   // Create the corresponding parameters for the conversion factors
//   std::string subDetector = "None" ;
//   if (partitionName.find("TEC",0) != string::npos) subDetector = "TEC" ;
//   if (partitionName.find("TIB",0) != string::npos) subDetector = "TIB" ;
//   if (partitionName.find("TID",0) != string::npos) subDetector = "TID" ;
//   if (partitionName.find("TOB",0) != string::npos) subDetector = "TOB" ;
//   if (subDetector == "None") {
//     std::cerr << "Warning: Unknown partition name, the sub detector for the DCU conversion factors should be TEC,TIB,TOB or TID, set the subdetector as TEC" << std::endl ;
//     subDetector = "TEC" ;
//   }

//   TkDcuConversionFactors tkDcuConversionFactorsStatic ( 0, subDetector, DCUFEH ) ;
//   tkDcuConversionFactorsStatic.setAdcGain0(2.144) ;
//   tkDcuConversionFactorsStatic.setAdcOffset0(0) ;
//   tkDcuConversionFactorsStatic.setAdcCal0(false) ;
//   tkDcuConversionFactorsStatic.setAdcInl0(0) ;
//   tkDcuConversionFactorsStatic.setAdcInl0OW(true) ;
//   tkDcuConversionFactorsStatic.setI20(0.02122);
//   tkDcuConversionFactorsStatic.setI10(0.01061);
//   tkDcuConversionFactorsStatic.setICal(false) ;
//   tkDcuConversionFactorsStatic.setKDiv(0.56) ;
//   tkDcuConversionFactorsStatic.setKDivCal(false) ;
//   tkDcuConversionFactorsStatic.setTsGain(8.9) ;
//   tkDcuConversionFactorsStatic.setTsOffset(2432) ;
//   tkDcuConversionFactorsStatic.setTsCal(false) ;
//   tkDcuConversionFactorsStatic.setR68(0) ;
//   tkDcuConversionFactorsStatic.setR68Cal(false) ;
//   tkDcuConversionFactorsStatic.setAdcGain2(0) ;
//   tkDcuConversionFactorsStatic.setAdcOffset2(0) ;
//   tkDcuConversionFactorsStatic.setAdcCal2(false) ;
//   tkDcuConversionFactorsStatic.setAdcGain3(0) ;
//   tkDcuConversionFactorsStatic.setAdcCal3(false) ;

//   // For each det id create the conversion factors
//   for (tkDcuInfoVector::iterator it = vDcuInfoPartition.begin() ; it != vDcuInfoPartition.end() ; it ++) {
    
//     TkDcuInfo *tkDcuInfo = *it ;

//     TkDcuConversionFactors *tkDcuConversionFactors = new TkDcuConversionFactors ( tkDcuConversionFactorsStatic ) ;
//     tkDcuConversionFactors->setDetId(tkDcuInfo->getDetId()) ;
//     tkDcuConversionFactors->setDcuHardId(tkDcuInfo->getDcuHardId()) ;
//     vConversionFactors.push_back(tkDcuConversionFactors) ;
//   }

//   // Display
//   //for (dcuConversionVector::iterator it = vConversionFactors.begin() ; it != vConversionFactors.end() ; it ++)
//   //(*it)->display() ;

//   if (deviceFactory.getDbUsed()) {
//     // Submit to the database
//     std::cout << "Upload database with the conversion factors for " << vConversionFactors.size() << " devices" << std::endl ;
//   }
//   else {
//     std::cout << "Upload the conversion factors in file " << "/tmp/conversionFactors.xml" << std::endl ;
//     deviceFactory.setOutputFileName ("/tmp/conversionFactors.xml") ;
//   }

//   // Upload
//   deviceFactory.setTkDcuConversionFactors ( vConversionFactors ) ;

//   // Delete the conversion factors
//   for (dcuConversionVector::iterator iti = vConversionFactors.begin() ; iti != vConversionFactors.end() ; iti ++) 
//     delete *iti ;
// }

// /** Give an index for each module
//  * \param vDcuInfo - list of module for one partition
//  * \param vModuleInfo - Module with the index (inherits from TkDcuInfo)
//  * \param partitionName - name of the partition
//  */
// void SiStripPopulateConfigDb::fillIndex ( tkDcuInfoVector vDcuInfoPartition, 
// 					  Sgi::hash_map<unsigned long, keyType> &detIdPosition, 
// 					  std::string partitionName ) {

//   // 10 modules per CCU, number of modules per FEC, then per Ring, then per CCU
//   unsigned int numberFEC  = (unsigned int)((vDcuInfoPartition.size() / 11) + 1) ;
//   unsigned int numberRing = (unsigned int)((numberFEC / 8) + 1) ;
//   unsigned int numberCCU  = (unsigned int)((numberRing / 10) + 1) ;

//   // Index
//   unsigned int fecSlot = 1 ;        // from 1 to 11
//   unsigned int fecRing = 0 ;        // from 0 to 7
//   unsigned int ccuAddress = 0x1 ;   // CCU number are linear
//   unsigned int i2cChannel = 0x11 ;  // 0x10 is kept for DOH and DCU on CCU

//   // For each det id
//   for (tkDcuInfoVector::iterator it = vDcuInfoPartition.begin() ; it != vDcuInfoPartition.end() ; it ++) {

//     TkDcuInfo *tkDcuInfo = *it ;

//     // A switch case can be introduced at that level to switch between TID, TIB, TOB, TEC
//     keyType index = buildCompleteKey(fecSlot,fecRing,ccuAddress,i2cChannel,0) ;
//     detIdPosition[tkDcuInfo->getDetId()] = index ;

//     //char msg[80] ; decodeKey(msg, index) ; std::cout << tkDcuInfo->getDetId() << ": " << msg << std::endl ;
    
//     // Next index: 
//     // 10 modules on every CCU
//     i2cChannel ++ ;
//     if (i2cChannel == 0x1B) { // next CCU
//       i2cChannel = 0x11 ;
//       ccuAddress ++ ; 
//       if (ccuAddress > numberCCU) { // next ring
// 	ccuAddress = 0x1 ;
// 	fecRing ++ ; 
// 	if (fecRing == 8) {
// 	  fecRing = 0 ;
// 	  fecSlot ++ ;
// 	  if (fecSlot == 12) {
// 	    std::cerr << "Error: the number of FEC, ring, CCU, channel cannot handle the number of modules:" << std::endl ;
// 	    std::cerr << "Number of modules in the partition " << partitionName << ": " << vDcuInfoPartition.size() << std::endl ;
// 	    std::cerr << "Repartition : " << std::endl ;
// 	    std::cerr << "\t Number of modules per FEC: " << numberFEC << std::endl ;
// 	    //std::cerr << "\t Number of rings: " << numberRing << std::endl ;
// 	    std::cerr << "\t Number of modules per CCU: " << numberCCU << std::endl ;
// 	    std::cerr << "Press a key to continue" ; getchar() ;
// 	    return ;
// 	  }
// 	}
//       }
//     }
//   }
// }

// /** Create and upload the devices to the DB
//  * \param vDcuInfo - list of module for one partition
//  * \param vModuleInfo - Module with the index (inherits from TkDcuInfo)
//  * \param partitionName - name of the partition
//  * \param crateNumber - crate number useed to change the value of the DCU hard id for the DCU on CCU
//  */
// void SiStripPopulateConfigDb::uploadDevices ( tkDcuInfoVector vDcuInfoPartition, 
// 					      Sgi::hash_map<unsigned long, keyType> detIdPosition, 
// 					      std::string partitionName, 
// 					      DeviceFactory &deviceFactory,
// 					      unsigned int crateNumber ) throw (FecExceptionHandler) {

//   // Default parameters
//   // Values for DOH
//   tscType8 gainDOH = 2 ;
//   tscType8 biasDOH[3] = {24, 24, 24} ;
//   tscType8 gainAOH = 2 ; 
//   tscType8 biasAOH[3] = {23, 23, 23} ;
//   apvDescription apvStatic ((tscType8)0x2b,
// 			    (tscType8)0x64,
// 			    (tscType8)0x4,
// 			    (tscType8)0x73,
// 			    (tscType8)0x3c,
// 			    (tscType8)0x32,
// 			    (tscType8)0x32,
// 			    (tscType8)0x32,
// 			    (tscType8)0x50,
// 			    (tscType8)0x32,
// 			    (tscType8)0x50,
// 			    (tscType8)0,    // Ispare
// 			    (tscType8)0x43,
// 			    (tscType8)0x43,
// 			    (tscType8)0x14,
// 			    (tscType8)0xFB,
// 			    (tscType8)0xFE,
// 			    (tscType8)0) ;
//   laserdriverDescription dohStatic (gainDOH,biasDOH) ;
//   laserdriverDescription aohStatic (gainAOH,biasAOH) ;
//   muxDescription muxStatic ((tscType16)0xFF) ;
//   pllDescription pllStatic ((tscType8)6,(tscType8)1) ;

//   // Create the corresponding parameters for the conversion factors
//   std::string subDetector = "None" ;
//   if (partitionName.find("TEC",0) != string::npos) subDetector = "TEC" ;
//   if (partitionName.find("TIB",0) != string::npos) subDetector = "TIB" ;
//   if (partitionName.find("TID",0) != string::npos) subDetector = "TID" ;
//   if (partitionName.find("TOB",0) != string::npos) subDetector = "TOB" ;
//   if (subDetector == "None") {
//     std::cerr << "Warning: Unknown partition name, the sub detector for the DCU conversion factors should be TEC,TIB,TOB or TID, set the subdetector as TEC" << std::endl ;
//     subDetector = "TEC" ;
//   }

//   TkDcuConversionFactors tkDcuConversionFactorsStatic ( 0, subDetector, DCUCCU ) ;
//   tkDcuConversionFactorsStatic.setAdcGain0(2.144) ;
//   tkDcuConversionFactorsStatic.setAdcOffset0(0) ;
//   tkDcuConversionFactorsStatic.setAdcCal0(false) ;
//   tkDcuConversionFactorsStatic.setAdcInl0(0) ;
//   tkDcuConversionFactorsStatic.setAdcInl0OW(true) ;
//   tkDcuConversionFactorsStatic.setI20(0.02122);
//   tkDcuConversionFactorsStatic.setI10(.01061);
//   tkDcuConversionFactorsStatic.setICal(false) ;
//   tkDcuConversionFactorsStatic.setKDiv(0.56) ;
//   tkDcuConversionFactorsStatic.setKDivCal(false) ;
//   tkDcuConversionFactorsStatic.setTsGain(8.9) ;
//   tkDcuConversionFactorsStatic.setTsOffset(2432) ;
//   tkDcuConversionFactorsStatic.setTsCal(false) ;
//   tkDcuConversionFactorsStatic.setR68(0) ;
//   tkDcuConversionFactorsStatic.setR68Cal(false) ;
//   tkDcuConversionFactorsStatic.setAdcGain2(0) ;
//   tkDcuConversionFactorsStatic.setAdcOffset2(0) ;
//   tkDcuConversionFactorsStatic.setAdcCal2(false) ;
//   tkDcuConversionFactorsStatic.setAdcGain3(0) ;
//   tkDcuConversionFactorsStatic.setAdcCal3(false) ;

//   // Vector
//   dcuConversionVector vConversionFactors ;
//   deviceVector vDevice ;
//   piaResetVector vPiaReset  ;

//   unsigned int oldCCU = 0, oldRing = 10 ; //, oldFec = 15 ;
//   unsigned int dcuHardId = vDcuInfoPartition.size() + 1 ;
//   if (crateNumber != 0) dcuHardId += (crateNumber << 16) ;

//   std::cout << std::hex << dcuHardId << std::endl ;
//   std::cout << std::dec << dcuHardId << std::endl ;

//   // For each det id
//   for (tkDcuInfoVector::iterator it = vDcuInfoPartition.begin() ; it != vDcuInfoPartition.end() ; it ++) {

//     TkDcuInfo *tkDcuInfo = *it ;
//     keyType index = detIdPosition[tkDcuInfo->getDetId()] ;
//     //std::cout << "FEC hardware Id = " << partitionName << toString(getFecKey(index)) << std::endl ;
//     std::string fecHardwareId = partitionName + toString(getFecKey(index)) ;

//     //char msg[80] ; decodeKey(msg,index) ; std::cout << msg << std::endl ;

//     // For each CCU create the corresponding DCU and add the conversion factors for it
//     // For each CCU create the corresponding PIA reset
//     if ( (oldCCU != getCcuKey(index)) || (oldRing != getRingKey(index)) ) {
      
//       // if CCU == 1 or 2 then create the corresponding DOH
//       if (getCcuKey(index) == 0x1) {
	
// 	//std::cout << "Adding DOH on CCU 0x1 0x10 0x70" << std::endl ;

// 	laserdriverDescription *dohd = new laserdriverDescription(dohStatic) ;
// 	dohd->setAccessKey(buildCompleteKey(getFecKey(index),getRingKey(index),getCcuKey(index),0x10,0x70)) ;
// 	//unsigned int fecHardId = getFecKey(index) ; //+ ((crateNumber & 0xF) << 28) ;
// 	dohd->setFecHardwareId(fecHardwareId) ;
// 	vDevice.push_back(dohd) ;
//       }
//       if (getCcuKey(index) == 0x2) {
	
// 	//std::cout << "Adding DOH on CCU 0x2 0x10 0x70" << std::endl ;
	
// 	laserdriverDescription *dohd = new laserdriverDescription(dohStatic) ;
// 	dohd->setAccessKey(buildCompleteKey(getFecKey(index),getRingKey(index),getCcuKey(index),0x10,0x70)) ;
// 	dohd->setFecHardwareId(fecHardwareId) ;
// 	vDevice.push_back(dohd) ;
//       }

//       //std::cout << "Adding DCU on FEC " << std::dec << getFecKey(index) << " Ring " << getRingKey(index) << " CCU 0x" << std::hex << getCcuKey(index) << " 0x10 0x0: " << std::dec << buildCompleteKey(getFecKey(index),getRingKey(index),getCcuKey(index),0x10,0x0) << std::endl ;
//       dcuDescription *dcud = new dcuDescription (buildCompleteKey(getFecKey(index),getRingKey(index),getCcuKey(index),0x10,0x0),
// 						 0,
//  						 dcuHardId,
//  						 0,0,0,0,0,0,0,0) ;
//       dcud->setFecHardwareId(fecHardwareId) ;
//       vDevice.push_back(dcud) ;
//       TkDcuConversionFactors *tkDcuConversionFactors = new TkDcuConversionFactors ( tkDcuConversionFactorsStatic ) ;
//       tkDcuConversionFactors->setDetId(0) ;
//       tkDcuConversionFactors->setDcuHardId(dcuHardId) ;
//       vConversionFactors.push_back(tkDcuConversionFactors) ;
//       dcuHardId ++ ;

//       //std::cout << "Adding PIA on CCU 0x" << std::hex << getCcuKey(index) << " 0x30 0x0" << std::endl ;

//       piaResetDescription *piad = new piaResetDescription (buildCompleteKey(getFecKey(index),getRingKey(index),getCcuKey(index),0x30,0x0),
// 							   10, 10000, 0xFF) ;
//       piad->setFecHardwareId(fecHardwareId) ;
//       vPiaReset.push_back(piad) ;

//       oldCCU = getCcuKey(index) ;
//     }

//     // For each ring add the corresponding dummy CCU with a DCU
//     if (oldRing != getRingKey(index)) {
      
//       //std::cout << "Adding DCU on FEC " << std::dec << getFecKey(index) << " Ring " << getRingKey(index) << " on CCU 0x7F 0x10 0x0: " << std::dec << buildCompleteKey(getFecKey(index),getRingKey(index),0x7F,0x10,0x0) << std::endl ;
//       dcuDescription *dcud = new dcuDescription (buildCompleteKey(getFecKey(index),getRingKey(index),0x7F,0x10,0x0),
// 						 0, 
// 						 dcuHardId,
// 						 0,0,0,0,0,0,0,0) ; 
//       dcud->setFecHardwareId(fecHardwareId) ;
//       vDevice.push_back(dcud) ;
//       TkDcuConversionFactors *tkDcuConversionFactors = new TkDcuConversionFactors ( tkDcuConversionFactorsStatic ) ;
//       tkDcuConversionFactors->setDetId(0) ; // No DET ID
//       tkDcuConversionFactors->setDcuHardId(dcuHardId) ;
//       vConversionFactors.push_back(tkDcuConversionFactors) ;
//       dcuHardId ++ ;


//       oldRing = getRingKey(index) ;
//     }

//     // Create the corresponding devices for the module

//     // APV
//     apvDescription *apv = new apvDescription (apvStatic) ;
//     apv->setAccessKey(index | setAddressKey(0x20)) ;
//     apv->setFecHardwareId(fecHardwareId) ;
//     vDevice.push_back(apv) ;

//     //std::cout << "Adding APV on CCU 0x" << std::hex << getCcuKey(index) << " 0x" << getChannelKey(index) << " 0x20" << std::endl ;

//     apv = new apvDescription (apvStatic) ;
//     apv->setAccessKey(index | setAddressKey(0x21)) ;
//     apv->setFecHardwareId(fecHardwareId) ;
//     vDevice.push_back(apv) ;

//     //std::cout << "Adding APV on CCU 0x" << std::hex << getCcuKey(index) << " 0x" << getChannelKey(index) << " 0x21" << std::endl ;

//     apv = new apvDescription (apvStatic) ;
//     apv->setAccessKey(index | setAddressKey(0x24)) ;
//     apv->setFecHardwareId(fecHardwareId) ;
//     vDevice.push_back(apv) ;

//     //std::cout << "Adding APV on CCU 0x" << std::hex << getCcuKey(index) << " 0x" << getChannelKey(index) << " 0x24" << std::endl ;

//     apv = new apvDescription (apvStatic) ;
//     apv->setAccessKey(index | setAddressKey(0x25)) ;
//     apv->setFecHardwareId(fecHardwareId) ;
//     vDevice.push_back(apv) ;

//     //std::cout << "Adding APV on CCU 0x" << std::hex << getCcuKey(index) << " 0x" << getChannelKey(index) << " 0x25" << std::endl ;

//     if (tkDcuInfo->getApvNumber() == 6) {
//       apv = new apvDescription (apvStatic) ;
//       apv->setAccessKey(index | setAddressKey(0x22)) ;
//       apv->setFecHardwareId(fecHardwareId) ;
//       vDevice.push_back(apv) ;

//       //std::cout << "Adding APV on CCU 0x" << std::hex << getCcuKey(index) << " 0x" << getChannelKey(index) << " 0x22" << std::endl ;

//       apv = new apvDescription (apvStatic) ;
//       apv->setAccessKey(index | setAddressKey(0x23)) ;
//       apv->setFecHardwareId(fecHardwareId) ;

//       //std::cout << "Adding APV on CCU 0x" << std::hex << getCcuKey(index) << " 0x" << getChannelKey(index) << " 0x23" << std::endl ;

//       vDevice.push_back(apv) ;
//     }

//     // DCU
//     dcuDescription *dcud = new dcuDescription (index,
// 					       0,
// 					       tkDcuInfo->getDcuHardId(),
// 					       0,0,0,0,0,0,0,0) ;
//     dcud->setFecHardwareId(fecHardwareId) ;
//     vDevice.push_back(dcud) ;
    
//     //std::cout << "Adding DCU on CCU 0x" << std::hex << getCcuKey(index) << " 0x" << getChannelKey(index) << " 0x0" << std::endl ;

//     // APV MUX
//     muxDescription *muxd = new muxDescription(muxStatic) ;
//     muxd->setAccessKey(index | setAddressKey(0x43)) ;
//     muxd->setFecHardwareId(fecHardwareId) ;
//     vDevice.push_back(muxd) ;

//     //std::cout << "Adding APV MUX on CCU 0x" << std::hex << getCcuKey(index) << " 0x" << getChannelKey(index) << " 0x43" << std::endl ;

//     // AOH
//     laserdriverDescription *aohd = new laserdriverDescription (aohStatic) ;
//     aohd->setAccessKey(index | setAddressKey(0x60)) ;
//     aohd->setFecHardwareId(fecHardwareId) ;
//     vDevice.push_back(aohd) ;

//     //std::cout << "Adding AOH on CCU 0x" << std::hex << getCcuKey(index) << " 0x" << getChannelKey(index) << " 0x43" << std::endl ;

//     // PLL
//     pllDescription *plld = new pllDescription (pllStatic) ;
//     plld->setAccessKey(index | setAddressKey(0x44)) ;
//     plld->setFecHardwareId(fecHardwareId) ;
//     vDevice.push_back(plld) ;

//     //std::cout << "Adding PLL on CCU 0x" << std::hex << getCcuKey(index) << " 0x" << getChannelKey(index) << " 0x44" << std::endl ;

//     tkDcuInfoVector::iterator it1 = (it + 1) ;

//     // Submit to the database for each FEC
//     //if ((deviceFactory.getDbUsed() && ((getFecKey(index) != oldFec) || (oldFec == 15))) ||
//     //(it1 == vDcuInfoPartition.end())) {
//     if (it1 == vDcuInfoPartition.end()) {

//       //checkDevices (vDevice) ;

//       // Upload in DB
//       if (deviceFactory.getDbUsed()) {
// 	unsigned int devMinor = 0, devMajor = 0, piaMinor = 0, piaMajor = 0 ;

// 	// Display
// 	std::cout << "Upload " << std::dec << vPiaReset.size() << " PIA reset descriptions" << std::endl ;
// 	std::cout << "Upload " << std::dec << vDevice.size() << " device descriptions" << std::endl ;
	
// // 	for (deviceVector::iterator it = vDevice.begin() ; it != vDevice.end() ; it ++) {
// // 	  std::string temp = partitionName + toString(getFecKey((*it)->getKey())) ;
// // 	  if (temp != (*it)->getFecHardwareId()) {
// // 	    std::cout << "FEC hardware id = " << (*it)->getFecHardwareId() << "/" << temp << std::endl ;
// // 	  }
// // 	}

// 	deviceFactory.createPartition(vDevice, vPiaReset, &devMajor, &devMinor, &piaMajor, &piaMinor, partitionName, partitionName) ;
	
// 	// Display
// 	std::cout << "Upload " << std::dec << vPiaReset.size() << " PIA reset version " << piaMajor << "." << piaMinor << " for partition " << partitionName << std::endl ;
// 	std::cout << "Upload " << std::dec << vDevice.size() << " devices version " << devMajor << "." << devMinor << " for partition " << partitionName << std::endl ;
	
// 	// Submit the conversion factors
// 	std::cout << "Upload " << std::dec << vConversionFactors.size() << " conversions factors" << std::endl ;
// 	for (dcuConversionVector::iterator iti = vConversionFactors.begin() ; iti != vConversionFactors.end() ; iti ++) {
// 	  try {
// 	    //std::cout << (*iti)->getDcuHardId() << std::endl ;
// 	    TkDcuConversionFactors *conversionFactors = deviceFactory.getTkDcuConversionFactors ( (*iti)->getDcuHardId() ) ;
// 	    std::cout << (*iti)->getDcuHardId() << " has conversion factors (problem, the DCU hard id cannot be duplicated)" << std::endl ;
// 	    delete conversionFactors ;
// 	    getchar() ;
// 	  }
// 	  catch (FecExceptionHandler e) {
// 	    //std::cout << "No conversion factors for DCU " << (*iti)->getDcuHardId() << std::endl ;
// 	  }
// 	}
// 	deviceFactory.setTkDcuConversionFactors ( vConversionFactors ) ;
//       }
//       else { // Upload in FILE

// 	deviceFactory.setFecDevicePiaDescriptions (vDevice, vPiaReset) ;
// 	std::cout << "Upload of the devices done in file " << std::endl ;
//       }
      
//       // Delete the conversion factors
//       for (dcuConversionVector::iterator iti = vConversionFactors.begin() ; iti != vConversionFactors.end() ; iti ++) 
// 	delete *iti ;
//       vConversionFactors.clear() ;
      
//       // Delete the PIA and devices
//       DeviceFactory::deleteVector(vPiaReset) ; vPiaReset.clear() ;
//       DeviceFactory::deleteVector(vDevice) ; vDevice.clear() ;
//     }
//   }
// }
















//     // Some debug
//     stringstream ss;
//     ss << "[SiStripPopulateConfigDb::beginJob]"
//        << " Found " << dcu_info.size() << " modules in ";
//     if      ( ipartition == 0 ) { ss << "TIB/TID"; }
//     else if ( ipartition == 1 ) { ss << "TOB"; }
//     else if ( ipartition == 2 ) { ss << "TEC+"; }
//     else if ( ipartition == 3 ) { ss << "TEC-"; }
//     ss << " partition";
//     edm::LogInfo("FedCabling") << ss.str();
      
//     // Some fixed constants
//     uint32_t crates_per_partition = 1;
//     uint32_t fecs_per_crate = 11;
//     uint32_t rings_per_fec = 8;
//     uint32_t ccus_per_ring = 10;
//     // Calculate mean number of modules per control ring
//     uint32_t average_modules_per_ring = dcu_info.size() / ( crates_per_partition * 
// 							    fecs_per_crate *
// 							    rings_per_fec );
//     // Calculate number of channels/CCU  
//     uint32_t channels_per_ccu = average_modules_per_ring / ccus_per_ring + 1;

//     edm::LogInfo("FedCabling") << "[SiStripPopulateConfigDb::beginJob]"
// 			       << "  Modules per ring: " << average_modules_per_ring;
    
//     // Create front-end devices
//     uint32_t idevice = 0;
//     for ( uint32_t imod = 0; imod < dcu_info.size(); imod++ ) {
//       for ( uint16_t ipair = 0; ipair < dcu_info[imod].nApvPairs_; ipair++ ) {
// 	uint16_t apv_addr = 0;
// 	if      ( dcu_info[imod].nApvPairs_ == 2 && ipair == 0 ) { apv_addr = 32; }
// 	else if ( dcu_info[imod].nApvPairs_ == 2 && ipair == 1 ) { apv_addr = 36; }
// 	else if ( dcu_info[imod].nApvPairs_ == 3 && ipair == 0 ) { apv_addr = 32; }
// 	else if ( dcu_info[imod].nApvPairs_ == 3 && ipair == 1 ) { apv_addr = 34; }
// 	else if ( dcu_info[imod].nApvPairs_ == 3 && ipair == 2 ) { apv_addr = 36; }
// 	else {
// 	  edm::LogError("FedCabling") << "[SiStripPopulateConfigDb::beginJob]"
// 				      << " Unexpected values: nPairs/ipair = "
// 				      << dcu_info[imod].nApvPairs_ << "/" << ipair;
// 	}

// 	// Control path 
// 	uint16_t fec_crate = ipartition + 1; 
// 	uint16_t fec_slot = (imod/(channels_per_ccu*ccus_per_ring*rings_per_fec)) % fecs_per_crate + 1;
// 	uint16_t fec_ring = (imod/(channels_per_ccu*ccus_per_ring)) % rings_per_fec + 1;
// 	uint16_t ccu_addr = (imod/(channels_per_ccu)) % ccus_per_ring + 1;
// 	uint16_t ccu_chan = (imod) % channels_per_ccu + 26;
// 	uint32_t key = SiStripControlKey::key( fec_crate,
// 					       fec_slot,
// 					       fec_ring,
// 					       ccu_addr,
// 					       ccu_chan );
// 	FedChannelConnection conn( fec_crate,
// 				   fec_slot,
// 				   fec_ring,
// 				   ccu_addr,
// 				   ccu_chan,
// 				   apv_addr,
// 				   apv_addr+1,
// 				   key, //@@ dcu id
// 				   dcu_info[imod].detId_,
// 				   dcu_info[imod].nApvPairs_,
// 				   0, //@@ fed id 
// 				   0, //@@ fed channel
// 				   (uint32_t)dcu_info[imod].fibreLength_,
// 				   true,   // dcu
// 				   true,   // pll
// 				   true,   // mux
// 				   true ); // lld
// 	fec_cabling->addDevices( conn );
// 	idevice += 2;
// 	LogDebug("FedCabling") << "[SiStripPopulateConfigDb::beginJob]"
// 			       << " Adding device number " << idevice-2
// 			       << " with crate/FEC/Ring/CCU/Module/APV0/APV1/DcuId/DetId/nApvPairs/FedId/FedCh/FibreLength/DCU/PLL/MUX/LLD: " 
// 			       << fec_crate << "/" 
// 			       << fec_slot << "/" 
// 			       << fec_ring << "/" 
// 			       << ccu_addr << "/" 
// 			       << ccu_chan << "/" 
// 			       << apv_addr << "/" 
// 			       << apv_addr+1<< "/" 
// 			       << key << "/"
// 			       << dcu_info[imod].detId_ << "/"
// 			       << dcu_info[imod].nApvPairs_ << "/"
// 			       << 0 << "/"
// 			       << 0 << "/"
// 			       << dcu_info[imod].fibreLength_ << "/"
// 			       << true << "/"
// 			       << true << "/"
// 			       << true << "/"
// 			       << true;
//       }
//     }


//     // Match TkDcuInfo with module
//     uint32_t cntr = 0;
//     for ( vector<SiStripFec>::const_iterator ifec = fec_cabling->fecs().begin(); ifec != fec_cabling->fecs().end(); ifec++ ) {
//       for ( vector<SiStripRing>::const_iterator iring = ifec->rings().begin(); iring != ifec->rings().end(); iring++ ) {
// 	for ( vector<SiStripCcu>::const_iterator iccu = iring->ccus().begin(); iccu != iring->ccus().end(); iccu++ ) {
// 	  for ( vector<SiStripModule>::const_iterator imod = iccu->modules().begin(); imod != iccu->modules().end(); imod++ ) {
// 	    uint32_t module_key = imod->detId();
// 	  }
// 	}
//       }
//     }

//     // Set "dummy" FED id / channel 
//     uint16_t fed_id = 50;
//     uint16_t fed_ch = 0;
//     for ( vector<SiStripFec>::const_iterator ifec = fec_cabling->fecs().begin(); ifec != fec_cabling->fecs().end(); ifec++ ) {
//       for ( vector<SiStripRing>::const_iterator iring = ifec->rings().begin(); iring != ifec->rings().end(); iring++ ) {
// 	for ( vector<SiStripCcu>::const_iterator iccu = iring->ccus().begin(); iccu != iring->ccus().end(); iccu++ ) {
// 	  for ( vector<SiStripModule>::const_iterator imod = iccu->modules().begin(); imod != iccu->modules().end(); imod++ ) {
// 	    if ( 96-fed_ch < imod->nApvPairs() ) { fed_id++; fed_ch = 0; } // move to next FED
// 	    for ( uint16_t ipair = 0; ipair < imod->nApvPairs(); ipair++ ) {
// 	      pair<uint16_t,uint16_t> addr = imod->activeApvPair( (*imod).lldChannel(ipair) );
// 	      pair<uint16_t,uint16_t> fed_channel = pair<uint16_t,uint16_t>( fed_id, fed_ch );
// 	      const_cast<SiStripModule&>(*imod).fedCh( addr.first, fed_channel );
// 	      LogDebug("FedCabling") << "[SiStripPopulateConfigDb::beginJob]"
// 				     << " Setting FED id/channel = " 
// 				     << fed_id << "/" << fed_ch
// 				     <<" for FEC/Ring/CCU/Module/Pair/APVaddr " 
// 				     << ifec->fecSlot() << "/" 
// 				     << iring->fecRing() << "/" 
// 				     << iccu->ccuAddr() << "/" 
// 				     << imod->ccuChan() << "/"
// 				     << ipair << "/"
// 				     << addr.first; 
// 	      fed_ch++;
// 	    }
// 	  }
// 	}
//       }
//     }
  
//     // Debug
//     fec_cabling->countDevices();
    
//     // Build FED cabling using FedChannelConnections
//     vector<FedChannelConnection> conns;
//     fec_cabling->connections( conns );
//     vector<FedChannelConnection>::iterator iconn;
//     for ( iconn = conns.begin(); iconn != conns.end(); iconn++ ) { iconn->print(); }
//     SiStripFedCabling* cabling = new SiStripFedCabling( conns );
//     edm::LogInfo("FedCabling") << "[SiStripPopulateConfigDb::beginJob]" 
// 			       << " Finished building FED cabling map!";
    
//     delete cabling;
    
//   } // partition loop
  
//   return NULL;














// -----------------------------------------------------------------------------
//
// void SiStripPopulateConfigDb::extractPartitionInfo( vector<SiStripDcuInfo::DcuInfo>& dcu_info ) {
  
//   stringstream ss;
//   ss << "[SiStripPopulateConfigDb::extractPartitionInfo]"
//      << " DetID info: \n";
//   vector<SiStripDcuInfo::DcuInfo>::const_iterator iter;

//   vector<uint32_t> tib_layer; tib_layer.clear(); 
//   vector<uint32_t>  tib_string_fwd; tib_string_fwd.clear();
//   vector<uint32_t>  tib_string_bwd; tib_string_bwd.clear();
//   uint32_t tib_string_int = 0, tib_string_ext = 0;
//   uint32_t tib_module = 0, tib_stereo = 0;
//   vector<uint32_t> dets; dets.clear(); dets.resize(10,0);
//   vector<uint32_t> subs; subs.clear(); subs.resize(10,0);
//   for ( iter = dcu_info.begin(); iter != dcu_info.end(); iter++ ) { 
//     DetId det(iter->detId_);
//     //StripSubdetector sub(iter->detId_);
//     dets[det.det()]++;
//     subs[det.subdetId()]++;
//     if ( det.subdetId() == 3 ) {
//       TIBDetId tib(iter->detId_);
//       ss << "TIB layer: " << tib.layer() 
// 	 << "  string fwd?: " << tib.string()[0]
// 	 << "  string ext?: " << tib.string()[1]
// 	 << "  string: " << tib.string()[2]
// 	 << "  module: " << tib.module()
// 	 << "  stereo: " << tib.stereo();
//       if ( find( tib_layer.begin(), tib_layer.end(), tib.layer() ) == tib_layer.end() ) { tib_layer.push_back( tib.layer() ); }
//       if ( find( tib_string_fwd.begin(), tib_string_fwd.end(), tib.string()[2] ) == tib_string_fwd.end() ) { tib_string_fwd.push_back( tib.string()[2] ); }
//       if ( find( tib_string_bwd.begin(), tib_string_bwd.end(), tib.string()[2] ) == tib_string_bwd.end() ) { tib_string_bwd.push_back( tib.string()[2] ); }
//       tib.string()[1] ? tib_string_ext++ : tib_string_int++;
//       tib_module++;
//       if ( tib.stereo() ) { tib_stereo++; }
//     }
//   }

//   cout << "DetId:"
//        << " Tracker: " << dets[1]
//        << " TIB: " << subs[3]
//        << " TID: " << subs[4]
//        << " TOB: " << subs[5]
//        << " TEC: " << subs[6]
//        << endl;

//   LogDebug("FedCabling") << ss.str() << endl;
  
//   stringstream sss;
//   sss << "TIB:"
//       << " nLayers: " << tib_layer.size() 
//       << " nFwdStrings: " << tib_string_fwd.size() 
//       << " nBwdStrings: " << tib_string_bwd.size() 
//       << " nExtStrings: " << tib_string_ext
//       << " nIntStrings: " << tib_string_int
//       << " nModules: " << tib_module
//       << " nStereo: " << tib_stereo;
//   edm::LogInfo("FedCabling") << sss.str() << endl;
 
// }  

// -----------------------------------------------------------------------------
//
// void SiStripPopulateConfigDb::createTkDcuInfoDescriptions( vector<SiStripDcuInfo::DcuInfo>& dcu_info ) {
  
//   stringstream ss;
//   ss << "[SiStripPopulateConfigDb::createTkDcuInfoDescriptions]"
//      << " dcuId/detId/nApvPairs/fibreLength: ";
//   vector<SiStripDcuInfo::DcuInfo>::const_iterator iter;
//   for ( iter = dcu_info.begin(); iter != dcu_info.end(); iter++ ) { 
//     ss << iter->dcuId_ << "/" 
//        << iter->detId_ << "/" 
//        << iter->nApvPairs_ << "/" 
//        << iter->fibreLength_ << " ";
//   }
//   LogDebug("FedCabling") << ss.str() << endl;
  
// }  
