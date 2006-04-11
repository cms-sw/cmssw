#include "OnlineDB/SiStripESSources/interface/SiStripFedCablingBuilderFromDb.h"
// fwk
#include "FWCore/MessageLogger/interface/MessageLogger.h"
// config db
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
// cabling
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "DQM/SiStripCommon/interface/SiStripGenerateKey.h"
// std
#include <ostream>
#include <string>
#include <vector>

// -----------------------------------------------------------------------------
/** */
SiStripFedCablingBuilderFromDb::SiStripFedCablingBuilderFromDb( const edm::ParameterSet& pset ) 
  : SiStripFedCablingESSource( pset ),
    db_(0)
{
  edm::LogInfo ("FedCabling") << "[SiStripFedCablingBuilderFromDb::SiStripFedCablingBuilderFromDb]"
			      << " Constructing object...";
  db_ = new SiStripConfigDb( pset.getUntrackedParameter<string>("User",""),
			     pset.getUntrackedParameter<string>("Passwd",""),
			     pset.getUntrackedParameter<string>("Path",""),
			     pset.getUntrackedParameter<string>("Partition","") );
  db_->fromXml( pset.getUntrackedParameter<bool>( "UseXmlFile", false ) );
  db_->xmlFile( pset.getUntrackedParameter<string>( "XmlFilename", "" ) );
}

// -----------------------------------------------------------------------------
/** */
SiStripFedCablingBuilderFromDb::~SiStripFedCablingBuilderFromDb() {
  edm::LogInfo("FedCabling") << "[SiStripFedCablingBuilderFromDb::~SiStripFedCablingBuilderFromDb]"
			     << " Destructing object...";
  if ( db_ ) { delete db_; } 
}

// -----------------------------------------------------------------------------
/** */
SiStripFedCabling* SiStripFedCablingBuilderFromDb::makeFedCabling() {
  edm::LogInfo("FedCabling") << "[SiStripFedCablingBuilderFromDb::makeFedCabling] Building FED cabling...";
  
  // Build FEC cabling object 
  SiStripFecCabling fec_cabling;
  
  vector<FedChannelConnectionDescription*> fed_conns = db_->fedConnections( true );
  
  if ( !fed_conns.empty() ) {
    
    vector<FedChannelConnectionDescription*>::iterator ifed;
    for ( ifed = fed_conns.begin(); ifed != fed_conns.end(); ifed++ ) {
      FedChannelConnection conn( static_cast<uint16_t>( 0 ), //@@ 
				 static_cast<uint16_t>( (*ifed)->getSlot() ),
				 static_cast<uint16_t>( (*ifed)->getRing() ),
				 static_cast<uint16_t>( (*ifed)->getCcu() ),
				 static_cast<uint16_t>( (*ifed)->getI2c() ),
				 static_cast<uint16_t>( (*ifed)->getApv() ),
				 static_cast<uint16_t>( (*ifed)->getApv()+1 ),
				 static_cast<uint32_t>( 0 ), //@@ (*ifed)->getDcuHardId() ),
				 static_cast<uint32_t>( 0 ), //@@ (*ifed)->getDcuHardId()+65536 ),
				 static_cast<uint32_t>( 0 ), //@@ 
				 static_cast<uint16_t>( (*ifed)->getFedId() ),
				 static_cast<uint16_t>( (*ifed)->getFedChannel() ) );
      fec_cabling.addDevices( conn );
      //       LogDebug ("FecCabling") 
      // 	<< conn.print(); 
    }
    
    uint32_t cntr = 0;
    for ( vector<SiStripFec>::const_iterator ifec = fec_cabling.fecs().begin(); ifec != fec_cabling.fecs().end(); ifec++ ) {
      for ( vector<SiStripRing>::const_iterator iring = ifec->rings().begin(); iring != ifec->rings().end(); iring++ ) {
	for ( vector<SiStripCcu>::const_iterator iccu = iring->ccus().begin(); iccu != iring->ccus().end(); iccu++ ) {
	  for ( vector<SiStripModule>::const_iterator imod = iccu->modules().begin(); imod != iccu->modules().end(); imod++ ) {
	    uint32_t module_key = SiStripGenerateKey::module( 0, // fec crate 
							      ifec->fecSlot(), 
							      iring->fecRing(), 
							      iccu->ccuAddr(), 
							      imod->ccuChan() );
	    SiStripModule& module = const_cast<SiStripModule&>(*imod);
	    module.dcuId( module_key );
	    module.detId( cntr+1 );
	    module.nApvPairs(0);
	    LogDebug("FedCabling") << "[SiStripFedCablingTrivialBuilder::makeFedCabling]"
				   << " Setting DcuId/DetId/nApvPairs = " 
				   << module_key << "/" << cntr+1 << "/" << module.nApvPairs()
				   << " for FEC/Ring/CCU/Module " 
				   << ifec->fecSlot() << "/" 
				   << iring->fecRing() << "/" 
				   << iccu->ccuAddr() << "/" 
				   << imod->ccuChan();
	    cntr++;
	  }
	}
      }
    }

  } else {
    
//     // Retrieve APV descriptions from DB and populate FEC cabling map 
//     vector<apvDescription*> apv_desc; 
//     db_->apvDescriptions( apv_desc );
//     vector<apvDescription*>::iterator iapv;
//     for ( iapv = apv_desc.begin(); iapv != apv_desc.end(); iapv++ ) {
//       SiStripConfigDb::DeviceAddress addr = db_->hwAddresses(**iapv);
//       FedChannelConnection conn( addr.fecSlot, 
// 				 addr.fecRing, 
// 				 addr.ccuAddr, 
// 				 addr.ccuChan, 
// 				 addr.i2cAddr ); 
//       fec_cabling.addDevices( conn );
//     }
    
//     // Retrieve DCU descriptions from DB and populate FEC cabling map 
//     vector<dcuDescription*> dcu_desc; 
//     db_->dcuDescriptions( dcu_desc );
//     vector<dcuDescription*>::iterator idcu;
//     for ( idcu = dcu_desc.begin(); idcu != dcu_desc.end(); idcu++ ) {
//       SiStripConfigDb::DeviceAddress addr = db_->hwAddresses(**idcu);
//       FedChannelConnection conn( addr.fecSlot, 
// 				 addr.fecRing, 
// 				 addr.ccuAddr, 
// 				 addr.ccuChan,
// 				 0, 0, /** APV I2C addresses */
// 				 (*idcu)->getDcuHardId() ); 
//       fec_cabling.dcuId( conn );
//     }
    
//     // Retrieve DCU descriptions from DB and populate FEC cabling map 
//     vector<unsigned short> fed_ids; 
//     db_->fedIds( fed_ids );
//     vector<unsigned short>::iterator idcu;

//   /** Returns FED identifiers. */
//   void fedIds( vector<unsigned short>& fed_ids ) {;}


//     // Retrieve connection objects and set FED id / channel 
//     uint32_t ichannel = 0;
//     vector<FedChannelConnection> conns;
//     fec_cabling.connections( conns );
//     for ( uint32_t iconn = 0; iconn < conns.size(); iconn++ ) {
//       if ( ichannel > (96-npairs) ) {
// 	(*iconn).fedId( (ichannel/96)+50 ); 
// 	(*iconn).fedCh( ichannel%96 ); 
//       }
//       ichannel++;
//     }
    
  }
  
  // Debug
  fec_cabling.countDevices();
  
  // Build FED cabling using FedChannelConnections
  vector<FedChannelConnection> conns;
  fec_cabling.connections( conns );
  vector<FedChannelConnection>::iterator iconn;
  for ( iconn = conns.begin(); iconn != conns.end(); iconn++ ) { iconn->print(); }
  SiStripFedCabling* cabling = new SiStripFedCabling( conns );
  edm::LogInfo("FedCabling") << "[SiStripFedCablingBuilderFromDb::makeFedCabling]" 
			     << " Finished building FED cabling map!";
  return cabling;
  
}








//   vector<apv_devices>::iterator imod;
//   for ( imod = modules_.begin(); imod != modules_.end(); imod++ ) {
//     // Iterate through DCU descriptions and match to ModuleConnection
//     vector<dcuDescription*>::iterator idcu;
//     for ( idcu = dcu_descriptions.begin(); idcu != dcu_descriptions.end(); idcu++ ) {
//       int fec_inst, fec_slot, fec_ring, ccu_addr, ccu_chan, i2c_addr;
//       fec_inst = fec_slot = fec_ring = ccu_addr = ccu_chan = i2c_addr = 0;
//       db_->hwAddresses( **idcu, fec_slot, fec_ring, ccu_addr, ccu_chan, i2c_addr );
//       //@@ HOW TO RETRIEVE FEC INSTANCE FROM DCU DESCRIPTION?
//       // Set DCU id if ModuleConnection has same hardware addresses
//       if ( (*imod).getFecInstance() == fec_inst &&
// 	   (*imod).getFecSlot() == fec_slot &&
// 	   (*imod).getRingSlot() == fec_ring &&
// 	   (*imod).getCcuAddress() == ccu_addr &&
// 	   (*imod).getI2CChannel() == ccu_chan ) {
// 	int dcu_id = (*idcu)->getDcuHardId();
// 	(*imod).setDcuHardId( dcu_id );
// 	LogDebug("FedCabling")V.debugOut << "[SiStripCreateControlConnections::attachDcuIdsToModules]"
// 		       << " Assigning DCU id " << (*idcu)->getDcuHardId()
// 		       << " to ModuleConnection with params: "
// 		       << " FECinst: " << fec_inst
// 		       << " FECslot: " << fec_slot
// 		       << " FECring: " << fec_ring
// 		       << " CCUaddr: " << ccu_addr
// 		       << " CCUchan: " << ccu_chan
// 		       << " I2Caddr: " << i2c_addr;
// 	dcu_descriptions.erase( idcu );
// 	break;
//       }
//     }
//   }
//   LogDebug("FedCabling")V.infoOut << "[SiStripCreateControlConnections::attachDcuIdsToModules] "
// 		<< "Assigned DCU ids to " << ndcu - dcu_descriptions.size() 
// 		<< " ModuleConnections (out of a possible " 
// 		<< modules_.size() << ")"<< endl;
  
//   // Check if all DCU description have been used when assigning DCU ids
//   if ( !dcu_descriptions.empty() ) {
//     edm::LogError("FedCabling") << warning("[SiStripCreateControlConnections::attachDcuIdsToModules] ")
// 	 << "There are " << dcu_descriptions.size() 
// 	 << " DCU ids that have not been assigned to ModuleConnections";
//   }

//   // Debug printout of hardware addresses
//   if ( LogDebug("FedCabling")V.testOut ) {
//     LogDebug("FedCabling") << "[SiStripCreateControlConnections::attachDcuIdsToModules] "
// 	 << "A list of ModuleConnections generated using "
// 	 << "descriptions retrieved from configuration database:";
//     vector<ModuleConnection>::iterator iter;
//     for ( iter = modules_.begin(); iter != modules_.end(); iter++ ) {
//       (*iter).print();
//     }    
//   }


// //  // SiStripFedCabling::SiStripFedCabling * local_controlcabling =  new SiStripFedCabling::SiStripFedCabling(); // simple test
// //   std::vector< std::vector< SiStripFedCablingRingItem > > controlcabling;
// //   for(unsigned short fec_it =0; fec_it != FECIdMax; fec_it++){
// //     std::vector< SiStripFedCablingRingItem > ringitems;
// //     for(unsigned short ri_it =0; ri_it != RingIdMax; ri_it++){
// //       for(unsigned short ccu_it =0; ccu_it != CCUIdMax; ccu_it++){
// //         std::vector<SiStripFedCablingDCUItem> i2c_lines;
// //         for(unsigned short dcu_it =0; dcu_it != DCUChannelMax; dcu_it++){
// //             uint32_t detraw_id = 11111; // change this
// //             SiStripFedCablingDCUItem * dcuitem = new SiStripFedCablingDCUItem::SiStripFedCablingDCUItem(dcu_it, detraw_id);
// //             i2c_lines.push_back(*dcuitem);
// //         }
// //         SiStripFedCablingRingItem * the_ritem = new SiStripFedCablingRingItem::SiStripFedCablingRingItem(ri_it, ccu_it, i2c_lines);
// //         ringitems.push_back(*the_ritem);
// //       }
// //     }
// //   controlcabling.push_back(ringitems);
// //   }

// //   SiStripFedCabling::SiStripFedCabling* local_controlcabling = new SiStripFedCabling::SiStripFedCabling( controlcabling );
// //   return local_controlcabling;
// // }


//   // Iterate through devices, pair them and create SiStripFedChannelConnection
//   std::vector<bool> used; used.clear(); used.resize( devs.size(), false );
//   for ( unsigned short idev = 0; idev < devs.size(); idev++ ) {
//     for ( unsigned short jdev = 0; jdev < devs.size(); jdev++ ) {
//       if ( used[jdev] == false && used[idev] == false &&  
// 	   devs[jdev].fecSlot() == devs[idev].fecSlot() &&
// 	   devs[jdev].fecRing() == devs[idev].fecRing() &&
// 	   devs[jdev].ccuAddr() == devs[idev].ccuAddr() &&
// 	   devs[jdev].ccuChan() == devs[idev].ccuChan() &&
// 	   ( devs[jdev].i2cAddr() % 2 ) // I2C address is odd
// 	   ( devs[jdev].i2cAddr()-devs[idev].i2cAddr() == 1 ) ) {
// 	connections.push_back( SiStripFedChannelConnection(devs[idev],devs[jdev]) );
// 	used[idev] = true; used[jdev] = true; 
//       }
//     }
//   }

//   // Iterate through devices without a pair and create "dummy pair" with null I2C address
//   for ( unsigned short kdev = 0; kdev < devs.size(); kdev++ ) {
//     if ( !used[kdev] ) {
//       SiStripDevice dev( devs[kdev].fecSlot(), 
// 			 devs[kdev].fecRing(), 
// 			 devs[kdev].ccuAddr(), 
// 			 devs[kdev].ccuChan(), 
// 			 0 ); // null I2C address
//       if ( devs[kdev].i2cAddr() % 2 )  { // I2C address is odd
// 	connections.push_back( SiStripFedChannelConnection(dev,devs[kdev]) );
//       } else {
// 	connections.push_back( SiStripFedChannelConnection(devs[kdev],dev) );
//       }
//     }
//   }
  
