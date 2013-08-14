// Last commit: $Id: PedsFullNoiseHistosUsingDb.cc,v 1.5 2010/04/28 08:46:16 lowette Exp $

#include "DQM/SiStripCommissioningDbClients/interface/PedsFullNoiseHistosUsingDb.h"
#include "CondFormats/SiStripObjects/interface/PedsFullNoiseAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
PedsFullNoiseHistosUsingDb::PedsFullNoiseHistosUsingDb( const edm::ParameterSet & pset,
                                                        DQMStore* bei,
                                                        SiStripConfigDb* const db ) 
  : CommissioningHistograms( pset.getParameter<edm::ParameterSet>("PedsFullNoiseParameters"),
                             bei,
                             sistrip::PEDS_FULL_NOISE ),
    CommissioningHistosUsingDb( db,
                                sistrip::PEDS_FULL_NOISE ),
    PedsFullNoiseHistograms( pset.getParameter<edm::ParameterSet>("PedsFullNoiseParameters"),
                             bei )
{
    LogTrace(mlDqmClient_) 
    << "[PedsFullNoiseHistosUsingDb::" << __func__ << "]"
    << " Constructing object...";
  highThreshold_ = this->pset().getParameter<double>("HighThreshold");
  lowThreshold_ = this->pset().getParameter<double>("LowThreshold");
  LogTrace(mlDqmClient_)
    << "[PedsFullNoiseHistosUsingDb::" << __func__ << "]"
    << " Set FED zero suppression high/low threshold to "
    << highThreshold_ << "/" << lowThreshold_;
  disableBadStrips_ = this->pset().getParameter<bool>("DisableBadStrips");
  keepStripsDisabled_ = this->pset().getParameter<bool>("KeepStripsDisabled");
  addBadStrips_ = this->pset().getParameter<bool>("AddBadStrips");
  LogTrace(mlDqmClient_)
    << "[PedestalsHistosUsingDb::" << __func__ << "]"
    << " Disabling strips: " << disableBadStrips_
    << " ; keeping previously disabled strips: " << keepStripsDisabled_;
}

// -----------------------------------------------------------------------------
/** */
PedsFullNoiseHistosUsingDb::~PedsFullNoiseHistosUsingDb() {
  LogTrace(mlDqmClient_) 
    << "[PedsFullNoiseHistosUsingDb::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
/** */
void PedsFullNoiseHistosUsingDb::uploadConfigurations() {
  LogTrace(mlDqmClient_) 
    << "[PedsFullNoiseHistosUsingDb::" << __func__ << "]";

  if ( !db() ) {
    edm::LogError(mlDqmClient_) 
      << "[PedsFullNoiseHistosUsingDb::" << __func__ << "]"
      << " NULL pointer to SiStripConfigDb interface!"
      << " Aborting upload...";
    return;
  }
  
  // Update FED descriptions with new peds/noise values
  SiStripConfigDb::FedDescriptionsRange feds = db()->getFedDescriptions(); 
  update( feds );
  if ( doUploadConf() ) { 
    edm::LogVerbatim(mlDqmClient_) 
      << "[PedsFullNoiseHistosUsingDb::" << __func__ << "]"
      << " Uploading pedestals/noise to DB...";
    db()->uploadFedDescriptions();
    edm::LogVerbatim(mlDqmClient_) 
      << "[PedsFullNoiseHistosUsingDb::" << __func__ << "]"
      << " Completed database upload of " << feds.size() 
      << " FED descriptions!";
  } else {
    edm::LogWarning(mlDqmClient_) 
      << "[PedsFullNoiseHistosUsingDb::" << __func__ << "]"
      << " TEST! No pedestals/noise values will be uploaded to DB...";
  }
  
}

// -----------------------------------------------------------------------------
/** */
void PedsFullNoiseHistosUsingDb::update( SiStripConfigDb::FedDescriptionsRange feds ) {
 
  // Iterate through feds and update fed descriptions
  uint16_t updated = 0;
  SiStripConfigDb::FedDescriptionsV::const_iterator ifed;
  for ( ifed = feds.begin(); ifed != feds.end(); ifed++ ) {
    
    for ( uint16_t ichan = 0; ichan < sistrip::FEDCH_PER_FED; ichan++ ) {

      // Build FED and FEC keys
      const FedChannelConnection& conn = cabling()->connection( (*ifed)->getFedId(), ichan );
      if ( conn.fecCrate()== sistrip::invalid_ ||
           conn.fecSlot() == sistrip::invalid_ ||
           conn.fecRing() == sistrip::invalid_ ||
           conn.ccuAddr() == sistrip::invalid_ ||
           conn.ccuChan() == sistrip::invalid_ ||
           conn.lldChannel() == sistrip::invalid_ ) { continue; }
           
      SiStripFedKey fed_key( conn.fedId(), 
                             SiStripFedKey::feUnit( conn.fedCh() ),
                             SiStripFedKey::feChan( conn.fedCh() ) );
      SiStripFecKey fec_key( conn.fecCrate(),
                             conn.fecSlot(),
                             conn.fecRing(),
                             conn.ccuAddr(),
                             conn.ccuChan(),
                             conn.lldChannel() );

      // Locate appropriate analysis object 
      Analyses::const_iterator iter = data().find( fec_key.key() );
      if ( iter != data().end() ) {
      
        PedsFullNoiseAnalysis* anal = dynamic_cast<PedsFullNoiseAnalysis*>( iter->second );
        if ( !anal ) { 
          edm::LogError(mlDqmClient_)
            << "[PedsFullNoiseHistosUsingDb::" << __func__ << "]"
            << " NULL pointer to analysis object!";
          continue; 
        }

        // Determine the pedestal shift to apply
        uint32_t pedshift = 127;
        for ( uint16_t iapv = 0; iapv < sistrip::APVS_PER_FEDCH; iapv++ ) {
          uint32_t pedmin = (uint32_t) anal->pedsMin()[iapv];
          pedshift = pedmin < pedshift ? pedmin : pedshift;
        }

        // Iterate through APVs and strips
        for ( uint16_t iapv = 0; iapv < sistrip::APVS_PER_FEDCH; iapv++ ) {
          for ( uint16_t istr = 0; istr < anal->peds()[iapv].size(); istr++ ) { 

            // get the information on the strip as it was on the db
            Fed9U::Fed9UAddress addr( ichan, iapv, istr );
            Fed9U::Fed9UStripDescription temp = (*ifed)->getFedStrips().getStrip( addr );
						if(temp.getDisable()) {
            	std::cout<<"Already Disabled: "<<conn.fecCrate()
							<<" "<<conn.fecSlot()
							<<" "<<conn.fecRing()
							<<" "<<conn.ccuAddr()
							<<" "<<conn.ccuChan()
							<<" "<<conn.lldChannel()
              <<" "<<iapv*128+istr<<std::endl;
            }
            // determine whether we need to disable the strip
            bool disableStrip = false;
            if ( addBadStrips_ ) {
              disableStrip = temp.getDisable();
              SiStripFedKey fed_key(anal->fedKey());              
              if(!disableStrip){
              	PedsFullNoiseAnalysis::VInt dead = anal->dead()[iapv];
              	if ( find( dead.begin(), dead.end(), istr ) != dead.end() ) {
                	disableStrip = true;
                  std::cout<<"Disabling Dead: "<<conn.fecCrate()
									<<" "<<conn.fecSlot()
									<<" "<<conn.fecRing()
									<<" "<<conn.ccuAddr()
									<<" "<<conn.ccuChan()
									<<" "<<conn.lldChannel()
              		<<" "<<iapv*128+istr<<std::endl;
                }
              	PedsFullNoiseAnalysis::VInt noisy = anal->noisy()[iapv];
              	if ( find( noisy.begin(), noisy.end(), istr ) != noisy.end() ) {
                	disableStrip = true;
                  std::cout<<"Disabling Noisy: "<<conn.fecCrate()
									<<" "<<conn.fecSlot()
									<<" "<<conn.fecRing()
									<<" "<<conn.ccuAddr()
									<<" "<<conn.ccuChan()
									<<" "<<conn.lldChannel()
              		<<" "<<iapv*128+istr<<std::endl;
                }
              }
            } else if ( keepStripsDisabled_ ) {
              disableStrip = temp.getDisable();
            } else if (disableBadStrips_) {
              PedsFullNoiseAnalysis::VInt dead = anal->dead()[iapv];
              if ( find( dead.begin(), dead.end(), istr ) != dead.end() ) {
              	disableStrip = true;              
                std::cout<<"Disabling Dead: "<<conn.fecCrate()
								<<" "<<conn.fecSlot()
								<<" "<<conn.fecRing()
								<<" "<<conn.ccuAddr()
								<<" "<<conn.ccuChan()
								<<" "<<conn.lldChannel()
              	<<" "<<iapv*128+istr<<std::endl;
              }
              PedsFullNoiseAnalysis::VInt noisy = anal->noisy()[iapv];
              if ( find( noisy.begin(), noisy.end(), istr ) != noisy.end() ) {
              	disableStrip = true;                
                std::cout<<"Disabling Noisy: "<<conn.fecCrate()
								<<" "<<conn.fecSlot()
								<<" "<<conn.fecRing()
								<<" "<<conn.ccuAddr()
								<<" "<<conn.ccuChan()
								<<" "<<conn.lldChannel()
              	<<" "<<iapv*128+istr<<std::endl;
              }
            }

            Fed9U::Fed9UStripDescription data( static_cast<uint32_t>( anal->peds()[iapv][istr]-pedshift ),
                                               highThreshold_,
                                               lowThreshold_,
                                               anal->noise()[iapv][istr],
                                               disableStrip );

            std::stringstream ss;
            if ( data.getDisable() && edm::isDebugEnabled() ) {
              ss << "[PedsFullNoiseHistosUsingDb::" << __func__ << "]"
                 << " Disabling strip in Fed9UStripDescription object..." << std::endl
                 << " for FED id/channel and APV/strip : "
                 << fed_key.fedId() << "/"
                 << fed_key.fedChannel() << " "
                 << iapv << "/"
                 << istr << std::endl 
                 << " and crate/FEC/ring/CCU/module    : "
                 << fec_key.fecCrate() << "/"
                 << fec_key.fecSlot() << "/"
                 << fec_key.fecRing() << "/"
                 << fec_key.ccuAddr() << "/"
                 << fec_key.ccuChan() << std::endl 
                 << " from ped/noise/high/low/disable  : "
                 << static_cast<uint16_t>( temp.getPedestal() ) << "/" 
                 << static_cast<uint16_t>( temp.getHighThreshold() ) << "/" 
                 << static_cast<uint16_t>( temp.getLowThreshold() ) << "/" 
                 << static_cast<uint16_t>( temp.getNoise() ) << "/" 
                 << static_cast<uint16_t>( temp.getDisable() ) << std::endl;
            }
            (*ifed)->getFedStrips().setStrip( addr, data );
            if ( data.getDisable() && edm::isDebugEnabled() ) {
              ss << " to ped/noise/high/low/disable    : "
                 << static_cast<uint16_t>( data.getPedestal() ) << "/" 
                 << static_cast<uint16_t>( data.getHighThreshold() ) << "/" 
                 << static_cast<uint16_t>( data.getLowThreshold() ) << "/" 
                 << static_cast<uint16_t>( data.getNoise() ) << "/" 
                 << static_cast<uint16_t>( data.getDisable() ) << std::endl;
              LogTrace(mlDqmClient_) << ss.str();
            }

          } // end loop on strips
        } // end loop on apvs
        updated++;

      } else {
        if ( deviceIsPresent(fec_key) ) {
          edm::LogWarning(mlDqmClient_) 
            << "[PedsFullNoiseHistosUsingDb::" << __func__ << "]"
            << " Unable to find pedestals/noise for FedKey/Id/Ch: " 
            << hex << setw(8) << setfill('0') << fed_key.key() << dec << "/"
            << (*ifed)->getFedId() << "/"
            << ichan
            << " and device with FEC/slot/ring/CCU/LLD " 
            << fec_key.fecCrate() << "/"
            << fec_key.fecSlot() << "/"
            << fec_key.fecRing() << "/"
            << fec_key.ccuAddr() << "/"
            << fec_key.ccuChan() << "/"
            << fec_key.channel();
        }
      }
    }
  }
  
  edm::LogVerbatim(mlDqmClient_) 
    << "[PedsFullNoiseHistosUsingDb::" << __func__ << "]"
    << " Updated FED pedestals/noise for " 
    << updated << " channels";
}

// -----------------------------------------------------------------------------
/** */
void PedsFullNoiseHistosUsingDb::create( SiStripConfigDb::AnalysisDescriptionsV& desc,
                                         Analysis analysis ) {

  PedsFullNoiseAnalysis* anal = dynamic_cast<PedsFullNoiseAnalysis*>( analysis->second );
  if ( !anal ) { return; }
  
  SiStripFecKey fec_key( anal->fecKey() );
  SiStripFedKey fed_key( anal->fedKey() );
  
  for ( uint16_t iapv = 0; iapv < 2; ++iapv ) {
    
    // Create description
    PedestalsAnalysisDescription* tmp;
    tmp = new PedestalsAnalysisDescription(
      anal->dead()[iapv],
      anal->noisy()[iapv],
      anal->pedsMean()[iapv],
      anal->pedsSpread()[iapv],
      anal->noiseMean()[iapv],
      anal->noiseSpread()[iapv],
      anal->rawMean()[iapv],
      anal->rawSpread()[iapv],
      anal->pedsMax()[iapv], 
      anal->pedsMin()[iapv], 
      anal->noiseMax()[iapv],
      anal->noiseMin()[iapv],
      anal->rawMax()[iapv],
      anal->rawMin()[iapv],
      fec_key.fecCrate(),
      fec_key.fecSlot(),
      fec_key.fecRing(),
      fec_key.ccuAddr(),
      fec_key.ccuChan(),
      SiStripFecKey::i2cAddr( fec_key.lldChan(), !iapv ), 
      db()->dbParams().partitions().begin()->second.partitionName(),
      db()->dbParams().partitions().begin()->second.runNumber(),
      anal->isValid(),
      "",
      fed_key.fedId(),
      fed_key.feUnit(),
      fed_key.feChan(),
      fed_key.fedApv()
    );
    
    // Add comments
    typedef std::vector<std::string> Strings;
    Strings errors = anal->getErrorCodes();
    Strings::const_iterator istr = errors.begin();
    Strings::const_iterator jstr = errors.end();
    for ( ; istr != jstr; ++istr ) { tmp->addComments( *istr ); }

    // Store description
    desc.push_back( tmp );
      
  }

}

