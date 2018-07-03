#include "DQM/SiStripCommissioningDbClients/interface/DaqScopeModeHistosUsingDb.h"
#include "CondFormats/SiStripObjects/interface/DaqScopeModeAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

using namespace sistrip;

// -----------------------------------------------------------------------------
DaqScopeModeHistosUsingDb::DaqScopeModeHistosUsingDb( const edm::ParameterSet & pset,
						      DQMStore* bei,
						      SiStripConfigDb* const db ) 
  : CommissioningHistograms( pset.getParameter<edm::ParameterSet>("DaqScopeModeParameters"),
                             bei,
                             sistrip::DAQ_SCOPE_MODE ),
    CommissioningHistosUsingDb( db,
                                sistrip::DAQ_SCOPE_MODE ),
    DaqScopeModeHistograms(pset.getParameter<edm::ParameterSet>("DaqScopeModeParameters"),
    			   bei )
{

  LogTrace(mlDqmClient_) 
    << "[DaqScopeModeHistosUsingDb::" << __func__ << "]"
    << " Constructing object...";
  highThreshold_ = this->pset().getParameter<double>("HighThreshold");
  lowThreshold_ = this->pset().getParameter<double>("LowThreshold");
  LogTrace(mlDqmClient_)
    << "[DaqScopeModeHistosUsingDb::" << __func__ << "]"
    << " Set FED zero suppression high/low threshold to "
    << highThreshold_ << "/" << lowThreshold_;
  disableBadStrips_   = this->pset().getParameter<bool>("DisableBadStrips");
  keepStripsDisabled_ = this->pset().getParameter<bool>("KeepStripsDisabled");
  LogTrace(mlDqmClient_)
    << "[DaqScopeModeHistosUsingDb::" << __func__ << "]"
    << " Disabling strips: " << disableBadStrips_
    << " ; keeping previously disabled strips: " << keepStripsDisabled_;
  allowSelectiveUpload_ = this->pset().existsAs<bool>("doSelectiveUpload")?this->pset().getParameter<bool>("doSelectiveUpload"):false;
  LogTrace(mlDqmClient_)
    << "[DaqScopeModeHistosUsingDb::" << __func__ << "]"
    << " Selective upload of modules set to : " << allowSelectiveUpload_;

  skipPedestalUpdate_ =  this->pset().existsAs<bool>("SkipPedestalUpdate")?this->pset().getParameter<bool>("SkipPedestalUpdate"):false;
  skipTickUpdate_     =  this->pset().existsAs<bool>("SkipTickUpdate")?this->pset().getParameter<bool>("SkipTickUpdate"):false;
  LogTrace(mlDqmClient_)
    << "[DaqScopeModeHistosUsingDb::" << __func__ << "]"
    << " Perform pedestal upload set to : " <<skipPedestalUpdate_;
  LogTrace(mlDqmClient_)
    << "[DaqScopeModeHistosUsingDb::" << __func__ << "]"
    << " Perform tick-mark upload set to : " <<skipTickUpdate_;
}

// -----------------------------------------------------------------------------
DaqScopeModeHistosUsingDb::~DaqScopeModeHistosUsingDb() {
  LogTrace(mlDqmClient_) 
    << "[DaqScopeModeHistosUsingDb::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
void DaqScopeModeHistosUsingDb::uploadConfigurations() {
  LogTrace(mlDqmClient_) 
    << "[DaqScopeModeHistosUsingDb::" << __func__ << "]";

  if ( !db() ) {
    edm::LogError(mlDqmClient_) 
      << "[DaqScopeModeHistosUsingDb::" << __func__ << "]"
      << " NULL pointer to SiStripConfigDb interface!"
      << " Aborting upload...";
    return;
  }
  
  // Update FED descriptions with new peds/noise values as well as tick-marks (no PLL delays, for these ones please use the Timing run
  SiStripConfigDb::FedDescriptionsRange feds = db()->getFedDescriptions();
  update( feds );

  if ( doUploadConf() ) { 
    edm::LogVerbatim(mlDqmClient_) 
      << "[DaqScopeModeHistosUsingDb::" << __func__ << "]"
      << " Uploading FED information to DB...";
    db()->uploadFedDescriptions();
    edm::LogVerbatim(mlDqmClient_) 
      << "[DaqScopeModeHistosUsingDb::" << __func__ << "]"
      << " Completed database upload of " << feds.size() 
      << " FED descriptions!";
  } else {
    edm::LogWarning(mlDqmClient_) 
      << "[DaqScopeModeHistosUsingDb::" << __func__ << "]"
      << " No FED values will be uploaded to DB...";
  }
}

// -----------------------------------------------------------------------------
void DaqScopeModeHistosUsingDb::update( SiStripConfigDb::FedDescriptionsRange feds ) {
  // Retrieve FED ids from cabling                                                                                                                                                                    
  auto ids = cabling()->fedIds();

  // Iterate through feds and update fed descriptions
  uint16_t updated_peds  = 0;
  uint16_t updated_ticks = 0;
  SiStripConfigDb::FedDescriptionsV::const_iterator ifed;
  for ( ifed = feds.begin(); ifed != feds.end(); ifed++ ) {

    // If FED id not found in list (from cabling), then continue                                                                                                                                  
    if ( find( ids.begin(), ids.end(), (*ifed)->getFedId() ) == ids.end() ) { continue; }
    
    for ( uint16_t ichan = 0; ichan < sistrip::FEDCH_PER_FED; ichan++ ) {

      // Build FED and FEC keys
      const FedChannelConnection& conn = cabling()->fedConnection( (*ifed)->getFedId(), ichan );

      if ( conn.fecCrate() == sistrip::invalid_ ||
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
      Analyses::const_iterator iter = data(allowSelectiveUpload_).find( fec_key.key() );

      if ( iter != data(allowSelectiveUpload_).end() ) {

         // Check if analysis is valid
         if ( !iter->second->isValid() ) { 
           continue; 
         }
	 
         DaqScopeModeAnalysis* anal = dynamic_cast<DaqScopeModeAnalysis*>( iter->second );
         if ( !anal ) { 
           edm::LogError(mlDqmClient_)
             << "[DaqScopeModeHistosUsingDb::" << __func__ << "]"
             << " NULL pointer to analysis object!";
           continue; 
         }
	 
	 /// Pedestal and noise uploads
	 if(not skipPedestalUpdate_){	   

	   // Determine the pedestal shift to apply
	   uint32_t pedshift = 127;
	   for ( uint16_t iapv = 0; iapv < sistrip::APVS_PER_FEDCH; iapv++ ) {
	     uint32_t pedmin = (uint32_t) anal->pedsMin()[iapv];
	     pedshift = pedmin < pedshift ? pedmin : pedshift;
	     std::stringstream ss;
	     ss << "iapv: " << iapv << " pedsMin()[iapv]: " << anal->pedsMin()[iapv] << " pedmin: " << pedmin << " pedshift: " << pedshift;	  
	     edm::LogWarning(mlDqmClient_) << ss.str();
	   }
	   
	   // Iterate through APVs and strips
	   for ( uint16_t iapv = 0; iapv < sistrip::APVS_PER_FEDCH; iapv++ ) {
	     for ( uint16_t istr = 0; istr < anal->peds()[iapv].size(); istr++ ) { 

	       // Patch added by R.B. (I'm back! ;-), requested by J.F. and S.L. (04/11/2010)
	       if ( anal->peds()[iapv][istr] < 1. ) { //@@ ie, zero
		 edm::LogWarning(mlDqmClient_) 
		   << "[DaqScopeModeHistosUsingDb::" << __func__ << "]"
		   << " Skipping ZERO pedestal value (ie, NO UPLOAD TO DB!) for FedKey/Id/Ch: " 
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
		 continue; //@@ do not upload
	       }
	       
	       // get the information on the strip as it was on the db
	       Fed9U::Fed9UAddress addr( ichan, iapv, istr );
	       Fed9U::Fed9UStripDescription temp = (*ifed)->getFedStrips().getStrip( addr );

	       // determine whether we need to disable the strip
	       bool disableStrip = false;
	       if ( keepStripsDisabled_ ) {
		 disableStrip = temp.getDisable();
	       } 
	       else if (disableBadStrips_) {
		 DaqScopeModeAnalysis::VInt dead = anal->dead()[iapv];
		 if ( find( dead.begin(), dead.end(), istr ) != dead.end() ) disableStrip = true;
		 DaqScopeModeAnalysis::VInt noisy = anal->noisy()[iapv];
		 if ( find( noisy.begin(), noisy.end(), istr ) != noisy.end() ) disableStrip = true;
	       }

	       Fed9U::Fed9UStripDescription data( static_cast<uint32_t>( anal->peds()[iapv][istr]-pedshift ),
						  highThreshold_,
						  lowThreshold_,
						  anal->noise()[iapv][istr],
						  disableStrip);

	       std::stringstream ss;
	       if ( data.getDisable() && edm::isDebugEnabled() ) {
		 ss << "[DaqScopeModeHistosUsingDb::" << __func__ << "]"
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
	       
	       // update strip inf
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
	   updated_peds++;
	 }
	 
	 // if one wants to update the frame finding threhsolds
	 if(not skipTickUpdate_){
	   
	   // Update frame finding threshold                                                                                                                                                      
	   Fed9U::Fed9UAddress addr( ichan );
	   uint16_t old_threshold = static_cast<uint16_t>( (*ifed)->getFrameThreshold( addr ) );
	   if(anal->isValid()) {
	     (*ifed)->setFrameThreshold( addr, anal->frameFindingThreshold() );
	     updated_ticks++;
	   }
	   uint16_t new_threshold = static_cast<uint16_t>( (*ifed)->getFrameThreshold( addr ) );

	   std::stringstream ss;
	   ss << "LLD channel : old frame threshold "<<old_threshold<<" new frame threshold "<<new_threshold<<std::endl;
	   edm::LogWarning(mlDqmClient_) << ss.str();
       
	   // Debug                                                                                                                                                                                   
	   ss.clear();
	   ss << "[DaqScopeModeHistosUsingDb::" << __func__ << "]";
	   if ( anal->isValid() ) {
	     ss << " Updating the frame-finding threshold"
		<< " from " << old_threshold
		<< " to " << new_threshold
		<< " using tick mark base/peak/height "
		<< anal->base() << "/"
		<< anal->peak() << "/"
		<< anal->height();
	   } else {
	     ss << " Cannot update the frame-finding threshold"
		<< " from " << old_threshold
		<< " to a new value using invalid analysis ";
	   }
	   ss << " for crate/FEC/ring/CCU/module/LLD "
	      << fec_key.fecCrate() << "/"
	      << fec_key.fecSlot() << "/"
	      << fec_key.fecRing() << "/"
	      << fec_key.ccuAddr() << "/"
	      << fec_key.ccuChan()
	      << fec_key.channel()
	      << " and FED id/ch "
	      << fed_key.fedId() << "/"
	      << fed_key.fedChannel();
	   anal->print(ss);
	   LogTrace(mlDqmClient_) << ss.str();                                                                                                                                                  
	 }
      }
      else{
	if ( deviceIsPresent(fec_key) ) {
	  edm::LogWarning(mlDqmClient_) 
	    << "[DaqScopeModeHistosUsingDb::" << __func__ << "]"
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
    << "[DaqScopeModeHistosUsingDb::" << __func__ << "]"
    << " Updated FED parameters for pedestal/noise " 
    << updated_peds << " channels" 
    << " Updated FED parameters for frame finding thresholds " 
    << updated_ticks << " channels";
}

// -----------------------------------------------------------------------------
void DaqScopeModeHistosUsingDb::create( SiStripConfigDb::AnalysisDescriptionsV& desc,
					Analysis analysis ) {

  DaqScopeModeAnalysis* anal = dynamic_cast<DaqScopeModeAnalysis*>( analysis->second );
  if ( !anal ) { return; }
  
  SiStripFecKey fec_key( anal->fecKey() );
  SiStripFedKey fed_key( anal->fedKey() );
  
  for ( uint16_t iapv = 0; iapv < 2; ++iapv ) {
    
    // Create description for the pedestal table
    PedestalsAnalysisDescription* peds_tmp;
    peds_tmp = new PedestalsAnalysisDescription( 
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
    for ( ; istr != jstr; ++istr ) { peds_tmp->addComments( *istr ); }

    // Store description
    desc.push_back( peds_tmp );

    // Create description                                                                                                                                                                           
    TimingAnalysisDescription* timing_tmp;
    timing_tmp = new TimingAnalysisDescription( 
        -1.,
	-1.,
        -1.,
        anal->height(),
        anal->base(),
        anal->peak(),
        anal->frameFindingThreshold(),
        -1.,
        DaqScopeModeAnalysis::tickMarkHeightThreshold_,
        true, //@@ APV timing analysis (not FED timing)                                                                                                             
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
        fed_key.fedApv() );

    istr = errors.begin();
    jstr = errors.end();
    for ( ; istr != jstr; ++istr ) { timing_tmp->addComments( *istr ); }
    desc.push_back(timing_tmp);    
  }
}
