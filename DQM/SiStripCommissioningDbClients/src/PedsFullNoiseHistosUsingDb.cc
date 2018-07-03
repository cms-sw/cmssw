
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
  : CommissioningHistograms(pset.getParameter<edm::ParameterSet>("PedsFullNoiseParameters"),bei,sistrip::PEDS_FULL_NOISE ),
    CommissioningHistosUsingDb(db,sistrip::PEDS_FULL_NOISE ),
    PedsFullNoiseHistograms( pset.getParameter<edm::ParameterSet>("PedsFullNoiseParameters"),bei )
{

  LogTrace(mlDqmClient_) 
    << "[PedsFullNoiseHistosUsingDb::" << __func__ << "]"
    << " Constructing object...";

  highThreshold_ = this->pset().getParameter<double>("HighThreshold");
  lowThreshold_  = this->pset().getParameter<double>("LowThreshold");

  LogTrace(mlDqmClient_)
    << "[PedsFullNoiseHistosUsingDb::" << __func__ << "]"
    << " Set FED zero suppression high/low threshold to "
    << highThreshold_ << "/" << lowThreshold_;

  disableBadStrips_   = this->pset().getParameter<bool>("DisableBadStrips");
  keepStripsDisabled_ = this->pset().getParameter<bool>("KeepStripsDisabled");
  skipEmptyStrips_    = this->pset().getParameter<bool>("SkipEmptyStrips");
  uploadOnlyStripBadChannelBit_ = this->pset().getParameter<bool>("UploadOnlyStripBadChannelBit");
  uploadPedsFullNoiseDBTable_ = this->pset().getParameter<bool>("UploadPedsFullNoiseDBTable");
  
  LogTrace(mlDqmClient_)
    << "[PedestalsHistosUsingDb::" << __func__ << "]"
    << " Disabling strips: " << disableBadStrips_
    << " ; keeping previously disabled strips: " << keepStripsDisabled_
    << " ; skip strips with no data: " << skipEmptyStrips_
    << " ; upload only bad channel bit: " << uploadOnlyStripBadChannelBit_;

  allowSelectiveUpload_ = this->pset().existsAs<bool>("doSelectiveUpload")?this->pset().getParameter<bool>("doSelectiveUpload"):false;
  LogTrace(mlDqmClient_)
    << "[PedestalsHistosUsingDb::" << __func__ << "]"
    << " Selective upload of modules set to : " << allowSelectiveUpload_;    
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
  if (doUploadConf()) {  // check whether the upload HD config is set to true

    edm::LogVerbatim(mlDqmClient_) 
      << "[PedsFullNoiseHistosUsingDb::" << __func__ << "]"
      << " Uploading pedestals/noise to DB...";

    db()->uploadFedDescriptions(); // change the FED version

    edm::LogVerbatim(mlDqmClient_) 
      << "[PedsFullNoiseHistosUsingDb::" << __func__ << "]"
      << " Completed database upload of " << feds.size() 
      << " FED descriptions!";
  } 
  else {
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

  for ( ifed = feds.begin(); ifed != feds.end(); ifed++ ) { // Loop on the FED for this partition

    for ( uint16_t ichan = 0; ichan < sistrip::FEDCH_PER_FED; ichan++ ) {
      // Build FED and FEC keys from the cabling object i.e. checking if there is a connection
      const FedChannelConnection& conn = cabling()->fedConnection( (*ifed)->getFedId(), ichan );
      if ( conn.fecCrate() == sistrip::invalid_ ||
           conn.fecSlot()  == sistrip::invalid_ ||
           conn.fecRing()  == sistrip::invalid_ ||
           conn.ccuAddr()  == sistrip::invalid_ ||
           conn.ccuChan()  == sistrip::invalid_ ||
           conn.lldChannel() == sistrip::invalid_ ) 
	continue; 

      // build the FED and FEC key from the connection object
      SiStripFedKey fed_key( conn.fedId(), 
                             SiStripFedKey::feUnit( conn.fedCh() ),
                             SiStripFedKey::feChan( conn.fedCh() ) );

      SiStripFecKey fec_key( conn.fecCrate(),
                             conn.fecSlot(),
                             conn.fecRing(),
                             conn.ccuAddr(),
                             conn.ccuChan(),
                             conn.lldChannel() );

      // Locate appropriate analysis object --> based on FEC keys cause they are per lldChannel
      Analyses::const_iterator iter = data(allowSelectiveUpload_).find( fec_key.key() );
      if ( iter != data(allowSelectiveUpload_).end() ) {
	
        PedsFullNoiseAnalysis* anal = dynamic_cast<PedsFullNoiseAnalysis*>( iter->second );

        if ( !anal ) { 
          edm::LogError(mlDqmClient_)
            << "[PedsFullNoiseHistosUsingDb::" << __func__ << "]"
            << " NULL pointer to analysis object!";
          continue; 
        }

        // Determine the pedestal shift to apply --> this is standard in the pedestal paylaod to avoid loss of signal from common-mode subtraction
        uint32_t pedshift = 127;
        for ( uint16_t iapv = 0; iapv < sistrip::APVS_PER_FEDCH; iapv++ ) {
          uint32_t pedmin = (uint32_t) anal->pedsMin()[iapv];
          pedshift = pedmin < pedshift ? pedmin : pedshift;
        }

        // Iterate through APVs and strips
        for ( uint16_t iapv = 0; iapv < sistrip::APVS_PER_FEDCH; iapv++ ) {
          for ( uint16_t istr = 0; istr < anal->peds()[iapv].size(); istr++ ) { // Loop on the pedestal for each APV

            if ( not uploadOnlyStripBadChannelBit_ and anal->peds()[iapv][istr] < 1. ) { //@@ ie, zero                                                                                       
	      edm::LogWarning(mlDqmClient_)
                << "[PedestalsHistosUsingDb::" << __func__ << "]"
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
	    std::stringstream ss_disable;

	    if(temp.getDisable()) { // strip already disabled in the database
	      ss_disable<<"Already Disabled: "<<conn.fecCrate()
			<<" "<<conn.fecSlot()
			<<" "<<conn.fecRing()
			<<" "<<conn.ccuAddr()
			<<" "<<conn.ccuChan()
			<<" "<<conn.lldChannel()
			<<" "<<iapv<<" "<<istr<<std::endl;
	      if(keepStripsDisabled_) disableStrip = true; // in case one wants to keep them disabled
            }

	    // to disable new strips
	    if(disableBadStrips_){		
	      SiStripFedKey fed_key(anal->fedKey());              
	      PedsFullNoiseAnalysis::VInt dead = anal->deadStrip()[iapv];
	      if (not skipEmptyStrips_ and  // if one don't want to skip dead strips
		  find( dead.begin(), dead.end(), istr ) != dead.end() ) {
		disableStrip = true;
		ss_disable<<"Disabling Dead Strip: "<<conn.fecCrate()		    
			  <<" "<<conn.fecSlot()
			  <<" "<<conn.fecRing()
			  <<" "<<conn.ccuAddr()
			  <<" "<<conn.ccuChan()
			  <<" "<<conn.lldChannel()
			  <<" "<<iapv<<" "<<istr<<std::endl;
	      }
	      
	      PedsFullNoiseAnalysis::VInt badcChan = anal->badStrip()[iapv]; // new feature --> this is the sample of the whole bad strips from the analysis
	      if(not disableStrip){
		if ( find( badcChan.begin(), badcChan.end(), istr ) != badcChan.end() ) {
		  disableStrip = true;
		  ss_disable<<"Disabling Bad strip: "<<conn.fecCrate()
			    <<" "<<conn.fecSlot()
			    <<" "<<conn.fecRing()
			    <<" "<<conn.ccuAddr()
			    <<" "<<conn.ccuChan()
			    <<" "<<conn.lldChannel()
			    <<" "<<iapv<<" "<<istr<<std::endl;

		}
	      }
	    }

	    if(edm::isDebugEnabled())
	      LogTrace(mlDqmClient_) << ss_disable.str();

	    uint32_t pedestalVal = 0;
            float    noiseVal = 0;
            float    lowThr   = 0;
            float    highThr  = 0;
	    
	    // download the previous pedestal/noise payload from the DB
	    if(uploadOnlyStripBadChannelBit_){ 
	      pedestalVal = static_cast<uint32_t>(temp.getPedestal());
              noiseVal    = static_cast<float>(temp.getNoise());
              lowThr      = static_cast<float>(temp.getLowThresholdFactor());
              highThr     = static_cast<float>(temp.getHighThresholdFactor());
 	    }
	    else{	      
	      pedestalVal = static_cast<uint32_t>(anal->peds()[iapv][istr]-pedshift);
              noiseVal    = anal->noise()[iapv][istr];
              lowThr      = lowThreshold_;
              highThr     = highThreshold_;	      
	    }
	    
	    //////
            Fed9U::Fed9UStripDescription data(pedestalVal,highThr,lowThr,noiseVal,disableStrip);

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
		 << static_cast<uint32_t>( temp.getPedestal() ) << "/"
                 << static_cast<float>( temp.getHighThresholdFactor() ) << "/"
                 << static_cast<float>( temp.getLowThresholdFactor() ) << "/"
                 << static_cast<float>( temp.getNoise() ) << "/"
                 << static_cast<uint16_t>( temp.getDisable() ) << std::endl;
            }

            (*ifed)->getFedStrips().setStrip( addr, data );

            if ( data.getDisable() && edm::isDebugEnabled() ) {
              ss << " to ped/noise/high/low/disable    : "
                 << static_cast<uint32_t>( data.getPedestal() ) << "/"
                 << static_cast<float>( data.getHighThresholdFactor() ) << "/"
                 << static_cast<float>( data.getLowThresholdFactor() ) << "/"
                 << static_cast<float>( data.getNoise() ) << "/"
                 << static_cast<uint16_t>( data.getDisable() ) << std::endl;
              LogTrace(mlDqmClient_) << ss.str();
            }	    
          } // end loop on strips
        } // end loop on apvs
        updated++;
      }
      else { // device not found in the analysis	
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

    // Create a description for the standard pedestal analysis                                                                                                                           
    PedestalsAnalysisDescription* pedestalDescription;
    pedestalDescription = new PedestalsAnalysisDescription(
					   anal->deadStripBit()[iapv],
					   anal->badStripBit()[iapv],
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
    for ( ; istr != jstr; ++istr ) {pedestalDescription->addComments( *istr ); }

    // Store description                                                                                                                                                                              
    desc.push_back(pedestalDescription);

    // Create description                                         
    if(uploadPedsFullNoiseDBTable_){
      PedsFullNoiseAnalysisDescription* pedsFullNoiseDescription;
      pedsFullNoiseDescription = new PedsFullNoiseAnalysisDescription(
								      anal->deadStrip()[iapv],
								      anal->badStrip()[iapv],
								      anal->shiftedStrip()[iapv], // bad strip-id within an APV due to offset
								      anal->lowNoiseStrip()[iapv], // bad strip-id within an APV due to noise
								      anal->largeNoiseStrip()[iapv], // bad strip-id within an APV due to noise
								      anal->largeNoiseSignificance()[iapv], // bad strip-id within an APV due to noise significance
								      anal->badFitStatus()[iapv], // bad strip-id within an APV due to fit status
								      anal->badADProbab()[iapv], // bad strip-id within an APV due to AD probab
								      anal->badKSProbab()[iapv], // bad strip-id within an APV due to KS probab 
								      anal->badJBProbab()[iapv], // bad strip-id within an APV due to JB probab 
								      anal->badChi2Probab()[iapv], // bad strip-id within an APV due to Chi2 probab 
								      anal->badTailStrip()[iapv], // bad strip-id within an APV due to tail
								      anal->badDoublePeakStrip()[iapv], // bad strip-id within an APV due to Double peaks					   
								      //////
								      anal->adProbab()[iapv], // one value oer strip
								      anal->ksProbab()[iapv], // one value oer strip 
								      anal->jbProbab()[iapv], // one value oer strip 
								      anal->chi2Probab()[iapv], // one value oer strip 
								      //// --> Per strip quantities
								      anal->residualRMS()[iapv],
								      anal->residualSigmaGaus()[iapv],
								      anal->noiseSignificance()[iapv],
								      anal->residualSkewness()[iapv],
								      anal->residualKurtosis()[iapv],
								      anal->residualIntegralNsigma()[iapv],
								      anal->residualIntegral()[iapv],
								      ////
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
      istr = errors.begin();
      jstr = errors.end();
      for ( ; istr != jstr; ++istr ) { 
	pedsFullNoiseDescription->addComments( *istr );
      }
      
      // Store description
      desc.push_back(pedsFullNoiseDescription);      
    }
  }
}

