#include <memory>
#include <ostream>

#include "CalibTracker/SiStripAPVAnalysis/interface/ApvAnalysisFactory.h"
#include "CalibTracker/SiStripAPVAnalysis/interface/ApvFactoryService.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DQM/SiStripCommon/interface/ExtractTObject.h"
#include "DQM/SiStripCommon/interface/UpdateTProfile.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQM/SiStripCommissioningSources/interface/NoiseTask.h"

using namespace sistrip;

std::ostream &operator <<( std::ostream &rOut, 
                           ApvAnalysis::PedestalType &rPEDS)
{
  for( ApvAnalysis::PedestalType::const_iterator pedsIter = rPEDS.begin();
       pedsIter != rPEDS.end();
       ++pedsIter)
    {
      rOut << ' ' << *pedsIter;
    }

  return rOut;
}

// -----------------------------------------------------------------------------
//
NoiseTask::NoiseTask( DQMStore *dqm,
                      const FedChannelConnection &conn)
  : CommissioningTask( dqm, conn, "NoiseTask")
{
  //@@ NOT GUARANTEED TO BE THREAD SAFE! 
  pApvFactory_ = edm::Service<ApvFactoryService>().operator->()->getApvFactory();
  
  LogTrace( mlDqmSource_)
    << "[NoiseTask::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
//
NoiseTask::~NoiseTask()
{
  LogTrace(mlDqmSource_)
    << "[NoiseTask::" << __func__ << "]"
    << " Destructing object...";

  // Have to delete pApvFactory_ manually even though we didn't create it
  // ourself. :(
  if( pApvFactory_) { delete pApvFactory_; } 
}

// -----------------------------------------------------------------------------
//
void NoiseTask::book() 
{
  LogTrace( mlDqmSource_) << "[NoiseTask::" << __func__ << "]";

  // CACHING
  static std::unique_ptr<SiStripPedestals> pDBPedestals;
  static std::unique_ptr<SiStripNoises>    pDBNoises;
  
  const uint16_t nBINS = 256;
  
  {
    // Pedestals
    std::string title = SiStripHistoTitle( sistrip::EXPERT_HISTO,
                                           sistrip::NOISE,
                                           sistrip::FED_KEY,
                                           fedKey(),
                                           sistrip::LLD_CHAN,
                                           connection().lldChannel(),
                                           sistrip::extrainfo::pedestals_).title();

    HistoSet oHSet;
    oHSet.isProfile_ = true;

    oHSet.vNumOfEntries_.resize( nBINS, 0);
    oHSet.vSumOfContents_.resize( nBINS, 0);
    oHSet.vSumOfSquares_.resize( nBINS, 0);

    oHSet.histo( dqm()->bookProfile( title, title,
				     nBINS, -0.5, nBINS * 1. - 0.5,
				     1025, 0., 1025.) );

    peds_.push_back( oHSet);
  }

  {
    // Corrected Noise
    std::string title = SiStripHistoTitle( sistrip::EXPERT_HISTO,
                                           sistrip::NOISE,
                                           sistrip::FED_KEY,
                                           fedKey(),
                                           sistrip::LLD_CHAN,
                                           connection().lldChannel(),
                                           sistrip::extrainfo::noise_).title();

    HistoSet oHSet;
    oHSet.isProfile_ = true;

    oHSet.vNumOfEntries_.resize( nBINS, 0);
    oHSet.vSumOfContents_.resize( nBINS, 0);
    oHSet.vSumOfSquares_.resize( nBINS, 0);

    oHSet.histo( dqm()->bookProfile( title, title,
				     nBINS, -0.5, nBINS * 1. - 0.5,
				     1025, 0., 1025.) );

    peds_.push_back( oHSet);
  }

  const uint16_t nCM_BINS = 1024;
  for( uint16_t nApv = 0; 2 > nApv; ++nApv)
    {
      std::string title = SiStripHistoTitle( sistrip::EXPERT_HISTO,
					     sistrip::PEDESTALS,
					     sistrip::FED_KEY,
					     fedKey(),
					     sistrip::APV,
					     connection().i2cAddr( nApv),
					     sistrip::extrainfo::commonMode_).title();

      HistoSet oHSet;

      oHSet.isProfile_ = false;

      oHSet.vNumOfEntries_.resize( nCM_BINS, 0);

      oHSet.histo( dqm()->book1D( title, title, 
				  nCM_BINS, 
				  nCM_BINS/2 * -1. - 0.5, 
				  nCM_BINS/2 *  1. - 0.5 ) );
    
      cm_.push_back( oHSet);
    }
  
  // Initialize Apv
  pApvFactory_->instantiateApvs( connection().detId(), connection().nApvs() );

  // --[ RETRIEVE PEDESTALS FROM DB ]--
  // Operation should be performed only once
  if( !pDBPedestals.get()) {
    LogTrace( mlDqmSource_)
      << "[NoiseTask::" << __func__ << "] "
      << "Retrieving Pedestals from DB";

    // Directly retrieve Pedestals from EventSetup
    edm::ESHandle<SiStripPedestals> pedestals;
    eventSetup()->get<SiStripPedestalsRcd>().get( pedestals);

    // Cache Pedestals
    pDBPedestals.reset( new SiStripPedestals( *pedestals) );

    LogTrace( mlDqmSource_)
      << "[NoiseTask::" << __func__ << "] "
      << "Done Retrieving Pedestals from DB";
  } // End retrieve Pedestals from DB

  // --[ RETRIEVE NOISES FROM DB ]--
  // Operation should be performed only once
  if( !pDBNoises.get()) {
    LogTrace( mlDqmSource_)
      << "[NoiseTask::" << __func__ << "] "
      << "Retrieving Noises from DB";

    // Directly retrieve Noises from EventSetup
    edm::ESHandle<SiStripNoises> noises;
    eventSetup()->get<SiStripNoisesRcd>().get( noises);

    // Cache Pedestals
    pDBNoises.reset( new SiStripNoises( *noises) );

    LogTrace( mlDqmSource_)
      << "[NoiseTask::" << __func__ << "] "
      << "Done Retrieving Noises from DB";
  } // End retrieve Noises from DB


  // Get ApvAnalysis associated with given DetId
  ApvAnalysisFactory::ApvAnalysisVector 
    apvAnalysisVector( pApvFactory_->getApvAnalysis( connection().detId()));

  SiStripPedestals::Range pedestalsRange( 
					 pDBPedestals->getRange( connection().detId() ) );
  SiStripNoises::Range noisesRange( 
				   pDBNoises->getRange( connection().detId() ) );

  // Cache Apv Pair #
  const uint16_t nAPV_PAIR = connection().apvPairNumber();

  for( uint16_t nLclApv = 0; 2 > nLclApv; ++nLclApv) {
    // Retrieve pedestals and noises associated with given DetId/Apv
    ApvAnalysis::PedestalType pedestals; 
    ApvAnalysis::PedestalType noises;
    for( uint16_t nStrip = nAPV_PAIR * 256 + nLclApv * 128, 
	   nMaxStrip = nStrip + 128; 
         nMaxStrip > nStrip; 
         ++nStrip) 
      {
	pedestals.push_back( pDBPedestals->getPed( nStrip, pedestalsRange));
	noises.push_back( pDBNoises->getNoise( nStrip, noisesRange));
      }

    try {
      // Checked access
      ApvAnalysisFactory::ApvAnalysisVector::reference rApvAnalysis =
        apvAnalysisVector.at( nAPV_PAIR * 2 + nLclApv);

      rApvAnalysis->pedestalCalculator().setPedestals( pedestals);
      rApvAnalysis->pedestalCalculator().setNoise( noises);

      /*
	std::stringstream out;
	LogTrace( mlDqmSource_)
        << "[NoiseTask::" << __func__ << "] "
        << "DetId|Apv# -> " 
        << connection().detId() << '|' << ( nAPV_PAIR * 2 + nLclApv)
        << "   Pedestals: " 
        << ( out << pedestals 
	<< "   Noises: " << noises, out.str());
      */
    } catch( std::out_of_range const& ) {
      // Hmm, didn't find appropriate Apv :((( -> VERY, VERY BAD
      LogTrace( mlDqmSource_)
        << "[NoiseTask::" << __func__ << "] "
        << "Could not set Pedestals/Noises for DetId|Apv# -> " 
        << connection().detId() << '|' << ( nAPV_PAIR * 2 + nLclApv)
        << ". !!! POSSIBLE BUG !!!";
    } // End Try block
  } // End Local Apvs loop
}

// -----------------------------------------------------------------------------
//
void NoiseTask::fill( const SiStripEventSummary         &rSummary,
                      const edm::DetSet<SiStripRawDigi> &rDigis) 
{
  pApvFactory_->updatePair( connection().detId(), 
			    connection().apvPairNumber(), 
			    rDigis);
}

// -----------------------------------------------------------------------------
//
void NoiseTask::update() 
{
  static UpdateTProfile updateTProfile;

  TProfile *pedsProf  = ExtractTObject<TProfile>().extract( peds_[0].histo() );
  TProfile *noiseProf = ExtractTObject<TProfile>().extract( peds_[1].histo() );

  for( uint16_t nLclApv = 2 * connection().apvPairNumber(),
	 nMaxLclApv = nLclApv + 2,
	 nApv = 0;
       nMaxLclApv > nLclApv;
       ++nLclApv, ++nApv)
    {
      ApvAnalysis::PedestalType lclPedestals;
      ApvAnalysis::PedestalType lclNoises;
      ApvAnalysis::PedestalType lclCommonMode( 
					      pApvFactory_->getCommonMode( connection().detId(), nLclApv) );

      pApvFactory_->getPedestal  ( connection().detId(), nLclApv, lclPedestals);
      pApvFactory_->getNoise     ( connection().detId(), nLclApv, lclNoises );

      const uint16_t nSTART_BIN = 128 * ( nLclApv % 2);

      for( uint16_t nBin = 0,
	     nAbsBin = nSTART_BIN + nBin + 1; 
	   128 > nBin; 
	   ++nBin, ++nAbsBin)
	{
	  updateTProfile.setBinContent( pedsProf, nAbsBin, 5, 
					lclPedestals[nBin], lclNoises[nBin]);
	  updateTProfile.setBinContent( noiseProf, nAbsBin, 5,
					lclNoises[nBin], 0);
	} // End loop over BINs

      // Samvel: Assume Once CM value is calculated per chip.
      //         In principle Chip can be divided into a several ranges. Then CM
      //         will be calculated per range. !!! UPDATE CODE THEN !!!
      for( ApvAnalysis::PedestalType::const_iterator cmIterator 
	     = lclCommonMode.begin();
	   cmIterator != lclCommonMode.end();
	   ++cmIterator)
	{
	  //uint32_t nCM = static_cast<uint32_t>( *cmIterator);
	  //if( nCM >= 1024) nCM = 1023;
	  //updateHistoSet( cm_[nApv], nCM);
	  float nCM = static_cast<float>( *cmIterator );
	  updateHistoSet( cm_[nApv], nCM );
	}
      
      std::stringstream out;
      LogTrace( mlDqmSource_)
	<< "[NoiseTask::" << __func__ << "] "
	<< "DET ID [" << connection().detId() 
	<< "] has Common Mode size " << lclCommonMode.size() << " : "
	<< ( out << lclCommonMode, out.str());
    } // End loop over Local Apvs

  updateHistoSet( cm_[0]);
  updateHistoSet( cm_[1]);
}
