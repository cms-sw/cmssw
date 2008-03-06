// Last commit: $Id: CalibrationHistosUsingDb.cc,v 1.2 2008/02/21 14:08:02 delaer Exp $

#include "DQM/SiStripCommissioningDbClients/interface/CalibrationHistosUsingDb.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include <iostream>

using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
CalibrationHistosUsingDb::CalibrationHistosUsingDb( DQMOldReceiver* mui,
					            const DbParams& params,
						    const sistrip::RunType& task )
  : CommissioningHistosUsingDb( params ),
    CalibrationHistograms( mui, task )
{
  LogTrace(mlDqmClient_) 
    << "[CalibrationHistosUsingDb::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
CalibrationHistosUsingDb::CalibrationHistosUsingDb( DQMOldReceiver* mui,
					      SiStripConfigDb* const db,
					      const sistrip::RunType& task )
  : CommissioningHistograms( mui, task ),
    CommissioningHistosUsingDb( db, mui, task),
    CalibrationHistograms( mui, task )
{
  LogTrace(mlDqmClient_) 
    << "[CalibrationHistosUsingDb::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
CalibrationHistosUsingDb::CalibrationHistosUsingDb( DQMStore* bei,
					      SiStripConfigDb* const db,
					      const sistrip::RunType& task ) 
  : CommissioningHistosUsingDb( db, task ),
    CalibrationHistograms( bei, task )
{
  LogTrace(mlDqmClient_) 
    << "[CalibrationHistosUsingDb::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
CalibrationHistosUsingDb::~CalibrationHistosUsingDb() {
  LogTrace(mlDqmClient_) 
    << "[CalibrationHistosUsingDb::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
/** */
void CalibrationHistosUsingDb::uploadConfigurations() {
  
  if ( !db() ) {
    edm::LogWarning(mlDqmClient_) 
      << "[CalibrationHistosUsingDb::" << __func__ << "]"
      << " NULL pointer to SiStripConfigDb interface!"
      << " Aborting upload...";
    return;
  }
 
  // nothing to update

}

// -----------------------------------------------------------------------------
/** */
void CalibrationHistosUsingDb::update( SiStripConfigDb::DeviceDescriptions&  ) {

  // nothing to update
}


// -----------------------------------------------------------------------------
/** */
void CalibrationHistosUsingDb::create( SiStripConfigDb::AnalysisDescriptions&,
				     Analysis ) {

  // nothing to update
  
}

