// Last commit: $Id: CalibrationHistosUsingDb.cc,v 1.1 2008/02/20 21:01:08 delaer Exp $

#include "DQM/SiStripCommissioningDbClients/interface/CalibrationHistosUsingDb.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include <iostream>

using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
CalibrationHistosUsingDb::CalibrationHistosUsingDb( MonitorUserInterface* mui,
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
CalibrationHistosUsingDb::CalibrationHistosUsingDb( MonitorUserInterface* mui,
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
CalibrationHistosUsingDb::CalibrationHistosUsingDb( DaqMonitorBEInterface* bei,
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

