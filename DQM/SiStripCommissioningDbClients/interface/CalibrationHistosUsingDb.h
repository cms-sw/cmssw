// Last commit: $Id: CalibrationHistosUsingDb.h,v 1.5 2008/05/06 12:38:06 bainbrid Exp $

#ifndef DQM_SiStripCommissioningClients_CalibrationHistosUsingDb_H
#define DQM_SiStripCommissioningClients_CalibrationHistosUsingDb_H

#include "DQM/SiStripCommissioningClients/interface/CalibrationHistograms.h"
#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include <boost/cstdint.hpp>
#include <string>
#include <map>

class TH1F;

class CalibrationHistosUsingDb : public CommissioningHistosUsingDb, public CalibrationHistograms {
  
 public:
  
  CalibrationHistosUsingDb( const edm::ParameterSet & pset,
                            DQMOldReceiver*,
			    SiStripConfigDb* const,
			    const sistrip::RunType& task = sistrip::CALIBRATION );
  
  CalibrationHistosUsingDb( const edm::ParameterSet & pset,
                            DQMStore*,
			    SiStripConfigDb* const,
			    const sistrip::RunType& task = sistrip::CALIBRATION );

  virtual ~CalibrationHistosUsingDb();

  virtual void uploadConfigurations();
  
 private:
  
  void update( SiStripConfigDb::DeviceDescriptionsRange& );
  
  void create( SiStripConfigDb::AnalysisDescriptionsV&, Analysis );

  TH1F *ishaHistogram_, *vfsHistogram_; 

};

#endif // DQM_SiStripCommissioningClients_CalibrationHistosUsingDb_H

