
#ifndef OnlineDB_SiStripESSources_SiStripNoiseBuilderFromDb_H
#define OnlineDB_SiStripESSources_SiStripNoiseBuilderFromDb_H

#include "CalibTracker/SiStripESProducers/interface/SiStripNoiseESSource.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "OnlineDB/SiStripESSources/interface/SiStripCondObjBuilderFromDb.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripDbParams.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <vector>
#include <string>

class SiStripNoiseBuilderFromDb : public SiStripNoiseESSource {
public:
  SiStripNoiseBuilderFromDb(const edm::ParameterSet&);
  ~SiStripNoiseBuilderFromDb() override;

  /** Builds pedestals using info from configuration database. */
  SiStripNoises* makeNoise() override;

protected:
  /** Virtual method that is called by makeNoise() to allow
      pedestals to be written to the conditions database. */
  virtual void writeNoiseToCondDb(const SiStripNoises&) { ; }

  /** Container for DB connection parameters. */
  SiStripDbParams dbParams_;

  /** Service to access onlineDB and extract pedestal/noise */
  edm::Service<SiStripCondObjBuilderFromDb> condObjBuilder;
};

#endif  // OnlineDB_SiStripESSources_SiStripNoiseBuilderFromDb_H
