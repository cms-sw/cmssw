
#ifndef OnlineDB_SiStripESSources_SiStripGainBuilderFromDb_H
#define OnlineDB_SiStripESSources_SiStripGainBuilderFromDb_H

#include "CalibTracker/SiStripESProducers/interface/SiStripGainESSource.h"

class SiStripGainBuilderFromDb : public SiStripGainESSource {
public:
  SiStripGainBuilderFromDb(const edm::ParameterSet&);

  ~SiStripGainBuilderFromDb() override;

  /** Builds pedestals using info from configuration database. */
  SiStripApvGain* makeGain() override;

protected:
  /** Virtual method that is called by makeGain() to allow
      gain to be written to the conditions database. */
  virtual void writeGainToCondDb(const SiStripApvGain&) { ; }
};

#endif  // OnlineDB_SiStripESSources_SiStripGainBuilderFromDb_H
