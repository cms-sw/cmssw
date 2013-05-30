// Last commit: $Id: SiStripGainBuilderFromDb.h,v 1.2 2008/09/29 13:20:28 bainbrid Exp $
// Latest tag:  $Name: V05-01-05 $
// Location:    $Source: /local/reps/CMSSW/CMSSW/OnlineDB/SiStripESSources/interface/SiStripGainBuilderFromDb.h,v $

#ifndef OnlineDB_SiStripESSources_SiStripGainBuilderFromDb_H
#define OnlineDB_SiStripESSources_SiStripGainBuilderFromDb_H

#include "CalibTracker/SiStripESProducers/interface/SiStripGainESSource.h"

class SiStripGainBuilderFromDb : public SiStripGainESSource {
  
 public:
  
  SiStripGainBuilderFromDb( const edm::ParameterSet& );

  virtual ~SiStripGainBuilderFromDb();
  
  /** Builds pedestals using info from configuration database. */
  virtual SiStripApvGain* makeGain();
  
 protected:
  
  /** Virtual method that is called by makeGain() to allow
      gain to be written to the conditions database. */
  virtual void writeGainToCondDb( const SiStripApvGain& ) {;}
  
};

#endif // OnlineDB_SiStripESSources_SiStripGainBuilderFromDb_H
