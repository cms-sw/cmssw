// Last commit: $Id: SiStripGainBuilderFromDb.h,v 1.3 2013/05/30 21:52:09 gartung Exp $
// Latest tag:  $Name: CMSSW_6_2_0 $
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
