#ifndef IOPool_Output_TimeoutPoolOutputModule_h
#define IOPool_Output_TimeoutPoolOutputModule_h

//////////////////////////////////////////////////////////////////////
//
// Class TimeoutPoolOutputModule. Output module to POOL file with file 
// closure based on timeout. First file has only one event, second
// file is closed after 15 seconds if at least one event was processed.
// Then timeout is increased to 30 seconds and 60 seconds. After that
// all other files are closed with timeout of 60 seconds.
//
// Created by Dmytro.Kovalskyi@cern.ch
//
//////////////////////////////////////////////////////////////////////

#include "IOPool/Output/interface/PoolOutputModule.h"

namespace edm {
  class ParameterSet;

  class TimeoutPoolOutputModule : public PoolOutputModule {
  public:
    explicit TimeoutPoolOutputModule(ParameterSet const& ps);
    virtual ~TimeoutPoolOutputModule(){};
  protected:
    virtual bool shouldWeCloseFile() const;
  private:
    mutable time_t m_lastEvent;
    mutable int    m_timeout;
  };
}

#endif
