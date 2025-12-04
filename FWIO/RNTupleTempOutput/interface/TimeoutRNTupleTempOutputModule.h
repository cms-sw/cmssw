#ifndef FWIO_RNTupleTempOutput_TimeoutRNTupleTempOutputModule_h
#define FWIO_RNTupleTempOutput_TimeoutRNTupleTempOutputModule_h

//////////////////////////////////////////////////////////////////////
//
// Class TimeoutRNTupleTempOutputModule. Output module to POOL file with file
// closure based on timeout. First file has only one event, second
// file is closed after 15 seconds if at least one event was processed.
// Then timeout is increased to 30 seconds and 60 seconds. After that
// all other files are closed with timeout of 60 seconds.
//
// Created by Dmytro.Kovalskyi@cern.ch
//
//////////////////////////////////////////////////////////////////////

#include "FWIO/RNTupleTempOutput/interface/RNTupleTempOutputModule.h"

namespace edm {
  class ConfigurationDescriptions;
  class ModuleCallingContext;
  class ParameterSet;
}  // namespace edm
namespace edm::rntuple_temp {

  class TimeoutRNTupleTempOutputModule : public RNTupleTempOutputModule {
  public:
    explicit TimeoutRNTupleTempOutputModule(ParameterSet const& ps);
    ~TimeoutRNTupleTempOutputModule() override {}
    TimeoutRNTupleTempOutputModule(TimeoutRNTupleTempOutputModule const&) = delete;  // Disallow copying and moving
    TimeoutRNTupleTempOutputModule& operator=(TimeoutRNTupleTempOutputModule const&) =
        delete;  // Disallow copying and moving

    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  protected:
    bool shouldWeCloseFile() const override;
    void write(EventForOutput const& e) override;

  private:
    mutable time_t m_lastEvent;
    mutable unsigned int eventsWrittenInCurrentFile;
    mutable int m_timeout;
  };
}  // namespace edm::rntuple_temp

#endif
