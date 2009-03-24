#ifndef DataFormats_Provenance_ProcessConfiguration_h
#define DataFormats_Provenance_ProcessConfiguration_h

#include <iosfwd>
#include <string>

#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/PassID.h"
#include "DataFormats/Provenance/interface/ReleaseVersion.h"
#include "DataFormats/Provenance/interface/ProcessConfigurationID.h"
#include "DataFormats/Provenance/interface/Transient.h"

namespace edm {
  class ProcessConfiguration {
  public:
    ProcessConfiguration();
    ProcessConfiguration(std::string const& procName,
			 ParameterSetID const& pSetID,
			 ReleaseVersion const& relVersion,
			 PassID const& pass);
    
    std::string const& processName() const {return processName_;}
    ParameterSetID const& parameterSetID() const {return parameterSetID_;}
    ReleaseVersion const& releaseVersion() const {return releaseVersion_;}
    PassID const& passID() const {return passID_;}
    ProcessConfigurationID id() const;

    struct Transients {
      Transients() : pcid_() {}
      ProcessConfigurationID pcid_;
    };

  private:
    ProcessConfigurationID & pcid() const {return transients_.get().pcid_;}
    std::string processName_;
    ParameterSetID parameterSetID_;
    ReleaseVersion releaseVersion_; 
    PassID passID_;
    mutable Transient<Transients> transients_;
  };

  bool
  operator<(ProcessConfiguration const& a, ProcessConfiguration const& b);

  inline
  bool
  operator==(ProcessConfiguration const& a, ProcessConfiguration const& b) {
    return a.processName() == b.processName() &&
    a.parameterSetID() == b.parameterSetID() &&
    a.releaseVersion() == b.releaseVersion() &&
    a.passID() == b.passID();
  }

  inline
  bool
  operator!=(ProcessConfiguration const& a, ProcessConfiguration const& b) {
    return !(a == b);
  }

  std::ostream&
  operator<< (std::ostream& os, ProcessConfiguration const& pc);
}

#endif
