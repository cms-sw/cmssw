#ifndef DataFormats_Provenance_ProcessConfiguration_h
#define DataFormats_Provenance_ProcessConfiguration_h

#include <iosfwd>
#include <string>

#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/PassID.h"
#include "DataFormats/Provenance/interface/ReleaseVersion.h"
#include "DataFormats/Provenance/interface/ProcessConfigurationID.h"

namespace edm {
  struct ProcessConfiguration {
    ProcessConfiguration() : processName_(), processGUID_(), parameterSetID_(), releaseVersion_(), passID_() {}
    ProcessConfiguration(std::string const& procName,
			 std::string const& procGUID,
			 ParameterSetID const& pSetID,
			 ReleaseVersion const& relVersion,
			 PassID const& pass) :
      processName_(procName),
      processGUID_(procGUID),
      parameterSetID_(pSetID),
      releaseVersion_(relVersion),
      passID_(pass) { }
    
    std::string const& processName() const {return processName_;}
    std::string const& processGUID() const {return processGUID_;}
    ParameterSetID const& parameterSetID() const {return parameterSetID_;}
    ReleaseVersion const& releaseVersion() const {return releaseVersion_;}
    PassID const& passID() const {return passID_;}
    ProcessConfigurationID id() const;

    std::string processName_;
    std::string processGUID_;
    ParameterSetID parameterSetID_;
    ReleaseVersion releaseVersion_; 
    PassID passID_;
  };

  inline
  bool
  operator==(ProcessConfiguration const& a, ProcessConfiguration const& b) {
    return a.processName() == b.processName() &&
    a.processGUID() == b.processGUID() &&
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
