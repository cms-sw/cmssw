#ifndef DataFormats_Provenance_ProcessConfiguration_h
#define DataFormats_Provenance_ProcessConfiguration_h

#include "DataFormats/Provenance/interface/HardwareResourcesDescription.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/ProcessConfigurationID.h"
#include "DataFormats/Provenance/interface/ReleaseVersion.h"

#include <iosfwd>
#include <string>
#include <vector>

namespace edm {
  class ProcessConfiguration {
  public:
    ProcessConfiguration();
    ProcessConfiguration(std::string const& procName,
                         ReleaseVersion const& relVersion,
                         HardwareResourcesDescription const& hwDescription);

    ProcessConfiguration(std::string const& procName,
                         ParameterSetID const& pSetID,
                         ReleaseVersion const& relVersion,
                         HardwareResourcesDescription const& hwDescription);

    std::string const& processName() const { return processName_; }
    ParameterSetID const& parameterSetID() const;
    bool isParameterSetValid() const { return parameterSetID_.isValid(); }
    ReleaseVersion const& releaseVersion() const { return releaseVersion_; }
    /// Note: this function parses the string on every call, so it should be called rarely
    HardwareResourcesDescription hardwareResourcesDescription() const { return HardwareResourcesDescription(passID_); }
    std::string const& hardwareResourcesDescriptionSerialized() const { return passID_; }
    ProcessConfigurationID id() const;

    void setParameterSetID(ParameterSetID const& pSetID);

    ProcessConfigurationID setProcessConfigurationID();

    void reduce();

    void initializeTransients() { transient_.reset(); }

    struct Transients {
      Transients() : pcid_(), isCurrentProcess_(false) {}
      explicit Transients(bool current) : pcid_(), isCurrentProcess_(current) {}
      void reset() {
        pcid_.reset();
        isCurrentProcess_ = false;
      }
      ProcessConfigurationID pcid_;
      bool isCurrentProcess_;
    };

  private:
    void setPCID(ProcessConfigurationID const& pcid) { transient_.pcid_ = pcid; }
    bool isCurrentProcess() const { return transient_.isCurrentProcess_; }
    void setCurrentProcess() { transient_.isCurrentProcess_ = true; }

    std::string processName_;
    ParameterSetID parameterSetID_;
    ReleaseVersion releaseVersion_;
    // The passID_ really holds the HardwareResourcesDescription in a
    // serialized form. Therefore the passID name is a complete
    // misnomer, but was kept to make forward-compabitility easier
    // (even if that not formally not supported, this construct is a
    // precaution in case a further Run3 use case would surface)
    std::string passID_;
    Transients transient_;
  };

  typedef std::vector<ProcessConfiguration> ProcessConfigurationVector;

  bool operator<(ProcessConfiguration const& a, ProcessConfiguration const& b);

  inline bool operator==(ProcessConfiguration const& a, ProcessConfiguration const& b) {
    return a.processName() == b.processName() && a.parameterSetID() == b.parameterSetID() &&
           a.releaseVersion() == b.releaseVersion() &&
           a.hardwareResourcesDescriptionSerialized() == b.hardwareResourcesDescriptionSerialized();
  }

  inline bool operator!=(ProcessConfiguration const& a, ProcessConfiguration const& b) { return !(a == b); }

  std::ostream& operator<<(std::ostream& os, ProcessConfiguration const& pc);
}  // namespace edm

#endif
