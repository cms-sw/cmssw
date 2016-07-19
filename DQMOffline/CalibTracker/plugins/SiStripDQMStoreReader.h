#ifndef DQMOffline_CalibTracker_SiStripDQMStoreReader_H
#define DQMOffline_CalibTracker_SiStripDQMStoreReader_H

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"

/**
 * @class SiStripDQMStoreReader
 * Base class with utilities to read from the DQMStore
 *
 * Split out of other classes
 */
class SiStripDQMStoreReader {
public:
  explicit SiStripDQMStoreReader(const edm::ParameterSet& pset)
    : m_accessDQMFile{pset.getParameter<bool>("accessDQMFile")}
    , m_fileName{pset.getUntrackedParameter<std::string>("FILE_NAME", "")}
  {}

private:
  bool m_accessDQMFile;
  std::string m_fileName;

protected:
  mutable DQMStore* dqmStore_;

  /// Uses DQMStore to access the DQM file
  void openRequestedFile() const;

  /// Uses DQM utilities to access the requested dir
  bool goToDir(const std::string& name) const;
  /// Fill the mfolders vector with the full list of directories for all the modules
  void getModuleFolderList(std::vector<std::string>& mfolders) const;

  /**
   * Returns a pointer to the monitoring element corresponding to the given detId and name. <br>
   * The name convention for module histograms is NAME__det__DETID. The name provided
   * must be NAME, removing all the __det__DETID part. This latter part will be built
   * and attached internally using the provided detId.
   */
  MonitorElement* getModuleHistogram(const uint32_t detId, const std::string & name) const;

  // Simple functor to remove unneeded ME
  struct StringNotMatch
  {
    StringNotMatch(const std::string & name) :
      name_(name)
    {
    }
    bool operator()(const MonitorElement * ME) const
    {
      return( ME->getName().find(name_) == std::string::npos );
    }
  protected:
    std::string name_;
  };
};

#endif // DQMOffline_CalibTracker_SiStripDQMStoreReader_H
