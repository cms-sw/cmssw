#ifndef IOPool_TFileAdaptor_TFileAdaptor_h
#define IOPool_TFileAdaptor_TFileAdaptor_h

#include "FWCore/Utilities/interface/propagate_const.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

class TPluginManager;

namespace edm {
  class ActivityRegistry;
  class ConfigurationDescriptions;
  class ParameterSet;
}  // namespace edm

// Driver for configuring ROOT plug-in manager to use TStorageFactoryFile.
class TFileAdaptor {
public:
  TFileAdaptor(edm::ParameterSet const& pset, edm::ActivityRegistry& ar);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  friend class TFileAdaptorUI;

private:
  // Write current Storage statistics on a ostream
  void termination(void) const;

  //Called by TFileAdaptorUI
  void stats(std::ostream& o) const;

  void statsXML(std::map<std::string, std::string>& data) const;

  static void addType(TPluginManager* mgr, char const* type, int altType = 0);
  bool native(char const* proto) const;

  bool enabled_;
  bool doStats_;
  bool enablePrefetching_;
  std::string cacheHint_;
  std::string readHint_;
  std::string tempDir_;
  double minFree_;
  unsigned int timeout_;
  unsigned int debugLevel_;
  std::vector<std::string> native_;
};

namespace edm {
  namespace service {
    inline bool isProcessWideService(TFileAdaptor const*) { return true; }
  }  // namespace service
}  // namespace edm

/*
 * wrapper to bind TFileAdaptor to root, python etc
 * loading IOPoolTFileAdaptor library and instantiating
 * TFileAdaptorUI will make root to use StorageAdaptor for I/O instead
 * of its own plugins
 */

class TFileAdaptorUI {
public:
  TFileAdaptorUI();
  ~TFileAdaptorUI();

  // print current Storage statistics on cout
  void stats() const;

private:
  edm::propagate_const<std::shared_ptr<TFileAdaptor>> me;
};

#endif
