#ifndef DQMSERVICES_COMPONENTS_DQMFILESAVERPB_H
#define DQMSERVICES_COMPONENTS_DQMFILESAVERPB_H

#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <sys/time.h>
#include <string>
#include <mutex>

#include "DQMFileSaverBase.h"

namespace dqm {

class DQMFileSaverPB : public DQMFileSaverBase {
 public:
  DQMFileSaverPB(const edm::ParameterSet &ps);
  ~DQMFileSaverPB();

  // used by the JsonWritingTimedPoolOutputModule,
  // fms will be nullptr in such case
  static boost::property_tree::ptree fillJson(
      int run, int lumi, const std::string &dataFilePathName, const std::string transferDestinationStr,
      evf::FastMonitoringService *fms);

 protected:
  virtual void initRun() const override;
  virtual void saveLumi(const FileParameters& fp) const override;
  virtual void saveRun(const FileParameters& fp) const override;

  bool fakeFilterUnitMode_;
  std::string streamLabel_;
  mutable std::string transferDestination_;

 public:
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
};

}  // dqm namespace

#endif  // DQMSERVICES_COMPONENTS_DQMFILESAVERPB_H
