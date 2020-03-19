#ifndef DQMSERVICES_COMPONENTS_DQMFILESAVERPB_H
#define DQMSERVICES_COMPONENTS_DQMFILESAVERPB_H

#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include <sys/time.h>
#include <string>
#include <mutex>

#include "DQMFileSaverBase.h"

namespace dqm {

  class DQMFileSaverPB : public DQMFileSaverBase {
  public:
    DQMFileSaverPB(const edm::ParameterSet& ps);
    ~DQMFileSaverPB() override;

    // used by the JsonWritingTimedPoolOutputModule,
    // fms will be nullptr in such case
    static boost::property_tree::ptree fillJson(int run,
                                                int lumi,
                                                const std::string& dataFilePathName,
                                                const std::string& transferDestinationStr,
                                                const std::string& mergeTypeStr,
                                                evf::FastMonitoringService* fms);

  protected:
    void initRun() const override;
    void saveLumi(const FileParameters& fp) const override;
    void saveRun(const FileParameters& fp) const override;
    void savePB(DQMStore* store, std::string const& filename, int run, int lumi) const;

    bool fakeFilterUnitMode_;
    std::string streamLabel_;
    mutable std::string transferDestination_;
    mutable std::string mergeType_;

  public:
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  };

}  // namespace dqm

#endif  // DQMSERVICES_COMPONENTS_DQMFILESAVERPB_H
