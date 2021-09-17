#ifndef DQMSERVICES_COMPONENTS_DQMFILESAVEROUTPUT_H
#define DQMSERVICES_COMPONENTS_DQMFILESAVEROUTPUT_H

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"

#include <sys/time.h>
#include <mutex>
#include <string>

#include "DQMFileSaverBase.h"

namespace dqm {

  class DQMFileSaverOnline : public DQMFileSaverBase {
  public:
    DQMFileSaverOnline(const edm::ParameterSet& ps);
    ~DQMFileSaverOnline() override;

    static const std::string fillOrigin(const std::string& filename, const std::string& final_filename);

  protected:
    void saveLumi(const FileParameters& fp) const override;
    void saveRun(const FileParameters& fp) const override;

  protected:
    int backupLumiCount_;
    bool keepBackupLumi_;

    // snapshot making
    struct SnapshotFiles {
      std::string data;
      std::string meta;
    };

    void makeSnapshot(const FileParameters& fp, bool final) const;
    void appendSnapshot(SnapshotFiles new_snap) const;

    mutable std::mutex snapshots_lock_;
    mutable std::list<SnapshotFiles> snapshots_;

    void checkError(const char* msg, const std::string& file, int status) const;

  public:
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  };

}  // namespace dqm

#endif  // DQMSERVICES_COMPONENTS_DQMFILESAVEROUTPUT_H
