#ifndef DQMSERVICES_COMPONENTS_DQMFILESAVEROUTPUT_H
#define DQMSERVICES_COMPONENTS_DQMFILESAVEROUTPUT_H

#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <sys/time.h>
#include <string>
#include <mutex>

#include "DQMFileSaverBase.h"

namespace dqm {

class DQMFileSaverOnline : public DQMFileSaverBase {
 public:
  DQMFileSaverOnline(const edm::ParameterSet &ps);
  ~DQMFileSaverOnline();

  static const std::string fillOrigin(const std::string filename,
                                  const std::string final_filename);

 protected:
  virtual void saveLumi(const FileParameters& fp) const override;
  virtual void saveRun(const FileParameters& fp) const override;

 protected:
  int backupLumiCount_;

  // snapshot making
  struct SnapshotFiles {
    std::string data;
    std::string meta;
  };

  void makeSnapshot(const FileParameters &fp, bool final) const;
  void appendSnapshot(SnapshotFiles new_snap) const;

  mutable std::mutex snapshots_lock_;
  mutable std::list<SnapshotFiles> snapshots_;

  void checkError(const char *msg, const std::string file, int status) const;

 public:
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
};

}  // dqm namespace

#endif  // DQMSERVICES_COMPONENTS_DQMFILESAVEROUTPUT_H
