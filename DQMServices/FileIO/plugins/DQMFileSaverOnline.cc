#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/JobReport.h"

#include "DQMFileSaverOnline.h"

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <utility>
#include <TString.h>
#include <TSystem.h>

using namespace dqm;

DQMFileSaverOnline::DQMFileSaverOnline(const edm::ParameterSet &ps)
    : DQMFileSaverBase(ps) {

  backupLumiCount_ = ps.getUntrackedParameter<int>("backupLumiCount", 1);
}

DQMFileSaverOnline::~DQMFileSaverOnline() {}

void DQMFileSaverOnline::saveLumi(FileParameters fp) const {
  if (backupLumiCount_ > 0) {
    if (fp.lumi_ % backupLumiCount_ == 0) {
  
      // actual saving is done here
      makeSnapshot(fp, false);
    }
  }
}

void DQMFileSaverOnline::saveRun(FileParameters fp) const {
  makeSnapshot(fp, true);
}

void DQMFileSaverOnline::makeSnapshot(const FileParameters& fp, bool final) const {
  int pid = getpid();
  char hostname[64];
  gethostname(hostname, 64);
  hostname[63] = 0;

  char suffix[128];
  if (!final) {
    snprintf(suffix, 127, ".ls%08ld_host%s_pid%08d", fp.lumi_, hostname, pid);
  } else {
    suffix[0] = 0;
  }

  std::string prefix = filename(fp, false);

  std::string root_fp = prefix + ".root" + suffix;
  std::string meta_fp = prefix + ".root.origin" + suffix;

  std::string tmp_root_fp = root_fp + ".tmp";
  std::string tmp_meta_fp = meta_fp + ".tmp";

  // run_ and lumi_ are ignored if dqmstore is not in multithread mode
  edm::Service<DQMStore> store;

  logFileAction("Writing DQM Root file: ", root_fp);
  //logFileAction("Writing DQM Origin file: ", meta_fp);

  char rewrite[128];
  snprintf(rewrite, 128, "\\1Run %ld/\\2/Run summary", fp.run_);

  store->save(tmp_root_fp,                       /* filename      */
              "",                                /* path          */
              "^(Reference/)?([^/]+)",           /* pattern       */
              rewrite,                           /* rewrite       */
              store->mtEnabled() ? fp.run_ : 0, /* run           */
              0,                                 /* lumi          */
              fp.saveReference_,                /* ref           */
              fp.saveReferenceQMin_,            /* ref minStatus */
              "RECREATE",                        /* fileupdate    */
              false                              /* resetMEs      */
              );

  // write metadata
  // format.origin: md5:d566a34b27f48d507150a332b189398b 294835 /home/dqmprolocal/output/DQM_V0001_FED_R000194224.root
  std::ofstream meta_fd(tmp_meta_fp);
  meta_fd << this->fillOrigin(tmp_root_fp, root_fp);
  meta_fd.close();

  checkError("Rename failed: ", root_fp, ::rename(tmp_root_fp.c_str(), root_fp.c_str()));
  checkError("Rename failed: ", meta_fp, ::rename(tmp_meta_fp.c_str(), meta_fp.c_str()));

  SnapshotFiles files = { root_fp, meta_fp };
  if (final) {
    // final will never be cleared
    appendSnapshot(SnapshotFiles{});

    saveJobReport(root_fp);
  } else {
    appendSnapshot(SnapshotFiles{ root_fp, meta_fp });
  }
}

void DQMFileSaverOnline::appendSnapshot(SnapshotFiles f) const {
  std::lock_guard<std::mutex> lock(snapshots_lock_);

  while (! snapshots_.empty()) {
    SnapshotFiles& x = snapshots_.front();

    //logFileAction("Deleting old snapshot (origin): ", x.meta);
    checkError("Unlink failed: ", x.meta, ::unlink(x.meta.c_str()));

    logFileAction("Deleting old snapshot (root): ", x.data);
    checkError("Unlink failed: ", x.data, ::unlink(x.data.c_str()));

    snapshots_.pop_front();
  }

  if (! f.data.empty()) {
    snapshots_.push_back(f);
  }
}

void DQMFileSaverOnline::checkError(const char *msg, const std::string file, int status) const {
  if (status != 0) {
    std::string actual_msg = msg;
    actual_msg += std::strerror(status);
    logFileAction(actual_msg, file);
  }
}

void DQMFileSaverOnline::fillDescriptions(
    edm::ConfigurationDescriptions& descriptions) {

  edm::ParameterSetDescription desc;
  desc.setComment("Saves histograms from DQM store, online workflow.");

  desc.addUntracked<int>("backupLumiCount", 10)->setComment(
      "How often the backup file will be generated, in lumisections (-1 disables).");

  DQMFileSaverBase::fillDescription(desc);

  descriptions.add("saver", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DQMFileSaverOnline);
