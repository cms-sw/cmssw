#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/LegacyIOHelper.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"

#include "DQMFileSaverOnline.h"

#include <TString.h>
#include <TSystem.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <openssl/md5.h>
#include <boost/filesystem.hpp>
#include <boost/iostreams/device/mapped_file.hpp>

using namespace dqm;

DQMFileSaverOnline::DQMFileSaverOnline(const edm::ParameterSet& ps) : DQMFileSaverBase(ps) {
  backupLumiCount_ = ps.getUntrackedParameter<int>("backupLumiCount", 1);
  keepBackupLumi_ = ps.getUntrackedParameter<bool>("keepBackupLumi", false);
}

DQMFileSaverOnline::~DQMFileSaverOnline() = default;

void DQMFileSaverOnline::saveLumi(const FileParameters& fp) const {
  if (backupLumiCount_ > 0) {
    if (fp.lumi_ % backupLumiCount_ == 0) {
      // actual saving is done here
      makeSnapshot(fp, false);
    }
  }
}

void DQMFileSaverOnline::saveRun(const FileParameters& fp) const { makeSnapshot(fp, true); }

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
  // logFileAction("Writing DQM Origin file: ", meta_fp);

  //char rewrite[128];
  //snprintf(rewrite, 128, "\\1Run %ld/\\2/Run summary", fp.run_);

  //store->save(tmp_root_fp,                      /* filename      */
  //            "",                               /* path          */
  //            "^(Reference/)?([^/]+)",          /* pattern       */
  //            rewrite,                          /* rewrite       */
  //            store->mtEnabled() ? fp.run_ : 0, /* run           */
  //            0,                                /* lumi          */
  //            fp.saveReference_,                /* ref           */
  //            fp.saveReferenceQMin_,            /* ref minStatus */
  //            "RECREATE");                      /* fileupdate    */
  // TODO: some parameters prepared here are now unused, and the code should
  // eventually be removed.
  LegacyIOHelper h(&*store);
  h.save(tmp_root_fp, "", fp.run_, /* saveall */ true, "RECREATE");

  // write metadata
  // format.origin: md5:d566a34b27f48d507150a332b189398b 294835
  // /home/dqmprolocal/output/DQM_V0001_FED_R000194224.root
  std::ofstream meta_fd(tmp_meta_fp);
  meta_fd << fillOrigin(tmp_root_fp, root_fp);
  meta_fd.close();

  checkError("Rename failed: ", root_fp, ::rename(tmp_root_fp.c_str(), root_fp.c_str()));
  checkError("Rename failed: ", meta_fp, ::rename(tmp_meta_fp.c_str(), meta_fp.c_str()));

  SnapshotFiles files = {root_fp, meta_fp};
  if (final) {
    // final will never be cleared
    appendSnapshot(SnapshotFiles{});

    saveJobReport(root_fp);
  } else {
    appendSnapshot(SnapshotFiles{root_fp, meta_fp});
  }
}

void DQMFileSaverOnline::appendSnapshot(SnapshotFiles f) const {
  std::lock_guard<std::mutex> lock(snapshots_lock_);

  if (!keepBackupLumi_) {
    while (!snapshots_.empty()) {
      SnapshotFiles& x = snapshots_.front();

      // logFileAction("Deleting old snapshot (origin): ", x.meta);
      checkError("Unlink failed: ", x.meta, ::unlink(x.meta.c_str()));

      logFileAction("Deleting old snapshot (root): ", x.data);
      checkError("Unlink failed: ", x.data, ::unlink(x.data.c_str()));

      snapshots_.pop_front();
    }
  }

  if (!f.data.empty()) {
    snapshots_.push_back(f);
  }
}

void DQMFileSaverOnline::checkError(const char* msg, const std::string& file, int status) const {
  if (status != 0) {
    std::string actual_msg = msg;
    actual_msg += std::strerror(status);
    logFileAction(actual_msg, file);
  }
}

const std::string DQMFileSaverOnline::fillOrigin(const std::string& filename, const std::string& final_filename) {
  // format.origin (one line):
  //   md5:d566a34b27f48d507150a332b189398b 294835 final_filename.root

  unsigned char md5[MD5_DIGEST_LENGTH];

  boost::iostreams::mapped_file_source fp(filename);

  MD5((unsigned char*)fp.data(), fp.size(), md5);

  std::ostringstream hash;
  for (unsigned char& i : md5) {
    hash << std::hex << std::setfill('0') << std::setw(2) << (int)i;
  }

  std::ostringstream out;
  out << "md5:" << hash.str() << " " << fp.size() << " " << final_filename;
  return out.str();
}

void DQMFileSaverOnline::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Saves histograms from DQM store, online workflow.");

  desc.addUntracked<int>("backupLumiCount", 10)
      ->setComment(
          "How often the backup file will be generated, in lumisections (-1 "
          "disables).");

  desc.addUntracked<bool>("keepBackupLumi", false)
      ->setComment(
          "Usually the backup old backup is deleted once the new file is "
          "available. Setting this to true ensures that no backup files are "
          "ever deleted. Useful for ML applications, which use backups as a "
          "'history' of what happened during the run.");

  DQMFileSaverBase::fillDescription(desc);

  // Changed to use addDefault instead of add here because previously
  // DQMFileSaverOnline and DQMFileSaverPB both used the module label
  // "saver" which caused conflicting cfi filenames to be generated.
  // add could be used if unique module labels were given.
  descriptions.addDefault(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DQMFileSaverOnline);
