#ifndef DQMSERVICES_CORE_LEGACYIOHELPER_H
#define DQMSERVICES_CORE_LEGACYIOHELPER_H

#include "DQMServices/Core/interface/DQMStore.h"
#include "TROOT.h"

// This class encapsulates the TDirectory based file format used for DQMGUI
// uploads and many other use cases.
// This should be part of `DQMFileSaver`, however since DQMServices/Components
// DQMFileSaver and DQMServices/FileIO DQMFileSaverOnline both write this
// format, the code is shared here (evnetually, these modules should become one
// again).
// This code is in DQMServices/Core to also allow the legacy DQMStore::save
// interface to use this without adding another dependency.
class LegacyIOHelper {
public:
  // use internal type here since we call this from the DQMStore itself.
  typedef dqm::implementation::DQMStore DQMStore;
  typedef dqm::legacy::MonitorElement MonitorElement;

  typedef dqm::harvesting::DQMStore HarvestedDQMStore;
  typedef dqm::harvesting::MonitorElement HarvestedMonitorElement;

  using MEMap = std::set<HarvestedMonitorElement*>;

  LegacyIOHelper(DQMStore* dqmstore) : dbe_(dqmstore){};
  LegacyIOHelper(HarvestedDQMStore* hdqmstore) : dbe_(hdqmstore){};
  // Replace or append to `filename`, a TDirectory ROOT file. If a run number
  // is passed, the paths are rewritten to the "Run Summary" format used by
  // DQMGUI. The run number does not affect which MEs are saved; this code only
  // supports non-threaded mode. `fileupdate` is passed to ROOT unchanged.
  // The run number passed in is added to the Directory structure inside the
  // file ("Run xxxxxx/.../Run Summary/...") if not 0. It is only used to
  // select only MEs for that run iff saveall is false, else all MEs (RUN, LUMI
  // and JOB) are saved.
  void save(std::string const& filename,
            std::string const& path = "",
            uint32_t const run = 0,
            bool saveall = true,
            std::string const& fileupdate = "RECREATE");

  bool open(std::string const& filename, std::string const& path = "", uint32_t const run = 0);

private:
  template <class T>
  void getMEName(T* h, const std::string& toppath, std::string& meName) {
    std::ostringstream fullpath;
    fullpath << gDirectory->GetPath() << "/" << h->GetName();
    std::string dirpath = fullpath.str();
    // Search for the substring in string
    size_t pos = dirpath.find(toppath);
    if (pos != std::string::npos) {
      dirpath.erase(pos, toppath.length());
    }
    std::string rsummary = "/Run summary";
    pos = dirpath.find(rsummary);
    if (pos != std::string::npos) {
      dirpath.erase(pos, rsummary.length());
    }
    meName = dirpath;
  }
  bool readdir(TDirectory* dir, const std::string& toppath);
  bool createDirectoryIfNeededAndCd(const std::string& path);
  DQMStore* dbe_;
  MEMap data_;
};

#endif
