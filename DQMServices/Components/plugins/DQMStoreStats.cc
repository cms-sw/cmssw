/*
 * \file DQMStoreStats.cc
 * \author Andreas Meyer
 * Last Update:
 *
 * Description: Print out statistics of histograms in DQMStore
*/

#include "DQMStoreStats.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/JobReport.h"

using namespace std;
using namespace edm;

template <class T>
static unsigned int getEmptyMetric(T* array, int lenx, int leny, int lenz) {
  // len{x,y,z} MUST include under/overflow bins.
  unsigned int len = lenx + leny + lenz;
  unsigned int result = 0;
  // start from 1 to exclude underflow bin. The comparison is accurate
  // since it takes properly into account under/overflow bins, for all
  // kind of histograms.
  for (unsigned int i = 1; i < len; ++i) {
    // get rid of under/overflow bins for x,y,z axis, to have a correct statistics.
    if (i % (lenx - 1) == 0)
      continue;
    if (i % lenx == 0)
      continue;
    if (i % (lenx + leny - 1) == 0)
      continue;
    if (i % (lenx + leny) == 0)
      continue;
    if (i % (lenx + leny + lenz - 1) == 0)
      continue;

    if (array[i] == 0)
      result += 1;
  }

  return result;
}

//==================================================================//
//================= Constructor and Destructor =====================//
//==================================================================//
DQMStoreStats::DQMStoreStats(const edm::ParameterSet& ps)
    : subsystem_(""),
      subfolder_(""),
      nbinsglobal_(0),
      nbinssubsys_(0),
      nmeglobal_(0),
      nmesubsys_(0),
      maxbinsglobal_(0),
      maxbinssubsys_(0),
      maxbinsmeglobal_(""),
      maxbinsmesubsys_(""),
      statsdepth_(1),
      pathnamematch_(""),
      verbose_(0) {
  parameters_ = ps;
  pathnamematch_ = ps.getUntrackedParameter<std::string>("pathNameMatch", pathnamematch_);
  statsdepth_ = ps.getUntrackedParameter<int>("statsDepth", statsdepth_);
  verbose_ = ps.getUntrackedParameter<int>("verbose", verbose_);
  dumpMemHistory_ = ps.getUntrackedParameter<bool>("dumpMemoryHistory", false);
  runonendrun_ = ps.getUntrackedParameter<bool>("runOnEndRun", true);
  runonendjob_ = ps.getUntrackedParameter<bool>("runOnEndJob", false);
  runonendlumi_ = ps.getUntrackedParameter<bool>("runOnEndLumi", false);
  runineventloop_ = ps.getUntrackedParameter<bool>("runInEventLoop", false);
  dumpToFWJR_ = ps.getUntrackedParameter<bool>("dumpToFWJR", false);

  startingTime_ = time(nullptr);
}

DQMStoreStats::~DQMStoreStats() = default;

void DQMStoreStats::calcIgProfDump(Folder& root) {
  std::ofstream stream("dqm-bin-stats.sql");
  stream << ""
            "    PRAGMA journal_mode=OFF;"
            "    PRAGMA count_changes=OFF;"
            "    DROP TABLE IF EXISTS files;"
            "    DROP TABLE IF EXISTS symbols;"
            "    DROP TABLE IF EXISTS mainrows;"
            "    DROP TABLE IF EXISTS children;"
            "    DROP TABLE IF EXISTS parents;"
            "    DROP TABLE IF EXISTS summary;"
            "    CREATE TABLE children ("
            "        self_id INTEGER CONSTRAINT self_exists REFERENCES mainrows(id),"
            "        parent_id INTEGER CONSTRAINT parent_exists REFERENCES mainrows(id),"
            "        from_parent_count INTEGER,"
            "        from_parent_calls INTEGER,"
            "        from_parent_paths INTEGER,"
            "        pct REAL"
            "    );"
            "    CREATE TABLE files ("
            "        id,"
            "        name TEXT"
            "    );"
            "    CREATE TABLE mainrows ("
            "        id INTEGER PRIMARY KEY,"
            "        symbol_id INTEGER CONSTRAINT symbol_id_exists REFERENCES symbols(id),"
            "        self_count INTEGER,"
            "        cumulative_count INTEGER,"
            "        kids INTEGER,"
            "        self_calls INTEGER,"
            "        total_calls INTEGER,"
            "        self_paths INTEGER,"
            "        total_paths INTEGER,"
            "        pct REAL"
            "    );"
            "    CREATE TABLE parents ("
            "        self_id INTEGER CONSTRAINT self_exists REFERENCES mainrows(id),"
            "        child_id INTEGER CONSTRAINT child_exists REFERENCES mainrows(id),"
            "        to_child_count INTEGER,"
            "        to_child_calls INTEGER,"
            "        to_child_paths INTEGER,"
            "        pct REAL"
            "    );"
            "    CREATE TABLE summary ("
            "        counter TEXT,"
            "        total_count INTEGER,"
            "        total_freq INTEGER,"
            "        tick_period REAL"
            "    );"
            "    CREATE TABLE symbols ("
            "        id,"
            "        name TEXT,"
            "        filename_id INTEGER CONSTRAINT file_id_exists REFERENCES files(id)"
            "    );"
            "    CREATE UNIQUE INDEX fileIndex ON files (id);"
            "    CREATE INDEX selfCountIndex ON mainrows(self_count);"
            "    CREATE UNIQUE INDEX symbolsIndex ON symbols (id);"
            "    CREATE INDEX totalCountIndex ON mainrows(cumulative_count);"
         << std::endl;

  std::string sql_statement("");

  root.files(sql_statement);
  root.symbols(sql_statement);
  root.mainrows_cumulative(sql_statement);
  root.summary(sql_statement);
  VIterator<Folder*> subsystems = root.CreateIterator();
  size_t ii = 0;
  for (subsystems.First(); !subsystems.IsDone(); subsystems.Next(), ++ii) {
    subsystems.CurrentItem()->mainrows(sql_statement);
    subsystems.CurrentItem()->parents(sql_statement);
    subsystems.CurrentItem()->children(sql_statement);
  }
  stream << sql_statement << std::endl;
}

///
/// do the stats here and produce output;
///
/// mode is coded in DQMStoreStats::statMode enum
/// (select subsets of ME, e.g. those with getLumiFlag() == true)
///
int DQMStoreStats::calcstats(int mode = DQMStoreStats::considerAllME) {
  ////---- initialise Event and LS counters
  nbinsglobal_ = 0;
  nbinssubsys_ = 0;
  maxbinsglobal_ = 0;
  maxbinssubsys_ = 0;
  std::string path = "";
  std::string subsystemname = "";
  std::string subfoldername = "";
  size_t subsysStringEnd = 0, subfolderStringBegin = 0, subfolderStringEnd = 0;

  std::vector<MonitorElement*> melist;
  melist = dbe_->getAllContents(pathnamematch_);

  Folder dbeFolder("root");
  DQMStoreStatsTopLevel dqmStoreStatsTopLevel;

  // loop all ME
  for (auto& it : melist) {
    // consider only ME with getLumiFlag() == true ?
    if (mode == DQMStoreStats::considerOnlyLumiProductME && !(it->getLumiFlag()))
      continue;

    // figure out subsystem/subfolder names
    const std::string& path = it->getPathname();

    subfolderStringBegin = 0;
    Folder* curr = &dbeFolder;
    while (true) {
      subfolderStringEnd = path.find('/', subfolderStringBegin);
      if (std::string::npos == subfolderStringEnd) {
        curr = curr->cd(path.substr(subfolderStringBegin, path.size() - subfolderStringBegin));
        break;
      }
      curr = curr->cd(path.substr(subfolderStringBegin, subfolderStringEnd - subfolderStringBegin));
      subfolderStringBegin = ++subfolderStringEnd < path.size() ? subfolderStringEnd : path.size();
    }

    // protection against ghost ME with empty paths
    if (path.empty())
      continue;

    subsysStringEnd = path.find('/', 0);
    if (std::string::npos == subsysStringEnd)
      subsysStringEnd = path.size();  // no subfolder

    // new subsystem?
    if (path.substr(0, subsysStringEnd) != subsystemname) {
      DQMStoreStatsSubsystem aSubsystem;
      subsystemname = path.substr(0, subsysStringEnd);
      aSubsystem.subsystemName_ = subsystemname;
      dqmStoreStatsTopLevel.push_back(aSubsystem);
      subfoldername = "";
    }

    // get subfolder name (if there is one..)
    if (path.size() == subsysStringEnd) {
      // no subfolders in subsystem, make dummy
      DQMStoreStatsSubfolder aSubfolder;
      aSubfolder.subfolderName_ = subsystemname;  // <-- for tagging this case
      dqmStoreStatsTopLevel.back().push_back(aSubfolder);
    }

    else {
      // there is a subfolder, get its name
      subfolderStringEnd = path.find('/', subsysStringEnd + 1);
      if (std::string::npos == subfolderStringEnd)
        subfolderStringEnd = path.size();

      // new subfolder?
      if (path.substr(subsysStringEnd + 1, subfolderStringEnd - subsysStringEnd - 1) != subfoldername) {
        subfoldername = path.substr(subsysStringEnd + 1, subfolderStringEnd - subsysStringEnd - 1);
        DQMStoreStatsSubfolder aSubfolder;
        aSubfolder.subfolderName_ = subfoldername;
        dqmStoreStatsTopLevel.back().push_back(aSubfolder);
      }
    }

    // shortcut
    DQMStoreStatsSubfolder& currentSubfolder = dqmStoreStatsTopLevel.back().back();

    switch (it->kind()) {
        // one-dim ME
      case MonitorElement::Kind::TH1F:
        currentSubfolder.AddBinsF(it->getNbinsX(), getEmptyMetric(it->getTH1F()->GetArray(), it->getTH1F()->fN, 0, 0));
        curr->update(it->getNbinsX(),
                     getEmptyMetric(it->getTH1F()->GetArray(), it->getTH1F()->fN, 0, 0),
                     it->getNbinsX() * sizeof(float));
        break;
      case MonitorElement::Kind::TH1S:
        currentSubfolder.AddBinsS(it->getNbinsX(), getEmptyMetric(it->getTH1S()->GetArray(), it->getTH1S()->fN, 0, 0));
        curr->update(it->getNbinsX(),
                     getEmptyMetric(it->getTH1S()->GetArray(), it->getTH1S()->fN, 0, 0),
                     it->getNbinsX() * sizeof(short));
        break;
      case MonitorElement::Kind::TH1D:
        currentSubfolder.AddBinsD(it->getNbinsX(), getEmptyMetric(it->getTH1D()->GetArray(), it->getTH1D()->fN, 0, 0));
        curr->update(it->getNbinsX(),
                     getEmptyMetric(it->getTH1D()->GetArray(), it->getTH1D()->fN, 0, 0),
                     it->getNbinsX() * sizeof(double));
        break;
      case MonitorElement::Kind::TPROFILE:
        currentSubfolder.AddBinsD(it->getNbinsX(),
                                  getEmptyMetric(it->getTProfile()->GetArray(), it->getTProfile()->fN, 0, 0));
        curr->update(it->getNbinsX(),
                     getEmptyMetric(it->getTProfile()->GetArray(), it->getTProfile()->fN, 0, 0),
                     it->getNbinsX() * sizeof(double));
        break;

        // two-dim ME
      case MonitorElement::Kind::TH2F:
        currentSubfolder.AddBinsF(
            it->getNbinsX() * it->getNbinsY(),
            getEmptyMetric(it->getTH2F()->GetArray(), it->getNbinsX() + 2, it->getNbinsY() + 2, 0));
        curr->update(it->getNbinsX() * it->getNbinsY(),
                     getEmptyMetric(it->getTH2F()->GetArray(), it->getNbinsX() + 2, it->getNbinsY() + 2, 0),
                     it->getNbinsX() * it->getNbinsY() * sizeof(float));
        break;
      case MonitorElement::Kind::TH2S:
        currentSubfolder.AddBinsS(
            it->getNbinsX() * it->getNbinsY(),
            getEmptyMetric(it->getTH2S()->GetArray(), it->getNbinsX() + 2, it->getNbinsY() + 2, 0));
        curr->update(it->getNbinsX() * it->getNbinsY(),
                     getEmptyMetric(it->getTH2S()->GetArray(), it->getNbinsX() + 2, it->getNbinsY() + 2, 0),
                     it->getNbinsX() * it->getNbinsY() * sizeof(short));
        break;
      case MonitorElement::Kind::TH2D:
        currentSubfolder.AddBinsD(
            it->getNbinsX() * it->getNbinsY(),
            getEmptyMetric(it->getTH2D()->GetArray(), it->getNbinsX() + 2, it->getNbinsY() + 2, 0));
        curr->update(it->getNbinsX() * it->getNbinsY(),
                     getEmptyMetric(it->getTH2D()->GetArray(), it->getNbinsX() + 2, it->getNbinsY() + 2, 0),
                     it->getNbinsX() * it->getNbinsY() * sizeof(double));
        break;
      case MonitorElement::Kind::TPROFILE2D:
        currentSubfolder.AddBinsD(
            it->getNbinsX() * it->getNbinsY(),
            getEmptyMetric(it->getTProfile2D()->GetArray(), it->getNbinsX() + 2, it->getNbinsY() + 2, 0));
        curr->update(it->getNbinsX() * it->getNbinsY(),
                     getEmptyMetric(it->getTProfile2D()->GetArray(), it->getNbinsX() + 2, it->getNbinsY() + 2, 0),
                     it->getNbinsX() * it->getNbinsY() * sizeof(double));
        break;

        // three-dim ME
      case MonitorElement::Kind::TH3F:
        currentSubfolder.AddBinsF(
            it->getNbinsX() * it->getNbinsY() * it->getNbinsZ(),
            getEmptyMetric(it->getTH3F()->GetArray(), it->getNbinsX() + 2, it->getNbinsY() + 2, it->getNbinsZ() + 2));
        curr->update(
            it->getNbinsX() * it->getNbinsY() * it->getNbinsZ(),
            getEmptyMetric(it->getTH3F()->GetArray(), it->getNbinsX() + 2, it->getNbinsY() + 2, it->getNbinsZ() + 2),
            it->getNbinsX() * it->getNbinsY() * it->getNbinsZ() * sizeof(float));
        break;

      default: {
      }
        // here we have a DQM_KIND_INVALID, DQM_KIND_INT, DQM_KIND_REAL or DQM_KIND_STRING
        // which we don't care much about. Alternatively:

        //   std::cerr << "[DQMStoreStats::calcstats] ** WARNING: monitor element of kind: "
        //               << (*it)->kind() << ", name: \"" << (*it)->getName() << "\"\n"
        //               << "  in path: \"" << path << "\" not considered." << std::endl;
    }
  }

  if (mode == DQMStoreStats::considerAllME)
    calcIgProfDump(dbeFolder);

  // OUTPUT

  std::cout << endl;
  std::cout << "==========================================================================================="
            << std::endl;
  std::cout << "[DQMStoreStats::calcstats] -- Dumping stats results ";
  if (mode == DQMStoreStats::considerAllME)
    std::cout << "FOR ALL ME" << std::endl;
  else if (mode == DQMStoreStats::considerOnlyLumiProductME)
    std::cout << "FOR LUMI PRODUCTS ONLY" << std::endl;
  std::cout << "==========================================================================================="
            << std::endl;
  std::cout << endl;

  std::cout << "------------------------------------------------------------------------------------------"
            << std::endl;
  std::cout << "Configuration:" << std::endl;
  std::cout << "------------------------------------------------------------------------------------------"
            << std::endl;
  std::cout << " > running ";
  if (runonendrun_)
    std::cout << "on run end." << std::endl;
  if (runonendlumi_)
    std::cout << "on lumi end." << std::endl;
  if (runonendjob_)
    std::cout << "on job end." << std::endl;
  if (runineventloop_)
    std::cout << "in event loop." << std::endl;
  std::cout << " > pathNameMatch = \"" << pathnamematch_ << "\"" << std::endl;
  std::cout << std::endl;

  // dump folder structure
  std::cout << "------------------------------------------------------------------------------------------"
            << std::endl;
  std::cout << "Top level folder tree:" << std::endl;
  std::cout << "------------------------------------------------------------------------------------------"
            << std::endl;
  for (auto it0 = dqmStoreStatsTopLevel.begin(); it0 < dqmStoreStatsTopLevel.end(); ++it0) {
    std::cout << it0->subsystemName_ << " (subsystem)" << std::endl;

    for (auto it1 = it0->begin(); it1 < it0->end(); ++it1) {
      std::cout << "  |--> " << it1->subfolderName_ << " (subfolder)" << std::endl;
    }
  }

  // dump mem/bin table

  unsigned int overallNHistograms = 0, overallNBins = 0, overallNEmptyBins = 0, overallNBytes = 0;

  std::cout << std::endl;
  std::cout << "------------------------------------------------------------------------------------------"
            << std::endl;
  std::cout << "Detailed ressource usage information ";
  if (mode == DQMStoreStats::considerAllME)
    std::cout << "FOR ALL ME" << std::endl;
  else if (mode == DQMStoreStats::considerOnlyLumiProductME)
    std::cout << "FOR LUMI PRODUCTS ONLY" << std::endl;
  std::cout << "------------------------------------------------------------------------------------------"
            << std::endl;
  std::cout << "subsystem/folder                          histograms       bins        Empty bins     Empty/Total      "
               "bins per       MB         kB per"
            << std::endl;
  std::cout << "                                           (total)        (total)        (total)                      "
               "histogram     (total)    histogram  "
            << std::endl;
  std::cout << "------------------------------------------------------------------------------------------"
            << std::endl;
  for (auto it0 = dqmStoreStatsTopLevel.begin(); it0 < dqmStoreStatsTopLevel.end(); ++it0) {
    std::cout << it0->subsystemName_ << std::endl;

    unsigned int nHistograms = 0, nBins = 0, nEmptyBins = 0, nBytes = 0;

    for (auto it1 = it0->begin(); it1 < it0->end(); ++it1) {
      // fixed-size working copy
      std::string thisSubfolderName(it1->subfolderName_);
      if (thisSubfolderName.size() > 30) {
        thisSubfolderName.resize(30);
        thisSubfolderName.replace(thisSubfolderName.size() - 3, 3, 3, '.');
      }

      std::cout << " -> " << std::setw(30) << std::left << thisSubfolderName;
      std::cout << std::setw(14) << std::right << it1->totalHistos_;
      std::cout << std::setw(14) << std::right << it1->totalBins_;
      std::cout << std::setw(14) << std::right << it1->totalEmptyBins_;
      std::cout << std::setw(14) << std::right << std::setprecision(3)
                << (float)it1->totalEmptyBins_ / (float)it1->totalBins_;

      // bins/histogram, need to catch nan if histos=0
      if (it1->totalHistos_) {
        std::cout << std::setw(14) << std::right << std::setprecision(3) << it1->totalBins_ / float(it1->totalHistos_);
      } else
        std::cout << std::setw(14) << std::right << "-";

      std::cout << std::setw(14) << std::right << std::setprecision(3) << it1->totalMemory_ / 1024. / 1024.;

      // mem/histogram, need to catch nan if histos=0
      if (it1->totalHistos_) {
        std::cout << std::setw(14) << std::right << std::setprecision(3)
                  << it1->totalMemory_ / 1024. / it1->totalHistos_;
      } else
        std::cout << std::setw(14) << std::right << "-";

      std::cout << std::endl;

      // collect totals
      nHistograms += it1->totalHistos_;
      nBins += it1->totalBins_;
      nEmptyBins += it1->totalEmptyBins_;
      nBytes += it1->totalMemory_;
    }

    overallNHistograms += nHistograms;
    overallNBins += nBins;
    overallNEmptyBins += nEmptyBins;
    overallNBytes += nBytes;

    // display totals
    std::cout << "    " << std::setw(30) << std::left << "SUBSYSTEM TOTAL";
    std::cout << std::setw(14) << std::right << nHistograms;
    std::cout << std::setw(14) << std::right << nBins;
    std::cout << std::setw(14) << std::right << nEmptyBins;
    std::cout << std::setw(14) << std::right << (float)nEmptyBins / (float)nBins;
    std::cout << std::setw(14) << std::right << std::setprecision(3) << nBins / float(nHistograms);
    std::cout << std::setw(14) << std::right << std::setprecision(3) << nBytes / 1024. / 1000.;
    std::cout << std::setw(14) << std::right << std::setprecision(3) << nBytes / 1024. / nHistograms;
    std::cout << std::endl;

    std::cout << ".........................................................................................."
              << std::endl;
  }

  // dump total
  std::cout << std::endl;
  std::cout << "------------------------------------------------------------------------------------------"
            << std::endl;
  std::cout << "Grand total ";
  if (mode == DQMStoreStats::considerAllME)
    std::cout << "FOR ALL ME:" << std::endl;
  else if (mode == DQMStoreStats::considerOnlyLumiProductME)
    std::cout << "FOR LUMI PRODUCTS ONLY:" << std::endl;
  std::cout << "------------------------------------------------------------------------------------------"
            << std::endl;
  std::cout << "Number of subsystems: " << dqmStoreStatsTopLevel.size() << std::endl;
  std::cout << "Total number of histograms: " << overallNHistograms << " with: " << overallNBins << " bins alltogether"
            << std::endl;
  std::cout << "Total memory occupied by histograms (excl. overhead): " << overallNBytes / 1024. / 1000. << " MB"
            << std::endl;

  std::cout << endl;
  std::cout << "==========================================================================================="
            << std::endl;
  std::cout << "[DQMStoreStats::calcstats] -- End of output ";
  if (mode == DQMStoreStats::considerAllME)
    std::cout << "FOR ALL ME." << std::endl;
  else if (mode == DQMStoreStats::considerOnlyLumiProductME)
    std::cout << "FOR LUMI PRODUCTS ONLY." << std::endl;
  std::cout << "==========================================================================================="
            << std::endl;
  std::cout << endl;

  // Put together a simplified version of the complete dump that is
  // sent to std::cout. Just dump the very basic information,
  // i.e. summary for each folder, both for run and LS products.
  if (dumpToFWJR_) {
    edm::Service<edm::JobReport> jr;
    // Do not even try if the FWJR service is not available.
    if (!jr.isAvailable())
      return 0;
    // Prepare appropriate map to store FWJR output.
    std::map<std::string, std::string> jrInfo;
    unsigned int overallNHistograms = 0, overallNBins = 0, overallNBytes = 0;

    jrInfo["Source"] = "DQMServices/Components";
    jrInfo["FileClass"] = "DQMStoreStats";
    if (runonendrun_)
      jrInfo["DumpType"] = "EndRun";
    if (runonendlumi_)
      jrInfo["DumpType"] = "EndLumi";
    if (runonendjob_)
      jrInfo["DumpType"] = "EndJob";
    if (runineventloop_)
      jrInfo["DumpType"] = "EventLoop";
    if (mode == DQMStoreStats::considerAllME)
      jrInfo["Type"] = "RunProduct";
    else if (mode == DQMStoreStats::considerOnlyLumiProductME)
      jrInfo["Type"] = "LumiProduct";

    jrInfo["pathNameMatch"] = pathnamematch_;

    for (auto it0 = dqmStoreStatsTopLevel.begin(); it0 < dqmStoreStatsTopLevel.end(); ++it0) {
      unsigned int nHistograms = 0, nBins = 0, nEmptyBins = 0, nBytes = 0;
      for (auto it1 = it0->begin(); it1 < it0->end(); ++it1) {
        // collect totals
        nHistograms += it1->totalHistos_;
        nBins += it1->totalBins_;
        nEmptyBins += it1->totalEmptyBins_;
        nBytes += it1->totalMemory_;
      }
      overallNHistograms += nHistograms;
      overallNBins += nBins;
      overallNBytes += nBytes;
      std::stringstream iss("");
      iss << nHistograms;
      jrInfo[it0->subsystemName_ + std::string("_h")] = iss.str();
      iss.str("");
      iss << nBins;
      jrInfo[it0->subsystemName_ + std::string("_b")] = iss.str();
      iss.str("");
      iss << nEmptyBins;
      jrInfo[it0->subsystemName_ + std::string("_be")] = iss.str();
      iss.str("");
      iss << ((float)nEmptyBins / (float)nBins);
      jrInfo[it0->subsystemName_ + std::string("_fbe")] = iss.str();
      iss.str("");
      iss << ((float)nBins / (float)nHistograms);
      jrInfo[it0->subsystemName_ + std::string("_b_h")] = iss.str();
      iss.str("");
      iss << nBytes / 1024. / 1024.;
      jrInfo[it0->subsystemName_ + std::string("_MB")] = iss.str();
      iss.str("");
      iss << nBytes / 1024. / nHistograms;
      jrInfo[it0->subsystemName_ + std::string("_Kb_h")] = iss.str();
    }
    jr->reportAnalysisFile("DQMStatsReport", jrInfo);
  }

  return 0;
}

///
///
///
void DQMStoreStats::dumpMemoryProfile() {
  std::cout << std::endl;
  std::cout << "------------------------------------------------------------------------------------------"
            << std::endl;
  std::cout << "Memory profile:" << std::endl;
  std::cout << "------------------------------------------------------------------------------------------"
            << std::endl;

  // determine virtual memory maximum
  std::pair<time_t, unsigned int> maxItem(0, 0);
  for (auto it = memoryHistoryVector_.begin(); it < memoryHistoryVector_.end(); ++it) {
    if (it->second > maxItem.second) {
      maxItem = *it;
    }
  }

  std::stringstream rootOutputFileName;
  rootOutputFileName << "dqmStoreStats_memProfile_" << getpid() << ".root";

  // dump memory history to root file
  if (dumpMemHistory_ && isOpenProcFileSuccessful_) {
    TFile outputFile(rootOutputFileName.str().c_str(), "RECREATE");

    int aTime;
    float aMb;

    TTree memHistoryTree("dqmstorestats_memhistory", "memory history");
    memHistoryTree.Branch("seconds", &aTime, "seconds/I");
    memHistoryTree.Branch("megabytes", &aMb, "megabytes/F");
    for (auto it = memoryHistoryVector_.begin(); it < memoryHistoryVector_.end(); ++it) {
      aTime = it->first - startingTime_;
      aMb = it->second / 1000.;
      memHistoryTree.Fill();
    }

    outputFile.Write();
    outputFile.Close();
  }

  std::cout << "Approx. maximum total virtual memory size of job: ";
  if (isOpenProcFileSuccessful_ && !memoryHistoryVector_.empty()) {
    std::cout << maxItem.second / 1000. << " MB (reached " << maxItem.first - startingTime_
              << " sec. after constructor called)," << std::endl;
    std::cout << " memory history written to: " << rootOutputFileName.str() << " (" << memoryHistoryVector_.size()
              << " samples)" << std::endl;
  } else {
    std::cout << "(could not be determined)" << std::endl;
  }

  std::cout << std::endl << std::endl;
}

///
///
///
void DQMStoreStats::print() {
  // subsystem info printout
  std::cout << " ---------- " << subsystem_ << " ---------- " << std::endl;
  std::cout << "  " << subfolder_ << ": ";
  std::cout << nmesubsys_ << " histograms with " << nbinssubsys_ << " bins. ";
  if (nmesubsys_ > 0)
    std::cout << nbinssubsys_ / nmesubsys_ << " bins/histogram ";
  std::cout << std::endl;
  std::cout << "  Largest histogram: " << maxbinsmesubsys_ << " with " << maxbinssubsys_ << " bins." << std::endl;
}

///
/// read virtual memory size from /proc/<pid>/status file
///
std::pair<unsigned int, unsigned int> DQMStoreStats::readMemoryEntry() const {
  // see if initial test reading was successful
  if (isOpenProcFileSuccessful_) {
    std::ifstream procFile(procFileName_.str().c_str(), ios::in);

    std::string readBuffer("");
    unsigned int memSize = 0;

    // scan procfile
    while (!procFile.eof()) {
      procFile >> readBuffer;
      if (std::string("VmSize:") == readBuffer) {
        procFile >> memSize;
        break;
      }
    }

    procFile.close();
    return std::pair<time_t, unsigned int>(time(nullptr), memSize);
  }

  return std::pair<time_t, unsigned int>(0, 0);
}

//==================================================================//
//========================= beginJob ===============================//
//==================================================================//
void DQMStoreStats::beginJob() {
  ////---- get DQM store interface
  dbe_ = Service<DQMStore>().operator->();

  // access the proc/ folder for memory information
  procFileName_ << "/proc/" << getpid() << "/status";

  // open for a test
  std::ifstream procFile(procFileName_.str().c_str(), ios::in);

  if (procFile.good()) {
    isOpenProcFileSuccessful_ = true;
  } else {
    std::cerr << " [DQMStoreStats::beginJob] ** WARNING: could not open file: " << procFileName_.str() << std::endl;
    std::cerr << "  Total memory profile will not be available." << std::endl;
    isOpenProcFileSuccessful_ = false;
  }

  procFile.close();
}

//==================================================================//
//========================= beginRun ===============================//
//==================================================================//
void DQMStoreStats::beginRun(const edm::Run& r, const EventSetup& context) {}

//==================================================================//
//==================== analyse (takes each event) ==================//
//==================================================================//
void DQMStoreStats::analyze(const Event& iEvent, const EventSetup& iSetup) {
  //now read virtual memory size from proc folder
  memoryHistoryVector_.emplace_back(readMemoryEntry());

  if (runineventloop_) {
    calcstats(DQMStoreStats::considerAllME);
    calcstats(DQMStoreStats::considerOnlyLumiProductME);
    dumpMemoryProfile();
  }
}

//==================================================================//
//========================= endLuminosityBlock =====================//
//==================================================================//
void DQMStoreStats::endLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& context) {
  if (runonendlumi_) {
    calcstats(DQMStoreStats::considerAllME);
    calcstats(DQMStoreStats::considerOnlyLumiProductME);
    dumpMemoryProfile();
  }
}

//==================================================================//
//============================= endRun =============================//
//==================================================================//
void DQMStoreStats::endRun(const Run& r, const EventSetup& context) {
  if (runonendrun_) {
    calcstats(DQMStoreStats::considerAllME);
    calcstats(DQMStoreStats::considerOnlyLumiProductME);
    dumpMemoryProfile();
  }
}

//==================================================================//
//============================= endJob =============================//
//==================================================================//
void DQMStoreStats::endJob() {
  if (runonendjob_) {
    calcstats(DQMStoreStats::considerAllME);
    calcstats(DQMStoreStats::considerOnlyLumiProductME);
    dumpMemoryProfile();
  }
}
