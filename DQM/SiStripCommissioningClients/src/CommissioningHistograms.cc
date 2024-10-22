#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"
#include "CondFormats/SiStripObjects/interface/CommissioningAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include "DataFormats/SiStripCommon/interface/SiStripDetKey.h"
#include "DQM/SiStripCommissioningSummary/interface/SummaryGenerator.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iomanip>

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
CommissioningHistograms::CommissioningHistograms(const edm::ParameterSet& pset,
                                                 DQMStore* bei,
                                                 const sistrip::RunType& task)
    : factory_(nullptr),
      task_(task),
      bei_(bei),
      data_(),
      histos_(),
      pset_(pset),
      mask_(pset.existsAs<bool>("vetoModules") ? pset.getParameter<bool>("vetoModules") : true),
      fedMaskVector_(pset.existsAs<std::vector<uint32_t> >("fedMaskVector")
                         ? pset.getParameter<std::vector<uint32_t> >("fedMaskVector")
                         : std::vector<uint32_t>()),
      fecMaskVector_(pset.existsAs<std::vector<uint32_t> >("fecMaskVector")
                         ? pset.getParameter<std::vector<uint32_t> >("fecMaskVector")
                         : std::vector<uint32_t>()),
      ringVector_(pset.existsAs<std::vector<uint32_t> >("ringVector")
                      ? pset.getParameter<std::vector<uint32_t> >("ringVector")
                      : std::vector<uint32_t>()),
      ccuVector_(pset.existsAs<std::vector<uint32_t> >("ccuVector")
                     ? pset.getParameter<std::vector<uint32_t> >("ccuVector")
                     : std::vector<uint32_t>()),
      i2cChanVector_(pset.existsAs<std::vector<uint32_t> >("i2cChanVector")
                         ? pset.getParameter<std::vector<uint32_t> >("i2cChanVector")
                         : std::vector<uint32_t>()),
      lldChanVector_(pset.existsAs<std::vector<uint32_t> >("lldChanVector")
                         ? pset.getParameter<std::vector<uint32_t> >("lldChanVector")
                         : std::vector<uint32_t>()),
      dataWithMask_(),
      dataWithMaskCached_(false) {
  LogTrace(mlDqmClient_) << "[" << __PRETTY_FUNCTION__ << "]"
                         << " Constructing object...";

  // DQMStore
  if (!bei_) {
    edm::LogError(mlDqmClient_) << "[CommissioningHistograms::" << __func__ << "]"
                                << " NULL pointer to DQMStore!";
  }

  clearHistosMap();
}

// -----------------------------------------------------------------------------
/** */
CommissioningHistograms::CommissioningHistograms()
    : factory_(nullptr),
      task_(sistrip::UNDEFINED_RUN_TYPE),
      bei_(nullptr),
      data_(),
      histos_(),
      mask_(true),
      dataWithMask_(),
      dataWithMaskCached_(false) {
  LogTrace(mlDqmClient_) << "[" << __PRETTY_FUNCTION__ << "]"
                         << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
CommissioningHistograms::~CommissioningHistograms() {
  LogTrace(mlDqmClient_) << "[" << __PRETTY_FUNCTION__ << "]"
                         << " Destructing object...";
  clearHistosMap();
  //@@ do not delete BEI ptrs!
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::Histo::print(std::stringstream& ss) const {
  ss << " [Histo::" << __func__ << "]" << std::endl
     << " Histogram title   : " << title_ << std::endl
     << " MonitorElement*   : 0x" << std::hex << std::setw(8) << std::setfill('0') << me_ << std::endl
     << std::dec << " CollateME*        : 0x" << std::hex << std::setw(8) << std::setfill('0') << cme_ << std::endl
     << std::dec;
}

// -----------------------------------------------------------------------------
//
uint32_t CommissioningHistograms::runNumber(DQMStore* const bei, const std::vector<std::string>& contents) {
  // Check if histograms present
  if (contents.empty()) {
    edm::LogError(mlDqmClient_) << "[CommissioningHistograms::" << __func__ << "]"
                                << " Found no histograms!";
    return 0;
  }

  // Iterate through added contents
  std::vector<std::string>::const_iterator istr = contents.begin();
  while (istr != contents.end()) {
    // Extract source directory path
    std::string source_dir = istr->substr(0, istr->find(":"));

    // Generate corresponding client path (removing trailing "/")
    SiStripFecKey path(source_dir);
    std::string client_dir = path.path();
    std::string slash = client_dir.substr(client_dir.size() - 1, 1);
    if (slash == sistrip::dir_) {
      client_dir = client_dir.substr(0, client_dir.size() - 1);
    }
    client_dir = std::string(sistrip::collate_) + sistrip::dir_ + client_dir;

    // Iterate though MonitorElements from source directory
    std::vector<MonitorElement*> me_list = bei->getContents(source_dir);
    std::vector<MonitorElement*>::iterator ime = me_list.begin();
    for (; ime != me_list.end(); ime++) {
      if (!(*ime)) {
        edm::LogError(mlDqmClient_) << "[CommissioningHistograms::" << __func__ << "]"
                                    << " NULL pointer to MonitorElement!";
        continue;
      }

      // Search for run type in string
      std::string title = (*ime)->getName();
      std::string::size_type pos = title.find(sistrip::runNumber_);

      // Extract run number from string
      if (pos != std::string::npos) {
        std::string value = title.substr(pos + sizeof(sistrip::runNumber_), std::string::npos);
        if (!value.empty()) {
          LogTrace(mlDqmClient_) << "[CommissioningHistograms::" << __func__ << "]"
                                 << " Found string \"" << title.substr(pos, std::string::npos) << "\" with value \""
                                 << value << "\"";
          if (!(bei->get(client_dir + "/" + title.substr(pos, std::string::npos)))) {
            bei->setCurrentFolder(client_dir);
            bei->bookString(title.substr(pos, std::string::npos), value);
            LogTrace(mlDqmClient_) << "[CommissioningHistograms::" << __func__ << "]"
                                   << " Booked string \"" << title.substr(pos, std::string::npos)
                                   << "\" in directory \"" << client_dir << "\"";
          }
          uint32_t run;
          std::stringstream ss;
          ss << value;
          ss >> std::dec >> run;
          return run;
        }
      }
    }

    istr++;
  }
  return 0;
}

// -----------------------------------------------------------------------------
/** Extract run type string from "added contents". */
sistrip::RunType CommissioningHistograms::runType(DQMStore* const bei, const std::vector<std::string>& contents) {
  // Check if histograms present
  if (contents.empty()) {
    edm::LogError(mlDqmClient_) << "[CommissioningHistograms::" << __func__ << "]"
                                << " Found no histograms!";
    return sistrip::UNKNOWN_RUN_TYPE;
  }

  // Iterate through added contents
  std::vector<std::string>::const_iterator istr = contents.begin();
  while (istr != contents.end()) {
    // Extract source directory path
    std::string source_dir = istr->substr(0, istr->find(":"));

    // Generate corresponding client path (removing trailing "/")
    SiStripFecKey path(source_dir);
    std::string client_dir = path.path();
    std::string slash = client_dir.substr(client_dir.size() - 1, 1);
    if (slash == sistrip::dir_) {
      client_dir = client_dir.substr(0, client_dir.size() - 1);
    }
    client_dir = std::string(sistrip::collate_) + sistrip::dir_ + client_dir;

    // Iterate though MonitorElements from source directory
    std::vector<MonitorElement*> me_list = bei->getContents(source_dir);

    if (me_list.empty()) {
      edm::LogError(mlDqmClient_) << "[CommissioningHistograms::" << __func__ << "]"
                                  << " No MonitorElements found in dir " << source_dir;
      return sistrip::UNKNOWN_RUN_TYPE;
    }

    std::vector<MonitorElement*>::iterator ime = me_list.begin();
    for (; ime != me_list.end(); ime++) {
      if (!(*ime)) {
        edm::LogError(mlDqmClient_) << "[CommissioningHistograms::" << __func__ << "]"
                                    << " NULL pointer to MonitorElement!";
        continue;
      }

      // Search for run type in string
      std::string title = (*ime)->getName();
      std::string::size_type pos = title.find(sistrip::taskId_);

      // Extract commissioning task from string
      if (pos != std::string::npos) {
        std::string value = title.substr(pos + sizeof(sistrip::taskId_), std::string::npos);
        if (!value.empty()) {
          LogTrace(mlDqmClient_) << "[CommissioningHistograms::" << __func__ << "]"
                                 << " Found string \"" << title.substr(pos, std::string::npos) << "\" with value \""
                                 << value << "\"";
          if (!(bei->get(client_dir + sistrip::dir_ + title.substr(pos, std::string::npos)))) {
            bei->setCurrentFolder(client_dir);
            bei->bookString(title.substr(pos, std::string::npos), value);
            LogTrace(mlDqmClient_) << "[CommissioningHistograms::" << __func__ << "]"
                                   << " Booked string \"" << title.substr(pos, std::string::npos)
                                   << "\" in directory \"" << client_dir << "\"";
          }
          return SiStripEnumsAndStrings::runType(value);
        }
      }
    }

    istr++;
  }

  edm::LogError(mlDqmClient_) << "[CommissioningHistograms::" << __func__ << "]"
                              << " Unable to extract RunType!";
  return sistrip::UNKNOWN_RUN_TYPE;
}

// -----------------------------------------------------------------------------
//
void CommissioningHistograms::copyCustomInformation(DQMStore* const bei, const std::vector<std::string>& contents) {
  // Check if histograms present
  if (contents.empty()) {
    edm::LogWarning(mlDqmClient_) << "[CommissioningHistograms::" << __func__ << "]"
                                  << " Found no histograms!";
    return;
  }

  // Iterate through added contents
  std::vector<std::string>::const_iterator istr = contents.begin();
  while (istr != contents.end()) {
    // Extract source directory path
    std::string source_dir = istr->substr(0, istr->find(":"));

    // Generate corresponding client path (removing trailing "/")
    SiStripFecKey path(source_dir);
    std::string client_dir = path.path();
    std::string slash = client_dir.substr(client_dir.size() - 1, 1);
    if (slash == sistrip::dir_) {
      client_dir = client_dir.substr(0, client_dir.size() - 1);
    }

    // Iterate though MonitorElements from source directory
    std::vector<MonitorElement*> me_list = bei->getContents(source_dir);
    std::vector<MonitorElement*>::iterator ime = me_list.begin();
    for (; ime != me_list.end(); ime++) {
      if (!(*ime)) {
        edm::LogWarning(mlDqmClient_) << "[CommissioningHistograms::" << __func__ << "]"
                                      << " NULL pointer to MonitorElement!";
        continue;
      }
      // Search for calchan, isha or vfs
      if ((*ime)->kind() == MonitorElement::Kind::INT) {
        std::string title = (*ime)->getName();
        std::string::size_type pos = title.find("calchan");
        if (pos == std::string::npos)
          pos = title.find("isha");
        if (pos == std::string::npos)
          pos = title.find("vfs");
        if (pos != std::string::npos) {
          int value = (*ime)->getIntValue();
          if (value >= 0) {
            edm::LogVerbatim(mlDqmClient_)
                << "[CommissioningHistograms::" << __func__ << "]"
                << " Found \"" << title.substr(pos, std::string::npos) << "\" with value \"" << value << "\"";
            if (!(bei->get(client_dir + "/" + title.substr(pos, std::string::npos)))) {
              bei->setCurrentFolder(client_dir);
              bei->bookInt(title.substr(pos, std::string::npos))->Fill(value);
              edm::LogVerbatim(mlDqmClient_)
                  << "[CommissioningHistograms::" << __func__ << "]"
                  << " Booked \"" << title.substr(pos, std::string::npos) << "\" in directory \"" << client_dir << "\"";
            }
          }
        }
      }
    }
    istr++;
  }
}

// -----------------------------------------------------------------------------

/** */
void CommissioningHistograms::extractHistograms(const std::vector<std::string>& contents) {
  LogTrace(mlDqmClient_) << "[CommissioningHistograms::" << __func__ << "]"
                         << " Extracting available histograms...";

  // Check pointer
  if (!bei_) {
    edm::LogError(mlDqmClient_) << "[CommissioningHistograms::" << __func__ << "]"
                                << " NULL pointer to DQMStore!";
    return;
  }

  // Check list of histograms
  if (contents.empty()) {
    edm::LogError(mlDqmClient_) << "[CommissioningHistograms::" << __func__ << "]"
                                << " Empty contents vector!";
    return;
  }

  // Iterate through list of histograms
  std::vector<std::string>::const_iterator idir;
  for (idir = contents.begin(); idir != contents.end(); idir++) {
    // Ignore "DQM source" directories if looking in client file
    if (idir->find(sistrip::collate_) == std::string::npos) {
      continue;
    }

    // Extract source directory path
    std::string source_dir = idir->substr(0, idir->find(":"));

    // Extract view and create key
    sistrip::View view = SiStripEnumsAndStrings::view(source_dir);
    SiStripKey path;
    if (view == sistrip::CONTROL_VIEW) {
      path = SiStripFecKey(source_dir);
    } else if (view == sistrip::READOUT_VIEW) {
      path = SiStripFedKey(source_dir);
    } else if (view == sistrip::DETECTOR_VIEW) {
      path = SiStripDetKey(source_dir);
    } else {
      path = SiStripKey();
    }

    // Check path is valid
    if (path.granularity() == sistrip::UNKNOWN_GRAN || path.granularity() == sistrip::UNDEFINED_GRAN) {
      continue;
    }

    // Generate corresponding client path (removing trailing "/")
    std::string client_dir(sistrip::undefinedView_);
    if (view == sistrip::CONTROL_VIEW) {
      client_dir = SiStripFecKey(path.key()).path();
    } else if (view == sistrip::READOUT_VIEW) {
      client_dir = SiStripFedKey(path.key()).path();
    } else if (view == sistrip::DETECTOR_VIEW) {
      client_dir = SiStripDetKey(path.key()).path();
    } else {
      client_dir = SiStripKey(path.key()).path();
    }
    std::string slash = client_dir.substr(client_dir.size() - 1, 1);
    if (slash == sistrip::dir_) {
      client_dir = client_dir.substr(0, client_dir.size() - 1);
    }
    client_dir = std::string(sistrip::collate_) + sistrip::dir_ + client_dir;

    // Retrieve MonitorElements from source directory
    std::vector<MonitorElement*> me_list = bei_->getContents(source_dir);

    // Iterate though MonitorElements and create CMEs
    std::vector<MonitorElement*>::iterator ime = me_list.begin();
    for (; ime != me_list.end(); ime++) {
      // Retrieve histogram title
      SiStripHistoTitle title((*ime)->getName());

      // Check histogram type
      //if ( title.histoType() != sistrip::EXPERT_HISTO ) { continue; }

      // Check granularity
      uint16_t channel = sistrip::invalid_;
      if (title.granularity() == sistrip::APV) {
        channel = SiStripFecKey::lldChan(title.channel());
      } else if (title.granularity() == sistrip::UNKNOWN_GRAN || title.granularity() == sistrip::UNDEFINED_GRAN) {
        std::stringstream ss;
        ss << "[CommissioningHistograms::" << __func__ << "]"
           << " Unexpected granularity for histogram title: " << std::endl
           << title << " found in path " << std::endl
           << path;
        edm::LogError(mlDqmClient_) << ss.str();
      } else {
        channel = title.channel();
      }

      // Build key
      uint32_t key = sistrip::invalid32_;

      if (view == sistrip::CONTROL_VIEW) {
        // for all runs except cabling
        SiStripFecKey temp(path.key());
        key = SiStripFecKey(temp.fecCrate(), temp.fecSlot(), temp.fecRing(), temp.ccuAddr(), temp.ccuChan(), channel)
                  .key();
        mapping_[title.keyValue()] = key;

      } else if (view == sistrip::READOUT_VIEW) {
        // for cabling run
        key = SiStripFedKey(path.key()).key();
        uint32_t temp = SiStripFecKey(sistrip::invalid_,
                                      sistrip::invalid_,
                                      sistrip::invalid_,
                                      sistrip::invalid_,
                                      sistrip::invalid_,
                                      channel)
                            .key();  // just record lld channel
        mapping_[title.keyValue()] = temp;

      } else if (view == sistrip::DETECTOR_VIEW) {
        SiStripDetKey temp(path.key());
        key = SiStripDetKey(temp.partition()).key();
        mapping_[title.keyValue()] = key;

      } else {
        key = SiStripKey(path.key()).key();
      }

      // Find CME in histos map
      Histo* histo = nullptr;
      HistosMap::iterator ihistos = histos_.find(key);
      if (ihistos != histos_.end()) {
        Histos::iterator ihis = ihistos->second.begin();
        while (!histo && ihis < ihistos->second.end()) {
          if ((*ime)->getName() == (*ihis)->title_) {
            histo = *ihis;
          }
          ihis++;
        }
      }

      // Create CollateME if it doesn't exist
      if (!histo) {
        histos_[key].push_back(new Histo());
        histo = histos_[key].back();
        histo->title_ = (*ime)->getName();

        // If histogram present in client directory, add to map
        if (source_dir.find(sistrip::collate_) != std::string::npos) {
          histo->me_ = bei_->get(client_dir + "/" + (*ime)->getName());
          if (!histo->me_) {
            edm::LogError(mlDqmClient_) << "[CommissioningHistograms::" << __func__ << "]"
                                        << " NULL pointer to MonitorElement!";
          }
        }
      }
    }
  }

  //printHistosMap();

  edm::LogVerbatim(mlDqmClient_) << "[CommissioningHistograms::" << __func__ << "]"
                                 << " Found histograms for " << histos_.size()
                                 << " structures in cached histogram map!";
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::histoAnalysis(bool debug) {
  LogTrace(mlDqmClient_) << "[CommissioningHistograms::" << __func__ << "]"
                         << " (Derived) implementation to come...";
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::printAnalyses() {
  Analyses::iterator ianal = data().begin();
  Analyses::iterator janal = data().end();
  for (; ianal != janal; ++ianal) {
    if (ianal->second) {
      std::stringstream ss;
      ianal->second->print(ss);
      if (ianal->second->isValid()) {
        LogTrace(mlDqmClient_) << ss.str();
      } else {
        edm::LogWarning(mlDqmClient_) << ss.str();
      }
    }
  }
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::printSummary() {
  std::stringstream good;
  std::stringstream bad;

  Analyses::iterator ianal = data().begin();
  Analyses::iterator janal = data().end();
  for (; ianal != janal; ++ianal) {
    if (ianal->second) {
      if (ianal->second->isValid()) {
        ianal->second->summary(good);
      } else {
        ianal->second->summary(bad);
      }
    }
  }

  if (good.str().empty()) {
    good << "None found!";
  }
  LogTrace(mlDqmClient_) << "[CommissioningHistograms::" << __func__ << "]"
                         << " Printing summary of good analyses:"
                         << "\n"
                         << good.str();

  if (bad.str().empty()) {
    return;
  }  //@@ bad << "None found!"; }
  LogTrace(mlDqmClient_) << "[CommissioningHistograms::" << __func__ << "]"
                         << " Printing summary of bad analyses:"
                         << "\n"
                         << bad.str();
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::printHistosMap() {
  LogTrace(mlDqmClient_) << "[CommissioningHistograms::" << __func__ << "]"
                         << " Printing histogram map, which has " << histos_.size() << " entries...";
  HistosMap::const_iterator ihistos = histos_.begin();
  for (; ihistos != histos_.end(); ihistos++) {
    std::stringstream ss;
    ss << " Found " << ihistos->second.size() << " histogram(s) for key: " << std::endl
       << SiStripFedKey(ihistos->first) << std::endl;
    Histos::const_iterator ihisto = ihistos->second.begin();
    for (; ihisto != ihistos->second.end(); ihisto++) {
      if (*ihisto) {
        (*ihisto)->print(ss);
      } else {
        ss << " NULL pointer to Histo object!";
      }
    }
    LogTrace(mlDqmClient_) << ss.str();
  }
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::clearHistosMap() {
  LogTrace(mlDqmClient_) << "[CommissioningHistograms::" << __func__ << "]"
                         << " Clearing histogram map...";
  HistosMap::iterator ihistos = histos_.begin();
  for (; ihistos != histos_.end(); ihistos++) {
    Histos::iterator ihisto = ihistos->second.begin();
    for (; ihisto != ihistos->second.end(); ihisto++) {
      if (*ihisto) {
        delete *ihisto;
      }
    }
    ihistos->second.clear();
  }
  histos_.clear();
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::createSummaryHisto(const sistrip::Monitorable& mon,
                                                 const sistrip::Presentation& pres,
                                                 const std::string& dir,
                                                 const sistrip::Granularity& gran) {
  LogTrace(mlDqmClient_) << "[CommissioningHistograms::" << __func__ << "]";

  // Check view
  sistrip::View view = SiStripEnumsAndStrings::view(dir);
  if (view == sistrip::UNKNOWN_VIEW) {
    return;
  }

  // Analyze histograms
  if (data().empty()) {
    histoAnalysis(false);
  }

  // Check
  if (data().empty()) {
    edm::LogError(mlDqmClient_) << "[CommissioningHistograms::" << __func__ << "]"
                                << " No analyses generated!";
    return;
  }

  // Extract data to be histogrammed
  uint32_t xbins = factory()->init(mon, pres, view, dir, gran, data());

  // Only create histograms if entries are found!
  if (!xbins) {
    return;
  }

  // Create summary histogram (if it doesn't already exist)
  TH1* summary = nullptr;
  if (pres != sistrip::HISTO_1D) {
    summary = histogram(mon, pres, view, dir, xbins);
  } else {
    summary = histogram(mon, pres, view, dir, sistrip::FED_ADC_RANGE, 0., sistrip::FED_ADC_RANGE * 1.);
  }

  // Fill histogram with data
  factory()->fill(*summary);
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::remove(std::string pattern) {
  // TODO: remove no longer supported in DQMStore.
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::save(std::string& path, uint32_t run_number, std::string partitionName) {
  // Construct path and filename
  std::stringstream ss;

  if (!path.empty()) {
    ss << path;
    if (ss.str().find(".root") == std::string::npos) {
      ss << ".root";
    }

  } else {
    // Retrieve SCRATCH directory
    std::string scratch = "SCRATCH";
    std::string dir = "";
    if (std::getenv(scratch.c_str()) != nullptr) {
      dir = std::getenv(scratch.c_str());
    }

    // Add directory path
    if (!dir.empty()) {
      ss << dir << "/";
    } else {
      ss << "/tmp/";
    }

    // Add filename with run number and ".root" extension
    if (partitionName.empty())
      ss << sistrip::dqmClientFileName_ << "_" << std::setfill('0') << std::setw(8) << run_number << ".root";
    else
      ss << sistrip::dqmClientFileName_ << "_" << partitionName << "_" << std::setfill('0') << std::setw(8)
         << run_number << ".root";
  }

  // Save file with appropriate filename
  LogTrace(mlDqmClient_) << "[CommissioningHistograms::" << __func__ << "]"
                         << " Saving histograms to root file"
                         << " (This may take some time!)";
  path = ss.str();
  bei_->save(path, sistrip::collate_);
  edm::LogVerbatim(mlDqmClient_) << "[CommissioningHistograms::" << __func__ << "]"
                                 << " Saved histograms to root file \"" << ss.str() << "\"!";
}

// -----------------------------------------------------------------------------
//
TH1* CommissioningHistograms::histogram(const sistrip::Monitorable& mon,
                                        const sistrip::Presentation& pres,
                                        const sistrip::View& view,
                                        const std::string& directory,
                                        const uint32_t& xbins,
                                        const float& xlow,
                                        const float& xhigh) {
  // Remember pwd
  std::string pwd = bei_->pwd();
  bei_->setCurrentFolder(std::string(sistrip::collate_) + sistrip::dir_ + directory);

  // Construct histogram name
  std::string name = SummaryGenerator::name(task_, mon, pres, view, directory);

  MonitorElement* me = bei_->get(bei_->pwd() + "/" + name);

  // Create summary plot
  float high = static_cast<float>(xbins);
  if (pres == sistrip::HISTO_1D) {
    if (xlow < 1. * sistrip::valid_ && xhigh < 1. * sistrip::valid_) {
      me = bei_->book1D(name, name, xbins, xlow, xhigh);
    } else {
      me = bei_->book1D(name, name, xbins, 0., high);
    }
  } else if (pres == sistrip::HISTO_2D_SUM) {
    me = bei_->book1D(name, name, xbins, 0., high);
  } else if (pres == sistrip::HISTO_2D_SCATTER) {
    me = bei_->book2D(name, name, xbins, 0., high, sistrip::FED_ADC_RANGE + 1, 0., sistrip::FED_ADC_RANGE * 1.);
  } else if (pres == sistrip::PROFILE_1D) {
    me = bei_->bookProfile(name, name, xbins, 0., high, sistrip::FED_ADC_RANGE + 1, 0., sistrip::FED_ADC_RANGE * 1.);
  } else {
    me = nullptr;
    edm::LogWarning(mlDqmClient_) << "[CommissioningHistograms::" << __func__ << "]"
                                  << " Unexpected presentation \"" << SiStripEnumsAndStrings::presentation(pres)
                                  << "\" Unable to create summary plot!";
  }

  // Check pointer
  if (me) {
    LogTrace(mlDqmClient_) << "[CommissioningHistograms::" << __func__ << "]"
                           << " Created summary plot with name \"" << me->getName() << "\" in directory \""
                           << bei_->pwd() << "\"!";
  } else {
    edm::LogWarning(mlDqmClient_) << "[CommissioningHistograms::" << __func__ << "]"
                                  << " NULL pointer to MonitorElement!"
                                  << " Unable to create summary plot!";
  }

  // Extract root object
  TH1* summary = ExtractTObject<TH1>().extract(me);
  if (!summary) {
    edm::LogWarning(mlDqmClient_) << "[CommissioningHistograms::" << __func__ << "]"
                                  << " Unable to extract root object!"
                                  << " Returning NULL pointer!";
  }

  // Return to pwd
  bei_->setCurrentFolder(pwd);

  return summary;
}

CommissioningHistograms::Analyses& CommissioningHistograms::data(bool getMaskedData) {
  if (!getMaskedData)
    return data_;
  else {
    if (dataWithMaskCached_)
      return dataWithMask_;
    else {
      Analyses::iterator ianal = data_.begin();
      Analyses::iterator janal = data_.end();
      for (; ianal != janal; ++ianal) {
        CommissioningAnalysis* anal = ianal->second;
        SiStripFedKey fedkey = anal->fedKey();
        SiStripFecKey feckey = anal->fecKey();
        bool maskThisAnal_ = false;
        for (std::size_t i = 0; i < fedMaskVector_.size(); i++) {
          if (fedkey.fedId() == fedMaskVector_[i])
            maskThisAnal_ = true;
        }
        for (std::size_t i = 0; i < fecMaskVector_.size(); i++) {
          if (fecMaskVector_.size() != ringVector_.size() || fecMaskVector_.size() != ccuVector_.size() ||
              fecMaskVector_.size() != i2cChanVector_.size() || fecMaskVector_.size() != lldChanVector_.size()) {
            continue;
          }
          if (feckey.fecSlot() == fecMaskVector_[i] && feckey.fecRing() == ringVector_[i] &&
              feckey.ccuAddr() == ccuVector_[i] && feckey.ccuChan() == i2cChanVector_[i] &&
              feckey.lldChan() == lldChanVector_[i]) {
            maskThisAnal_ = true;
          }
        }
        if (mask_ && !maskThisAnal_)
          dataWithMask_[ianal->first] = ianal->second;
        else if (!mask_ && maskThisAnal_)
          dataWithMask_[ianal->first] = ianal->second;
      }
      dataWithMaskCached_ = true;
      return dataWithMask_;
    }
  }
}
