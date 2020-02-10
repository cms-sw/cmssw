#include "DQM/SiStripCommissioningClients/interface/CalibrationHistograms.h"
#include "CondFormats/SiStripObjects/interface/CalibrationAnalysis.h"
#include "CondFormats/SiStripObjects/interface/CalibrationScanAnalysis.h"
#include "DQM/SiStripCommissioningAnalysis/interface/CalibrationAlgorithm.h"
#include "DQM/SiStripCommissioningAnalysis/interface/CalibrationScanAlgorithm.h"
#include "DQM/SiStripCommissioningSummary/interface/CalibrationSummaryFactory.h"
#include "DQM/SiStripCommissioningSummary/interface/CalibrationScanSummaryFactory.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DQM/SiStripCommon/interface/ExtractTObject.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include "TH1F.h"
#include "TFile.h"
#include "TMultiGraph.h"
#include "TGraph.h"

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
CalibrationHistograms::CalibrationHistograms(const edm::ParameterSet& pset, DQMStore* bei, const sistrip::RunType& task)
    : CommissioningHistograms(pset.getParameter<edm::ParameterSet>("CalibrationParameters"), bei, task) {
  LogTrace(mlDqmClient_) << "[CalibrationHistograms::" << __func__ << "]"
                         << " Constructing object...";

  if (task == sistrip::CALIBRATION_SCAN or task == sistrip::CALIBRATION_SCAN_DECO)
    factory_ = auto_ptr<CalibrationScanSummaryFactory>(new CalibrationScanSummaryFactory);
  else
    factory_ = auto_ptr<CalibrationSummaryFactory>(new CalibrationSummaryFactory);

  targetRiseTime_ =
      this->pset().existsAs<double>("targetRiseTime") ? this->pset().getParameter<double>("targetRiseTime") : 50;
  targetDecayTime_ =
      this->pset().existsAs<double>("targetDecayTime") ? this->pset().getParameter<double>("targetDecayTime") : 125;
  tuneSimultaneously_ =
      this->pset().existsAs<bool>("tuneSimultaneously") ? this->pset().getParameter<bool>("tuneSimultaneously") : false;
}

// -----------------------------------------------------------------------------
/** */
CalibrationHistograms::~CalibrationHistograms() {
  LogTrace(mlDqmClient_) << "[CalibrationHistograms::" << __func__ << "]"
                         << " Deleting object...";
}

// -----------------------------------------------------------------------------
/** */
void CalibrationHistograms::histoAnalysis(bool debug) {
  // Clear map holding analysis objects
  Analyses::iterator ianal;
  for (ianal = data().begin(); ianal != data().end(); ianal++) {
    if (ianal->second) {
      delete ianal->second;
    }
  }
  data().clear();

  // Iterate through map containing vectors of profile histograms
  HistosMap::const_iterator iter = histos().begin();

  // One entry for each LLD channel --> differnt thousand entries
  for (; iter != histos().end(); iter++) {
    if (iter->second.empty()) {
      edm::LogWarning(mlDqmClient_) << "[CalibrationHistograms::" << __func__ << "]"
                                    << " Zero collation histograms found!";
      continue;
    }

    // Retrieve pointers to 1D histos for this FED channel --> all strips in the fiber = 256
    vector<TH1*> profs;
    Histos::const_iterator ihis = iter->second.begin();
    for (; ihis != iter->second.end(); ihis++) {
      TH1F* prof = ExtractTObject<TH1F>().extract((*ihis)->me_);
      if (prof) {
        profs.push_back(prof);
      }
    }

    // Perform histo analysis
    bool isdeconv = false;
    if (task() == sistrip::CALIBRATION_DECO or task() == sistrip::CALIBRATION_SCAN_DECO)
      isdeconv = true;

    if (task() == sistrip::CALIBRATION_SCAN or task() == sistrip::CALIBRATION_SCAN_DECO) {
      CalibrationScanAnalysis* anal = new CalibrationScanAnalysis(iter->first, isdeconv);
      CalibrationScanAlgorithm algo(this->pset(), anal);
      algo.analysis(profs);
      data()[iter->first] = anal;

      // tune the parameters for this a given target
      for (int iapv = 0; iapv < 2; iapv++) {
        if (tuneSimultaneously_)
          algo.tuneSimultaneously(iapv, targetRiseTime_, targetDecayTime_);
        else
          algo.tuneIndependently(iapv, targetRiseTime_, targetDecayTime_);
        algo.fillTunedObservables(iapv);
      }
    } else {
      CalibrationAnalysis* anal = new CalibrationAnalysis(iter->first, isdeconv);
      CalibrationAlgorithm algo(this->pset(), anal);
      algo.analysis(profs);
      data()[iter->first] = anal;
    }
  }
}

// -----------------------------------------------------------------------------
/** */
void CalibrationHistograms::printAnalyses() {
  Analyses::iterator ianal = data().begin();
  Analyses::iterator janal = data().end();
  for (; ianal != janal; ++ianal) {
    if (ianal->second) {
      std::stringstream ss;
      ianal->second->print(ss, 0);
      ianal->second->print(ss, 1);
      if (ianal->second->isValid()) {
        LogTrace(mlDqmClient_) << ss.str();
      } else {
        edm::LogWarning(mlDqmClient_) << ss.str();
      }
    }
  }
}

//-----------------------------------------------------------------------------
/** */
void CalibrationHistograms::save(std::string& path, uint32_t run_number, std::string partitionName) {
  // Construct path and filename
  std::stringstream ss;
  if (!path.empty()) {  // create with a specific outputName
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
  bei()->save(path, sistrip::collate_);
  edm::LogVerbatim(mlDqmClient_) << "[CommissioningHistograms::" << __func__ << "]"
                                 << " Saved histograms to root file \"" << ss.str() << "\"!";

  // In case of calibration-scan, add also the TGraphs
  // re-open the file
  TFile* outputFile = TFile::Open(path.c_str(), "UPDATE");
  outputFile->cd();

  auto contents = bei()->getAllContents("");

  TMultiGraph* graph_isha = new TMultiGraph("riseTime_vs_isha", "");
  TMultiGraph* graph_vfs = new TMultiGraph("decayTime_vs_vfs", "");

  bool save_graph_isha = false;
  bool save_graph_vfs = false;

  // loop on the analysis objects which are storing all relevant results
  Analyses::iterator ianal = data().begin();
  Analyses::iterator janal = data().end();
  for (; ianal != janal; ++ianal) {
    if (ianal->second) {
      CalibrationScanAnalysis* anal = dynamic_cast<CalibrationScanAnalysis*>(ianal->second);
      SiStripFecKey feckey = anal->fecKey();

      TString directory;
      for (auto me : contents) {
        directory = me->getPathname();
        if (directory.Contains(Form("FecCrate%d", feckey.fecCrate())) and
            directory.Contains(Form("FecRing%d", feckey.fecRing())) and
            directory.Contains(Form("FecSlot%d", feckey.fecSlot())) and
            directory.Contains(Form("CcuAddr%d", feckey.ccuAddr())) and
            directory.Contains(Form("CcuChan%d", feckey.ccuChan())))
          break;
      }

      outputFile->cd("DQMData/" + directory);

      for (size_t igraph = 0; igraph < anal->decayTimeVsVFS().size(); igraph++) {
        graph_vfs->Add(anal->decayTimeVsVFS()[igraph]);
        anal->decayTimeVsVFS()[igraph]->Write();
        save_graph_vfs = true;
      }

      for (size_t igraph = 0; igraph < anal->riseTimeVsISHA().size(); igraph++) {
        graph_isha->Add(anal->riseTimeVsISHA()[igraph]);
        anal->riseTimeVsISHA()[igraph]->Write();
        save_graph_isha = true;
      }

      for (size_t igraph = 0; igraph < anal->riseTimeVsISHAVsVFS().size(); igraph++)
        anal->riseTimeVsISHAVsVFS()[igraph]->Write();

      for (size_t igraph = 0; igraph < anal->decayTimeVsISHAVsVFS().size(); igraph++)
        anal->decayTimeVsISHAVsVFS()[igraph]->Write();

      outputFile->cd();
    }
  }

  outputFile->cd();
  outputFile->cd("DQMData/Collate/SiStrip/ControlView");

  if (save_graph_isha)
    graph_isha->Write("riseTime_vs_isha");
  if (save_graph_vfs)
    graph_vfs->Write("decayTime_vs_vfs");

  outputFile->Close();
}
