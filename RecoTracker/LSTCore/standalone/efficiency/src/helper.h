#ifndef helper_h
#define helper_h

#include "LSTEff.h"
#include "cxxopts.h"
#include "rooutil.h"

class AnalysisConfig {
public:
  // TString that holds the input file list (comma separated)
  TString input_file_list_tstring;

  // TString that holds the name of the TTree to open for each input files
  TString input_tree_name;

  // Output TFile
  TFile* output_tfile;

  // Number of events to loop over
  int n_events;

  // Minimum pt cut
  float pt_cut;

  // Max eta cut
  float eta_cut;

  // Jobs to split (if this number is positive, then we will skip certain number of events)
  // If there are N events, and was asked to split 2 ways, then depending on job_index, it will run over first half or latter half
  int nsplit_jobs;

  // Job index (assuming nsplit_jobs is set, the job_index determine where to loop over)
  int job_index;

  // Debug boolean
  bool debug;

  // TChain that holds the input TTree's
  TChain* events_tchain;

  // Custom Looper object to facilitate looping over many files
  RooUtil::Looper<LSTEff> looper;

  // Custom Cutflow framework
  RooUtil::Cutflow cutflow;

  // Custom Histograms object compatible with RooUtil::Cutflow framework
  RooUtil::Histograms histograms;

  // Custom TTree object to hold intermediate variables
  RooUtil::TTreeX tx;

  // pt binning options
  int ptbound_mode;

  // pdgid
  int pdgid;

  // pdgids to filter
  std::vector<int> pdgids;

  // do lower level
  bool do_lower_level;

  AnalysisConfig();
};

extern AnalysisConfig ana;

class SimTrackSetDefinition {
public:
  TString set_name;
  int pdgid;
  int q;
  std::function<bool(unsigned int)> pass;
  std::function<bool(unsigned int)> sel;  // subset of sim track selection
  SimTrackSetDefinition(TString, int, int, std::function<bool(unsigned int)>, std::function<bool(unsigned int)>);
};

class RecoTrackSetDefinition {
public:
  TString set_name;
  std::function<bool(unsigned int)> pass;
  std::function<bool(unsigned int)> sel;
  std::function<const std::vector<float>()> pt;
  std::function<const std::vector<float>()> eta;
  std::function<const std::vector<float>()> phi;
  std::function<const std::vector<int>()> type;
  RecoTrackSetDefinition(TString,
                         std::function<bool(unsigned int)>,
                         std::function<bool(unsigned int)>,  // subsect of reco track selection
                         std::function<const std::vector<float>()>,
                         std::function<const std::vector<float>()>,
                         std::function<const std::vector<float>()>,
                         std::function<const std::vector<int>()>);
};

void parseArguments(int argc, char** argv);
void initializeInputsAndOutputs();
std::vector<float> getPtBounds(int mode);

#endif
