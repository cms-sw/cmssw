#include "helper.h"

// Global variable
AnalysisConfig ana;

void parseArguments(int argc, char** argv) {
  //********************************************************************************
  //
  // 1. Parsing options
  //
  //********************************************************************************

  // cxxopts is just a tool to parse argc, and argv easily

  // Grand option setting
  cxxopts::Options options("\n  $ doAnalysis",
                           "\n         **********************\n         *                    *\n         *       "
                           "Looper       *\n         *                    *\n         **********************\n");

  // Read the options
  options.add_options()("i,input",
                        "Comma separated input file list OR if just a directory is provided it will glob all in the "
                        "directory BUT must end with '/' for the path",
                        cxxopts::value<std::string>())("o,output", "Output file name", cxxopts::value<std::string>())(
      "n,nevents", "N events to loop over", cxxopts::value<int>()->default_value("-1"))(
      "j,nsplit_jobs", "Enable splitting jobs by N blocks (--job_index must be set)", cxxopts::value<int>())(
      "I,job_index",
      "job_index of split jobs (--nsplit_jobs must be set. index starts from 0. i.e. 0, 1, 2, 3, etc...)",
      cxxopts::value<int>())(
      "g,pdgid", "additional pdgid filtering to use (must be comma separated)", cxxopts::value<std::string>())(
      "p,pt_cut", "Transverse momentum cut", cxxopts::value<float>()->default_value("50"))( // Kasia changed from 0.9
      "e,eta_cut", "Pseudorapidity cut", cxxopts::value<float>()->default_value("4.5"))(
      "d,debug", "Run debug job. i.e. overrides output option to 'debug.root' and 'recreate's the file.")("h,help",
                                                                                                          "Print help");

  auto result = options.parse(argc, argv);

  // NOTE: When an option was provided (e.g. -i or --input), then the result.count("<option name>") is more than 0
  // Therefore, the option can be parsed easily by asking the condition if (result.count("<option name>");
  // That's how the several options are parsed below

  //_______________________________________________________________________________
  // --help
  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    exit(1);
  }

  //_______________________________________________________________________________
  // --input
  if (result.count("input")) {
    ana.input_file_list_tstring = result["input"].as<std::string>();
  } else {
    std::cout << options.help() << std::endl;
    std::cout << "ERROR: Input list is not provided! Check your arguments" << std::endl;
    exit(1);
  }

  ana.input_tree_name = "tree";

  //_______________________________________________________________________________
  // --pdgid
  if (result.count("pdgid")) {
    std::vector<TString> pdgid_strs = RooUtil::StringUtil::split(result["pdgid"].as<std::string>(), ",");
    for (auto& pdgid_str : pdgid_strs) {
      ana.pdgids.push_back(pdgid_str.Atoi());
    }
  }

  //_______________________________________________________________________________
  // --pt_cut
  ana.pt_cut = result["pt_cut"].as<float>();

  //_______________________________________________________________________________
  // --eta_cut
  ana.eta_cut = result["eta_cut"].as<float>();

  //_______________________________________________________________________________
  // --debug
  if (result.count("debug")) {
    ana.output_tfile = new TFile("debug.root", "recreate");
  } else {
    //_______________________________________________________________________________
    // --output
    if (result.count("output")) {
      ana.output_tfile = new TFile(result["output"].as<std::string>().c_str(), "create");
      if (not ana.output_tfile->IsOpen()) {
        std::cout << options.help() << std::endl;
        std::cout << "ERROR: output already exists! provide new output name or delete old file. OUTPUTFILE="
                  << result["output"].as<std::string>() << std::endl;
        exit(1);
      }
    } else {
      std::cout << options.help() << std::endl;
      std::cout << "ERROR: Output file name is not provided! Check your arguments" << std::endl;
      exit(1);
    }
  }

  //_______________________________________________________________________________
  // --nevents
  ana.n_events = result["nevents"].as<int>();

  //_______________________________________________________________________________
  // --nsplit_jobs
  if (result.count("nsplit_jobs")) {
    ana.nsplit_jobs = result["nsplit_jobs"].as<int>();
    if (ana.nsplit_jobs <= 0) {
      std::cout << options.help() << std::endl;
      std::cout << "ERROR: option string --nsplit_jobs" << ana.nsplit_jobs << " has zero or negative value!"
                << std::endl;
      std::cout << "I am not sure what this means..." << std::endl;
      exit(1);
    }
  } else {
    ana.nsplit_jobs = -1;
  }

  //_______________________________________________________________________________
  // --nsplit_jobs
  if (result.count("job_index")) {
    ana.job_index = result["job_index"].as<int>();
    if (ana.job_index < 0) {
      std::cout << options.help() << std::endl;
      std::cout << "ERROR: option string --job_index" << ana.job_index << " has negative value!" << std::endl;
      std::cout << "I am not sure what this means..." << std::endl;
      exit(1);
    }
  } else {
    ana.job_index = -1;
  }

  // Sanity check for split jobs (if one is set the other must be set too)
  if (result.count("job_index") or result.count("nsplit_jobs")) {
    // If one is not provided then throw error
    if (not(result.count("job_index") and result.count("nsplit_jobs"))) {
      std::cout << options.help() << std::endl;
      std::cout << "ERROR: option string --job_index and --nsplit_jobs must be set at the same time!" << std::endl;
      exit(1);
    }
    // If it is set then check for sanity
    else {
      if (ana.job_index >= ana.nsplit_jobs) {
        std::cout << options.help() << std::endl;
        std::cout << "ERROR: --job_index >= --nsplit_jobs ! This does not make sense..." << std::endl;
        exit(1);
      }
    }
  }

  //
  // Printing out the option settings overview
  //
  std::cout << "=========================================================" << std::endl;
  std::cout << " Setting of the analysis job based on provided arguments " << std::endl;
  std::cout << "---------------------------------------------------------" << std::endl;
  std::cout << " ana.input_file_list_tstring: " << ana.input_file_list_tstring << std::endl;
  std::cout << " ana.output_tfile: " << ana.output_tfile->GetName() << std::endl;
  std::cout << " ana.n_events: " << ana.n_events << std::endl;
  std::cout << " ana.pt_cut: " << ana.pt_cut << std::endl;
  std::cout << " ana.eta_cut: " << ana.eta_cut << std::endl;
  std::cout << " ana.nsplit_jobs: " << ana.nsplit_jobs << std::endl;
  std::cout << " ana.job_index: " << ana.job_index << std::endl;
  std::cout << "=========================================================" << std::endl;
}

AnalysisConfig::AnalysisConfig() : tx("variable", "variable") {}

SimTrackSetDefinition::SimTrackSetDefinition(TString set_name_,
                                             int pdgid_,
                                             int q_,
                                             std::function<bool(unsigned int)> pass_,
                                             std::function<bool(unsigned int)> sel_) {
  set_name = set_name_;
  pdgid = pdgid_;
  q = q_;
  pass = pass_;
  sel = sel_;
}

RecoTrackSetDefinition::RecoTrackSetDefinition(TString set_name_,
                                               std::function<bool(unsigned int)> pass_,
                                               std::function<bool(unsigned int)> sel_,
                                               std::function<const std::vector<float>()> pt_,
                                               std::function<const std::vector<float>()> eta_,
                                               std::function<const std::vector<float>()> phi_,
                                               std::function<const std::vector<int>()> type_)
    : pt(pt_), eta(eta_), phi(phi_), type(type_) {
  set_name = set_name_;
  pass = pass_;
  sel = sel_;
}

void initializeInputsAndOutputs() {
  // Create the TChain that holds the TTree's of the baby ntuples
  ana.events_tchain = RooUtil::FileUtil::createTChain(ana.input_tree_name, ana.input_file_list_tstring);

  // This is a standard thing SNT does pretty much every looper we use
  ana.looper.init(ana.events_tchain, &lstEff, ana.n_events);

  // Set the cutflow object output file
  ana.cutflow.setTFile(ana.output_tfile);

  // Copy the information of the code
  std::vector<TString> vstr = RooUtil::StringUtil::split(ana.input_file_list_tstring, ",");
  TFile* firstFile = new TFile(vstr[0]);

  // Obtain the info from the input file
  TString code_tag_data = firstFile->Get("code_tag_data")->GetTitle();
  TString gitdiff = firstFile->Get("gitdiff")->GetTitle();
  TString input = firstFile->Get("input")->GetTitle();

  // cd to output file
  ana.output_tfile->cd();

  // Write the githash after parsing whether there is any gitdiff
  TString githash = RooUtil::StringUtil::split(code_tag_data, "\n")[0];
  githash = githash(0, 6);
  std::cout << " gitdiff.Length(): " << gitdiff.Length() << std::endl;
  if (gitdiff.Length() > 0)
    githash += "D";
  std::cout << " githash: " << githash << std::endl;
  TNamed githash_tnamed("githash", githash.Data());
  githash_tnamed.Write();

  // Write the sample information
  if (input.Contains("PU200"))
    input = "PU200";
  std::cout << " input: " << input << std::endl;
  TNamed input_tnamed("input", input.Data());
  input_tnamed.Write();

  ana.do_lower_level = false;  // default is false
  TObjArray* brobjArray = ana.events_tchain->GetListOfBranches();
  for (unsigned int ibr = 0; ibr < (unsigned int)brobjArray->GetEntries(); ++ibr) {
    TString brname = brobjArray->At(ibr)->GetName();
    if (brname.EqualTo("t5_pt"))
      ana.do_lower_level = true;  // if it has the branch it is set to true
  }
}

std::vector<float> getPtBounds(int mode) {
  std::vector<float> pt_boundaries;
  if (mode == 0)
    pt_boundaries = {0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0, 10, 15., 25, 50};
  else if (mode == 1)
    pt_boundaries = {0.988, 0.99, 0.992, 0.994, 0.996, 0.998, 1.0, 1.002, 1.004, 1.006, 1.008, 1.01, 1.012};  // lowpt
  else if (mode == 2)
    pt_boundaries = {0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995, 1.00,
                     1.005, 1.01, 1.015, 1.02, 1.025, 1.03, 1.035, 1.04, 1.045, 1.05};  // pt 0p95 1p05
  else if (mode == 3)
    pt_boundaries = {
        0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06, 1.08, 1.1, 1.2, 1.5};  // lowpt
  else if (mode == 4)
    pt_boundaries = {0.5,  0.52, 0.54, 0.56, 0.58, 0.6,  0.62, 0.64, 0.66, 0.68, 0.7,  0.72, 0.74,
                     0.76, 0.78, 0.8,  0.82, 0.84, 0.86, 0.88, 0.9,  0.92, 0.94, 0.96, 0.98, 1.0,
                     1.02, 1.04, 1.06, 1.08, 1.1,  1.12, 1.14, 1.16, 1.18, 1.2,  1.22, 1.24, 1.26,
                     1.28, 1.3,  1.32, 1.34, 1.36, 1.38, 1.4,  1.5,  1.6,  1.7,  1.8,  1.9,  2.0};  // lowpt
  else if (mode == 5)
    pt_boundaries = {0.5,  0.52, 0.54, 0.56, 0.58, 0.6,  0.62, 0.64, 0.66, 0.68, 0.7,  0.72, 0.74,
                     0.76, 0.78, 0.8,  0.82, 0.84, 0.86, 0.88, 0.9,  0.92, 0.94, 0.96, 0.98, 1.0,
                     1.02, 1.04, 1.06, 1.08, 1.1,  1.12, 1.14, 1.16, 1.18, 1.2,  1.24, 1.28, 1.32,
                     1.36, 1.4,  1.6,  1.8,  2.0,  2.2,  2.4,  2.6,  2.8,  3.0};  // lowpt
  else if (mode == 6)
    pt_boundaries = {0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 3.0, 4.0, 5.0};  // lowpt
  else if (mode == 7)
    pt_boundaries = {0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64, 0.66, 0.68, 0.7, 0.72, 0.74, 0.76, 0.78,
                     0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06, 1.08,
                     1.1, 1.12, 1.14, 1.16, 1.18, 1.2, 1.22, 1.24, 1.26, 1.28, 1.3, 1.32, 1.34, 1.36, 1.38,
                     1.4, 1.5,  1.6,  1.7,  1.8,  1.9, 2.0,  2.2,  2.4,  2.6,  2.8, 3,    4,    5,    6,
                     7,   8,    9,    10,   15,   20,  25,   30,   35,   40,   45,  50};  // lowpt
  else if (mode == 8)
    pt_boundaries = {0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10, 15., 25, 50};
  else if (mode == 9) {
    for (int i = 0; i < 41; ++i) {
      pt_boundaries.push_back(pow(10., -1. + 4. * i / 40.));
    }
  }
  return pt_boundaries;
}
