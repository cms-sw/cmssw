#include <memory>
#include <vector>
#include <iostream>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// #include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentEngine2016.h"
#include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentEngine2017.h"
#include "L1Trigger/L1TMuonEndCap/interface/PtLUTWriter.h"

#include "helper.h"
#include "progress_bar.h"

class MakePtLUT : public edm::one::EDAnalyzer<> {
public:
  explicit MakePtLUT(const edm::ParameterSet&);
  ~MakePtLUT() override;

private:
  //virtual void beginJob();
  //virtual void endJob();

  //virtual void beginRun(const edm::Run&, const edm::EventSetup&);
  //virtual void endRun(const edm::Run&, const edm::EventSetup&);

  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  void makeLUT();

  void checkAddresses();

private:
  std::unique_ptr<PtAssignmentEngine> pt_assign_engine_;

  PtLUTWriter ptlut_writer_;

  const edm::ParameterSet config_;

  int verbose_;
  int num_;
  int denom_;

  std::string xml_dir_;
  std::string outfile_;

  bool onlyCheck_;
  std::vector<unsigned long long> addressesToCheck_;

  bool done_;
};

// _____________________________________________________________________________
#define PTLUT_SIZE (1 << 30)

MakePtLUT::MakePtLUT(const edm::ParameterSet& iConfig)
    :  // pt_assign_engine_(new PtAssignmentEngine2016()),
      pt_assign_engine_(new PtAssignmentEngine2017()),
      ptlut_writer_(),
      config_(iConfig),
      verbose_(iConfig.getUntrackedParameter<int>("verbosity")),
      num_(iConfig.getParameter<int>("numerator")),
      denom_(iConfig.getParameter<int>("denominator")),
      outfile_(iConfig.getParameter<std::string>("outfile")),
      onlyCheck_(iConfig.getParameter<bool>("onlyCheck")),
      addressesToCheck_(iConfig.getParameter<std::vector<unsigned long long> >("addressesToCheck")),
      done_(false) {
  auto ptLUTVersion = iConfig.getParameter<int>("PtLUTVersion");

  const edm::ParameterSet spPAParams16 = config_.getParameter<edm::ParameterSet>("spPAParams16");
  auto bdtXMLDir = spPAParams16.getParameter<std::string>("BDTXMLDir");
  auto readPtLUTFile = spPAParams16.getParameter<bool>("ReadPtLUTFile");
  auto fixMode15HighPt = spPAParams16.getParameter<bool>("FixMode15HighPt");
  auto bug9BitDPhi = spPAParams16.getParameter<bool>("Bug9BitDPhi");
  auto bugMode7CLCT = spPAParams16.getParameter<bool>("BugMode7CLCT");
  auto bugNegPt = spPAParams16.getParameter<bool>("BugNegPt");

  ptlut_writer_.set_version(ptLUTVersion);

  pt_assign_engine_->configure(verbose_, readPtLUTFile, fixMode15HighPt, bug9BitDPhi, bugMode7CLCT, bugNegPt);

  xml_dir_ = bdtXMLDir;
}

MakePtLUT::~MakePtLUT() {}

void MakePtLUT::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  if (done_)
    return;

  if (onlyCheck_) {
    checkAddresses();

  } else {
    makeLUT();
  }

  done_ = true;
  return;
}

void MakePtLUT::makeLUT() {
  // Load XMLs inside function
  std::cout << "Inside makeLUT() - loading XMLs" << std::endl;
  pt_assign_engine_->read(config_.getParameter<int>("PtLUTVersion"), xml_dir_);

  std::cout << "Calculating pT for " << PTLUT_SIZE / denom_ << " addresses, please sit tight..." << std::endl;

  if (num_ - 1 < 0)
    std::cout << "ERROR: tried to fill address < 0.  KILL!!!" << std::endl;
  PtLUTWriter::address_t address = abs((num_ - 1) * (PTLUT_SIZE / denom_));

  float xmlpt = 0.;
  float pt = 0.;
  int gmt_pt = 0;

  for (; address < (PtLUTWriter::address_t)abs(num_ * (PTLUT_SIZE / denom_)); ++address) {
    if (address % (PTLUT_SIZE / (denom_ * 128)) == 0)
      show_progress_bar(address, PTLUT_SIZE);

    //int mode_inv = (address >> (30-4)) & ((1<<4)-1);

    // floats
    xmlpt = pt_assign_engine_->calculate_pt(address);
    pt = (xmlpt < 0.) ? 1. : xmlpt;             // Matt used fabs(-1) when mode is invalid
    pt *= pt_assign_engine_->scale_pt(pt, 15);  // Multiply by some factor to achieve 90% efficiency at threshold

    // integers
    gmt_pt = (pt * 2) + 1;
    gmt_pt = (gmt_pt > 511) ? 511 : gmt_pt;

    //if (address % (1<<20) == 0)
    //  std::cout << mode_inv << " " << address << " " << print_subaddresses(address) << " " << gmt_pt << std::endl;

    ptlut_writer_.push_back(gmt_pt);
  }

  std::cout << "\nAbout to write file " << outfile_ << " for part " << num_ << "/" << denom_ << std::endl;
  ptlut_writer_.write(outfile_, num_, denom_);
  std::cout << "Wrote file! DONE!" << std::endl;
}

void MakePtLUT::checkAddresses() {
  unsigned int n = addressesToCheck_.size();
  std::cout << "Calculating pT for " << n << " addresses, please sit tight..." << std::endl;

  PtLUTWriter::address_t address = 0;

  float xmlpt = 0.;
  float pt = 0.;
  int gmt_pt = 0;

  for (unsigned int i = 0; i < n; ++i) {
    //show_progress_bar(i, n);

    address = addressesToCheck_.at(i);

    int mode_inv = (address >> (30 - 4)) & ((1 << 4) - 1);

    // floats
    xmlpt = pt_assign_engine_->calculate_pt(address);
    pt = (xmlpt < 0.) ? 1. : xmlpt;  // Matt used fabs(-1) when mode is invalid
    pt *= 1.4;  // multiply by 1.4 to keep efficiency above 90% when the L1 trigger pT cut is applied

    // integers
    gmt_pt = (pt * 2) + 1;
    gmt_pt = (gmt_pt > 511) ? 511 : gmt_pt;

    std::cout << mode_inv << " " << address << " " << print_subaddresses(address) << " " << gmt_pt << std::endl;
  }
}

// DEFINE THIS AS A PLUG-IN
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MakePtLUT);
