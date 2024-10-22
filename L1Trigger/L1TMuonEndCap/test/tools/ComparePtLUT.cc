#include <map>
#include <memory>
#include <iostream>

#include "TH1F.h"
#include "TFile.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "L1Trigger/L1TMuonEndCap/interface/PtLUTReader.h"

#include "helper.h"
#include "progress_bar.h"

class ComparePtLUT : public edm::one::EDAnalyzer<> {
public:
  explicit ComparePtLUT(const edm::ParameterSet&);
  ~ComparePtLUT() override;

private:
  //virtual void beginJob();
  //virtual void endJob();

  //virtual void beginRun(const edm::Run&, const edm::EventSetup&);
  //virtual void endRun(const edm::Run&, const edm::EventSetup&);

  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  void compareLUTs();

private:
  PtLUTReader ptlut_reader1_;
  PtLUTReader ptlut_reader2_;

  const edm::ParameterSet config_;

  int verbose_;

  std::string infile1_;
  std::string infile2_;

  bool done_;
};

// _____________________________________________________________________________
#define PTLUT_SIZE (1 << 30)

ComparePtLUT::ComparePtLUT(const edm::ParameterSet& iConfig)
    : ptlut_reader1_(),
      ptlut_reader2_(),
      config_(iConfig),
      verbose_(iConfig.getUntrackedParameter<int>("verbosity")),
      infile1_(iConfig.getParameter<std::string>("infile1")),
      infile2_(iConfig.getParameter<std::string>("infile2")),
      done_(false) {
  ptlut_reader1_.read(infile1_);
  ptlut_reader2_.read(infile2_);
}

ComparePtLUT::~ComparePtLUT() {}

void ComparePtLUT::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  if (done_)
    return;

  compareLUTs();

  done_ = true;
  return;
}

void ComparePtLUT::compareLUTs() {
  TFile* f = TFile::Open("diff.root", "RECREATE");

  std::map<int, TH1F*> histograms;
  int diff_limit = 20;
  for (int mode_inv = 0; mode_inv < 16; ++mode_inv) {
    histograms[mode_inv] = new TH1F(
        Form("diff_mode_inv_%i", mode_inv), "", 2 * diff_limit + 1, -1.0 * diff_limit - 0.5, 1.0 * diff_limit + 0.5);
  }

  for (int ivar = 0; ivar < 10; ++ivar) {
    histograms[100 + ivar] = new TH1F(Form("diff_subaddress_%i", ivar), "", 32 + 1, 0. - 0.5, 32. + 0.5);
  }

  PtLUTReader::address_t address = 0;

  for (; address < PTLUT_SIZE; ++address) {
    show_progress_bar(address, PTLUT_SIZE);

    int mode_inv = (address >> (30 - 4)) & ((1 << 4) - 1);

    int gmt_pt1 = ptlut_reader1_.lookup(address);
    int gmt_pt2 = ptlut_reader2_.lookup(address);

    int diff = gmt_pt2 - gmt_pt1;
    diff = std::min(std::max(-diff_limit, diff), diff_limit);

    //if (address % (1<<20) == 0)
    //  std::cout << mode_inv << " " << address << " " << print_subaddresses(address) << " " << gmt_pt1 << " " << gmt_pt2 << " " << gmt_pt2 - gmt_pt1 << std::endl;

    if (std::abs(gmt_pt2 - gmt_pt1) > diff_limit) {
      std::cout << mode_inv << " " << address << " " << print_subaddresses(address) << " " << gmt_pt1 << " " << gmt_pt2
                << " " << gmt_pt2 - gmt_pt1 << std::endl;

      //if (mode_inv == 5) {  // debug
      //  histograms[100+0]->Fill(get_subword(address,25,21));  // theta
      //  histograms[100+1]->Fill(get_subword(address,20,20));  // FR3
      //  histograms[100+2]->Fill(get_subword(address,19,19));  // FR1
      //  histograms[100+3]->Fill(get_subword(address,18,18));  // CLCT3Sign
      //  histograms[100+4]->Fill(get_subword(address,17,16));  // CLCT3
      //  histograms[100+5]->Fill(get_subword(address,15,15));  // CLCT1Sign
      //  histograms[100+6]->Fill(get_subword(address,14,13));  // CLCT1
      //  histograms[100+7]->Fill(get_subword(address,12,10));  // dTheta13
      //  histograms[100+8]->Fill(get_subword(address, 9, 9));  // sign13
      //  histograms[100+9]->Fill(get_subword(address, 8, 0));  // dPhi13
      //}

      //if (mode_inv == 14) {  // debug
      //  histograms[100+0]->Fill(get_subword(address,25,21));  // theta
      //  histograms[100+1]->Fill(get_subword(address,20,20));  // CLCT2Sign
      //  histograms[100+2]->Fill(get_subword(address,19,18));  // CLCT2
      //  histograms[100+3]->Fill(get_subword(address,17,15));  // dTheta24
      //  histograms[100+4]->Fill(get_subword(address,14,14));  // sign34
      //  histograms[100+5]->Fill(get_subword(address,13,13));  // sign23
      //  histograms[100+6]->Fill(get_subword(address,12, 7));  // dPhi34
      //  histograms[100+7]->Fill(get_subword(address, 6, 0));  // dPhi23
      //}
    }

    histograms[mode_inv]->Fill(diff);
  }

  for (const auto& kv : histograms) {
    kv.second->Write();
  }
  f->Close();
}

// DEFINE THIS AS A PLUG-IN
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ComparePtLUT);
