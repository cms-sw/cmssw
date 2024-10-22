#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <TCanvas.h>
#include <TFile.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TTree.h>
#include <TPaveStats.h>

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout << "run with: ./<exe> <path to input file> <path to output file>\n";
    exit(0);
  }

  // branches to use
  edm::Wrapper<EBDigiCollection>*wgpuEB = nullptr, *wcpuEB = nullptr;
  edm::Wrapper<EEDigiCollection>*wgpuEE = nullptr, *wcpuEE = nullptr;

  std::string inFileName{argv[1]};
  std::string outFileName{argv[2]};

  // prep output
  TFile rfout{outFileName.c_str(), "recreate"};

  int const nbins = 400;
  float const last = 4096.;
  auto hADCEBGPU = new TH1D("hADCEBGPU", "hADCEBGPU", nbins, 0, last);
  auto hADCEBCPU = new TH1D("hADCEBCPU", "hADCEBCPU", nbins, 0, last);
  auto hADCEEGPU = new TH1D("hADCEEGPU", "hADCEEGPU", nbins, 0, last);
  auto hADCEECPU = new TH1D("hADCEECPU", "hADCEECPU", nbins, 0, last);

  auto hGainEBGPU = new TH1D("hGainEBGPU", "hGainEBGPU", 4, 0, 4);
  auto hGainEBCPU = new TH1D("hGainEBCPU", "hGainEBCPU", 4, 0, 4);
  auto hGainEEGPU = new TH1D("hGainEEGPU", "hGainEEGPU", 4, 0, 4);
  auto hGainEECPU = new TH1D("hGainEECPU", "hGainEECPU", 4, 0, 4);

  auto hADCEBGPUvsCPU = new TH2D("hADCEBGPUvsCPU", "hADCEBGPUvsCPU", nbins, 0, last, nbins, 0, last);
  auto hADCEEGPUvsCPU = new TH2D("hADCEEGPUvsCPU", "hADCEEGPUvsCPU", nbins, 0, last, nbins, 0, last);
  auto hGainEBGPUvsCPU = new TH2D("hGainEBGPUvsCPU", "hGainEBGPUvsCPU", 4, 0, 4, 4, 0, 4);
  auto hGainEEGPUvsCPU = new TH2D("hGainEEGPUvsCPU", "hGainEEGPUvsCPU", 4, 0, 4, 4, 0, 4);

  // prep input
  TFile rfin{inFileName.c_str()};
  TTree* rt = (TTree*)rfin.Get("Events");
  rt->SetBranchAddress("EBDigiCollection_ecalCPUDigisProducer_ebDigis_RECO.", &wgpuEB);
  rt->SetBranchAddress("EEDigiCollection_ecalCPUDigisProducer_eeDigis_RECO.", &wgpuEE);
  rt->SetBranchAddress("EBDigiCollection_ecalDigis_ebDigis_RECO.", &wcpuEB);
  rt->SetBranchAddress("EEDigiCollection_ecalDigis_eeDigis_RECO.", &wcpuEE);

  // accumulate
  auto const nentries = rt->GetEntries();
  std::cout << ">>> nentries = " << nentries << std::endl;
  for (int ie = 0; ie < nentries; ++ie) {
    rt->GetEntry(ie);

    auto const ngpuebs = wgpuEB->bareProduct().size();
    auto const ncpuebs = wcpuEB->bareProduct().size();
    auto const ngpuees = wgpuEE->bareProduct().size();
    auto const ncpuees = wcpuEE->bareProduct().size();

    if (ngpuebs != ncpuebs or ngpuees != ncpuees) {
      std::cerr << "*** mismatch in ndigis: "
                << "ie = " << ie << "  ngpuebs = " << ngpuebs << "  ncpuebs = " << ncpuebs << "  ngpuees = " << ngpuees
                << "  ncpuees = " << ncpuees << std::endl;

      // this is a must for now
      //assert(ngpuebs==ncpuebs);
      //assert(ngpuees==ncpuees);
    }

    // assume identical sizes
    auto const& idsgpuEB = wgpuEB->bareProduct().ids();
    auto const& datagpuEB = wgpuEB->bareProduct().data();
    auto const& idscpuEB = wcpuEB->bareProduct().ids();
    auto const& datacpuEB = wcpuEB->bareProduct().data();
    for (uint32_t ieb = 0; ieb < ngpuebs; ++ieb) {
      auto const& idgpu = idsgpuEB[ieb];
      auto iter2idcpu = std::find(idscpuEB.begin(), idscpuEB.end(), idgpu);
      // FIXME
      assert(idgpu == *iter2idcpu);

      auto const ptrdiff = iter2idcpu - idscpuEB.begin();
      for (uint32_t s = 0u; s < 10u; s++) {
        EcalMGPASample sampleGPU{datagpuEB[ieb * 10 + s]};
        EcalMGPASample sampleCPU{datacpuEB[ptrdiff * 10 + s]};

        hADCEBGPU->Fill(sampleGPU.adc());
        hGainEBGPU->Fill(sampleGPU.gainId());
        hADCEBCPU->Fill(sampleCPU.adc());
        hGainEBCPU->Fill(sampleCPU.gainId());
        hADCEBGPUvsCPU->Fill(sampleCPU.adc(), sampleGPU.adc());
        hGainEBGPUvsCPU->Fill(sampleCPU.gainId(), sampleGPU.gainId());
      }
    }

    auto const& idsgpuEE = wgpuEE->bareProduct().ids();
    auto const& datagpuEE = wgpuEE->bareProduct().data();
    auto const& idscpuEE = wcpuEE->bareProduct().ids();
    auto const& datacpuEE = wcpuEE->bareProduct().data();
    for (uint32_t iee = 0; iee < ngpuees; ++iee) {
      auto const& idgpu = idsgpuEE[iee];
      auto iter2idcpu = std::find(idscpuEE.begin(), idscpuEE.end(), idgpu);
      // FIXME
      assert(idgpu == *iter2idcpu);

      // get the digis
      auto const ptrdiff = iter2idcpu - idscpuEE.begin();
      for (uint32_t s = 0u; s < 10u; s++) {
        EcalMGPASample sampleGPU{datagpuEE[iee * 10 + s]};
        EcalMGPASample sampleCPU{datacpuEE[ptrdiff * 10 + s]};

        hADCEEGPU->Fill(sampleGPU.adc());
        hGainEEGPU->Fill(sampleGPU.gainId());
        hADCEECPU->Fill(sampleCPU.adc());
        hGainEECPU->Fill(sampleCPU.gainId());
        hADCEEGPUvsCPU->Fill(sampleCPU.adc(), sampleGPU.adc());
        hGainEEGPUvsCPU->Fill(sampleCPU.gainId(), sampleGPU.gainId());
      }
    }
  }

  {
    TCanvas c{"plots", "plots", 4200, 6200};
    c.Divide(2, 4);
    c.cd(1);
    {
      gPad->SetLogy();
      hADCEBCPU->SetLineColor(kBlack);
      hADCEBCPU->SetLineWidth(1.);
      hADCEBCPU->Draw("");
      hADCEBGPU->SetLineColor(kBlue);
      hADCEBGPU->SetLineWidth(1.);
      hADCEBGPU->Draw("sames");
      gPad->Update();
      auto stats = (TPaveStats*)hADCEBGPU->FindObject("stats");
      auto y2 = stats->GetY2NDC();
      auto y1 = stats->GetY1NDC();
      stats->SetY2NDC(y1);
      stats->SetY1NDC(y1 - (y2 - y1));
    }
    c.cd(2);
    {
      gPad->SetLogy();
      hADCEECPU->SetLineColor(kBlack);
      hADCEECPU->SetLineWidth(1.);
      hADCEECPU->Draw("");
      hADCEEGPU->SetLineColor(kBlue);
      hADCEEGPU->SetLineWidth(1.);
      hADCEEGPU->Draw("sames");
      gPad->Update();
      auto stats = (TPaveStats*)hADCEEGPU->FindObject("stats");
      auto y2 = stats->GetY2NDC();
      auto y1 = stats->GetY1NDC();
      stats->SetY2NDC(y1);
      stats->SetY1NDC(y1 - (y2 - y1));
    }
    c.cd(3);
    {
      gPad->SetLogy();
      hGainEBCPU->SetLineColor(kBlack);
      hGainEBCPU->SetLineWidth(1.);
      hGainEBCPU->Draw("");
      hGainEBGPU->SetLineColor(kBlue);
      hGainEBGPU->SetLineWidth(1.);
      hGainEBGPU->Draw("sames");
      gPad->Update();
      auto stats = (TPaveStats*)hGainEBGPU->FindObject("stats");
      auto y2 = stats->GetY2NDC();
      auto y1 = stats->GetY1NDC();
      stats->SetY2NDC(y1);
      stats->SetY1NDC(y1 - (y2 - y1));
    }
    c.cd(4);
    {
      gPad->SetLogy();
      hGainEECPU->SetLineColor(kBlack);
      hGainEECPU->SetLineWidth(1.);
      hGainEECPU->Draw("");
      hGainEEGPU->SetLineColor(kBlue);
      hGainEEGPU->SetLineWidth(1.);
      hGainEEGPU->Draw("sames");
      gPad->Update();
      auto stats = (TPaveStats*)hGainEEGPU->FindObject("stats");
      auto y2 = stats->GetY2NDC();
      auto y1 = stats->GetY1NDC();
      stats->SetY2NDC(y1);
      stats->SetY1NDC(y1 - (y2 - y1));
    }
    c.cd(5);
    hADCEBGPUvsCPU->Draw("colz");
    c.cd(6);
    hADCEEGPUvsCPU->Draw("colz");
    c.cd(7);
    hGainEBGPUvsCPU->Draw("colz");
    c.cd(8);
    hGainEEGPUvsCPU->Draw("colz");
    c.SaveAs("plots.pdf");
  }

  rfin.Close();
  rfout.Write();
  rfout.Close();
}
