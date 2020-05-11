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
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "CUDADataFormats/EcalRecHitSoA/interface/EcalUncalibratedRecHit_soa.h"

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cout << "run with: ./validateGPU <path to input file> <output file>\n";
    exit(0);
  }

  edm::Wrapper<ecal::UncalibratedRecHit<ecal::Tag::soa>> *wgpuEB = nullptr;
  edm::Wrapper<ecal::UncalibratedRecHit<ecal::Tag::soa>> *wgpuEE = nullptr;
  edm::Wrapper<EBUncalibratedRecHitCollection> *wcpuEB = nullptr;
  edm::Wrapper<EEUncalibratedRecHitCollection> *wcpuEE = nullptr;

  std::string fileName = argv[1];
  std::string outFileName = argv[2];

  // output
  TFile rfout{outFileName.c_str(), "recreate"};

  int nbins = 300;
  float last = 3000.;

  int nbins_chi2 = 1000;
  float last_chi2 = 1000.;

  int nbins_delta = 201;  // use an odd number to center around 0
  float delta = 0.2;

  auto hSOIAmplitudesEBGPU = new TH1D("hSOIAmplitudesEBGPU", "hSOIAmplitudesEBGPU", nbins, 0, last);
  auto hSOIAmplitudesEEGPU = new TH1D("hSOIAmplitudesEEGPU", "hSOIAmplitudesEEGPU", nbins, 0, last);
  auto hSOIAmplitudesEBCPU = new TH1D("hSOIAmplitudesEBCPU", "hSOIAmplitudesEBCPU", nbins, 0, last);
  auto hSOIAmplitudesEECPU = new TH1D("hSOIAmplitudesEECPU", "hSOIAmplitudesEECPU", nbins, 0, last);
  auto hSOIAmplitudesEBGPUCPUratio =
      new TH1D("SOIAmplitudesEBGPU/CPUratio", "SOIAmplitudesEBGPU/CPUratio; GPU/CPU", 200, 0.9, 1.1);
  auto hSOIAmplitudesEEGPUCPUratio =
      new TH1D("SOIAmplitudesEEGPU/CPUratio", "SOIAmplitudesEEGPU/CPUratio; GPU/CPU", 200, 0.9, 1.1);

  auto hChi2EBGPU = new TH1D("hChi2EBGPU", "hChi2EBGPU", nbins_chi2, 0, last_chi2);
  auto hChi2EEGPU = new TH1D("hChi2EEGPU", "hChi2EEGPU", nbins_chi2, 0, last_chi2);
  auto hChi2EBCPU = new TH1D("hChi2EBCPU", "hChi2EBCPU", nbins_chi2, 0, last_chi2);
  auto hChi2EECPU = new TH1D("hChi2EECPU", "hChi2EECPU", nbins_chi2, 0, last_chi2);

  auto hSOIAmplitudesEBGPUvsCPU =
      new TH2D("hSOIAmplitudesEBGPUvsCPU", "hSOIAmplitudesEBGPUvsCPU", nbins, 0, last, nbins, 0, last);
  auto hSOIAmplitudesEEGPUvsCPU =
      new TH2D("hSOIAmplitudesEEGPUvsCPU", "hSOIAmplitudesEEGPUvsCPU", nbins, 0, last, nbins, 0, last);
  auto hSOIAmplitudesEBdeltavsCPU =
      new TH2D("hSOIAmplitudesEBdeltavsCPU", "hSOIAmplitudesEBdeltavsCPU", nbins, 0, last, nbins_delta, -delta, delta);
  auto hSOIAmplitudesEEdeltavsCPU =
      new TH2D("hSOIAmplitudesEEdeltavsCPU", "hSOIAmplitudesEEdeltavsCPU", nbins, 0, last, nbins_delta, -delta, delta);

  auto hChi2EBGPUvsCPU =
      new TH2D("hChi2EBGPUvsCPU", "hChi2EBGPUvsCPU", nbins_chi2, 0, last_chi2, nbins_chi2, 0, last_chi2);
  auto hChi2EEGPUvsCPU =
      new TH2D("hChi2EEGPUvsCPU", "hChi2EEGPUvsCPU", nbins_chi2, 0, last_chi2, nbins_chi2, 0, last_chi2);
  auto hChi2EBdeltavsCPU =
      new TH2D("hChi2EBdeltavsCPU", "hChi2EBdeltavsCPU", nbins_chi2, 0, last_chi2, nbins_delta, -delta, delta);
  auto hChi2EEdeltavsCPU =
      new TH2D("hChi2EEdeltavsCPU", "hChi2EEdeltavsCPU", nbins_chi2, 0, last_chi2, nbins_delta, -delta, delta);

  // input
  std::cout << "validating file " << fileName << std::endl;
  TFile rf{fileName.c_str()};
  TTree *rt = (TTree *)rf.Get("Events");
  rt->SetBranchAddress("ecalTagsoaecalUncalibratedRecHit_ecalCPUUncalibRecHitProducer_EcalUncalibRecHitsEB_RECO.",
                       &wgpuEB);
  rt->SetBranchAddress("ecalTagsoaecalUncalibratedRecHit_ecalCPUUncalibRecHitProducer_EcalUncalibRecHitsEE_RECO.",
                       &wgpuEE);
  rt->SetBranchAddress("EcalUncalibratedRecHitsSorted_ecalMultiFitUncalibRecHit_EcalUncalibRecHitsEB_RECO.", &wcpuEB);
  rt->SetBranchAddress("EcalUncalibratedRecHitsSorted_ecalMultiFitUncalibRecHit_EcalUncalibRecHitsEE_RECO.", &wcpuEE);

  constexpr float eps_diff = 1e-3;

  // accumulate
  auto const nentries = rt->GetEntries();
  std::cout << "#events to validate over: " << nentries << std::endl;
  for (int ie = 0; ie < nentries; ++ie) {
    rt->GetEntry(ie);

    const char *ordinal[] = {"th", "st", "nd", "rd", "th", "th", "th", "th", "th", "th"};
    auto cpu_eb_size = wcpuEB->bareProduct().size();
    auto cpu_ee_size = wcpuEE->bareProduct().size();
    auto gpu_eb_size = wgpuEB->bareProduct().amplitude.size();
    auto gpu_ee_size = wgpuEE->bareProduct().amplitude.size();
    if (cpu_eb_size != gpu_eb_size or cpu_ee_size != gpu_ee_size) {
      std::cerr << ie << ordinal[ie % 10] << " entry:\n"
                << "  EB size: " << std::setw(4) << cpu_eb_size << " (cpu) vs " << std::setw(4) << gpu_eb_size
                << " (gpu)\n"
                << "  EE size: " << std::setw(4) << cpu_ee_size << " (cpu) vs " << std::setw(4) << gpu_ee_size
                << " (gpu)" << std::endl;
      continue;
    }

    assert(wgpuEB->bareProduct().amplitude.size() == wcpuEB->bareProduct().size());
    assert(wgpuEE->bareProduct().amplitude.size() == wcpuEE->bareProduct().size());
    auto const neb = wcpuEB->bareProduct().size();
    auto const nee = wcpuEE->bareProduct().size();

    for (uint32_t i = 0; i < neb; ++i) {
      auto const did_gpu = wgpuEB->bareProduct().did[i];
      auto const soi_amp_gpu = wgpuEB->bareProduct().amplitude[i];
      auto const cpu_iter = wcpuEB->bareProduct().find(DetId{did_gpu});
      if (cpu_iter == wcpuEB->bareProduct().end()) {
        std::cerr << ie << ordinal[ie % 10] << " entry\n"
                  << "  Did not find a DetId " << did_gpu << " in a CPU collection\n";
        continue;
      }
      auto const soi_amp_cpu = cpu_iter->amplitude();
      auto const chi2_gpu = wgpuEB->bareProduct().chi2[i];
      auto const chi2_cpu = cpu_iter->chi2();

      hSOIAmplitudesEBGPU->Fill(soi_amp_gpu);
      hSOIAmplitudesEBCPU->Fill(soi_amp_cpu);
      hSOIAmplitudesEBGPUvsCPU->Fill(soi_amp_cpu, soi_amp_gpu);
      hSOIAmplitudesEBdeltavsCPU->Fill(soi_amp_cpu, soi_amp_gpu - soi_amp_cpu);
      hSOIAmplitudesEBGPUCPUratio->Fill((float)soi_amp_gpu / soi_amp_cpu);
      hChi2EBGPU->Fill(chi2_gpu);
      hChi2EBCPU->Fill(chi2_cpu);
      hChi2EBGPUvsCPU->Fill(chi2_cpu, chi2_gpu);
      hChi2EBdeltavsCPU->Fill(chi2_cpu, chi2_gpu - chi2_cpu);

      if ((std::abs(soi_amp_gpu - soi_amp_cpu) >= eps_diff) or (std::abs(chi2_gpu - chi2_cpu) >= eps_diff) or
          std::isnan(chi2_gpu)) {
        printf("EB eventid = %d chid = %d amp_gpu = %f amp_cpu %f chi2_gpu = %f chi2_cpu = %f\n",
               ie,
               i,
               soi_amp_gpu,
               soi_amp_cpu,
               chi2_gpu,
               chi2_cpu);
        if (std::isnan(chi2_gpu))
          printf("*** nan ***\n");
      }
    }

    for (uint32_t i = 0; i < nee; ++i) {
      auto const did_gpu = wgpuEE->bareProduct().did[i];
      auto const soi_amp_gpu = wgpuEE->bareProduct().amplitude[i];
      auto const cpu_iter = wcpuEE->bareProduct().find(DetId{did_gpu});
      if (cpu_iter == wcpuEE->bareProduct().end()) {
        std::cerr << ie << ordinal[ie % 10] << " entry\n"
                  << "  did not find a DetId " << did_gpu << " in a CPU collection\n";
        continue;
      }
      auto const soi_amp_cpu = cpu_iter->amplitude();
      auto const chi2_gpu = wgpuEE->bareProduct().chi2[i];
      auto const chi2_cpu = cpu_iter->chi2();

      hSOIAmplitudesEEGPU->Fill(soi_amp_gpu);
      hSOIAmplitudesEECPU->Fill(soi_amp_cpu);
      hSOIAmplitudesEEGPUvsCPU->Fill(soi_amp_cpu, soi_amp_gpu);
      hSOIAmplitudesEEdeltavsCPU->Fill(soi_amp_cpu, soi_amp_gpu - soi_amp_cpu);
      hSOIAmplitudesEEGPUCPUratio->Fill((float)soi_amp_gpu / soi_amp_cpu);
      hChi2EEGPU->Fill(chi2_gpu);
      hChi2EECPU->Fill(chi2_cpu);
      hChi2EEGPUvsCPU->Fill(chi2_cpu, chi2_gpu);
      hChi2EEdeltavsCPU->Fill(chi2_cpu, chi2_gpu - chi2_cpu);

      if ((std::abs(soi_amp_gpu - soi_amp_cpu) >= eps_diff) or (std::abs(chi2_gpu - chi2_cpu) >= eps_diff) or
          std::isnan(chi2_gpu)) {
        printf("EE eventid = %d chid = %d amp_gpu = %f amp_cpu %f chi2_gpu = %f chi2_cpu = %f\n",
               ie,
               static_cast<int>(neb + i),
               soi_amp_gpu,
               soi_amp_cpu,
               chi2_gpu,
               chi2_cpu);
        if (std::isnan(chi2_gpu))
          printf("*** nan ***\n");
      }
    }
  }

  {
    TCanvas c("plots", "plots", 4200, 6200);
    c.Divide(2, 4);

    c.cd(1);
    {
      gPad->SetLogy();
      hSOIAmplitudesEBCPU->SetLineColor(kBlack);
      hSOIAmplitudesEBCPU->SetLineWidth(1.);
      hSOIAmplitudesEBCPU->Draw("");
      hSOIAmplitudesEBGPU->SetLineColor(kBlue);
      hSOIAmplitudesEBGPU->SetLineWidth(1.);
      hSOIAmplitudesEBGPU->Draw("sames");
      gPad->Update();
      auto stats = (TPaveStats *)hSOIAmplitudesEBGPU->FindObject("stats");
      auto y2 = stats->GetY2NDC();
      auto y1 = stats->GetY1NDC();
      stats->SetY2NDC(y1);
      stats->SetY1NDC(y1 - (y2 - y1));
    }
    c.cd(2);
    {
      gPad->SetLogy();
      hSOIAmplitudesEECPU->SetLineColor(kBlack);
      hSOIAmplitudesEECPU->SetLineWidth(1.);
      hSOIAmplitudesEECPU->Draw("");
      hSOIAmplitudesEEGPU->SetLineColor(kBlue);
      hSOIAmplitudesEEGPU->SetLineWidth(1.);
      hSOIAmplitudesEEGPU->Draw("sames");
      gPad->Update();
      auto stats = (TPaveStats *)hSOIAmplitudesEEGPU->FindObject("stats");
      auto y2 = stats->GetY2NDC();
      auto y1 = stats->GetY1NDC();
      stats->SetY2NDC(y1);
      stats->SetY1NDC(y1 - (y2 - y1));
    }
    c.cd(3);
    hSOIAmplitudesEBGPUvsCPU->Draw("COLZ");
    c.cd(4);
    hSOIAmplitudesEEGPUvsCPU->Draw("COLZ");
    c.cd(5);
    hSOIAmplitudesEBdeltavsCPU->Draw("COLZ");
    c.cd(6);
    hSOIAmplitudesEEdeltavsCPU->Draw("COLZ");
    c.cd(7);
    {
      gPad->SetLogy();
      hSOIAmplitudesEBGPUCPUratio->SetLineColor(kBlack);
      hSOIAmplitudesEBGPUCPUratio->SetLineWidth(1.);
      hSOIAmplitudesEBGPUCPUratio->Draw("");
    }
    c.cd(8);
    {
      gPad->SetLogy();
      hSOIAmplitudesEEGPUCPUratio->SetLineColor(kBlack);
      hSOIAmplitudesEEGPUCPUratio->SetLineWidth(1.);
      hSOIAmplitudesEEGPUCPUratio->Draw("");
    }

    c.SaveAs("ecal-amplitudes.pdf");
  }
  {
    TCanvas c("plots", "plots", 4200, 6200);
    c.Divide(2, 3);

    c.cd(1);
    {
      gPad->SetLogy();
      hChi2EBCPU->SetLineColor(kBlack);
      hChi2EBCPU->SetLineWidth(1.);
      hChi2EBCPU->Draw("");
      hChi2EBGPU->SetLineColor(kBlue);
      hChi2EBGPU->SetLineWidth(1.);
      hChi2EBGPU->Draw("sames");
      gPad->Update();
      auto stats = (TPaveStats *)hChi2EBGPU->FindObject("stats");
      auto y2 = stats->GetY2NDC();
      auto y1 = stats->GetY1NDC();
      stats->SetY2NDC(y1);
      stats->SetY1NDC(y1 - (y2 - y1));
    }
    c.cd(2);
    {
      gPad->SetLogy();
      hChi2EECPU->SetLineColor(kBlack);
      hChi2EECPU->SetLineWidth(1.);
      hChi2EECPU->Draw("");
      hChi2EEGPU->SetLineColor(kBlue);
      hChi2EEGPU->SetLineWidth(1.);
      hChi2EEGPU->Draw("sames");
      gPad->Update();
      auto stats = (TPaveStats *)hChi2EEGPU->FindObject("stats");
      auto y2 = stats->GetY2NDC();
      auto y1 = stats->GetY1NDC();
      stats->SetY2NDC(y1);
      stats->SetY1NDC(y1 - (y2 - y1));
    }
    c.cd(3);
    hChi2EBGPUvsCPU->Draw("COLZ");
    c.cd(4);
    hChi2EEGPUvsCPU->Draw("COLZ");
    c.cd(5);
    hChi2EBdeltavsCPU->Draw("COLZ");
    c.cd(6);
    hChi2EEdeltavsCPU->Draw("COLZ");

    c.SaveAs("ecal-chi2.pdf");
  }

  rf.Close();
  rfout.Write();
  rfout.Close();

  return 0;
}
