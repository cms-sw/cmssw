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
#include "CUDADataFormats/EcalRecHitSoA/interface/EcalUncalibratedRecHit.h"

#include "TStyle.h"

void setAxis(TH2D *histo) {
  histo->GetXaxis()->SetTitle("cpu");
  histo->GetYaxis()->SetTitle("gpu");
}

void setAxisDelta(TH2D *histo) {
  histo->GetXaxis()->SetTitle("cpu");
  histo->GetYaxis()->SetTitle("#Delta gpu-cpu");
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cout << "run with: ./validateGPU <path to input file> <output file>\n";
    exit(0);
  }

  gStyle->SetOptStat("ourme");

  edm::Wrapper<ecal::UncalibratedRecHit<calo::common::VecStoragePolicy<calo::common::CUDAHostAllocatorAlias>>> *wgpuEB =
      nullptr;
  edm::Wrapper<ecal::UncalibratedRecHit<calo::common::VecStoragePolicy<calo::common::CUDAHostAllocatorAlias>>> *wgpuEE =
      nullptr;
  edm::Wrapper<EBUncalibratedRecHitCollection> *wcpuEB = nullptr;
  edm::Wrapper<EEUncalibratedRecHitCollection> *wcpuEE = nullptr;

  std::string fileName = argv[1];
  std::string outFileName = argv[2];

  // output
  TFile rfout{outFileName.c_str(), "recreate"};

  int nbins_count = 200;
  float last_count = 5000.;
  int nbins_count_delta = 201;

  int nbins = 300;
  float last = 3000.;

  //     int nbins_chi2 = 1000;
  //     float last_chi2 = 1000.;
  int nbins_chi2 = 1000;
  float last_chi2 = 200.;

  int nbins_flags = 100;
  float last_flags = 100.;
  float delta_flags = 20;

  int nbins_delta = 201;  // use an odd number to center around 0
  float delta = 0.2;

  // RecHits plots for EB and EE on both GPU and CPU
  auto hRechitsEBGPU = new TH1D("RechitsEBGPU", "RechitsEBGPU; No. of Rechits", nbins_count, 0, last_count);
  auto hRechitsEBCPU = new TH1D("RechitsEBCPU", "RechitsEBCPU; No. of Rechits", nbins_count, 0, last_count);
  auto hRechitsEEGPU = new TH1D("RechitsEEGPU", "RechitsEEGPU; No. of Rechits", nbins_count, 0, last_count);
  auto hRechitsEECPU = new TH1D("RechitsEECPU", "RechitsEECPU; No. of Rechits", nbins_count, 0, last_count);
  auto hRechitsEBGPUCPUratio = new TH1D("RechitsEBGPU/CPUratio", "RechitsEBGPU/CPUratio; GPU/CPU", 50, 0.9, 1.1);
  auto hRechitsEEGPUCPUratio = new TH1D("RechitsEEGPU/CPUratio", "RechitsEEGPU/CPUratio; GPU/CPU", 50, 0.9, 1.1);

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
  auto hChi2EBGPUCPUratio = new TH1D("Chi2EBGPU/CPUratio", "Chi2EBGPU/CPUratio; GPU/CPU", 200, 0.9, 1.1);
  auto hChi2EEGPUCPUratio = new TH1D("Chi2EEGPU/CPUratio", "Chi2EEGPU/CPUratio; GPU/CPU", 200, 0.9, 1.1);

  auto hFlagsEBGPU = new TH1D("hFlagsEBGPU", "hFlagsEBGPU", nbins_flags, 0, last_flags);
  auto hFlagsEEGPU = new TH1D("hFlagsEEGPU", "hFlagsEEGPU", nbins_flags, 0, last_flags);
  auto hFlagsEBCPU = new TH1D("hFlagsEBCPU", "hFlagsEBCPU", nbins_flags, 0, last_flags);
  auto hFlagsEECPU = new TH1D("hFlagsEECPU", "hFlagsEECPU", nbins_flags, 0, last_flags);
  auto hFlagsEBGPUCPUratio = new TH1D("FlagsEBGPU/CPUratio", "FlagsEBGPU/CPUratio; GPU/CPU", 200, 0.9, 1.1);
  auto hFlagsEEGPUCPUratio = new TH1D("FlagsEEGPU/CPUratio", "FlagsEEGPU/CPUratio; GPU/CPU", 200, 0.9, 1.1);

  auto hSOIAmplitudesEBGPUvsCPU =
      new TH2D("hSOIAmplitudesEBGPUvsCPU", "hSOIAmplitudesEBGPUvsCPU", nbins, 0, last, nbins, 0, last);
  setAxis(hSOIAmplitudesEBGPUvsCPU);
  auto hSOIAmplitudesEEGPUvsCPU =
      new TH2D("hSOIAmplitudesEEGPUvsCPU", "hSOIAmplitudesEEGPUvsCPU", nbins, 0, last, nbins, 0, last);
  setAxis(hSOIAmplitudesEEGPUvsCPU);
  auto hSOIAmplitudesEBdeltavsCPU =
      new TH2D("hSOIAmplitudesEBdeltavsCPU", "hSOIAmplitudesEBdeltavsCPU", nbins, 0, last, nbins_delta, -delta, delta);
  setAxisDelta(hSOIAmplitudesEBdeltavsCPU);
  auto hSOIAmplitudesEEdeltavsCPU =
      new TH2D("hSOIAmplitudesEEdeltavsCPU", "hSOIAmplitudesEEdeltavsCPU", nbins, 0, last, nbins_delta, -delta, delta);
  setAxisDelta(hSOIAmplitudesEEdeltavsCPU);

  auto hChi2EBGPUvsCPU =
      new TH2D("hChi2EBGPUvsCPU", "hChi2EBGPUvsCPU", nbins_chi2, 0, last_chi2, nbins_chi2, 0, last_chi2);
  setAxis(hChi2EBGPUvsCPU);
  auto hChi2EEGPUvsCPU =
      new TH2D("hChi2EEGPUvsCPU", "hChi2EEGPUvsCPU", nbins_chi2, 0, last_chi2, nbins_chi2, 0, last_chi2);
  setAxis(hChi2EEGPUvsCPU);
  auto hChi2EBdeltavsCPU =
      new TH2D("hChi2EBdeltavsCPU", "hChi2EBdeltavsCPU", nbins_chi2, 0, last_chi2, nbins_delta, -delta, delta);
  setAxisDelta(hChi2EBdeltavsCPU);
  auto hChi2EEdeltavsCPU =
      new TH2D("hChi2EEdeltavsCPU", "hChi2EEdeltavsCPU", nbins_chi2, 0, last_chi2, nbins_delta, -delta, delta);
  setAxisDelta(hChi2EEdeltavsCPU);

  auto hFlagsEBGPUvsCPU =
      new TH2D("hFlagsEBGPUvsCPU", "hFlagsEBGPUvsCPU", nbins_flags, 0, last_flags, nbins_flags, 0, last_flags);
  setAxis(hFlagsEBGPUvsCPU);
  auto hFlagsEEGPUvsCPU =
      new TH2D("hFlagsEEGPUvsCPU", "hFlagsEEGPUvsCPU", nbins_flags, 0, last_flags, nbins_flags, 0, last_flags);
  setAxis(hFlagsEEGPUvsCPU);
  auto hFlagsEBdeltavsCPU = new TH2D(
      "hFlagsEBdeltavsCPU", "hFlagsEBdeltavsCPU", nbins_flags, 0, last_flags, nbins_delta, -delta_flags, delta_flags);
  setAxisDelta(hFlagsEBdeltavsCPU);
  auto hFlagsEEdeltavsCPU = new TH2D(
      "hFlagsEEdeltavsCPU", "hFlagsEEdeltavsCPU", nbins_flags, 0, last_flags, nbins_delta, -delta_flags, delta_flags);
  setAxisDelta(hFlagsEEdeltavsCPU);

  auto hRechitsEBGPUvsCPU = new TH2D(
      "RechitsEBGPUvsCPU", "RechitsEBGPUvsCPU; CPU; GPU", last_count, 0, last_count, last_count, 0, last_count);
  setAxis(hRechitsEBGPUvsCPU);
  auto hRechitsEEGPUvsCPU = new TH2D(
      "RechitsEEGPUvsCPU", "RechitsEEGPUvsCPU; CPU; GPU", last_count, 0, last_count, last_count, 0, last_count);
  setAxis(hRechitsEEGPUvsCPU);
  auto hRechitsEBdeltavsCPU = new TH2D(
      "RechitsEBdeltavsCPU", "RechitsEBdeltavsCPU", nbins_count, 0, last_count, nbins_count_delta, -delta, delta);
  setAxisDelta(hRechitsEBdeltavsCPU);
  auto hRechitsEEdeltavsCPU = new TH2D(
      "RechitsEEdeltavsCPU", "RechitsEEdeltavsCPU", nbins_count, 0, last_count, nbins_count_delta, -delta, delta);
  setAxisDelta(hRechitsEEdeltavsCPU);

  // input
  std::cout << "validating file " << fileName << std::endl;
  TFile rf{fileName.c_str()};
  TTree *rt = (TTree *)rf.Get("Events");
  rt->SetBranchAddress(
      "calocommonCUDAHostAllocatorAliascalocommonVecStoragePolicyecalUncalibratedRecHit_ecalCPUUncalibRecHitProducer_"
      "EcalUncalibRecHitsEB_RECO.",
      &wgpuEB);
  rt->SetBranchAddress(
      "calocommonCUDAHostAllocatorAliascalocommonVecStoragePolicyecalUncalibratedRecHit_ecalCPUUncalibRecHitProducer_"
      "EcalUncalibRecHitsEE_RECO.",
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

    float eb_ratio = (float)gpu_eb_size / cpu_eb_size;
    float ee_ratio = (float)gpu_ee_size / cpu_ee_size;

    // Filling up the histograms on events sizes for EB and EE on both GPU and CPU
    hRechitsEBGPU->Fill(gpu_eb_size);
    hRechitsEBCPU->Fill(cpu_eb_size);
    hRechitsEEGPU->Fill(gpu_ee_size);
    hRechitsEECPU->Fill(cpu_ee_size);
    hRechitsEBGPUvsCPU->Fill(cpu_eb_size, gpu_eb_size);
    hRechitsEEGPUvsCPU->Fill(cpu_ee_size, gpu_ee_size);
    hRechitsEBGPUCPUratio->Fill(eb_ratio);
    hRechitsEEGPUCPUratio->Fill(ee_ratio);
    hRechitsEBdeltavsCPU->Fill(cpu_eb_size, gpu_eb_size - cpu_eb_size);
    hRechitsEEdeltavsCPU->Fill(cpu_ee_size, gpu_ee_size - cpu_ee_size);

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

      auto const flags_gpu = wgpuEB->bareProduct().flags[i];
      auto const flags_cpu = cpu_iter->flags();

      hSOIAmplitudesEBGPU->Fill(soi_amp_gpu);
      hSOIAmplitudesEBCPU->Fill(soi_amp_cpu);
      hSOIAmplitudesEBGPUvsCPU->Fill(soi_amp_cpu, soi_amp_gpu);
      hSOIAmplitudesEBdeltavsCPU->Fill(soi_amp_cpu, soi_amp_gpu - soi_amp_cpu);
      if (soi_amp_cpu > 0)
        hSOIAmplitudesEBGPUCPUratio->Fill((float)soi_amp_gpu / soi_amp_cpu);

      hChi2EBGPU->Fill(chi2_gpu);
      hChi2EBCPU->Fill(chi2_cpu);
      hChi2EBGPUvsCPU->Fill(chi2_cpu, chi2_gpu);
      hChi2EBdeltavsCPU->Fill(chi2_cpu, chi2_gpu - chi2_cpu);
      if (chi2_cpu > 0)
        hChi2EBGPUCPUratio->Fill((float)chi2_gpu / chi2_cpu);

      if (std::abs(chi2_gpu / chi2_cpu - 1) > 0.05 || std::abs(soi_amp_gpu / soi_amp_cpu - 1) > 0.05) {
        std::cout << " ---- EB  " << std::endl;
        std::cout << " eventid = " << ie << " xtal = " << i << std::endl;
        std::cout << " chi2_gpu    = " << chi2_gpu << " chi2_cpu =    " << chi2_cpu << std::endl;
        std::cout << " soi_amp_gpu = " << soi_amp_gpu << " soi_amp_cpu = " << soi_amp_cpu << std::endl;
        std::cout << " flags_gpu   = " << flags_gpu << " flags_cpu =   " << flags_cpu << std::endl;
      }

      hFlagsEBGPU->Fill(flags_gpu);
      hFlagsEBCPU->Fill(flags_cpu);
      hFlagsEBGPUvsCPU->Fill(flags_cpu, flags_gpu);
      hFlagsEBdeltavsCPU->Fill(flags_cpu, flags_gpu - flags_cpu);
      if (flags_cpu > 0)
        hFlagsEBGPUCPUratio->Fill((float)flags_gpu / flags_cpu);

      if (flags_cpu != flags_gpu) {
        std::cout << "    >>  No! Different flag cpu:gpu = " << flags_cpu << " : " << flags_gpu;
        std::cout << std::endl;
      }

      if ((std::abs(soi_amp_gpu - soi_amp_cpu) >= eps_diff) or (std::abs(chi2_gpu - chi2_cpu) >= eps_diff) or
          std::isnan(chi2_gpu) or (flags_cpu != flags_gpu)) {
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

      auto const flags_gpu = wgpuEE->bareProduct().flags[i];
      auto const flags_cpu = cpu_iter->flags();

      hSOIAmplitudesEEGPU->Fill(soi_amp_gpu);
      hSOIAmplitudesEECPU->Fill(soi_amp_cpu);
      hSOIAmplitudesEEGPUvsCPU->Fill(soi_amp_cpu, soi_amp_gpu);
      hSOIAmplitudesEEdeltavsCPU->Fill(soi_amp_cpu, soi_amp_gpu - soi_amp_cpu);
      if (soi_amp_cpu > 0)
        hSOIAmplitudesEEGPUCPUratio->Fill((float)soi_amp_gpu / soi_amp_cpu);

      hChi2EEGPU->Fill(chi2_gpu);
      hChi2EECPU->Fill(chi2_cpu);
      hChi2EEGPUvsCPU->Fill(chi2_cpu, chi2_gpu);
      hChi2EEdeltavsCPU->Fill(chi2_cpu, chi2_gpu - chi2_cpu);
      if (chi2_cpu > 0)
        hChi2EEGPUCPUratio->Fill((float)chi2_gpu / chi2_cpu);

      if (std::abs(chi2_gpu / chi2_cpu - 1) > 0.05 || std::abs(soi_amp_gpu / soi_amp_cpu - 1) > 0.05) {
        std::cout << " ---- EE  " << std::endl;
        std::cout << " eventid = " << ie << " xtal = " << i << std::endl;
        std::cout << " chi2_gpu    = " << chi2_gpu << " chi2_cpu =    " << chi2_cpu << std::endl;
        std::cout << " soi_amp_gpu = " << soi_amp_gpu << " soi_amp_cpu = " << soi_amp_cpu << std::endl;
        std::cout << " flags_gpu   = " << flags_gpu << " flags_cpu =   " << flags_cpu << std::endl;
      }

      hFlagsEEGPU->Fill(flags_gpu);
      hFlagsEECPU->Fill(flags_cpu);
      hFlagsEEGPUvsCPU->Fill(flags_cpu, flags_gpu);
      hFlagsEEdeltavsCPU->Fill(flags_cpu, flags_gpu - flags_cpu);
      if (flags_cpu > 0)
        hFlagsEEGPUCPUratio->Fill((float)flags_gpu / flags_cpu);

      if (flags_cpu != flags_gpu) {
        std::cout << "    >>  No! Different flag cpu:gpu = " << flags_cpu << " : " << flags_gpu;
        std::cout << std::endl;
      }

      if ((std::abs(soi_amp_gpu - soi_amp_cpu) >= eps_diff) or (std::abs(chi2_gpu - chi2_cpu) >= eps_diff) or
          std::isnan(chi2_gpu) or (flags_cpu != flags_gpu)) {
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
    TCanvas c("plots", "plots", 1750, 860);
    c.Divide(3, 2);

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

    c.cd(4);
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

    c.cd(2);
    gPad->SetGrid();
    hSOIAmplitudesEBGPUvsCPU->Draw("COLZ");

    c.cd(5);
    gPad->SetGrid();
    hSOIAmplitudesEEGPUvsCPU->Draw("COLZ");

    c.cd(3);

    hSOIAmplitudesEBGPUCPUratio->Draw("");

    c.cd(6);

    hSOIAmplitudesEEGPUCPUratio->Draw("");

    c.SaveAs("ecal-amplitudes.root");
    c.SaveAs("ecal-amplitudes.png");

    // chi2

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

    c.cd(4);
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

    c.cd(2);
    gPad->SetGrid();
    hChi2EBGPUvsCPU->Draw("COLZ");

    c.cd(5);
    gPad->SetGrid();
    hChi2EEGPUvsCPU->Draw("COLZ");

    c.cd(3);

    hChi2EBGPUCPUratio->Draw("");

    c.cd(6);

    hChi2EEGPUCPUratio->Draw("");

    c.SaveAs("ecal-chi2.root");
    c.SaveAs("ecal-chi2.png");

    // flags

    c.cd(1);
    {
      gPad->SetLogy();
      hFlagsEBCPU->SetLineColor(kBlack);
      hFlagsEBCPU->SetLineWidth(1.);
      hFlagsEBCPU->Draw("");
      hFlagsEBGPU->SetLineColor(kBlue);
      hFlagsEBGPU->SetLineWidth(1.);
      hFlagsEBGPU->Draw("sames");
      gPad->Update();
      auto stats = (TPaveStats *)hFlagsEBGPU->FindObject("stats");
      auto y2 = stats->GetY2NDC();
      auto y1 = stats->GetY1NDC();
      stats->SetY2NDC(y1);
      stats->SetY1NDC(y1 - (y2 - y1));
    }

    c.cd(4);
    {
      gPad->SetLogy();
      hFlagsEECPU->SetLineColor(kBlack);
      hFlagsEECPU->SetLineWidth(1.);
      hFlagsEECPU->Draw("");
      hFlagsEEGPU->SetLineColor(kBlue);
      hFlagsEEGPU->SetLineWidth(1.);
      hFlagsEEGPU->Draw("sames");
      gPad->Update();
      auto stats = (TPaveStats *)hFlagsEEGPU->FindObject("stats");
      auto y2 = stats->GetY2NDC();
      auto y1 = stats->GetY1NDC();
      stats->SetY2NDC(y1);
      stats->SetY1NDC(y1 - (y2 - y1));
    }

    c.cd(2);
    gPad->SetGrid();
    hFlagsEBGPUvsCPU->Draw("COLZ");

    c.cd(5);
    gPad->SetGrid();
    hFlagsEEGPUvsCPU->Draw("COLZ");

    c.cd(3);
    hFlagsEBGPUCPUratio->Draw("");

    c.cd(6);
    hFlagsEEGPUCPUratio->Draw("");

    c.SaveAs("ecal-flags.root");
    c.SaveAs("ecal-flags.png");

    TCanvas cRechits("Rechits", "Rechits", 1750, 860);
    cRechits.Divide(3, 2);

    // Plotting the sizes of GPU vs CPU for each event of EB
    cRechits.cd(1);
    {
      gPad->SetLogy();
      hRechitsEBCPU->SetLineColor(kRed);
      hRechitsEBCPU->SetLineWidth(2);
      hRechitsEBCPU->Draw("");
      hRechitsEBGPU->SetLineColor(kBlue);
      hRechitsEBGPU->SetLineWidth(2);
      hRechitsEBGPU->Draw("sames");
      cRechits.Update();
      auto stats = (TPaveStats *)hRechitsEBGPU->FindObject("stats");
      auto y2 = stats->GetY2NDC();
      auto y1 = stats->GetY1NDC();
      stats->SetY2NDC(y1);
      stats->SetY1NDC(y1 - (y2 - y1));
    }
    cRechits.cd(4);
    {
      gPad->SetLogy();
      hRechitsEECPU->SetLineColor(kRed);
      hRechitsEECPU->SetLineWidth(2);
      hRechitsEECPU->Draw("");
      hRechitsEEGPU->SetLineColor(kBlue);
      hRechitsEEGPU->SetLineWidth(2);
      hRechitsEEGPU->Draw("sames");
      cRechits.Update();
      auto stats = (TPaveStats *)hRechitsEEGPU->FindObject("stats");
      auto y2 = stats->GetY2NDC();
      auto y1 = stats->GetY1NDC();
      stats->SetY2NDC(y1);
      stats->SetY1NDC(y1 - (y2 - y1));
    }
    cRechits.cd(2);
    { hRechitsEBGPUvsCPU->Draw("COLZ"); }
    cRechits.cd(5);
    { hRechitsEEGPUvsCPU->Draw("COLZ"); }
    cRechits.cd(3);
    {
      gPad->SetLogy();
      hRechitsEBGPUCPUratio->Draw("");
    }
    cRechits.cd(6);
    {
      gPad->SetLogy();
      hRechitsEEGPUCPUratio->Draw("");
    }
    cRechits.SaveAs("ecal-rechits.root");
    cRechits.SaveAs("ecal-rechits.png");
  }

  rf.Close();
  rfout.Write();
  rfout.Close();

  return 0;
}
