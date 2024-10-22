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

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/Common/interface/Wrapper.h"
//#include "CUDADataFormats/HcalRecHitSoA/interface/RecHitCollection.h"

#define CREATE_HIST_1D(varname, nbins, first, last) auto varname = new TH1D(#varname, #varname, nbins, first, last)

#define CREATE_HIST_2D(varname, nbins, first, last) \
  auto varname = new TH2D(#varname, #varname, nbins, first, last, nbins, first, last)

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout << "run with: ./<exe> <path to input file> <path to output file>\n";
    exit(0);
  }

  std::string inFileName{argv[1]};
  std::string outFileName{argv[2]};

  // branches to use
  edm::Wrapper<HBHERecHitCollection>* wcpu = nullptr;
  edm::Wrapper<HBHERecHitCollection>* wgpu = nullptr;
  //    edm::Wrapper<hcal::RecHitCollection<calo::common::VecStoragePolicy<calo::common::CUDAHostAllocatorAlias>>> *wgpu=nullptr;

  // prep output
  TFile rfout{outFileName.c_str(), "recreate"};

  CREATE_HIST_1D(hEnergyM0HBGPU, 1000, 0, 100);
  CREATE_HIST_1D(hEnergyM0HEGPU, 1000, 0, 100);
  CREATE_HIST_1D(hEnergyM0HBCPU, 1000, 0, 100);
  CREATE_HIST_1D(hEnergyM0HECPU, 1000, 0, 100);

  CREATE_HIST_1D(hEnergyHBGPU, 1000, 0, 100);
  CREATE_HIST_1D(hEnergyHBCPU, 1000, 0, 100);
  CREATE_HIST_1D(hEnergyHEGPU, 1000, 0, 100);
  CREATE_HIST_1D(hEnergyHECPU, 1000, 0, 100);

  CREATE_HIST_1D(hChi2HBGPU, 1000, 0, 100);
  CREATE_HIST_1D(hChi2HBCPU, 1000, 0, 100);
  CREATE_HIST_1D(hChi2HEGPU, 1000, 0, 100);
  CREATE_HIST_1D(hChi2HECPU, 1000, 0, 100);

  CREATE_HIST_2D(hEnergyHBGPUvsCPU, 1000, 0, 100);
  CREATE_HIST_2D(hEnergyHEGPUvsCPU, 1000, 0, 100);
  CREATE_HIST_2D(hChi2HBGPUvsCPU, 1000, 0, 100);
  CREATE_HIST_2D(hChi2HEGPUvsCPU, 1000, 0, 100);

  CREATE_HIST_2D(hEnergyM0HBGPUvsCPU, 1000, 0, 100);
  CREATE_HIST_2D(hEnergyM0HEGPUvsCPU, 1000, 0, 100);

  // prep input
  TFile rfin{inFileName.c_str()};
  TTree* rt = (TTree*)rfin.Get("Events");
  rt->SetBranchAddress("HBHERecHitsSorted_hcalCPURecHitsProducer_recHitsLegacyHBHE_RECO.", &wgpu);
  //    rt->SetBranchAddress("hcalCUDAHostAllocatorAliashcalcommonVecStoragePolicyhcalRecHitCollection_hcalCPURecHitsProducer_recHitsM0LabelOut_RECO.", &wgpu);
  rt->SetBranchAddress("HBHERecHitsSorted_hbheprereco__RECO.", &wcpu);

  // accumulate
  auto const nentries = rt->GetEntries();
  std::cout << ">>> nentries = " << nentries << std::endl;
  for (int ie = 0; ie < nentries; ++ie) {
    rt->GetEntry(ie);

    auto const& gpuProduct = wgpu->bareProduct();
    auto const& cpuProduct = wcpu->bareProduct();

    auto const ncpu = cpuProduct.size();
    auto const ngpu = gpuProduct.size();
    //        auto const ngpu = gpuProduct.energy.size();

    if (ngpu != ncpu) {
      std::cerr << "*** mismatch in number of rec hits for event " << ie << std::endl
                << ">>> ngpu = " << ngpu << std::endl
                << ">>> ncpu = " << ncpu << std::endl;
    }

    for (uint32_t ich = 0; ich < ncpu; ich++) {
      auto const& cpurh = cpuProduct[ich];
      auto const& did = cpurh.id();
      auto iter2gpu = gpuProduct.find(did);
      //            auto iter2idgpu = std::find(
      //                gpuProduct.did.begin(), gpuProduct.did.end(), did.rawId());

      if (iter2gpu == gpuProduct.end()) {
        std::cerr << "missing " << did << std::endl;
        continue;
      }

      assert(iter2gpu->id().rawId() == did.rawId());

      auto const gpu_energy_m0 = iter2gpu->eraw();
      auto const cpu_energy_m0 = cpurh.eraw();
      auto const gpu_energy = iter2gpu->energy();
      auto const cpu_energy = cpurh.energy();
      auto const gpu_chi2 = iter2gpu->chi2();
      auto const cpu_chi2 = cpurh.chi2();

      if (did.subdetId() == HcalBarrel) {
        hEnergyM0HBGPU->Fill(gpu_energy_m0);
        hEnergyM0HBCPU->Fill(cpu_energy_m0);
        hEnergyM0HBGPUvsCPU->Fill(cpu_energy_m0, gpu_energy_m0);

        hEnergyHBGPU->Fill(gpu_energy);
        hEnergyHBCPU->Fill(cpu_energy);
        hEnergyHBGPUvsCPU->Fill(cpu_energy, gpu_energy);
        hChi2HBGPU->Fill(gpu_chi2);
        hChi2HBCPU->Fill(cpu_chi2);
        hChi2HBGPUvsCPU->Fill(cpu_chi2, gpu_chi2);
      } else if (did.subdetId() == HcalEndcap) {
        hEnergyM0HEGPU->Fill(gpu_energy_m0);
        hEnergyM0HECPU->Fill(cpu_energy_m0);
        hEnergyM0HEGPUvsCPU->Fill(cpu_energy_m0, gpu_energy_m0);

        hEnergyHEGPU->Fill(gpu_energy);
        hEnergyHECPU->Fill(cpu_energy);
        hEnergyHEGPUvsCPU->Fill(cpu_energy, gpu_energy);

        hChi2HEGPU->Fill(gpu_chi2);
        hChi2HECPU->Fill(cpu_chi2);
        hChi2HEGPUvsCPU->Fill(cpu_chi2, gpu_chi2);
      }
    }
  }

  {
    TCanvas c{"plots", "plots", 4200, 6200};
    c.Divide(4, 3);
    c.cd(1);
    {
      gPad->SetLogy();
      hEnergyM0HBCPU->SetLineColor(kBlack);
      hEnergyM0HBCPU->SetLineWidth(1.);
      hEnergyM0HBCPU->Draw("");
      hEnergyM0HBGPU->SetLineColor(kBlue);
      hEnergyM0HBGPU->SetLineWidth(1.);
      hEnergyM0HBGPU->Draw("sames");
      gPad->Update();
      auto stats = (TPaveStats*)hEnergyM0HBGPU->FindObject("stats");
      auto y2 = stats->GetY2NDC();
      auto y1 = stats->GetY1NDC();
      stats->SetY2NDC(y1);
      stats->SetY1NDC(y1 - (y2 - y1));
    }
    c.cd(2);
    {
      gPad->SetLogz();
      hEnergyM0HBGPUvsCPU->GetXaxis()->SetTitle("cpu");
      hEnergyM0HBGPUvsCPU->GetYaxis()->SetTitle("gpu");
      hEnergyM0HBGPUvsCPU->Draw("colz");
    }
    c.cd(3);
    {
      gPad->SetLogy();
      hEnergyM0HECPU->SetLineColor(kBlack);
      hEnergyM0HECPU->SetLineWidth(1.);
      hEnergyM0HECPU->Draw("");
      hEnergyM0HEGPU->SetLineColor(kBlue);
      hEnergyM0HEGPU->SetLineWidth(1.);
      hEnergyM0HEGPU->Draw("sames");
      gPad->Update();
      auto stats = (TPaveStats*)hEnergyM0HEGPU->FindObject("stats");
      auto y2 = stats->GetY2NDC();
      auto y1 = stats->GetY1NDC();
      stats->SetY2NDC(y1);
      stats->SetY1NDC(y1 - (y2 - y1));
    }
    c.cd(4);
    {
      gPad->SetLogz();
      hEnergyM0HEGPUvsCPU->GetXaxis()->SetTitle("cpu");
      hEnergyM0HEGPUvsCPU->GetYaxis()->SetTitle("gpu");
      hEnergyM0HEGPUvsCPU->Draw("colz");
    }
    c.cd(5);
    {
      gPad->SetLogy();
      hEnergyHBCPU->SetLineColor(kBlack);
      hEnergyHBCPU->SetLineWidth(1.);
      hEnergyHBCPU->Draw("");
      hEnergyHBGPU->SetLineColor(kBlue);
      hEnergyHBGPU->SetLineWidth(1.);
      hEnergyHBGPU->Draw("sames");
      gPad->Update();
      auto stats = (TPaveStats*)hEnergyHBGPU->FindObject("stats");
      auto y2 = stats->GetY2NDC();
      auto y1 = stats->GetY1NDC();
      stats->SetY2NDC(y1);
      stats->SetY1NDC(y1 - (y2 - y1));
    }
    c.cd(6);
    {
      gPad->SetLogz();
      hEnergyHBGPUvsCPU->GetXaxis()->SetTitle("cpu");
      hEnergyHBGPUvsCPU->GetYaxis()->SetTitle("gpu");
      hEnergyHBGPUvsCPU->Draw("colz");
    }
    c.cd(7);
    {
      gPad->SetLogy();
      hEnergyHECPU->SetLineColor(kBlack);
      hEnergyHECPU->SetLineWidth(1.);
      hEnergyHECPU->Draw("");
      hEnergyHEGPU->SetLineColor(kBlue);
      hEnergyHEGPU->SetLineWidth(1.);
      hEnergyHEGPU->Draw("sames");
      gPad->Update();
      auto stats = (TPaveStats*)hEnergyHEGPU->FindObject("stats");
      auto y2 = stats->GetY2NDC();
      auto y1 = stats->GetY1NDC();
      stats->SetY2NDC(y1);
      stats->SetY1NDC(y1 - (y2 - y1));
    }
    c.cd(8);
    {
      gPad->SetLogz();
      hEnergyHEGPUvsCPU->GetXaxis()->SetTitle("cpu");
      hEnergyHEGPUvsCPU->GetYaxis()->SetTitle("gpu");
      hEnergyHEGPUvsCPU->Draw("colz");
    }
    c.cd(9);
    {
      gPad->SetLogy();
      hChi2HBCPU->SetLineColor(kBlack);
      hChi2HBCPU->SetLineWidth(1.);
      hChi2HBCPU->Draw("");
      hChi2HBGPU->SetLineColor(kBlue);
      hChi2HBGPU->SetLineWidth(1.);
      hChi2HBGPU->Draw("sames");
      gPad->Update();
      auto stats = (TPaveStats*)hChi2HBGPU->FindObject("stats");
      auto y2 = stats->GetY2NDC();
      auto y1 = stats->GetY1NDC();
      stats->SetY2NDC(y1);
      stats->SetY1NDC(y1 - (y2 - y1));
    }
    c.cd(10);
    {
      gPad->SetLogz();
      hChi2HBGPUvsCPU->GetXaxis()->SetTitle("cpu");
      hChi2HBGPUvsCPU->GetYaxis()->SetTitle("gpu");
      hChi2HBGPUvsCPU->Draw("colz");
    }
    c.cd(11);
    {
      gPad->SetLogy();
      hChi2HECPU->SetLineColor(kBlack);
      hChi2HECPU->SetLineWidth(1.);
      hChi2HECPU->Draw("");
      hChi2HEGPU->SetLineColor(kBlue);
      hChi2HEGPU->SetLineWidth(1.);
      hChi2HEGPU->Draw("sames");
      gPad->Update();
      auto stats = (TPaveStats*)hChi2HEGPU->FindObject("stats");
      auto y2 = stats->GetY2NDC();
      auto y1 = stats->GetY1NDC();
      stats->SetY2NDC(y1);
      stats->SetY1NDC(y1 - (y2 - y1));
    }
    c.cd(12);
    {
      gPad->SetLogz();
      hChi2HEGPUvsCPU->GetXaxis()->SetTitle("cpu");
      hChi2HEGPUvsCPU->GetYaxis()->SetTitle("gpu");
      hChi2HEGPUvsCPU->Draw("colz");
    }
    c.SaveAs("plots.pdf");
  }

  rfin.Close();
  rfout.Write();
  rfout.Close();
}
