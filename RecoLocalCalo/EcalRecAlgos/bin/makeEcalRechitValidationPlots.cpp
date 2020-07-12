#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>

#include <TCanvas.h>
#include <TStyle.h>
#include <TPad.h>
#include <TFile.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TTree.h>
#include <TPaveStats.h>

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "CUDADataFormats/EcalRecHitSoA/interface/EcalRecHit.h"

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cout << "run with: ./makeEcalRechitValidationPlots <path to input file> <output file>\n";
    exit(0);
  }
  // Set the GPU and CPU pointers for both EB and EE
  edm::Wrapper<ecal::RecHit<calo::common::VecStoragePolicy<calo::common::CUDAHostAllocatorAlias>>> *wgpuEB = nullptr;
  edm::Wrapper<ecal::RecHit<calo::common::VecStoragePolicy<calo::common::CUDAHostAllocatorAlias>>> *wgpuEE = nullptr;
  edm::Wrapper<EBRecHitCollection> *wcpuEB = nullptr;
  edm::Wrapper<EERecHitCollection> *wcpuEE = nullptr;

  std::string fileName = argv[1];     // The input file containing the data to be validated (i.e. result.root)
  std::string outFileName = argv[2];  //The output file in which the validation results will be saved (i.e. output.root)

  //output
  TFile rfout{outFileName.c_str(), "recreate"};

  int nbins = 200;
  int last = 5000.;

  int nbins_energy = 300;
  float last_energy = 2.;

  int nbins_chi2 = 200;
  float last_chi2 = 100.;

  int nbins_flag = 40;
  //   int nbins_flag = 1000;
  int last_flag = 1500;
  //   int nbins_flag = 40;
  //   int last_flag = 10000;

  int nbins_extra = 200;
  int last_extra = 200;

  int nbins_delta = 201;  // use an odd number to center around 0
  float delta = 0.2;

  // RecHits plots for EB and EE on both GPU and CPU
  auto hRechitsEBGPU = new TH1D("RechitsEBGPU", "RechitsEBGPU; No. of Rechits. No Filter GPU", nbins, 0, last);
  auto hRechitsEBCPU = new TH1D("RechitsEBCPU", "RechitsEBCPU; No. of Rechits. No Filter GPU", nbins, 0, last);
  auto hRechitsEEGPU = new TH1D("RechitsEEGPU", "RechitsEEGPU; No. of Rechits. No Filter GPU", nbins, 0, last);
  auto hRechitsEECPU = new TH1D("RechitsEECPU", "RechitsEECPU; No. of Rechits. No Filter GPU", nbins, 0, last);
  auto hRechitsEBGPUvsCPU =
      new TH2D("RechitsEBGPUvsCPU", "RechitsEBGPUvsCPU; CPU; GPU. No Filter GPU", last, 0, last, last, 0, last);
  auto hRechitsEEGPUvsCPU =
      new TH2D("RechitsEEGPUvsCPU", "RechitsEEGPUvsCPU; CPU; GPU. No Filter GPU", last, 0, last, last, 0, last);
  auto hRechitsEBGPUCPUratio =
      new TH1D("RechitsEBGPU/CPUratio", "RechitsEBGPU/CPUratio; GPU/CPU. No Filter GPU", 200, 0.95, 1.05);
  auto hRechitsEEGPUCPUratio =
      new TH1D("RechitsEEGPU/CPUratio", "RechitsEEGPU/CPUratio; GPU/CPU. No Filter GPU", 200, 0.95, 1.05);
  auto hRechitsEBdeltavsCPU =
      new TH2D("RechitsEBdeltavsCPU", "RechitsEBdeltavsCPU. No Filter GPU", nbins, 0, last, nbins_delta, -delta, delta);
  auto hRechitsEEdeltavsCPU =
      new TH2D("RechitsEEdeltavsCPU", "RechitsEEdeltavsCPU. No Filter GPU", nbins, 0, last, nbins_delta, -delta, delta);

  // RecHits plots for EB and EE on both GPU and CPU
  auto hSelectedRechitsEBGPU = new TH1D("RechitsEBGPU", "RechitsEBGPU; No. of Rechits", nbins, 0, last);
  auto hSelectedRechitsEBCPU = new TH1D("RechitsEBCPU", "RechitsEBCPU; No. of Rechits", nbins, 0, last);
  auto hSelectedRechitsEEGPU = new TH1D("RechitsEEGPU", "RechitsEEGPU; No. of Rechits", nbins, 0, last);
  auto hSelectedRechitsEECPU = new TH1D("RechitsEECPU", "RechitsEECPU; No. of Rechits", nbins, 0, last);
  auto hSelectedRechitsEBGPUvsCPU =
      new TH2D("RechitsEBGPUvsCPU", "RechitsEBGPUvsCPU; CPU; GPU", last, 0, last, last, 0, last);
  auto hSelectedRechitsEEGPUvsCPU =
      new TH2D("RechitsEEGPUvsCPU", "RechitsEEGPUvsCPU; CPU; GPU", last, 0, last, last, 0, last);
  auto hSelectedRechitsEBGPUCPUratio =
      new TH1D("RechitsEBGPU/CPUratio", "RechitsEBGPU/CPUratio; GPU/CPU", 200, 0.95, 1.05);
  auto hSelectedRechitsEEGPUCPUratio =
      new TH1D("RechitsEEGPU/CPUratio", "RechitsEEGPU/CPUratio; GPU/CPU", 200, 0.95, 1.05);
  auto hSelectedRechitsEBdeltavsCPU =
      new TH2D("RechitsEBdeltavsCPU", "RechitsEBdeltavsCPU", nbins, 0, last, nbins_delta, -delta, delta);
  auto hSelectedRechitsEEdeltavsCPU =
      new TH2D("RechitsEEdeltavsCPU", "RechitsEEdeltavsCPU", nbins, 0, last, nbins_delta, -delta, delta);

  // RecHits plots for EB and EE on both GPU and CPU
  auto hPositiveRechitsEBGPU = new TH1D("RechitsEBGPU", "RechitsEBGPU; No. of Rechits", nbins, 0, last);
  auto hPositiveRechitsEBCPU = new TH1D("RechitsEBCPU", "RechitsEBCPU; No. of Rechits", nbins, 0, last);
  auto hPositiveRechitsEEGPU = new TH1D("RechitsEEGPU", "RechitsEEGPU; No. of Rechits", nbins, 0, last);
  auto hPositiveRechitsEECPU = new TH1D("RechitsEECPU", "RechitsEECPU; No. of Rechits", nbins, 0, last);
  auto hPositiveRechitsEBGPUvsCPU =
      new TH2D("RechitsEBGPUvsCPU", "RechitsEBGPUvsCPU; CPU; GPU", last, 0, last, last, 0, last);
  auto hPositiveRechitsEEGPUvsCPU =
      new TH2D("RechitsEEGPUvsCPU", "RechitsEEGPUvsCPU; CPU; GPU", last, 0, last, last, 0, last);
  auto hPositiveRechitsEBGPUCPUratio =
      new TH1D("RechitsEBGPU/CPUratio", "RechitsEBGPU/CPUratio; GPU/CPU", 200, 0.95, 1.05);
  auto hPositiveRechitsEEGPUCPUratio =
      new TH1D("RechitsEEGPU/CPUratio", "RechitsEEGPU/CPUratio; GPU/CPU", 200, 0.95, 1.05);
  auto hPositiveRechitsEBdeltavsCPU =
      new TH2D("RechitsEBdeltavsCPU", "RechitsEBdeltavsCPU", nbins, 0, last, nbins_delta, -delta, delta);
  auto hPositiveRechitsEEdeltavsCPU =
      new TH2D("RechitsEEdeltavsCPU", "RechitsEEdeltavsCPU", nbins, 0, last, nbins_delta, -delta, delta);

  // Energies plots for EB and EE on both GPU and CPU
  auto hEnergiesEBGPU = new TH1D("EnergiesEBGPU", "EnergiesEBGPU; Energy [GeV]", nbins_energy, 0, last_energy);
  auto hEnergiesEEGPU = new TH1D("EnergiesEEGPU", "EnergiesEEGPU; Energy [GeV]", nbins_energy, 0, last_energy);
  auto hEnergiesEBCPU = new TH1D("EnergiesEBCPU", "EnergiesEBCPU; Energy [GeV]", nbins_energy, 0, last_energy);
  auto hEnergiesEECPU = new TH1D("EnergiesEECPU", "EnergiesEECPU; Energy [GeV]", nbins_energy, 0, last_energy);
  auto hEnergiesEBGPUvsCPU = new TH2D(
      "EnergiesEBGPUvsCPU", "EnergiesEBGPUvsCPU; CPU; GPU", nbins_energy, 0, last_energy, nbins_energy, 0, last_energy);
  auto hEnergiesEEGPUvsCPU = new TH2D(
      "EnergiesEEGPUvsCPU", "EnergiesEEGPUvsCPU; CPU; GPU", nbins_energy, 0, last_energy, nbins_energy, 0, last_energy);
  auto hEnergiesEBGPUCPUratio = new TH1D("EnergiesEBGPU/CPUratio", "EnergiesEBGPU/CPUratio; GPU/CPU", 100, 0.8, 1.2);
  auto hEnergiesEEGPUCPUratio = new TH1D("EnergiesEEGPU/CPUratio", "EnergiesEEGPU/CPUratio; GPU/CPU", 100, 0.8, 1.2);
  auto hEnergiesEBdeltavsCPU =
      new TH2D("EnergiesEBdeltavsCPU", "EnergiesEBdeltavsCPU", nbins, 0, last, nbins_delta, -delta, delta);
  auto hEnergiesEEdeltavsCPU =
      new TH2D("EnergiesEEdeltavsCPU", "EnergiesEEdeltavsCPU", nbins, 0, last, nbins_delta, -delta, delta);

  // Chi2 plots for EB and EE on both GPU and CPU
  auto hChi2EBGPU = new TH1D("Chi2EBGPU", "Chi2EBGPU; Ch^{2}", nbins_chi2, 0, last_chi2);
  auto hChi2EEGPU = new TH1D("Chi2EEGPU", "Chi2EEGPU; Ch^{2}", nbins_chi2, 0, last_chi2);
  auto hChi2EBCPU = new TH1D("Chi2EBCPU", "Chi2EBCPU; Ch^{2}", nbins_chi2, 0, last_chi2);
  auto hChi2EECPU = new TH1D("Chi2EECPU", "Chi2EECPU; Ch^{2}", nbins_chi2, 0, last_chi2);
  auto hChi2EBGPUvsCPU = new TH2D("Chi2EBGPUvsCPU", "Chi2EBGPUvsCPU; CPU; GPU", nbins_chi2, 0, 100, nbins_chi2, 0, 100);
  auto hChi2EEGPUvsCPU = new TH2D("Chi2EEGPUvsCPU", "Chi2EEGPUvsCPU; CPU; GPU", nbins_chi2, 0, 100, nbins_chi2, 0, 100);
  auto hChi2EBGPUCPUratio = new TH1D("Chi2EBGPU/CPUratio", "Chi2EBGPU/CPUratio; GPU/CPU", 100, 0.8, 1.2);
  auto hChi2EEGPUCPUratio = new TH1D("Chi2EEGPU/CPUratio", "Chi2EEGPU/CPUratio; GPU/CPU", 100, 0.8, 1.2);
  auto hChi2EBdeltavsCPU =
      new TH2D("Chi2EBdeltavsCPU", "Chi2EBdeltavsCPU", nbins_chi2, 0, last_chi2, nbins_delta, -delta, delta);
  auto hChi2EEdeltavsCPU =
      new TH2D("Chi2EEdeltavsCPU", "Chi2EEdeltavsCPU", nbins_chi2, 0, last_chi2, nbins_delta, -delta, delta);

  // Flags plots for EB and EE on both GPU and CPU
  auto hFlagsEBGPU = new TH1D("FlagsEBGPU", "FlagsEBGPU; Flags", nbins_flag, -10, last_flag);
  auto hFlagsEBCPU = new TH1D("FlagsEBCPU", "FlagsEBCPU; Flags", nbins_flag, -10, last_flag);
  auto hFlagsEEGPU = new TH1D("FlagsEEGPU", "FlagsEEGPU; Flags", nbins_flag, -10, last_flag);
  auto hFlagsEECPU = new TH1D("FlagsEECPU", "FlagsEECPU; Flags", nbins_flag, -10, last_flag);
  auto hFlagsEBGPUvsCPU =
      new TH2D("FlagsEBGPUvsCPU", "FlagsEBGPUvsCPU; CPU; GPU", nbins_flag, -10, last_flag, nbins_flag, -10, last_flag);
  auto hFlagsEEGPUvsCPU =
      new TH2D("FlagsEEGPUvsCPU", "FlagsEEGPUvsCPU; CPU; GPU", nbins_flag, -10, last_flag, nbins_flag, -10, last_flag);
  auto hFlagsEBGPUCPUratio = new TH1D("FlagsEBGPU/CPUratio", "FlagsEBGPU/CPUratio; GPU/CPU", 50, -5, 10);
  auto hFlagsEEGPUCPUratio = new TH1D("FlagsEEGPU/CPUratio", "FlagsEEGPU/CPUratio; GPU/CPU", 50, -5, 10);
  auto hFlagsEBdeltavsCPU =
      new TH2D("FlagsEBdeltavsCPU", "FlagsEBdeltavsCPU", nbins_flag, -10, last_flag, nbins_delta, -delta, delta);
  auto hFlagsEEdeltavsCPU =
      new TH2D("FlagsEEdeltavsCPU", "FlagsEEdeltavsCPU", nbins_flag, -10, last_flag, nbins_delta, -delta, delta);

  // Extras plots for EB and EE on both GPU and CPU
  auto hExtrasEBGPU = new TH1D("ExtrasEBGPU", "ExtrasEBGPU; No. of Extras", nbins_extra, 0, last_extra);
  auto hExtrasEBCPU = new TH1D("ExtrasEBCPU", "ExtrasEBCPU; No. of Extras", nbins_extra, 0, last_extra);
  auto hExtrasEEGPU = new TH1D("ExtrasEEGPU", "ExtrasEEGPU; No. of Extras", nbins_extra, 0, last_extra);
  auto hExtrasEECPU = new TH1D("ExtrasEECPU", "ExtrasEECPU; No. of Extras", nbins_extra, 0, last_extra);
  auto hExtrasEBGPUvsCPU = new TH2D(
      "ExtrasEBGPUvsCPU", "ExtrasEBGPUvsCPU; CPU; GPU", nbins_extra, 0, last_extra, nbins_extra, 0, last_extra);
  auto hExtrasEEGPUvsCPU = new TH2D(
      "ExtrasEEGPUvsCPU", "ExtrasEEGPUvsCPU; CPU; GPU", nbins_extra, 0, last_extra, nbins_extra, 0, last_extra);
  auto hExtrasEBGPUCPUratio = new TH1D("ExtrasEBGPU/CPUratio", "ExtrasEBGPU/CPUratio; GPU/CPU", 50, 0.0, 2.0);
  auto hExtrasEEGPUCPUratio = new TH1D("ExtrasEEGPU/CPUratio", "ExtrasEEGPU/CPUratio; GPU/CPU", 50, 0.0, 2.0);
  auto hExtrasEBdeltavsCPU =
      new TH2D("ExtrasEBdeltavsCPU", "ExtrasEBdeltavsCPU", nbins_extra, 0, last_extra, nbins_delta, -delta, delta);
  auto hExtrasEEdeltavsCPU =
      new TH2D("ExtrasEEdeltavsCPU", "ExtrasEEdeltavsCPU", nbins_extra, 0, last_extra, nbins_delta, -delta, delta);

  // input file setup for tree
  std::cout << "validating file " << fileName << std::endl;
  TFile rf{fileName.c_str()};
  TTree *rt = (TTree *)rf.Get("Events");

  // Allocating the appropriate data to their respective pointers
  rt->SetBranchAddress("ecalTagsoaecalRecHit_ecalCPURecHitProducer_EcalRecHitsEB_RECO.", &wgpuEB);
  rt->SetBranchAddress("ecalTagsoaecalRecHit_ecalCPURecHitProducer_EcalRecHitsEE_RECO.", &wgpuEE);
  rt->SetBranchAddress("EcalRecHitsSorted_ecalRecHit_EcalRecHitsEB_RECO.", &wcpuEB);
  rt->SetBranchAddress("EcalRecHitsSorted_ecalRecHit_EcalRecHitsEE_RECO.", &wcpuEE);

  // constexpr float eps_diff = 1e-3;

  // accumulate sizes for events and sizes of each event on both GPU and CPU
  //   auto const nentries = rt->GetEntries();
  int nentries = rt->GetEntries();

  //---- AM: tests
  if (nentries > 1000) {
    nentries = 1000;
  }
  //   nentries = 1;

  std::cout << "#events to validate over: " << nentries << std::endl;
  for (int ie = 0; ie < nentries; ++ie) {
    rt->GetEntry(ie);

    //     const char* ordinal[] = { "th", "st", "nd", "rd", "th", "th", "th", "th", "th", "th" };
    auto cpu_eb_size = wcpuEB->bareProduct().size();
    auto cpu_ee_size = wcpuEE->bareProduct().size();
    auto gpu_eb_size = wgpuEB->bareProduct().energy.size();
    auto gpu_ee_size = wgpuEE->bareProduct().energy.size();
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

    /*    
     *    // condition that sizes on GPU and CPU should be the same for EB or EE
     *       if (cpu_eb_size != gpu_eb_size or cpu_ee_size != gpu_ee_size) {
     *         std::cerr << ie << ordinal[ie % 10] << " entry:\n"
     *                   << "  EB size: " << std::setw(4) << cpu_eb_size << " (cpu) vs " << std::setw(4) << gpu_eb_size << " (gpu)\n"
     *                   << "  EE size: " << std::setw(4) << cpu_ee_size << " (cpu) vs " << std::setw(4) << gpu_ee_size << " (gpu)" << std::endl;
     *                  
     *         continue;
  }
  assert(wgpuEB->bareProduct().energy.size() == wcpuEB->bareProduct().size());
  assert(wgpuEE->bareProduct().energy.size() == wcpuEE->bareProduct().size()); 
  auto const neb = wcpuEB->bareProduct().size(); //like cpu_eb_size but set to constant
  auto const nee = wcpuEE->bareProduct().size(); //like cpu_ee_size but set to constant
  */

    uint selected_gpu_eb_size = 0;
    uint selected_gpu_ee_size = 0;

    uint positive_gpu_eb_size = 0;
    uint positive_gpu_ee_size = 0;

    // EB:
    for (uint32_t i = 0; i < gpu_eb_size; ++i) {
      auto const did_gpu = wgpuEB->bareProduct().did[i];  // set the did for the current RecHit
      // Set the variables for GPU
      auto const enr_gpu = wgpuEB->bareProduct().energy[i];
      auto const chi2_gpu = wgpuEB->bareProduct().chi2[i];
      auto const flag_gpu = wgpuEB->bareProduct().flagBits[i];
      auto const extra_gpu = wgpuEB->bareProduct().extra[i];

      // you have "-1" if the crystal is not selected
      if (enr_gpu >= 0) {
        selected_gpu_eb_size++;

        if (enr_gpu > 0) {
          positive_gpu_eb_size++;
        }

        // find the Rechit on CPU reflecting the same did
        auto const cpu_iter = wcpuEB->bareProduct().find(DetId{did_gpu});
        if (cpu_iter == wcpuEB->bareProduct().end()) {
          //           std::cerr << ie << ordinal[ie % 10] << " entry\n"
          //                   << "  Did not find a DetId " << did_gpu_eb
          //                 << " in a CPU collection\n";
          std::cerr << "  Did not find a DetId " << did_gpu << " in a CPU collection\n";
          continue;
        }
        // Set the variables for CPU
        auto const enr_cpu = cpu_iter->energy();
        auto const chi2_cpu = cpu_iter->chi2();
        //         auto const flag_cpu = cpu_iter->flagBits();
        auto const flag_cpu = 1;
        //         auto const extra_cpu = cpu_iter->extra();
        auto const extra_cpu = 1;
        //       auto const flag_cpu = cpu_iter->flagBits() ? cpu_iter->flagBits():-1;
        //       auto const extra_cpu = cpu_iter->extra() ? cpu_iter->extra():-1;

        // AM: TEST
        //       if (extra_cpu != 10) continue;

        // Fill the energy and Chi2 histograms for GPU and CPU and their comparisons with delta
        hEnergiesEBGPU->Fill(enr_gpu);
        hEnergiesEBCPU->Fill(enr_cpu);
        //       std::cout<<"EB CPU Energy:\t"<<enr_cpu<<std::endl;
        hEnergiesEBGPUvsCPU->Fill(enr_cpu, enr_gpu);
        hEnergiesEBGPUCPUratio->Fill(enr_gpu / enr_cpu);
        hEnergiesEBdeltavsCPU->Fill(enr_cpu, enr_gpu - enr_cpu);

        hChi2EBGPU->Fill(chi2_gpu);
        hChi2EBCPU->Fill(chi2_cpu);
        hChi2EBGPUvsCPU->Fill(chi2_cpu, chi2_gpu);
        hChi2EBGPUCPUratio->Fill(chi2_gpu / chi2_cpu);
        hChi2EBdeltavsCPU->Fill(chi2_cpu, chi2_gpu - chi2_cpu);

        hFlagsEBGPU->Fill(flag_gpu);
        hFlagsEBCPU->Fill(flag_cpu);
        hFlagsEBGPUvsCPU->Fill(flag_cpu, flag_gpu);
        hFlagsEBGPUCPUratio->Fill(flag_cpu ? flag_gpu / flag_cpu : -1);
        hFlagsEBdeltavsCPU->Fill(flag_cpu, flag_gpu - flag_cpu);

        hExtrasEBGPU->Fill(extra_gpu);
        hExtrasEBCPU->Fill(extra_cpu);
        hExtrasEBGPUvsCPU->Fill(extra_cpu, extra_gpu);
        hExtrasEBGPUCPUratio->Fill(extra_cpu ? extra_gpu / extra_cpu : -1);
        hExtrasEBdeltavsCPU->Fill(extra_cpu, extra_gpu - extra_cpu);

        // Check if abs difference between GPU and CPU values for energy and Chi2 are smaller than eps, if not print message
        // if ((std::abs(enr_gpu - enr_cpu) >= eps_diff) or
        //      (std::abs(chi2_gpu - chi2_cpu) >= eps_diff) or std::isnan(chi2_gpu))
        //  {
        //      printf("EB eventid = %d chid = %d energy_gpu = %f energy_cpu %f chi2_gpu = %f chi2_cpu = %f\n",
        //          ie, i, enr_gpu, enr_cpu, chi2_gpu, chi2_cpu);
        //      if (std::isnan(chi2_gpu))
        //        printf("*** nan ***\n");
        //  }
      }
    }

    // EE:
    for (uint32_t i = 0; i < gpu_ee_size; ++i) {
      auto const did_gpu = wgpuEE->bareProduct().did[i];  // set the did for the current RecHit
      // Set the variables for GPU
      auto const enr_gpu = wgpuEE->bareProduct().energy[i];
      auto const chi2_gpu = wgpuEE->bareProduct().chi2[i];
      auto const flag_gpu = wgpuEE->bareProduct().flagBits[i];
      auto const extra_gpu = wgpuEE->bareProduct().extra[i];

      // you have "-1" if the crystal is not selected
      if (enr_gpu >= 0) {
        selected_gpu_ee_size++;

        if (enr_gpu > 0) {
          positive_gpu_ee_size++;
        }

        // find the Rechit on CPU reflecting the same did
        auto const cpu_iter = wcpuEE->bareProduct().find(DetId{did_gpu});
        if (cpu_iter == wcpuEE->bareProduct().end()) {
          //    std::cerr << ie << ordinal[ie % 10] << " entry\n"
          //            << "  Did not find a DetId " << did_gpu
          //          << " in a CPU collection\n";
          std::cerr << "  Did not find a DetId " << did_gpu << " in a CPU collection\n";
          continue;
        }
        // Set the variables for CPU
        auto const enr_cpu = cpu_iter->energy();
        auto const chi2_cpu = cpu_iter->chi2();
        //         auto const flag_cpu = cpu_iter->flagBits();
        auto const flag_cpu = 1;
        //         auto const extra_cpu = cpu_iter->extra();
        auto const extra_cpu = 1;
        //       auto const flag_cpu = cpu_iter->flagBits()?cpu_iter->flagBits():-1;
        //       auto const extra_cpu = cpu_iter->extra()?cpu_iter->extra():-1;

        // AM: TEST
        //       if (extra_cpu != 10) continue;

        // Fill the energy and Chi2 histograms for GPU and CPU and their comparisons with delta
        hEnergiesEEGPU->Fill(enr_gpu);
        hEnergiesEECPU->Fill(enr_cpu);
        hEnergiesEEGPUvsCPU->Fill(enr_cpu, enr_gpu);
        hEnergiesEEGPUCPUratio->Fill(enr_gpu / enr_cpu);
        hEnergiesEEdeltavsCPU->Fill(enr_cpu, enr_gpu - enr_cpu);

        hChi2EEGPU->Fill(chi2_gpu);
        hChi2EECPU->Fill(chi2_cpu);
        hChi2EEGPUvsCPU->Fill(chi2_cpu, chi2_gpu);
        hChi2EEGPUCPUratio->Fill(chi2_gpu / chi2_cpu);
        hChi2EEdeltavsCPU->Fill(chi2_cpu, chi2_gpu - chi2_cpu);

        hFlagsEEGPU->Fill(flag_gpu);
        hFlagsEECPU->Fill(flag_cpu);
        hFlagsEEGPUvsCPU->Fill(flag_cpu, flag_gpu);
        hFlagsEEGPUCPUratio->Fill(flag_cpu ? flag_gpu / flag_cpu : -1);
        hFlagsEEdeltavsCPU->Fill(flag_cpu, flag_gpu - flag_cpu);

        hExtrasEEGPU->Fill(extra_gpu);
        hExtrasEECPU->Fill(extra_cpu);
        hExtrasEEGPUvsCPU->Fill(extra_cpu, extra_gpu);
        hExtrasEEGPUCPUratio->Fill(extra_cpu ? extra_gpu / extra_cpu : -1);
        hExtrasEEdeltavsCPU->Fill(extra_cpu, extra_gpu - extra_cpu);

        // Check if abs difference between GPU and CPU values for energy and Chi2 are smaller than eps, if not print message
        // if ((std::abs(enr_gpu - enr_cpu) >= eps_diff) or
        //      (std::abs(chi2_gpu - chi2_cpu) >= eps_diff) or std::isnan(chi2_gpu))
        //  {
        //      printf("EE eventid = %d chid = %d energy_gpu = %f energy_cpu %f chi2_gpu = %f chi2_cpu = %f\n",
        //          ie, i, enr_gpu, enr_cpu, chi2_gpu, chi2_cpu);
        //      if (std::isnan(chi2_gpu))
        //        printf("*** nan ***\n");
        //  }
      }
    }

    //
    // now the rechit counting
    //
    float selected_eb_ratio = (float)selected_gpu_eb_size / cpu_eb_size;
    float selected_ee_ratio = (float)selected_gpu_ee_size / cpu_ee_size;

    // Filling up the histograms on events sizes for EB and EE on both GPU and CPU
    hSelectedRechitsEBGPU->Fill(selected_gpu_eb_size);
    hSelectedRechitsEBCPU->Fill(cpu_eb_size);
    hSelectedRechitsEEGPU->Fill(selected_gpu_ee_size);
    hSelectedRechitsEECPU->Fill(cpu_ee_size);
    hSelectedRechitsEBGPUvsCPU->Fill(cpu_eb_size, selected_gpu_eb_size);
    hSelectedRechitsEEGPUvsCPU->Fill(cpu_ee_size, selected_gpu_ee_size);
    hSelectedRechitsEBGPUCPUratio->Fill(selected_eb_ratio);
    hSelectedRechitsEEGPUCPUratio->Fill(selected_ee_ratio);
    hSelectedRechitsEBdeltavsCPU->Fill(cpu_eb_size, selected_gpu_eb_size - cpu_eb_size);
    hSelectedRechitsEEdeltavsCPU->Fill(cpu_ee_size, selected_gpu_ee_size - cpu_ee_size);

    //
    // now the rechit counting
    //

    uint positive_cpu_eb_size = 0;
    uint positive_cpu_ee_size = 0;

    // EB:
    for (uint32_t i = 0; i < cpu_eb_size; ++i) {
      auto const enr_cpu = wcpuEB->bareProduct()[i].energy();
      if (enr_cpu > 0) {
        positive_cpu_eb_size++;
      }
    }
    // EE:
    for (uint32_t i = 0; i < cpu_ee_size; ++i) {
      auto const enr_cpu = wcpuEE->bareProduct()[i].energy();
      if (enr_cpu > 0) {
        positive_cpu_ee_size++;
      }
    }

    float positive_eb_ratio = (float)positive_gpu_eb_size / positive_cpu_eb_size;
    float positive_ee_ratio = (float)positive_gpu_ee_size / positive_cpu_ee_size;

    // Filling up the histograms on events sizes for EB and EE on both GPU and CPU
    hPositiveRechitsEBGPU->Fill(positive_gpu_eb_size);
    hPositiveRechitsEBCPU->Fill(positive_cpu_eb_size);
    hPositiveRechitsEEGPU->Fill(positive_gpu_ee_size);
    hPositiveRechitsEECPU->Fill(positive_cpu_ee_size);
    hPositiveRechitsEBGPUvsCPU->Fill(positive_cpu_eb_size, positive_gpu_eb_size);
    hPositiveRechitsEEGPUvsCPU->Fill(positive_cpu_ee_size, positive_gpu_ee_size);
    hPositiveRechitsEBGPUCPUratio->Fill(positive_eb_ratio);
    hPositiveRechitsEEGPUCPUratio->Fill(positive_ee_ratio);
    hPositiveRechitsEBdeltavsCPU->Fill(positive_cpu_eb_size, positive_gpu_eb_size - positive_cpu_eb_size);
    hPositiveRechitsEEdeltavsCPU->Fill(positive_cpu_ee_size, positive_gpu_ee_size - positive_cpu_ee_size);

    if (cpu_eb_size != selected_gpu_eb_size or cpu_ee_size != selected_gpu_ee_size) {
      //       std::cerr << ie << ordinal[ie % 10] << " entry:\n"
      std::cerr << ie << " entry:\n"
                << "  EB size: " << std::setw(4) << cpu_eb_size << " (cpu) vs " << std::setw(4) << selected_gpu_eb_size
                << " (gpu)\n"
                << "  EE size: " << std::setw(4) << cpu_ee_size << " (cpu) vs " << std::setw(4) << selected_gpu_ee_size
                << " (gpu)" << std::endl;
    }
  }

  // Plotting the results:
  {
    // Canvases Setup:
    TCanvas cAllRechits("AllRechits", "AllRechits", 1750, 860);
    cAllRechits.Divide(3, 2);
    TCanvas cRechits("Rechits", "Rechits", 1750, 860);
    cRechits.Divide(3, 2);
    TCanvas cRechitsPositive("RechitsPositive", "RechitsPositive", 1750, 860);
    cRechitsPositive.Divide(3, 2);
    TCanvas cEnergies("Energies", "Energies", 1750, 860);
    cEnergies.Divide(3, 2);
    TCanvas cChi2("Chi2", "Chi2", 1750, 860);
    cChi2.Divide(3, 2);
    TCanvas cFlags("Flags", "Flags", 1750, 860);
    cFlags.Divide(3, 2);
    TCanvas cExtras("Extras", "Extras", 1750, 860);
    cExtras.Divide(3, 2);

    // Plotting the sizes of GPU vs CPU for each event of EB
    cAllRechits.cd(1);
    {
      gPad->SetLogy();
      hRechitsEBCPU->SetLineColor(kRed);
      hRechitsEBCPU->SetLineWidth(2);
      hRechitsEBCPU->Draw("");
      hRechitsEBGPU->SetLineColor(kBlue);
      hRechitsEBGPU->SetLineWidth(2);
      hRechitsEBGPU->Draw("sames");
      cAllRechits.Update();
      auto stats = (TPaveStats *)hRechitsEBGPU->FindObject("stats");
      auto y2 = stats->GetY2NDC();
      auto y1 = stats->GetY1NDC();
      stats->SetY2NDC(y1);
      stats->SetY1NDC(y1 - (y2 - y1));
    }
    cAllRechits.cd(4);
    {
      gPad->SetLogy();
      hRechitsEECPU->SetLineColor(kRed);
      hRechitsEECPU->SetLineWidth(2);
      hRechitsEECPU->Draw("");
      hRechitsEEGPU->SetLineColor(kBlue);
      hRechitsEEGPU->SetLineWidth(2);
      hRechitsEEGPU->Draw("sames");
      cAllRechits.Update();
      auto stats = (TPaveStats *)hRechitsEEGPU->FindObject("stats");
      auto y2 = stats->GetY2NDC();
      auto y1 = stats->GetY1NDC();
      stats->SetY2NDC(y1);
      stats->SetY1NDC(y1 - (y2 - y1));
    }
    cAllRechits.cd(2);
    {
      gStyle->SetPalette(55);
      hRechitsEBGPUvsCPU->Draw("COLZ");
    }
    cAllRechits.cd(5);
    {
      gStyle->SetPalette(55);
      hRechitsEEGPUvsCPU->Draw("COLZ");
    }
    cAllRechits.cd(3);
    {
      gPad->SetLogy();
      //hRechitsEBdeltavsCPU->Draw("COLZ");
      hRechitsEBGPUCPUratio->Draw("");
    }
    cAllRechits.cd(6);
    {
      gPad->SetLogy();
      //hRechitsEEdeltavsCPU->Draw("COLZ");
      hRechitsEEGPUCPUratio->Draw("");
    }
    cAllRechits.SaveAs("ecal-allrechits.root");
    cAllRechits.SaveAs("ecal-allrechits.png");

    // Plotting the sizes of GPU vs CPU for each event of EB
    cRechits.cd(1);
    {
      gPad->SetLogy();
      hSelectedRechitsEBCPU->SetLineColor(kRed);
      hSelectedRechitsEBCPU->SetLineWidth(2);
      hSelectedRechitsEBCPU->Draw("");
      hSelectedRechitsEBGPU->SetLineColor(kBlue);
      hSelectedRechitsEBGPU->SetLineWidth(2);
      hSelectedRechitsEBGPU->Draw("sames");
      cRechits.Update();
      auto stats = (TPaveStats *)hSelectedRechitsEBGPU->FindObject("stats");
      auto y2 = stats->GetY2NDC();
      auto y1 = stats->GetY1NDC();
      stats->SetY2NDC(y1);
      stats->SetY1NDC(y1 - (y2 - y1));
    }
    cRechits.cd(4);
    {
      gPad->SetLogy();
      hSelectedRechitsEECPU->SetLineColor(kRed);
      hSelectedRechitsEECPU->SetLineWidth(2);
      hSelectedRechitsEECPU->Draw("");
      hSelectedRechitsEEGPU->SetLineColor(kBlue);
      hSelectedRechitsEEGPU->SetLineWidth(2);
      hSelectedRechitsEEGPU->Draw("sames");
      cRechits.Update();
      auto stats = (TPaveStats *)hSelectedRechitsEEGPU->FindObject("stats");
      auto y2 = stats->GetY2NDC();
      auto y1 = stats->GetY1NDC();
      stats->SetY2NDC(y1);
      stats->SetY1NDC(y1 - (y2 - y1));
    }
    cRechits.cd(2);
    {
      gStyle->SetPalette(55);
      hSelectedRechitsEBGPUvsCPU->Draw("COLZ");
    }
    cRechits.cd(5);
    {
      gStyle->SetPalette(55);
      hSelectedRechitsEEGPUvsCPU->Draw("COLZ");
    }
    cRechits.cd(3);
    {
      gPad->SetLogy();
      //hSelectedRechitsEBdeltavsCPU->Draw("COLZ");
      hSelectedRechitsEBGPUCPUratio->Draw("");
    }
    cRechits.cd(6);
    {
      gPad->SetLogy();
      //hSelectedRechitsEEdeltavsCPU->Draw("COLZ");
      hSelectedRechitsEEGPUCPUratio->Draw("");
    }
    cRechits.SaveAs("ecal-rechits.root");
    cRechits.SaveAs("ecal-rechits.png");

    // Plotting the sizes of GPU vs CPU for each event of EB
    cRechitsPositive.cd(1);
    {
      gPad->SetLogy();
      hPositiveRechitsEBCPU->SetLineColor(kRed);
      hPositiveRechitsEBCPU->SetLineWidth(2);
      hPositiveRechitsEBCPU->Draw("");
      hPositiveRechitsEBGPU->SetLineColor(kBlue);
      hPositiveRechitsEBGPU->SetLineWidth(2);
      hPositiveRechitsEBGPU->Draw("sames");
      cRechitsPositive.Update();
      auto stats = (TPaveStats *)hPositiveRechitsEBGPU->FindObject("stats");
      auto y2 = stats->GetY2NDC();
      auto y1 = stats->GetY1NDC();
      stats->SetY2NDC(y1);
      stats->SetY1NDC(y1 - (y2 - y1));
    }
    cRechitsPositive.cd(4);
    {
      gPad->SetLogy();
      hPositiveRechitsEECPU->SetLineColor(kRed);
      hPositiveRechitsEECPU->SetLineWidth(2);
      hPositiveRechitsEECPU->Draw("");
      hPositiveRechitsEEGPU->SetLineColor(kBlue);
      hPositiveRechitsEEGPU->SetLineWidth(2);
      hPositiveRechitsEEGPU->Draw("sames");
      cRechitsPositive.Update();
      auto stats = (TPaveStats *)hPositiveRechitsEEGPU->FindObject("stats");
      auto y2 = stats->GetY2NDC();
      auto y1 = stats->GetY1NDC();
      stats->SetY2NDC(y1);
      stats->SetY1NDC(y1 - (y2 - y1));
    }
    cRechitsPositive.cd(2);
    {
      gStyle->SetPalette(55);
      hPositiveRechitsEBGPUvsCPU->Draw("COLZ");
    }
    cRechitsPositive.cd(5);
    {
      gStyle->SetPalette(55);
      hPositiveRechitsEEGPUvsCPU->Draw("COLZ");
    }
    cRechitsPositive.cd(3);
    {
      gPad->SetLogy();
      //hPositiveRechitsEBdeltavsCPU->Draw("COLZ");
      hPositiveRechitsEBGPUCPUratio->Draw("");
    }
    cRechitsPositive.cd(6);
    {
      gPad->SetLogy();
      //hPositiveRechitsEEdeltavsCPU->Draw("COLZ");
      hPositiveRechitsEEGPUCPUratio->Draw("");
    }
    cRechitsPositive.SaveAs("ecal-rechits-positive.root");
    cRechitsPositive.SaveAs("ecal-rechits-positive.png");

    cEnergies.cd(1);
    {
      gPad->SetLogy();
      hEnergiesEBCPU->SetLineColor(kBlack);
      hEnergiesEBCPU->SetLineWidth(2);
      hEnergiesEBCPU->Draw("");
      hEnergiesEBGPU->SetLineColor(kBlue);
      hEnergiesEBGPU->SetLineWidth(2);
      hEnergiesEBGPU->Draw("sames");
      gPad->Update();
      auto stats = (TPaveStats *)hEnergiesEBGPU->FindObject("stats");
      auto y2 = stats->GetY2NDC();
      auto y1 = stats->GetY1NDC();
      stats->SetY2NDC(y1);
      stats->SetY1NDC(y1 - (y2 - y1));
    }
    cEnergies.cd(4);
    {
      gPad->SetLogy();
      hEnergiesEECPU->SetLineColor(kBlack);
      hEnergiesEECPU->SetLineWidth(2);
      hEnergiesEECPU->Draw("");
      hEnergiesEEGPU->SetLineColor(kBlue);
      hEnergiesEEGPU->SetLineWidth(2);
      hEnergiesEEGPU->Draw("sames");
      gPad->Update();
      auto stats = (TPaveStats *)hEnergiesEEGPU->FindObject("stats");
      auto y2 = stats->GetY2NDC();
      auto y1 = stats->GetY1NDC();
      stats->SetY2NDC(y1);
      stats->SetY1NDC(y1 - (y2 - y1));
    }
    cEnergies.cd(2);
    { hEnergiesEBGPUvsCPU->Draw("COLZ"); }
    cEnergies.cd(5);
    { hEnergiesEEGPUvsCPU->Draw("COLZ"); }
    cEnergies.cd(3);
    {
      gPad->SetLogy();
      //hEnergiesEBdeltavsCPU->Draw("COLZ");
      hEnergiesEBGPUCPUratio->Draw("");
    }
    cEnergies.cd(6);
    {
      gPad->SetLogy();
      //hEnergiesEEdeltavsCPU->Draw("COLZ");
      hEnergiesEEGPUCPUratio->Draw("");
    }
    cEnergies.SaveAs("ecal-energies.root");
    cEnergies.SaveAs("ecal-energies.png");

    cChi2.cd(1);
    {
      gPad->SetLogy();
      hChi2EBCPU->SetLineColor(kBlack);
      hChi2EBCPU->SetLineWidth(2);
      hChi2EBCPU->Draw("");
      hChi2EBGPU->SetLineColor(kBlue);
      hChi2EBGPU->SetLineWidth(2);
      hChi2EBGPU->Draw("sames");
      gPad->Update();
      auto stats = (TPaveStats *)hChi2EBGPU->FindObject("stats");
      auto y2 = stats->GetY2NDC();
      auto y1 = stats->GetY1NDC();
      stats->SetY2NDC(y1);
      stats->SetY1NDC(y1 - (y2 - y1));
    }
    cChi2.cd(4);
    {
      gPad->SetLogy();
      hChi2EECPU->SetLineColor(kBlack);
      hChi2EECPU->SetLineWidth(2);
      hChi2EECPU->Draw("");
      hChi2EEGPU->SetLineColor(kBlue);
      hChi2EEGPU->SetLineWidth(2);
      hChi2EEGPU->Draw("sames");
      gPad->Update();
      auto stats = (TPaveStats *)hChi2EEGPU->FindObject("stats");
      auto y2 = stats->GetY2NDC();
      auto y1 = stats->GetY1NDC();
      stats->SetY2NDC(y1);
      stats->SetY1NDC(y1 - (y2 - y1));
    }
    cChi2.cd(2);
    { hChi2EBGPUvsCPU->Draw("COLZ"); }
    cChi2.cd(5);
    { hChi2EEGPUvsCPU->Draw("COLZ"); }
    cChi2.cd(3);
    {
      gPad->SetLogy();
      //hChi2EBdeltavsCPU->Draw("COLZ");
      hChi2EBGPUCPUratio->Draw("");
    }
    cChi2.cd(6);
    {
      gPad->SetLogy();
      //hChi2EEdeltavsCPU->Draw("COLZ");
      hChi2EEGPUCPUratio->Draw("");
    }
    cChi2.SaveAs("ecal-chi2.root");
    cChi2.SaveAs("ecal-chi2.png");

    cFlags.cd(1);
    {
      gPad->SetLogy();
      hFlagsEBCPU->SetLineColor(kBlack);
      hFlagsEBCPU->SetLineWidth(2);
      hFlagsEBCPU->Draw("");
      hFlagsEBGPU->SetLineColor(kBlue);
      hFlagsEBGPU->SetLineWidth(2);
      hFlagsEBGPU->Draw("sames");
      gPad->Update();
      auto stats = (TPaveStats *)hFlagsEBGPU->FindObject("stats");
      auto y2 = stats->GetY2NDC();
      auto y1 = stats->GetY1NDC();
      stats->SetY2NDC(y1);
      stats->SetY1NDC(y1 - (y2 - y1));
    }
    cFlags.cd(4);
    {
      gPad->SetLogy();
      hFlagsEECPU->SetLineColor(kBlack);
      hFlagsEECPU->SetLineWidth(2);
      hFlagsEECPU->Draw("");
      hFlagsEEGPU->SetLineColor(kBlue);
      hFlagsEEGPU->SetLineWidth(2);
      hFlagsEEGPU->Draw("sames");
      gPad->Update();
      auto stats = (TPaveStats *)hFlagsEEGPU->FindObject("stats");
      auto y2 = stats->GetY2NDC();
      auto y1 = stats->GetY1NDC();
      stats->SetY2NDC(y1);
      stats->SetY1NDC(y1 - (y2 - y1));
    }
    cFlags.cd(2);
    { hFlagsEBGPUvsCPU->Draw("COLZ"); }
    cFlags.cd(5);
    { hFlagsEEGPUvsCPU->Draw("COLZ"); }
    cFlags.cd(3);
    {
      gPad->SetLogy();
      //hFlagsEBdeltavsCPU->Draw("COLZ");
      hFlagsEBGPUCPUratio->Draw("");
    }
    cFlags.cd(6);
    {
      gPad->SetLogy();
      //hFlagsEEdeltavsCPU->Draw("COLZ");
      hFlagsEEGPUCPUratio->Draw("");
    }
    cFlags.SaveAs("ecal-flags.root");
    cFlags.SaveAs("ecal-flags.png");

    cExtras.cd(1);
    {
      gPad->SetLogy();
      hExtrasEBCPU->SetLineColor(kBlack);
      hExtrasEBCPU->SetLineWidth(2);
      hExtrasEBCPU->Draw("");
      hExtrasEBGPU->SetLineColor(kBlue);
      hExtrasEBGPU->SetLineWidth(2);
      hExtrasEBGPU->Draw("sames");
      gPad->Update();
      auto stats = (TPaveStats *)hExtrasEBGPU->FindObject("stats");
      auto y2 = stats->GetY2NDC();
      auto y1 = stats->GetY1NDC();
      stats->SetY2NDC(y1);
      stats->SetY1NDC(y1 - (y2 - y1));
    }
    cExtras.cd(4);
    {
      gPad->SetLogy();
      hExtrasEECPU->SetLineColor(kBlack);
      hExtrasEECPU->SetLineWidth(2);
      hExtrasEECPU->Draw("");
      hExtrasEEGPU->SetLineColor(kBlue);
      hExtrasEEGPU->SetLineWidth(2);
      hExtrasEEGPU->Draw("sames");
      gPad->Update();
      auto stats = (TPaveStats *)hExtrasEEGPU->FindObject("stats");
      auto y2 = stats->GetY2NDC();
      auto y1 = stats->GetY1NDC();
      stats->SetY2NDC(y1);
      stats->SetY1NDC(y1 - (y2 - y1));
    }
    cExtras.cd(2);
    { hExtrasEBGPUvsCPU->Draw("COLZ"); }
    cExtras.cd(5);
    { hExtrasEEGPUvsCPU->Draw("COLZ"); }
    cExtras.cd(3);
    {
      gPad->SetLogy();
      //hExtrasEBdeltavsCPU->Draw("COLZ");
      hExtrasEBGPUCPUratio->Draw("");
    }
    cExtras.cd(6);
    {
      gPad->SetLogy();
      //hExtrasEEdeltavsCPU->Draw("COLZ");
      hExtrasEEGPUCPUratio->Draw("");
    }
    cExtras.SaveAs("ecal-extras.root");
    cExtras.SaveAs("ecal-extras.png");
  }

  // Close all open files
  rf.Close();
  rfout.Write();
  rfout.Close();

  return 0;
}
