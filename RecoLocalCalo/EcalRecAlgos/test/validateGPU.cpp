#include <iostream>
#include <string>
#include <vector>

#include "TFile.h"
#include "TTree.h"
#include "TH1D.h"
#include "TH2D.h"

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "CUDADataFormats/EcalRecHitSoA/interface/EcalUncalibratedRecHit_soa.h"

int main(int argc, char *argv[]) {
    if (argc<3) {
        std::cout << "run with: ./validateGPU <path to input file> <output file>\n";
        exit(0);
    }

    edm::Wrapper<ecal::UncalibratedRecHit<ecal::Tag::soa>> *wgpuEB=nullptr;
    edm::Wrapper<ecal::UncalibratedRecHit<ecal::Tag::soa>> *wgpuEE=nullptr;
    edm::Wrapper<EBUncalibratedRecHitCollection> *wcpuEB = nullptr;
    edm::Wrapper<EEUncalibratedRecHitCollection> *wcpuEE = nullptr;


    std::string fileName = argv[1];
    std::string outFileName = argv[2];

    // output
    TFile rfout{outFileName.c_str(), "recreate"};
    TH1D *hSOIAmplitudesEBGPU, *hSOIAmplitudesEEGPU;
    TH1D *hSOIAmplitudesEBCPU, *hSOIAmplitudesEECPU;
    TH1D *hChi2EBGPU, *hChi2EEGPU;
    TH1D *hChi2EBCPU, *hChi2EECPU;
    TH2D *hSOIAmplitudesEBGPUvsCPU, *hSOIAmplitudesEEGPUvsCPU;
    TH2D *hChi2EBGPUvsCPU, *hChi2EEGPUvsCPU;
    TH1D *hAmplitudesEB, *hAmplitudesEE;

    int nbins = 100; int last = 1000;
    hSOIAmplitudesEBGPU = new TH1D("hSOIAmplitudesEBGPU", "hSOIAmplitudesEBGPU",
        nbins, 0, last);
    hSOIAmplitudesEEGPU = new TH1D("hSOIAmplitudesEEGPU", "hSOIAmplitudesEEGPU",
        nbins, 0, last);
    hSOIAmplitudesEBCPU = new TH1D("hSOIAmplitudesEBCPU", "hSOIAmplitudesEBCPU",
        nbins, 0, last);
    hSOIAmplitudesEECPU = new TH1D("hSOIAmplitudesEECPU", "hSOIAmplitudesEECPU",
        nbins, 0, last);
    
    int nbins_chi2 = 100; int last_chi2 = 100;
    hChi2EBGPU = new TH1D("hChi2EBGPU", "hChi2EBGPU",
        nbins_chi2, 0, last_chi2);
    hChi2EEGPU = new TH1D("hChi2EEGPU", "hChi2EEGPU",
        nbins_chi2, 0, last_chi2);
    hChi2EBCPU = new TH1D("hChi2EBCPU", "hChi2EBCPU",
        nbins_chi2, 0, last_chi2);
    hChi2EECPU = new TH1D("hChi2EECPU", "hChi2EECPU",
        nbins_chi2, 0, last_chi2);
    
    hSOIAmplitudesEBGPUvsCPU = new TH2D("hSOIAmplitudesEBGPUvsCPU", 
        "hSOIAmplitudesEBGPUvsCPU", nbins, 0, last, nbins, 0, last);
    hSOIAmplitudesEEGPUvsCPU = new TH2D("hSOIAmplitudesEEGPUvsCPU", 
        "hSOIAmplitudesEEGPUvsCPU", nbins, 0, last, nbins, 0, last);
    
    hChi2EBGPUvsCPU = new TH2D("hChi2EBGPUvsCPU", 
        "hChi2EBGPUvsCPU", nbins_chi2, 0, last_chi2, nbins_chi2, 0, last_chi2);
    hChi2EEGPUvsCPU = new TH2D("hChi2EEGPUvsCPU", 
        "hChi2EEGPUvsCPU", nbins_chi2, 0, last_chi2, nbins_chi2, 0, last_chi2);

    // input
    std::cout << "validating file " << fileName << std::endl;
    TFile rf{fileName.c_str()};
    TTree *rt = (TTree*)rf.Get("Events");
    rt->SetBranchAddress("ecalTagsoaecalUncalibratedRecHit_ecalUncalibRecHitProducerGPU_EcalUncalibRecHitsEB_RECO.", &wgpuEB);
    rt->SetBranchAddress("ecalTagsoaecalUncalibratedRecHit_ecalUncalibRecHitProducerGPU_EcalUncalibRecHitsEE_RECO.", &wgpuEE);
    rt->SetBranchAddress("EcalUncalibratedRecHitsSorted_ecalMultiFitUncalibRecHit_EcalUncalibRecHitsEB_RECO.", &wcpuEB);
    rt->SetBranchAddress("EcalUncalibratedRecHitsSorted_ecalMultiFitUncalibRecHit_EcalUncalibRecHitsEE_RECO.", &wcpuEE);

    constexpr float eps_diff = 1e-3;

    // accumulate
    auto const nentries = rt->GetEntries();
    std::cout << "#events to validate over: " << nentries << std::endl;
    for (int ie=0; ie<nentries; ++ie) {
        rt->GetEntry(ie);

        assert(wgpuEB->bareProduct().amplitude.size() == wcpuEB->bareProduct().size());
        assert(wgpuEE->bareProduct().amplitude.size() == wcpuEE->bareProduct().size());
        auto const neb = wcpuEB->bareProduct().size();
        auto const nee = wcpuEE->bareProduct().size();

        for (uint32_t i=0; i<neb; ++i) {
            auto const soi_amp_gpu = wgpuEB->bareProduct().amplitude[i];
            auto const soi_amp_cpu = wcpuEB->bareProduct()[i].amplitude();
            auto const chi2_gpu = wgpuEB->bareProduct().chi2[i];
            auto const chi2_cpu = wcpuEB->bareProduct()[i].chi2();

            hSOIAmplitudesEBGPU->Fill(soi_amp_gpu);
            hSOIAmplitudesEBCPU->Fill(soi_amp_cpu);
            hSOIAmplitudesEBGPUvsCPU->Fill(soi_amp_cpu, soi_amp_gpu);
            hChi2EBGPU->Fill(chi2_gpu);
            hChi2EBCPU->Fill(chi2_cpu);
            hChi2EBGPUvsCPU->Fill(chi2_cpu, chi2_gpu);

            if (std::abs(soi_amp_gpu - soi_amp_cpu) >= eps_diff)
                printf("eb eventid = %d chid = %d amp_gpu = %f amp_cpu %f chi2_gpu = %f chi2_cpu = %f\n",
                    ie, i, soi_amp_gpu, soi_amp_cpu, chi2_gpu, chi2_cpu);
            
            if (std::abs(chi2_gpu - chi2_cpu) >= eps_diff || std::isnan(chi2_gpu))
                printf("eb eventid = %d chid = %d amp_gpu = %f amp_cpu %f chi2_gpu = %f chi2_cpu = %f\n",
                    ie, i, soi_amp_gpu, soi_amp_cpu, chi2_gpu, chi2_cpu);
            if (std::isnan(chi2_gpu))
                printf("*** nan ***\n");
        }

        for (uint32_t i=0; i<nee; ++i) {
            auto const soi_amp_gpu = wgpuEE->bareProduct().amplitude[i];
            auto const soi_amp_cpu = wcpuEE->bareProduct()[i].amplitude();
            auto const chi2_gpu = wgpuEE->bareProduct().chi2[i];
            auto const chi2_cpu = wcpuEE->bareProduct()[i].chi2();

            hSOIAmplitudesEEGPU->Fill(soi_amp_gpu);
            hSOIAmplitudesEECPU->Fill(soi_amp_cpu);
            hSOIAmplitudesEEGPUvsCPU->Fill(soi_amp_cpu, soi_amp_gpu);
            hChi2EEGPU->Fill(chi2_gpu);
            hChi2EECPU->Fill(chi2_cpu);
            hChi2EEGPUvsCPU->Fill(chi2_cpu, chi2_gpu);
            
            if (std::abs(soi_amp_gpu - soi_amp_cpu) >= eps_diff)
                printf("ee eventid = %d chid = %d amp_gpu = %f amp_cpu %f chi2_gpu = %f chi2_cpu = %f\n",
                    ie, static_cast<int>(i+neb), soi_amp_gpu, soi_amp_cpu, chi2_gpu, chi2_cpu);
            
            if (std::abs(chi2_gpu - chi2_cpu) >= eps_diff || std::isnan(chi2_gpu))
                printf("ee eventid = %d chid = %d amp_gpu = %f amp_cpu %f chi2_gpu = %f chi2_cpu = %f\n",
                    ie, static_cast<int>(neb+i), soi_amp_gpu, soi_amp_cpu, chi2_gpu, chi2_cpu);
            if (std::isnan(chi2_gpu))
                printf("*** nan ***\n");
        }
    }

    rf.Close();
    rfout.Write();
    rfout.Close();
    return 0;
}
