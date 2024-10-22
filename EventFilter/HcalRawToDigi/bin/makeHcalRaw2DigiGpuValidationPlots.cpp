#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <TCanvas.h>
#include <TFile.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TPaveStats.h>
#include <TTree.h>

#include "CUDADataFormats/HcalDigi/interface/DigiCollection.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

#define CREATE_HIST_1D(varname, nbins, first, last) auto varname = new TH1D(#varname, #varname, nbins, first, last)

#define CREATE_HIST_2D(varname, nbins, first, last) \
  auto varname = new TH2D(#varname, #varname, nbins, first, last, nbins, first, last)

QIE11DigiCollection filterQIE11(QIE11DigiCollection const& coll) {
  QIE11DigiCollection out;
  out.reserve(coll.size());

  for (uint32_t i = 0; i < coll.size(); i++) {
    auto const df = coll[i];
    auto const id = HcalDetId{df.id()};
    if (id.subdetId() != HcalEndcap)
      continue;

    out.push_back(QIE11DataFrame{df});
  }

  return out;
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout << "run with: ./<exe> <path to input file> <path to output file>\n";
    exit(0);
  }

  auto filterf01HE = [](QIE11DigiCollection const& coll) {
    QIE11DigiCollection out{coll.samples(), coll.subdetId()};
    out.reserve(coll.size());

    for (uint32_t i = 0; i < coll.size(); i++) {
      auto const df = QIE11DataFrame{coll[i]};
      auto const id = HcalDetId{df.id()};
      if ((df.flavor() == 0 or df.flavor() == 1) and id.subdetId() == HcalEndcap)
        out.push_back(df);
    }

    return out;
  };

  auto filterf3HB = [](QIE11DigiCollection const& coll) {
    QIE11DigiCollection out{coll.samples(), coll.subdetId()};
    out.reserve(coll.size());

    for (uint32_t i = 0; i < coll.size(); i++) {
      auto const df = QIE11DataFrame{coll[i]};
      auto const did = HcalDetId{df.id()};
      if (df.flavor() == 3 and did.subdetId() == HcalBarrel)
        out.push_back(df);
    }

    return out;
  };

  // branches to use
  using Collectionf01 =
      hcal::DigiCollection<hcal::Flavor1, calo::common::VecStoragePolicy<calo::common::CUDAHostAllocatorAlias>>;
  using Collectionf5 =
      hcal::DigiCollection<hcal::Flavor5, calo::common::VecStoragePolicy<calo::common::CUDAHostAllocatorAlias>>;
  using Collectionf3 =
      hcal::DigiCollection<hcal::Flavor3, calo::common::VecStoragePolicy<calo::common::CUDAHostAllocatorAlias>>;
  edm::Wrapper<Collectionf01>* wgpuf01he = nullptr;
  edm::Wrapper<Collectionf5>* wgpuf5hb = nullptr;
  edm::Wrapper<Collectionf3>* wgpuf3hb = nullptr;
  edm::Wrapper<QIE11DigiCollection>* wcpuf01he = nullptr;
  edm::Wrapper<HBHEDigiCollection>* wcpuf5hb = nullptr;

  std::string inFileName{argv[1]};
  std::string outFileName{argv[2]};

  // prep output
  TFile rfout{outFileName.c_str(), "recreate"};

  CREATE_HIST_1D(hADCf01HEGPU, 256, 0, 256);
  CREATE_HIST_1D(hADCf01HECPU, 256, 0, 256);
  CREATE_HIST_1D(hADCf5HBGPU, 128, 0, 128);
  CREATE_HIST_1D(hADCf5HBCPU, 128, 0, 128);
  CREATE_HIST_1D(hADCf3HBGPU, 256, 0, 256);
  CREATE_HIST_1D(hADCf3HBCPU, 256, 0, 256);
  CREATE_HIST_1D(hTDCf01HEGPU, 64, 0, 64);
  CREATE_HIST_1D(hTDCf01HECPU, 64, 0, 64);

  CREATE_HIST_2D(hADCf01HEGPUvsCPU, 256, 0, 256);
  CREATE_HIST_2D(hADCf3HBGPUvsCPU, 256, 0, 256);
  CREATE_HIST_2D(hADCf5HBGPUvsCPU, 128, 0, 128);
  CREATE_HIST_2D(hTDCf01HEGPUvsCPU, 64, 0, 64);
  CREATE_HIST_2D(hTDCf3HBGPUvsCPU, 4, 0, 4);

  // prep input
  TFile rfin{inFileName.c_str()};
  TTree* rt = (TTree*)rfin.Get("Events");
  rt->SetBranchAddress("QIE11DataFrameHcalDataFrameContainer_hcalDigis__RECO.", &wcpuf01he);
  rt->SetBranchAddress("HBHEDataFramesSorted_hcalDigis__RECO.", &wcpuf5hb);
  rt->SetBranchAddress(
      "hcalFlavor5calocommonCUDAHostAllocatorAliascalocommonVecStoragePolicyhcalDigiCollection_hcalCPUDigisProducer_"
      "f5HBDigis_RECO.",
      &wgpuf5hb);
  rt->SetBranchAddress(
      "hcalFlavor1calocommonCUDAHostAllocatorAliascalocommonVecStoragePolicyhcalDigiCollection_hcalCPUDigisProducer_"
      "f01HEDigis_RECO.",
      &wgpuf01he);
  rt->SetBranchAddress(
      "hcalFlavor3calocommonCUDAHostAllocatorAliascalocommonVecStoragePolicyhcalDigiCollection_hcalCPUDigisProducer_"
      "f3HBDigis_RECO.",
      &wgpuf3hb);

  // accumulate
  auto const nentries = rt->GetEntries();
  std::cout << ">>> nentries = " << nentries << std::endl;
  for (int ie = 0; ie < nentries; ++ie) {
    rt->GetEntry(ie);

    auto const& f01HEProduct = wgpuf01he->bareProduct();
    auto const& f5HBProduct = wgpuf5hb->bareProduct();
    auto const& f3HBProduct = wgpuf3hb->bareProduct();
    auto const& qie11Product = wcpuf01he->bareProduct();
    auto const qie11Filteredf01 = filterf01HE(qie11Product);
    auto const qie11Filteredf3 = filterf3HB(qie11Product);
    auto const& qie8Product = wcpuf5hb->bareProduct();

    auto const ngpuf01he = f01HEProduct.ids.size();
    auto const ngpuf5hb = f5HBProduct.ids.size();
    auto const ngpuf3hb = f3HBProduct.ids.size();
    auto const ncpuf01he = qie11Filteredf01.size();
    auto const ncpuf5hb = qie8Product.size();
    auto const ncpuf3hb = qie11Filteredf3.size();

    /*
        printf("ngpuf01he = %u nqie11 = %u ncpuf01he = %u ngpuf5hb = %u ncpuf5hb = %u\n",
            f01HEProduct.size(), qie11Product.size(), qie11Filtered.size(), 
            f5HBProduct.size(),
            static_cast<uint32_t>(qie8Product.size()));
            */

    if (ngpuf01he != ncpuf01he) {
      std::cerr << "*** mismatch in number of flavor 01 digis for event " << ie << std::endl
                << ">>> ngpuf01he = " << ngpuf01he << std::endl
                << ">>> ncpuf01he = " << ncpuf01he << std::endl;
    }

    {
      auto const& idsgpu = f01HEProduct.ids;
      auto const& datagpu = f01HEProduct.data;

      for (uint32_t ich = 0; ich < ncpuf01he; ich++) {
        auto const cpudf = QIE11DataFrame{qie11Filteredf01[ich]};
        auto const cpuid = cpudf.id();
        auto iter2idgpu = std::find(idsgpu.begin(), idsgpu.end(), cpuid);

        if (iter2idgpu == idsgpu.end()) {
          std::cerr << "missing " << HcalDetId{cpuid} << std::endl;
          continue;
        }

        // FIXME: cna fail...
        assert(*iter2idgpu == cpuid);

        auto const ptrdiff = iter2idgpu - idsgpu.begin();
        auto const nsamples_gpu = hcal::compute_nsamples<hcal::Flavor1>(f01HEProduct.stride);
        auto const nsamples_cpu = qie11Filteredf01.samples();
        assert(static_cast<uint32_t>(nsamples_cpu) == nsamples_gpu);

        uint32_t ichgpu = ptrdiff;
        uint32_t offset = ichgpu * f01HEProduct.stride;
        uint16_t const* df_start = datagpu.data() + offset;
        for (uint32_t sample = 0u; sample < nsamples_gpu; sample++) {
          auto const cpuadc = cpudf[sample].adc();
          auto const gpuadc = hcal::adc_for_sample<hcal::Flavor1>(df_start, sample);
          auto const cputdc = cpudf[sample].tdc();
          auto const gputdc = hcal::tdc_for_sample<hcal::Flavor1>(df_start, sample);
          auto const cpucapid = cpudf[sample].capid();
          auto const gpucapid = hcal::capid_for_sample<hcal::Flavor1>(df_start, sample);

          hADCf01HEGPU->Fill(gpuadc);
          hADCf01HECPU->Fill(cpuadc);
          hTDCf01HEGPU->Fill(gputdc);
          hTDCf01HECPU->Fill(cputdc);
          hADCf01HEGPUvsCPU->Fill(cpuadc, gpuadc);
          hTDCf01HEGPUvsCPU->Fill(cputdc, gputdc);

          // At RAW Decoding level there must not be any mistmatches
          // in the adc values at all!
          assert(static_cast<uint8_t>(cpuadc) == gpuadc);
          assert(static_cast<uint8_t>(cputdc) == gputdc);
          assert(static_cast<uint8_t>(cpucapid) == gpucapid);
        }
      }
    }

    if (ngpuf3hb != ncpuf3hb) {
      std::cerr << "*** mismatch in number of flavor 3 digis for event " << ie << std::endl
                << ">>> ngpuf01he = " << ngpuf3hb << std::endl
                << ">>> ncpuf01he = " << ncpuf3hb << std::endl;
    }

    {
      auto const& idsgpu = f3HBProduct.ids;
      auto const& datagpu = f3HBProduct.data;

      for (uint32_t ich = 0; ich < ncpuf3hb; ich++) {
        auto const cpudf = QIE11DataFrame{qie11Filteredf3[ich]};
        auto const cpuid = cpudf.id();
        auto iter2idgpu = std::find(idsgpu.begin(), idsgpu.end(), cpuid);

        if (iter2idgpu == idsgpu.end()) {
          std::cerr << "missing " << HcalDetId{cpuid} << std::endl;
          continue;
        }

        // FIXME: cna fail...
        assert(*iter2idgpu == cpuid);

        auto const ptrdiff = iter2idgpu - idsgpu.begin();
        auto const nsamples_gpu = hcal::compute_nsamples<hcal::Flavor3>(f3HBProduct.stride);
        auto const nsamples_cpu = qie11Filteredf3.samples();
        assert(static_cast<uint32_t>(nsamples_cpu) == nsamples_gpu);

        uint32_t ichgpu = ptrdiff;
        uint32_t offset = ichgpu * f3HBProduct.stride;
        uint16_t const* df_start = datagpu.data() + offset;
        for (uint32_t sample = 0u; sample < nsamples_gpu; sample++) {
          auto const cpuadc = cpudf[sample].adc();
          auto const gpuadc = hcal::adc_for_sample<hcal::Flavor3>(df_start, sample);
          auto const cputdc = cpudf[sample].tdc();
          auto const gputdc = hcal::tdc_for_sample<hcal::Flavor3>(df_start, sample);

          hADCf3HBGPU->Fill(gpuadc);
          hADCf3HBCPU->Fill(cpuadc);
          hADCf3HBGPUvsCPU->Fill(cpuadc, gpuadc);
          hTDCf3HBGPUvsCPU->Fill(cputdc, gputdc);

          // At RAW Decoding level there must not be any mistmatches
          // in the adc values at all!
          assert(static_cast<uint8_t>(cpuadc) == gpuadc);
          assert(static_cast<uint8_t>(cputdc) == gputdc);
        }
      }
    }

    if (ngpuf5hb != ncpuf5hb) {
      std::cerr << "*** mismatch in number of flavor 5 digis for event " << ie << std::endl
                << ">>> ngpuf5hb = " << ngpuf5hb << std::endl
                << ">>> ncpuf5hb = " << ncpuf5hb << std::endl;
    }

    {
      auto const& idsgpu = f5HBProduct.ids;
      auto const& datagpu = f5HBProduct.data;
      for (uint32_t i = 0; i < ncpuf5hb; i++) {
        auto const cpudf = qie8Product[i];
        auto const cpuid = cpudf.id().rawId();
        auto iter2idgpu = std::find(idsgpu.begin(), idsgpu.end(), cpuid);
        if (iter2idgpu == idsgpu.end()) {
          std::cerr << "missing " << HcalDetId{cpuid} << std::endl;
          continue;
        }

        assert(*iter2idgpu == cpuid);

        auto const ptrdiff = iter2idgpu - idsgpu.begin();
        auto const nsamples_gpu = hcal::compute_nsamples<hcal::Flavor5>(f5HBProduct.stride);
        auto const nsamples_cpu = qie8Product[0].size();
        assert(static_cast<uint32_t>(nsamples_cpu) == nsamples_gpu);

        uint32_t offset = ptrdiff * f5HBProduct.stride;
        uint16_t const* df_start = datagpu.data() + offset;
        for (uint32_t sample = 0u; sample < nsamples_gpu; sample++) {
          auto const cpuadc = cpudf.sample(sample).adc();
          auto const gpuadc = hcal::adc_for_sample<hcal::Flavor5>(df_start, sample);
          auto const cpucapid = cpudf.sample(sample).capid();
          auto const gpucapid = hcal::capid_for_sample<hcal::Flavor1>(df_start, sample);

          hADCf5HBGPU->Fill(gpuadc);
          hADCf5HBCPU->Fill(cpuadc);
          hADCf5HBGPUvsCPU->Fill(cpuadc, gpuadc);

          // the must for us at RAW Decoding stage
          assert(static_cast<uint8_t>(cpuadc) == gpuadc);
          assert(static_cast<uint8_t>(cpucapid) == gpucapid);
        }
      }
    }
  }

  {
    TCanvas c{"plots", "plots", 4200, 6200};
    c.Divide(3, 3);
    c.cd(1);
    {
      gPad->SetLogy();
      hADCf01HECPU->SetLineColor(kBlack);
      hADCf01HECPU->SetLineWidth(1.);
      hADCf01HECPU->Draw("");
      hADCf01HEGPU->SetLineColor(kBlue);
      hADCf01HEGPU->SetLineWidth(1.);
      hADCf01HEGPU->Draw("sames");
      gPad->Update();
      auto stats = (TPaveStats*)hADCf01HEGPU->FindObject("stats");
      auto y2 = stats->GetY2NDC();
      auto y1 = stats->GetY1NDC();
      stats->SetY2NDC(y1);
      stats->SetY1NDC(y1 - (y2 - y1));
    }
    c.cd(2);
    {
      gPad->SetLogy();
      hADCf5HBCPU->SetLineColor(kBlack);
      hADCf5HBCPU->SetLineWidth(1.);
      hADCf5HBCPU->Draw("");
      hADCf5HBGPU->SetLineColor(kBlue);
      hADCf5HBGPU->SetLineWidth(1.);
      hADCf5HBGPU->Draw("sames");
      gPad->Update();
      auto stats = (TPaveStats*)hADCf5HBGPU->FindObject("stats");
      auto y2 = stats->GetY2NDC();
      auto y1 = stats->GetY1NDC();
      stats->SetY2NDC(y1);
      stats->SetY1NDC(y1 - (y2 - y1));
    }
    c.cd(3);
    {
      gPad->SetLogy();
      hADCf3HBCPU->SetLineColor(kBlack);
      hADCf3HBCPU->SetLineWidth(1.);
      hADCf3HBCPU->Draw("");
      hADCf3HBGPU->SetLineColor(kBlue);
      hADCf3HBGPU->SetLineWidth(1.);
      hADCf3HBGPU->Draw("sames");
      gPad->Update();
      auto stats = (TPaveStats*)hADCf3HBGPU->FindObject("stats");
      auto y2 = stats->GetY2NDC();
      auto y1 = stats->GetY1NDC();
      stats->SetY2NDC(y1);
      stats->SetY1NDC(y1 - (y2 - y1));
    }
    c.cd(4);
    hADCf01HEGPUvsCPU->Draw("colz");
    c.cd(5);
    hADCf5HBGPUvsCPU->Draw("colz");
    c.cd(6);
    hADCf3HBGPUvsCPU->Draw("colz");
    c.cd(7);
    {
      gPad->SetLogy();
      hTDCf01HECPU->SetLineColor(kBlack);
      hTDCf01HECPU->SetLineWidth(1.);
      hTDCf01HECPU->Draw("");
      hTDCf01HEGPU->SetLineColor(kBlue);
      hTDCf01HEGPU->SetLineWidth(1.);
      hTDCf01HEGPU->Draw("sames");
      gPad->Update();
      auto stats = (TPaveStats*)hTDCf01HEGPU->FindObject("stats");
      auto y2 = stats->GetY2NDC();
      auto y1 = stats->GetY1NDC();
      stats->SetY2NDC(y1);
      stats->SetY1NDC(y1 - (y2 - y1));
    }
    c.cd(8);
    hTDCf01HEGPUvsCPU->Draw("colz");
    c.cd(9);
    hTDCf3HBGPUvsCPU->Draw("colz");

    c.SaveAs("plots.pdf");
  }

  rfin.Close();
  rfout.Write();
  rfout.Close();
}
