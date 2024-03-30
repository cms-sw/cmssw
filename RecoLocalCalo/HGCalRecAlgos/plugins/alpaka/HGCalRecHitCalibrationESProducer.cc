#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESProducts.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "DataFormats/Math/interface/libminifloat.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "DataFormats/HGCalDigi/interface/HGCalElectronicsId.h"
//#include "CondFormats/DataRecord/interface/HGCalCondSerializableModuleInfoRcd.h"
//#include "CondFormats/HGCalObjects/interface/HGCalCondSerializableModuleInfo.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h"
#include "CondFormats/DataRecord/interface/HGCalMappingModuleIndexerRcd.h"

//#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingParameterHostCollection.h"
#include "CondFormats/HGCalObjects/interface/alpaka/HGCalMappingParameterDeviceCollection.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalCalibrationParameterHostCollection.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/alpaka/HGCalCalibrationParameterDeviceCollection.h"

#include <string>
#include <iostream>
#include <iomanip> // for std::setw
#include <fstream> // needed to read json file with std::ifstream
#include <sstream>
#include <nlohmann/json.hpp>
using json = nlohmann::json;


namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace hgcalrechit {

    class HGCalCalibrationESProducer : public ESProducer {
    public:

      HGCalCalibrationESProducer(const edm::ParameterSet& iConfig)
        : ESProducer(iConfig),
          filename_(iConfig.getParameter<std::string>("filename")) {
        auto cc = setWhatProduced(this);
        moduleIndexerToken_ = cc.consumes(iConfig.getParameter<edm::ESInputTag>("moduleIndexerSource"));
        std::cout << "HGCalCalibrationESProducer::HGCalCalibrationESProducer: file=" << filename_ << std::endl;
      }

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
        edm::ParameterSetDescription desc;
        desc.add<std::string>("filename", "/home/hgcdaq00/DPG/test/hackathon_2024Mar/ineuteli/calibration_parameters_v2.json");
        desc.add<edm::ESInputTag>("moduleIndexerSource",edm::ESInputTag(""))->setComment("Label for module info to calculate size");
        descriptions.addWithDefaultLabel(desc);
      }

      template<typename T>
      static const std::vector<T> fill_SoA_column(T* col_SoA, const json col_json, const int offset=0, const int nrows_exp=-1) {
        // fill SoA column with data from JSON list for any type
        const std::vector<T> values = col_json.get<std::vector<T>>();
        const int nrows = values.size();
        if(nrows_exp>=0 and nrows_exp!=nrows) { // sanity check
          edm::LogWarning("HGCalCalibrationESProducer") << " Found two columns with different number or rows: "
                                                        << nrows_exp << " vs. " << nrows << "!";
        }
        ::memcpy(&col_SoA[offset], values.data(), sizeof(T)*nrows); // use mybool (=std::byte) instead of bool
        return values;
      }

      std::optional<hgcalrechit::HGCalCalibParamHostCollection> produce(const HGCalMappingModuleIndexerRcd& iRecord) {
        auto const& moduleMap = iRecord.get(moduleIndexerToken_);

        // load dense indexing
        const uint32_t size = moduleMap.getMaxDataSize(); // channel-level size
        hgcalrechit::HGCalCalibParamHostCollection product(size, cms::alpakatools::host());
        product.view().map() = moduleMap; // set dense indexing in SoA
        std::cout << "HGCalCalibrationESProducer: moduleMap.getMaxDataSize()=" << size
                  << ", moduleMap.getMaxERxSize()=" << moduleMap.getMaxERxSize() << std::endl;

        // load calib parameters from JSON
        std::cout << "HGCalCalibrationESProducer: filename_=" << filename_ << std::endl;
        //std::ifstream infile("/home/hgcdaq00/DPG/test/hackathon_2024Mar/ineuteli/level0_calib_params.json");
        std::ifstream infile(filename_);
        json calib_data = json::parse(infile);
        for(const auto& it: calib_data.items()){
          std::string module = it.key(); // module typecode, e.g. "ML-F3PT-TX-0003"
          uint32_t offset = moduleMap.getIndexForModuleData(module); // convert electronics ID to dense index for this module
          int nrows = calib_data[module]["Channel"].size();
          std::cout << "HGCalCalibrationESProducer: calib_data[\"" << module << "\"][\"Channel\"].size() = "
                    << nrows  << ", offset=" << offset << std::endl;
          fill_SoA_column<float>(product.view().ADC_ped(),   calib_data[module]["ADC_ped"],   offset,nrows);
          fill_SoA_column<float>(product.view().CM_slope(),  calib_data[module]["CM_slope"],  offset,nrows);
          fill_SoA_column<float>(product.view().CM_ped(),    calib_data[module]["CM_ped"],    offset,nrows);
          fill_SoA_column<float>(product.view().ADCtofC(),   calib_data[module]["ADCtofC"],   offset,nrows);
          fill_SoA_column<float>(product.view().BXm1_slope(),calib_data[module]["BXm1_slope"],offset,nrows);
          fill_SoA_column<float>(product.view().TOTtofC(),   calib_data[module]["TOTtofC"],   offset,nrows);
          fill_SoA_column<float>(product.view().TOT_ped(),   calib_data[module]["TOT_ped"],   offset,nrows);
          fill_SoA_column<float>(product.view().TOT_lin(),   calib_data[module]["TOT_lin"],   offset,nrows);
          fill_SoA_column<float>(product.view().TOT_P0(),    calib_data[module]["TOT_P0"],    offset,nrows);
          fill_SoA_column<float>(product.view().TOT_P1(),    calib_data[module]["TOT_P1"],    offset,nrows);
          fill_SoA_column<float>(product.view().TOT_P2(),    calib_data[module]["TOT_P2"],    offset,nrows);
          fill_SoA_column<float>(product.view().TOAtops(),   calib_data[module]["TOAtops"],   offset,nrows);
          fill_SoA_column<mybool>(product.view().valid(),    calib_data[module]["Valid"],     offset,nrows); // mybool (=std::byte) defined in HGCalCalibrationParameterSoA.h
          std::cout << "HGCalCalibrationESProducer: memcpied all columns !" << std::endl;
        }

        //// OLD: load calib parameters from txt
        //// https://github.com/CMS-HGCAL/cmssw/blob/hgcal-condformat-HGCalNANO-13_2_0_pre3/RecoLocalCalo/HGCalRecAlgos/plugins/alpaka/HGCalRecHitCalibrationESProducer.cc
        //std::cout << "HGCalCalibrationESProducer: Comparing to txt file for validation..." << std::endl;
        ////edm::FileInPath fip("/eos/cms/store/group/dpg_hgcal/comm_hgcal/ykao/calibration_parameters_v2.txt");
        //edm::FileInPath fip("/eos/cms/store/group/dpg_hgcal/tb_hgcal/2023/CMSSW/ReReco_Oct10/Run1695563673/26428d8a-6d27-11ee-8957-fa163e8039dc/calibs/level0_calib_params.txt");
        //std::ifstream file(fip.fullPath());
        //std::string line;
        //uint32_t id;
        //float ped, noise, cm_slope, cm_offset, bxm1_slope, bxm1_offset;
        //while(std::getline(file, line)) {
        //  if(line.find("Channel")!=std::string::npos || line.find("#")!=std::string::npos) continue;
        //
        //  std::istringstream stream(line);
        //  //stream >> std::hex >> id >> std::dec >> ped >> noise >> cm_slope >> cm_offset >> bxm1_slope >> bxm1_offset;
        //  stream >> std::hex >> id >> std::dec >> ped >> noise >> cm_offset >> cm_slope >> bxm1_slope >> bxm1_offset;  // columns got switched in txt file
        //
        //  //reduce to half-point float and fill the pedestals of this channel
        //  std::cout << "HGCalCalibrationESProducer: getting idx for id=" << id << std::endl;
        //  HGCalElectronicsId elecid(id);
        //  std::cout << elecid.localFEDId() << ", " << elecid.captureBlock() << ", " << elecid.econdIdx() << ", " << elecid.econdeRx() << ", " << elecid.halfrocChannel();
        //  uint32_t idx = moduleMap.getIndexForModuleData(elecid); // convert electronics ID to dense index
        //  std::cout << "HGCalCalibrationESProducer: got idx=" << idx << std::endl;
        //
        //  // Comment: if planning to use MiniFloatConverter::float32to16(), a host function,
        //  // one needs to think how to perform MiniFloatConverter::float16to32() in kernels running on GPU (HGCalCalibrationAlgorithms.dev.cc)
        //  std::cout << "HGCalCalibrationESProducer: id=" << std::setw(3) << id;
        //  std::cout << ", idx=" << std::setw(3) << idx;
        //  std::cout << ", json ped=" << std::setw(4) << product.view()[idx].ADC_ped()
        //            << ", txt ped=" << std::setw(4) << ped << ", txt cm_slope=" << cm_slope
        //            << ", json cm_slope=" << std::setw(6) << product.view()[idx].CM_slope() << std::endl;
        //}

        return product;
      }  // end of produce()

    private:
      edm::ESGetToken<HGCalMappingModuleIndexer, HGCalMappingModuleIndexerRcd> moduleIndexerToken_;
      const std::string filename_;
    };

  }  // namespace hgcalrechit

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(hgcalrechit::HGCalCalibrationESProducer);
