#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"
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
#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h"
//#include "CondFormats/DataRecord/interface/HGCalModuleConfigurationRcd.h"
#include "CondFormats/DataRecord/interface/HGCalMappingModuleIndexerRcd.h"
#include "CondFormats/HGCalObjects/interface/HGCalCalibrationParameterHost.h"
#include "CondFormats/HGCalObjects/interface/alpaka/HGCalCalibrationParameterDevice.h"

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
        //configToken_ = cc.consumes(iConfig.getParameter<edm::ESInputTag>("configSource"));
        std::cout << "HGCalCalibrationESProducer::HGCalCalibrationESProducer: file=" << filename_ << std::endl;
      }

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
        edm::ParameterSetDescription desc;
        desc.add<std::string>("filename","HGCalCommissioning/LocalCalibration/data/calibration_parameters_v2.json");
        desc.add<edm::ESInputTag>("moduleIndexerSource",edm::ESInputTag(""))->setComment("Label for module info to set SoA size");
        //desc.add<edm::ESInputTag>("configSource",edm::ESInputTag(""))->setComment("Label for ROC configuration parameters");
        descriptions.addWithDefaultLabel(desc);
      }

      template<typename T>
      static void fill_SoA_column(T* col_SoA, const std::vector<T> values, const int offset, const int nrows, int arr_offset=0) {
        // fill SoA column with data from vector for any type
        const int nrows_vals = values.size();
        if (arr_offset<0) {
          arr_offset = 0;
          if (nrows_vals!=arr_offset+nrows) {
            edm::LogWarning("HGCalCalibrationESProducer") << " Expected " << nrows << " rows, but got " << nrows_vals << "!";
          }
        } else if (nrows_vals<arr_offset+nrows) {
          edm::LogWarning("HGCalCalibrationESProducer") << " Tried to copy " << nrows << " rows to SoA with offset " << arr_offset
                                                        << ", but only have " << nrows_vals << " values in JSON!";
        }
        ::memcpy(&col_SoA[offset], &values.data()[arr_offset], sizeof(T)*nrows); // use mybool (=std::byte) instead of bool
        //return values;
      }

      std::optional<hgcalrechit::HGCalCalibParamHost> produce(const HGCalMappingModuleIndexerRcd& iRecord) {
        auto const& moduleMap = iRecord.get(moduleIndexerToken_);
        //auto const& configParamDevice = iRecord.get(configToken_);

        // load dense indexing
        const uint32_t size = moduleMap.getMaxDataSize(); // channel-level size
        const uint32_t nmod = moduleMap.getMaxERxSize(); // ROC-level size
        hgcalrechit::HGCalCalibParamHost product(size, cms::alpakatools::host());
        //product.view().map() = moduleMap; // set dense indexing in SoA (now redundant & NOT thread safe !?)
        std::cout << "HGCalCalibrationESProducer::produce: moduleMap.getMaxDataSize()=" << size
                  << ", moduleMap.getMaxERxSize()=" << nmod << std::endl;

        // load calib parameters from JSON
        std::cout << "HGCalCalibrationESProducer::produce: filename_=" << filename_ << std::endl;
        //std::ifstream infile("/home/hgcdaq00/DPG/test/hackathon_2024Mar/ineuteli/level0_calib_params.json");
        std::ifstream infile(filename_);
        json calib_data = json::parse(infile);
        for (const auto& it: calib_data.items()) {
          std::string module = it.key(); // module typecode, e.g. "ML-F3PT-TX-0003"
          if (module=="Metadata") continue; // ignore metadata fields
          uint32_t offset = moduleMap.getIndexForModuleData(module); // convert module typecode to dense index for this module
          int nrows = calib_data[module]["Channel"].size(); // number of channels to compare with other columns
          std::cout << "HGCalCalibrationESProducer::produce: calib_data[\"" << module << "\"][\"Channel\"].size() = "
                    << nrows  << ", offset=" << offset << std::endl;
          
          // retrieve gains from configuration (placeholder with gain=1 for now)
          // TODO: retrieve from Configurations ESProducer / SoA
          std::vector<uint8_t> gains(6,1); // 3 ROCs x 2 halves = 6 half-ROCs (ECON eRxs) per module
          
          // loop over ECON eRx blocks to fill columns for gain-dependent calibration parameters
          uint32_t nchans = (nrows%39==0 ? 39 : 37); // number of channels per eRx (37 excl. common modes)
          for (std::size_t i_eRx=0; i_eRx<gains.size(); ++i_eRx) {
            uint32_t i_gain = gains[i_eRx]; // index of JSON array corresponding to (index,gain) = (0,80fC), (1,160fC), (2,320fC)
            //uint32_t i_gain = gains[i_eRx]==1 ? 0 : (gains[i_eRx]==2 ? 1 : 2); // (index,gain,charge) = (0,1,80fC), (1,2,160fC), (2,4,320fC)
            uint32_t offset_arr = i_eRx*nchans; // dense index offset for JSON array (input to SoA)
            uint32_t offset_soa = offset + offset_arr; // dense index offset for SoA
            std::cout << "HGCalCalibrationESProducer::produce:   i_eRx=" << i_eRx << ", nchans=" << nchans
                      << " => offset_soa=" << offset_soa << ", offset_arr=" << offset_arr << std::endl;
            fill_SoA_column<float>(product.view().ADCtofC(),   calib_data[module]["ADCtofC"   ][i_gain],offset_soa,nchans,offset_arr);
            fill_SoA_column<float>(product.view().ADC_ped(),   calib_data[module]["ADC_ped"   ][i_gain],offset_soa,nchans,offset_arr);
            fill_SoA_column<float>(product.view().Noise(),     calib_data[module]["Noise"     ][i_gain],offset_soa,nchans,offset_arr);
            fill_SoA_column<float>(product.view().CM_slope(),  calib_data[module]["CM_slope"  ][i_gain],offset_soa,nchans,offset_arr);
            fill_SoA_column<float>(product.view().CM_ped(),    calib_data[module]["CM_ped"    ][i_gain],offset_soa,nchans,offset_arr);
            fill_SoA_column<float>(product.view().BXm1_slope(),calib_data[module]["BXm1_slope"][i_gain],offset_soa,nchans,offset_arr);
          }
          
          // fill columns for gain-dependent calibration parameters
          fill_SoA_column<float>(product.view().TOTtofC(),   calib_data[module]["TOTtofC"],   offset,nrows);
          fill_SoA_column<float>(product.view().TOT_ped(),   calib_data[module]["TOT_ped"],   offset,nrows);
          fill_SoA_column<float>(product.view().TOT_lin(),   calib_data[module]["TOT_lin"],   offset,nrows);
          fill_SoA_column<float>(product.view().TOT_P0(),    calib_data[module]["TOT_P0"],    offset,nrows);
          fill_SoA_column<float>(product.view().TOT_P1(),    calib_data[module]["TOT_P1"],    offset,nrows);
          fill_SoA_column<float>(product.view().TOT_P2(),    calib_data[module]["TOT_P2"],    offset,nrows);
          fill_SoA_column<float>(product.view().TOAtops(),   calib_data[module]["TOAtops"],   offset,nrows);
          fill_SoA_column<float>(product.view().MIPS_scale(),calib_data[module]["MIPS_scale"],offset,nrows);
          fill_SoA_column<mybool>(product.view().valid(),    calib_data[module]["Valid"],     offset,nrows); // mybool (=std::byte) defined in HGCalCalibrationParameterSoA.h
          
          std::cout << "HGCalCalibrationESProducer::produce: memcpied all columns !" << std::endl;
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
        //while (std::getline(file, line)) {
        //  if (line.find("Channel")!=std::string::npos || line.find("#")!=std::string::npos) continue;
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
      //edm::ESGetToken<HGCalCondSerializableConfig, HGCalModuleConfigurationRcd> configToken_;
      //edm::ESGetToken<hgcalrechit::HGCalConfigParamDevice, HGCalModuleConfigurationRcd> configToken_;
      //device::ESGetToken<hgcalrechit::HGCalConfigParamDevice, HGCalModuleConfigurationRcd> configToken_;
      const std::string filename_;
    };

  }  // namespace hgcalrechit

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(hgcalrechit::HGCalCalibrationESProducer);
