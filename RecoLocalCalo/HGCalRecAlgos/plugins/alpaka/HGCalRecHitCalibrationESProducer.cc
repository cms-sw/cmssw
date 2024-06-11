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
#include "CondFormats/DataRecord/interface/HGCalElectronicsMappingRcd.h"
#include "CondFormats/DataRecord/interface/HGCalModuleConfigurationRcd.h" // depends on HGCalElectronicsMappingRcd
#include "CondFormats/HGCalObjects/interface/HGCalCalibrationParameterHost.h" // for HGCalConfigParamHost
#include "CondFormats/HGCalObjects/interface/alpaka/HGCalCalibrationParameterDevice.h"

#include <string>
#include <iostream>
#include <iomanip> // for std::setw
#include <sstream>
#include <fstream> // needed to read json file with std::ifstream
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
        indexToken_ = cc.consumes(iConfig.getParameter<edm::ESInputTag>("indexSource"));
        configToken_ = cc.consumes(iConfig.getParameter<edm::ESInputTag>("configSource"));
        std::cout << "HGCalCalibrationESProducer::HGCalCalibrationESProducer: file=" << filename_ << std::endl;
      }

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
        edm::ParameterSetDescription desc;
        desc.add<std::string>("filename","HGCalCommissioning/LocalCalibration/data/calibration_parameters_v2.json");
        desc.add<edm::ESInputTag>("indexSource",edm::ESInputTag(""))->setComment("Label for module indexer to set SoA size");
        desc.add<edm::ESInputTag>("configSource",edm::ESInputTag(""))->setComment("Label for ROC configuration parameters");
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

      std::optional<hgcalrechit::HGCalCalibParamHost> produce(const HGCalModuleConfigurationRcd& iRecord) {
        auto const& configHost = iRecord.get(configToken_);
        auto const& moduleMap = iRecord.getRecord<HGCalElectronicsMappingRcd>().get(indexToken_);

        std::cout << "HGCalCalibrationESProducer::produce: configHost.size()="
                  << configHost.view().metadata().size() << std::endl;

        // load dense indexing
        const uint32_t size = moduleMap.getMaxDataSize(); // channel-level size
        const uint32_t nmod = moduleMap.getMaxERxSize(); // ROC-level size (number of ECON eRx)
        hgcalrechit::HGCalCalibParamHost product(size, cms::alpakatools::host());
        //product.view().map() = moduleMap; // set dense indexing in SoA (now redundant & NOT thread safe !?)
        std::cout << "HGCalCalibrationESProducer::produce: moduleMap.getMaxDataSize()=" << size
                  << ", moduleMap.getMaxERxSize()=" << nmod << std::endl;

        // load calib parameters from JSON
        std::cout << "HGCalCalibrationESProducer::produce: filename_=" << filename_ << std::endl;
        std::ifstream infile(filename_);
        json calib_data = json::parse(infile);
        for (const auto& it: calib_data.items()) { // loop over module typecodes in JSON file
          std::string module = it.key(); // module typecode, e.g. "ML-F3PT-TX-0003"
          if (module=="Metadata") continue; // ignore metadata fields
          uint32_t offset = moduleMap.getIndexForModuleData(module); // convert module typecode to dense index for this module
          uint32_t nrows = calib_data[module]["Channel"].size(); // number of channels to compare with JSON arrays
          std::cout << "HGCalCalibrationESProducer::produce: calib_data[\"" << module << "\"][\"Channel\"].size() = "
                    << nrows  << ", offset=" << offset << std::endl;

          // retrieve gains from configuration (placeholder with gain=1 for now)
          // TODO: retrieve from Configurations ESProducer / SoA
          std::vector<uint8_t> gains(6,1); // 3 ROCs x 2 halves = 6 half-ROCs (ECON eRxs) per module

          // check number of channels make sense
          uint32_t nchans = (nrows%39==0 ? 39 : 37); // number of channels per eRx (37 excl. common modes)
          if (nrows%37!=0 and nrows%39!=0) {
            edm::LogWarning("HGCalCalibrationESProducer") << " nchannels%nchannels_per_erX!=0 nchannels="
                                                          << nrows << ", nchannels_per_erX=37 or 39!";
          }

          // loop over ECON eRx blocks to fill columns for gain-dependent calibration parameters
          for (std::size_t i_eRx=0; i_eRx<gains.size(); ++i_eRx) {
            uint32_t i_gain = gains[i_eRx]; // index of JSON array corresponding to (index,gain) = (0,80fC), (1,160fC), (2,320fC)
            //uint32_t i_gain = gains[i_eRx]==1 ? 0 : (gains[i_eRx]==2 ? 1 : 2); // (index,gain,charge) = (0,1,80fC), (1,2,160fC), (2,4,320fC)
            uint32_t offset_arr = i_eRx*nchans; // dense index offset for JSON array (input to SoA)
            uint32_t offset_soa = offset + offset_arr; // dense index offset for SoA
            std::cout << "HGCalCalibrationESProducer::produce:   i_eRx=" << i_eRx << ", nchans=" << nchans
                      << " => offset_soa=" << offset_soa << ", offset_arr=" << offset_arr << std::endl;
            if (offset_arr+nchans>nrows) {
              edm::LogWarning("HGCalCalibrationESProducer") << " offset + nchannels_per_eRx = " << offset_arr << " + " << nchans
                                                            << " = " << offset_arr+nchans << " > " << nrows << " = nchannels ";
            }
            fill_SoA_column<float>(product.view().ADCtofC(),   calib_data[module]["ADCtofC"   ][i_gain],offset_soa,nchans,offset_arr);
            fill_SoA_column<float>(product.view().ADC_ped(),   calib_data[module]["ADC_ped"   ][i_gain],offset_soa,nchans,offset_arr);
            fill_SoA_column<float>(product.view().Noise(),     calib_data[module]["Noise"     ][i_gain],offset_soa,nchans,offset_arr);
            fill_SoA_column<float>(product.view().CM_slope(),  calib_data[module]["CM_slope"  ][i_gain],offset_soa,nchans,offset_arr);
            fill_SoA_column<float>(product.view().CM_ped(),    calib_data[module]["CM_ped"    ][i_gain],offset_soa,nchans,offset_arr);
            fill_SoA_column<float>(product.view().BXm1_slope(),calib_data[module]["BXm1_slope"][i_gain],offset_soa,nchans,offset_arr);
          }

          // fill columns for gain-independent calibration parameters
          fill_SoA_column<float>(product.view().TOTtofC(),   calib_data[module]["TOTtofC"],   offset,nrows);
          fill_SoA_column<float>(product.view().TOT_ped(),   calib_data[module]["TOT_ped"],   offset,nrows);
          fill_SoA_column<float>(product.view().TOT_lin(),   calib_data[module]["TOT_lin"],   offset,nrows);
          fill_SoA_column<float>(product.view().TOT_P0(),    calib_data[module]["TOT_P0"],    offset,nrows);
          fill_SoA_column<float>(product.view().TOT_P1(),    calib_data[module]["TOT_P1"],    offset,nrows);
          fill_SoA_column<float>(product.view().TOT_P2(),    calib_data[module]["TOT_P2"],    offset,nrows);
          fill_SoA_column<float>(product.view().TOAtops(),   calib_data[module]["TOAtops"],   offset,nrows);
          fill_SoA_column<float>(product.view().MIPS_scale(),calib_data[module]["MIPS_scale"],offset,nrows);
          fill_SoA_column<mybool>(product.view().valid(),    calib_data[module]["Valid"],     offset,nrows); // mybool (=std::byte) defined in HGCalCalibrationParameterSoA.h

          //std::cout << "HGCalCalibrationESProducer::produce: memcpied all columns !" << std::endl;
        }

        return product;
      }  // end of produce()

    private:
      edm::ESGetToken<HGCalMappingModuleIndexer, HGCalElectronicsMappingRcd> indexToken_;
      edm::ESGetToken<hgcalrechit::HGCalConfigParamHost, HGCalModuleConfigurationRcd> configToken_;
      //device::ESGetToken<hgcalrechit::HGCalConfigParamDevice, HGCalModuleConfigurationRcd> configToken_;
      const std::string filename_;
    };

  }  // namespace hgcalrechit

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(hgcalrechit::HGCalCalibrationESProducer);
