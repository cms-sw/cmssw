#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "DataFormats/HGCalDigi/interface/HGCalElectronicsId.h"
#include "CondFormats/HGCalObjects/interface/HGCalConfiguration.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h"
#include "CondFormats/DataRecord/interface/HGCalElectronicsMappingRcd.h"
#include "CondFormats/DataRecord/interface/HGCalModuleConfigurationRcd.h"      // depends on HGCalElectronicsMappingRcd
#include "CondFormats/HGCalObjects/interface/HGCalCalibrationParameterHost.h"  // for HGCalConfigParamHost
#include "CondFormats/HGCalObjects/interface/alpaka/HGCalCalibrationParameterDevice.h"

#include <string>
#include <iostream>
#include <iomanip>  // for std::setw
#include <sstream>
#include <fstream>  // needed to read json file with std::ifstream
#include <nlohmann/json.hpp>
using json = nlohmann::json;

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace hgcalrechit {

    class HGCalCalibrationESProducer : public ESProducer {
    public:
      HGCalCalibrationESProducer(const edm::ParameterSet& iConfig)
          : ESProducer(iConfig), filename_(iConfig.getParameter<edm::FileInPath>("filename")) {
        auto cc = setWhatProduced(this);
        indexToken_ = cc.consumes(iConfig.getParameter<edm::ESInputTag>("indexSource"));
        configToken_ = cc.consumes(iConfig.getParameter<edm::ESInputTag>("configSource"));
      }

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
        edm::ParameterSetDescription desc;
        desc.add<edm::FileInPath>("filename")->setComment("Path to JSON file with calibration parameters");
        desc.add<edm::ESInputTag>("indexSource", edm::ESInputTag(""))
            ->setComment("Label for module indexer to set SoA size");
        desc.add<edm::ESInputTag>("configSource", edm::ESInputTag(""))
            ->setComment("Label for ROC configuration parameters");
        descriptions.addWithDefaultLabel(desc);
      }

      template <typename T>
      static void fill_SoA_column(
          T* column_SoA, const std::vector<T>& values, const int offset, const int nrows, int arr_offset = 0) {
        // fill SoA column with data from vector for any type
        const int nrows_vals = values.size();
        if (arr_offset < 0) {
          arr_offset = 0;
          if (nrows_vals != arr_offset + nrows) {
            edm::LogWarning("HGCalCalibrationESProducer")
                << " Expected " << nrows << " rows, but got " << nrows_vals << "!";
          }
        } else if (nrows_vals < arr_offset + nrows) {
          edm::LogWarning("HGCalCalibrationESProducer")
              << " Tried to copy " << nrows << " rows to SoA with offset " << arr_offset << ", but only have "
              << nrows_vals << " values in JSON!";
        }
        auto begin = values.begin() + arr_offset;
        auto end = (begin + nrows > values.end()) ? values.end() : begin + nrows;
        std::copy(begin, end, &column_SoA[offset]);
      }

      std::optional<hgcalrechit::HGCalCalibParamHost> produce(const HGCalModuleConfigurationRcd& iRecord) {
        auto const& moduleMap = iRecord.get(indexToken_);
        auto const& config = iRecord.get(configToken_);

        // load dense indexing
        const uint32_t nchans = moduleMap.getMaxDataSize();  // channel-level size
        hgcalrechit::HGCalCalibParamHost product(nchans, cms::alpakatools::host());

        // load calib parameters from JSON
        std::ifstream infile(filename_.fullPath().c_str());
        json calib_data = json::parse(infile);
        for (const auto& it : calib_data.items()) {  // loop over module typecodes in JSON file
          const std::string& module = it.key();      // module typecode, e.g. "ML-F3PT-TX-0003"
          if (module == "Metadata")
            continue;  // ignore metadata fields
          const auto& [ifed, imod] = moduleMap.getIndexForFedAndModule(module);
          const uint32_t offset =
              moduleMap.getIndexForModuleData(module);  // convert module typecode to dense index for this module
          const uint32_t nrows =
              calib_data[module]["Channel"].size();  // number of channels to compare with JSON arrays
          const uint32_t nrocs =
              config.feds[ifed].econds[imod].rocs.size();  // number of channels to compare with JSON arrays

          // check number of channels & ROCs make sense
          uint32_t nchans = (nrows % 39 == 0 ? 39 : 37);  // number of channels per eRx (37 excl. common modes)
          if (nrows % 37 != 0 and nrows % 39 != 0) {
            edm::LogWarning("HGCalCalibrationESProducer")
                << " nchannels%nchannels_per_erX!=0 nchannels=" << nrows << ", nchannels_per_erX=37 or 39!";
          }

          // loop over ECON eRx blocks to fill columns for gain-dependent calibration parameters
          for (std::size_t iroc = 0; iroc < nrocs; ++iroc) {
            const uint32_t offset_arr = iroc * nchans;        // dense index offset for JSON array (input to SoA)
            const uint32_t offset_soa = offset + offset_arr;  // dense index offset for SoA
            if (offset_arr + nchans > nrows) {
              edm::LogWarning("HGCalCalibrationESProducer")
                  << " offset + nchannels_per_eRx = " << offset_arr << " + " << nchans << " = " << offset_arr + nchans
                  << " > " << nrows << " = nchannels ";
            }
            fill_SoA_column<float>(
                product.view().ADC_ped(), calib_data[module]["ADC_ped"], offset_soa, nchans, offset_arr);
            fill_SoA_column<float>(product.view().Noise(), calib_data[module]["Noise"], offset_soa, nchans, offset_arr);
            fill_SoA_column<float>(
                product.view().CM_slope(), calib_data[module]["CM_slope"], offset_soa, nchans, offset_arr);
            fill_SoA_column<float>(
                product.view().CM_ped(), calib_data[module]["CM_ped"], offset_soa, nchans, offset_arr);
            fill_SoA_column<float>(
                product.view().BXm1_slope(), calib_data[module]["BXm1_slope"], offset_soa, nchans, offset_arr);
          }
          // fill columns for gain-independent calibration parameters
          fill_SoA_column<float>(product.view().TOTtoADC(), calib_data[module]["TOTtoADC"], offset, nrows);
          fill_SoA_column<float>(product.view().TOT_ped(), calib_data[module]["TOT_ped"], offset, nrows);
          fill_SoA_column<float>(product.view().TOT_lin(), calib_data[module]["TOT_lin"], offset, nrows);
          fill_SoA_column<float>(product.view().TOT_P0(), calib_data[module]["TOT_P0"], offset, nrows);
          fill_SoA_column<float>(product.view().TOT_P1(), calib_data[module]["TOT_P1"], offset, nrows);
          fill_SoA_column<float>(product.view().TOT_P2(), calib_data[module]["TOT_P2"], offset, nrows);
          fill_SoA_column<float>(product.view().TOAtops(), calib_data[module]["TOAtops"], offset, nrows);
          fill_SoA_column<float>(product.view().MIPS_scale(), calib_data[module]["MIPS_scale"], offset, nrows);
          fill_SoA_column<unsigned char>(product.view().valid(), calib_data[module]["Valid"], offset, nrows);
        }

        return product;
      }  // end of produce()

    private:
      edm::ESGetToken<HGCalMappingModuleIndexer, HGCalElectronicsMappingRcd> indexToken_;
      edm::ESGetToken<HGCalConfiguration, HGCalModuleConfigurationRcd> configToken_;
      const edm::FileInPath filename_;
    };

  }  // namespace hgcalrechit

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(hgcalrechit::HGCalCalibrationESProducer);
