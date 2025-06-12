#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "DataFormats/HGCalDigi/interface/HGCalElectronicsId.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h"
#include "CondFormats/HGCalObjects/interface/HGCalCalibrationParameterHost.h"
#include "CondFormats/HGCalObjects/interface/alpaka/HGCalCalibrationParameterDevice.h"
#include "CondFormats/DataRecord/interface/HGCalElectronicsMappingRcd.h"
#include "CondFormats/DataRecord/interface/HGCalModuleConfigurationRcd.h"  // depends on HGCalElectronicsMappingRcd
#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalESProducerTools.h"    // for json, search_modkey

#include <string>
//#include <iostream>
//#include <sstream>
#include <fstream>  // needed to read json file with std::ifstream

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace hgcalrechit {

    class HGCalCalibrationESProducer : public ESProducer {
    public:
      HGCalCalibrationESProducer(const edm::ParameterSet& iConfig)
          : ESProducer(iConfig), filename_(iConfig.getParameter<edm::FileInPath>("filename")) {
        auto cc = setWhatProduced(this);
        indexToken_ = cc.consumes(iConfig.getParameter<edm::ESInputTag>("indexSource"));
      }

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
        edm::ParameterSetDescription desc;
        desc.add<edm::FileInPath>("filename")->setComment("Path to JSON file with calibration parameters");
        desc.add<edm::ESInputTag>("indexSource", edm::ESInputTag(""))
            ->setComment("Label for module indexer to set SoA size");
        descriptions.addWithDefaultLabel(desc);
      }

      // @short fill SoA column with data from vector for any type with some offset
      template <typename T>
      static void fill_SoA_column(
          T* column_SoA, const std::vector<T>& values, const int offset, const int nrows, int arr_offset = 0) {
        const int nrows_vals = values.size();
        if (arr_offset < 0) {
          arr_offset = 0;
          if (nrows_vals != arr_offset + nrows) {
            throw edm::Exception(edm::errors::LogicError, "HGCalCalibrationESProducer")
                << " Expected " << nrows << " rows, but got " << nrows_vals << "!";
          }
        } else if (nrows_vals < arr_offset + nrows) {
          throw edm::Exception(edm::errors::LogicError, "HGCalCalibrationESProducer")
              << " Tried to copy " << nrows << " rows to SoA with offset " << arr_offset << ", but only have "
              << nrows_vals << " values in JSON!";
        }
        auto begin = values.begin() + arr_offset;
        auto end = (begin + nrows > values.end()) ? values.end() : begin + nrows;
        std::copy(begin, end, &column_SoA[offset]);
      }

      // @short fill full SoA column with data from vector for any type
      template <typename T, typename P>
      static void fill_SoA_eigen_row(P& soa, const std::vector<std::vector<T>>& values, const size_t row) {
        if (row >= values.size())
          throw edm::Exception(edm::errors::LogicError, "HGCalCalibrationESProducer")
              << " Tried to copy row " << row << " to SoA, but only have " << values.size() << " values in JSON!";
        if (!values.empty() && int(values[row].size()) != soa.size())
          throw edm::Exception(edm::errors::LogicError, "HGCalCalibrationESProducer")
              << " Expected " << soa.size() << " elements in Eigen vector, but got " << values[row].size() << "!";
        for (int i = 0; i < soa.size(); i++)
          soa(i) = values[row][i];
      }

      // @short create the ESProducer product: a SoA with channel-level calibration constants
      std::optional<hgcalrechit::HGCalCalibParamHost> produce(const HGCalModuleConfigurationRcd& iRecord) {
        auto const& moduleMap = iRecord.get(indexToken_);
        edm::LogInfo("HGCalCalibrationESProducer") << "produce: filename=" << filename_.fullPath().c_str();

        // load dense indexing
        const uint32_t nchans = moduleMap.getMaxDataSize();  // channel-level size
        hgcalrechit::HGCalCalibParamHost product(nchans, cms::alpakatools::host());

        // load calib parameters from JSON
        std::ifstream infile(filename_.fullPath().c_str());
        json calib_data = json::parse(infile, nullptr, true, /*ignore_comments*/ true);
        for (const auto& it : moduleMap.getTypecodeMap()) {  // loop over all module typecodes
          std::string const& module = it.first;              // module typecode, e.g. "ML-F3PT-TX-0003"

          // retrieve matching key (glob patterns allowed)
          const auto modkey = hgcal::search_modkey(module, calib_data, filename_.fullPath());
          auto calib_data_ = calib_data[modkey];

          // get dimensions
          const auto firstkey = calib_data_.begin().key();
          const uint32_t offset = moduleMap.getIndexForModuleData(module);  // first channel index
          const uint32_t nchans = moduleMap.getNumChannels(module);         // number of channels in mapper
          uint32_t nrows = calib_data_[firstkey].size();                    // number of channels in JSON

          // check number of channels & ROCs make sense
          if (nrows % 37 != 0) {
            edm::LogWarning("HGCalCalibrationESProducer")
                << "nchannels=" << nrows << ", which is not divisible by 37 (#channels per e-Rx)!";
          }
          if (nchans != nrows) {
            edm::LogWarning("HGCalCalibrationESProducer")
                << "nchannels does not match between module indexer ('" << module << "'," << nchans << ") and JSON ('"
                << modkey << "'," << nrows << "')!";
            nrows = std::min(nrows, nchans);  // take smallest to avoid overlap
          }

          // fill calibration parameters for ADC, CM, TOT, MIPS scale, ...
          fill_SoA_column<float>(product.view().ADC_ped(), calib_data_["ADC_ped"], offset, nrows);
          fill_SoA_column<float>(product.view().Noise(), calib_data_["Noise"], offset, nrows);
          fill_SoA_column<float>(product.view().CM_slope(), calib_data_["CM_slope"], offset, nrows);
          fill_SoA_column<float>(product.view().CM_ped(), calib_data_["CM_ped"], offset, nrows);
          fill_SoA_column<float>(product.view().BXm1_slope(), calib_data_["BXm1_slope"], offset, nrows);
          fill_SoA_column<float>(product.view().TOTtoADC(), calib_data_["TOTtoADC"], offset, nrows);
          fill_SoA_column<float>(product.view().TOT_ped(), calib_data_["TOT_ped"], offset, nrows);
          fill_SoA_column<float>(product.view().TOT_lin(), calib_data_["TOT_lin"], offset, nrows);
          fill_SoA_column<float>(product.view().TOT_P0(), calib_data_["TOT_P0"], offset, nrows);
          fill_SoA_column<float>(product.view().TOT_P1(), calib_data_["TOT_P1"], offset, nrows);
          fill_SoA_column<float>(product.view().TOT_P2(), calib_data_["TOT_P2"], offset, nrows);
          fill_SoA_column<float>(product.view().MIPS_scale(), calib_data_["MIPS_scale"], offset, nrows);
          fill_SoA_column<unsigned char>(product.view().valid(), calib_data_["Valid"], offset, nrows);

          // default parameters for fine-grain TDC, coarse-grain TDC and time-walk corrections that allow passthrough
          // (for backwards compatibility of older calibration JSONs)
          if (calib_data_.find("TOA_CTDC") == calib_data_.end())
            calib_data_["TOA_CTDC"] = std::vector<std::vector<float>>(nrows, std::vector<float>(32, 0.));
          if (calib_data_.find("TOA_FTDC") == calib_data_.end())
            calib_data_["TOA_FTDC"] = std::vector<std::vector<float>>(nrows, std::vector<float>(8, 0.));
          if (calib_data_.find("TOA_TW") == calib_data_.end())
            calib_data_["TOA_TW"] = std::vector<std::vector<float>>(nrows, std::vector<float>(3, 0.));

          // fill vectors for ToA correction parameters
          for (size_t n = 0; n < nrows; n++) {
            auto vi = product.view()[offset + n];
            fill_SoA_eigen_row<float>(vi.TOA_CTDC(), calib_data_["TOA_CTDC"], n);
            fill_SoA_eigen_row<float>(vi.TOA_FTDC(), calib_data_["TOA_FTDC"], n);
            fill_SoA_eigen_row<float>(vi.TOA_TW(), calib_data_["TOA_TW"], n);
          }
        }

        return product;
      }  // end of produce()

    private:
      edm::ESGetToken<HGCalMappingModuleIndexer, HGCalElectronicsMappingRcd> indexToken_;
      const edm::FileInPath filename_;
    };

  }  // namespace hgcalrechit

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(hgcalrechit::HGCalCalibrationESProducer);
