// Author: Izaak Neutelings (March 2024)

// includes for CMSSW
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// includes for Alpaka
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

// includes for HGCal, calibration, and configuration parameters
#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h"
#include "CondFormats/HGCalObjects/interface/HGCalCalibrationParameterHost.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingParameterHost.h"
#include "CondFormats/HGCalObjects/interface/alpaka/HGCalCalibrationParameterDevice.h"
#include "CondFormats/DataRecord/interface/HGCalElectronicsMappingRcd.h"
#include "CondFormats/DataRecord/interface/HGCalModuleConfigurationRcd.h"  // depends on HGCalElectronicsMappingRcd
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"            // for HGCSiliconDetId::waferType
#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalESProducerTools.h"    // for json, search_modkey

// includes for standard libraries
#include <string>
#include <fstream>    // needed to read json file with std::ifstream
#include <algorithm>  // for std::fill

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace hgcalrechit {
    using namespace ::hgcal;  // for check_keys, fill_SoA

    class HGCalCalibrationESProducer : public ESProducer {
    public:
      HGCalCalibrationESProducer(const edm::ParameterSet& iConfig)
          : ESProducer(iConfig),
            filename_(iConfig.getParameter<edm::FileInPath>("filename")),
            filenameEnergy_(iConfig.getParameter<edm::FileInPath>("filenameEnergyLoss")) {
        auto cc = setWhatProduced(this);
        indexToken_ = cc.consumes(iConfig.getParameter<edm::ESInputTag>("indexSource"));
        mapToken_ = cc.consumes(iConfig.getParameter<edm::ESInputTag>("mapSource"));
      }

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
        edm::ParameterSetDescription desc;
        desc.add<edm::FileInPath>("filename")->setComment("Path to JSON file with calibration parameters");
        desc.add<edm::FileInPath>("filenameEnergyLoss")
            ->setComment("Path to JSON file with energy loss & thickness corrections");
        desc.add<edm::ESInputTag>("indexSource", edm::ESInputTag(""))
            ->setComment("Label for module indexer to set SoA size");
        desc.add<edm::ESInputTag>("mapSource", edm::ESInputTag(""))
            ->setComment("Label for SoA with module mapper/information");
        descriptions.addWithDefaultLabel(desc);
      }

      // @short compute thickness correction to energy loss
      // 0: CE_E_120um, 1: CE_E_200um, 2: CE_E_300um,
      // 3: CE_H_120um, 4: CE_H_200um, 5: CE_H_300um
      float getThicknessCorrection(const std::vector<float>& sfs,
                                   const uint32_t& idetid,
                                   const int& celltype,
                                   const std::string& fname) {
        using waferType = HGCSiliconDetId::waferType;
        const HGCSiliconDetId detid(idetid);
        const bool isCEE = (detid.det() == DetId::HGCalEE);  //layer<=17;
        uint32_t idx = -1;
        if (celltype == waferType::HGCalHD120)
          idx = (isCEE ? 0 : 3);
        else if (celltype == waferType::HGCalHD200 or celltype == waferType::HGCalLD200)
          idx = (isCEE ? 1 : 4);
        else if (celltype == waferType::HGCalLD300)
          idx = (isCEE ? 2 : 5);
        else {
          cms::Exception ex("InvalidData");
          ex << "Could not find thickness correction for celltype " << celltype << " in layer" << detid.layer()
             << "in '" << fname << "'!";
          ex.addContext("Calling hgcal::getThicknessCorrection()");
        }
        if (idx >= sfs.size()) {
          cms::Exception ex("InvalidData");
          ex << "The index of the thickness correction ()" << idx << ") for celltype " << celltype << " in layer"
             << detid.layer() << "is too large for '" << fname << "'(" << sfs.size() << ")!";
          ex.addContext("Calling hgcal::getThicknessCorrection()");
        }
        return sfs[idx];
      }

      // @short create the ESProducer product: a SoA with channel-level calibration constants
      std::optional<hgcalrechit::HGCalCalibParamHost> produce(const HGCalModuleConfigurationRcd& iRecord) {
        auto const& moduleIndexer = iRecord.get(indexToken_);
        auto const& moduleMapper = iRecord.get(mapToken_);
        edm::LogInfo("HGCalCalibrationESProducer") << "produce: filename=" << filename_.fullPath().c_str();

        // load dense indexing
        const uint32_t nchans = moduleIndexer.maxDataSize();  // channel-level size
        hgcalrechit::HGCalCalibParamHost product(nchans, cms::alpakatools::host());

        // load calib parameters from JSON
        std::ifstream infile(filename_.fullPath().c_str());
        std::ifstream infileEnergy(filenameEnergy_.fullPath().c_str());
        json calib_data = json::parse(infile, nullptr, true, /*ignore_comments*/ true);
        json energy_data = json::parse(infileEnergy, nullptr, true, /*ignore_comments*/ true);

        // check keys
        const std::vector<std::string> energy_keys = {"dEdx", "SF_thickness_Si", "SF_thickness_SiPM"};
        check_keys(energy_data, energy_keys, filenameEnergy_.fullPath());
        const float nlayers = energy_data["dEdx"].size();  // number of absorber layers
        if (nlayers != 47)                                 // TODO: retrieve from nlayers from Geometry
          edm::LogError("HGCalCalibrationESProducer")
              << "Expected 47 layers, but got " << nlayers << " in " << filenameEnergy_.fullPath();
        const std::vector<float> energylosses = energy_data["dEdx"].get<std::vector<float>>();

        // loop over all module typecodes, e.g. "ML-F3PT-TX-0003"
        for (const auto& [module, ids] : moduleIndexer.typecodeMap()) {
          const auto [fedid, modid] = ids;

          // retrieve matching key (glob patterns allowed)
          const auto modkey = search_modkey(module, calib_data, filename_.fullPath());
          auto calib_data_ = calib_data[modkey];

          // get dimensions
          const auto firstkey = calib_data_.begin().key();
          const uint32_t imod = moduleIndexer.getIndexForModule(fedid, modid);  // dense index in module SoA
          const uint32_t offset = moduleIndexer.getIndexForModuleData(module);  // first channel index
          const uint32_t nchans = moduleIndexer.getNumChannels(module);         // number of channels in mapper
          uint32_t nrows = calib_data_[firstkey].size();                        // number of channels in JSON

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

          // average energy loss of absorption layers that sandwich the sensors (do not average last layer)
          // https://twiki.cern.ch/twiki/pub/CMS/HGCALSimulationAndPerformance/CalibratedRecHits.pdf
          const int layer = moduleMapper.view().plane()[imod];  // counts from 1
          float dEdx =
              (layer < nlayers ? (energylosses[layer - 1] + energylosses[layer]) / 2 : energylosses[nlayers - 1]);

          // compute thickness correction
          float sf;
          const bool isSiPM = moduleMapper.view().isSiPM()[imod];
          const int celltype = moduleMapper.view().celltype()[imod];
          const uint32_t detid = moduleMapper.view().detid()[imod];
          if (isSiPM)  // scintillator
            sf = energy_data["SF_thickness_SiPM"][0];
          else  // Si module
            sf = getThicknessCorrection(energy_data["SF_thickness_Si"], detid, celltype, filenameEnergy_.fullPath());
          edm::LogInfo("HGCalCalibrationESProducer")
              << "layer=" << layer << ", celltype=" << celltype << ", isSiPM=" << isSiPM << ", dEdx=" << dEdx
              << ", sf=" << sf << std::endl;
          dEdx *= sf * 1e-3;  // apply correction and convert from MeV to GeV
          fill_SoA_column_single<float>(product.view().EM_scale().data(), dEdx, offset, nrows);

        }  // end of loop over modules

        return product;
      }  // end of produce()

    private:
      edm::ESGetToken<HGCalMappingModuleIndexer, HGCalElectronicsMappingRcd> indexToken_;
      edm::ESGetToken<hgcal::HGCalMappingModuleParamHost, HGCalElectronicsMappingRcd> mapToken_;
      const edm::FileInPath filename_;
      const edm::FileInPath filenameEnergy_;
    };

  }  // namespace hgcalrechit

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(hgcalrechit::HGCalCalibrationESProducer);
