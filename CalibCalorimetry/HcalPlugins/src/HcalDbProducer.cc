// -*- C++ -*-
//
// Package:    HcalDbProducer
// Class:      HcalDbProducer
//
/**\class HcalDbProducer HcalDbProducer.h CalibFormats/HcalDbProducer/interface/HcalDbProducer.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Fedor Ratnikov
//         Created:  Tue Aug  9 19:10:10 CDT 2005
//
//

// system include files
#include <iostream>
#include <fstream>
#include <memory>

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESProductHost.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Utilities/interface/ReusableObjectHolder.h"
#include "FWCore/Utilities/interface/TypeDemangler.h"

#include "CondFormats/HcalObjects/interface/AllObjects.h"
#include "CondFormats/DataRecord/interface/HcalAllRcds.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

class HcalDbProducer : public edm::ESProducer {
public:
  HcalDbProducer(const edm::ParameterSet&);
  ~HcalDbProducer() override;

  std::shared_ptr<HcalDbService> produce(const HcalDbRecord&);

private:
  using HostType = edm::ESProductHost<HcalDbService,
                                      HcalPedestalsRcd,
                                      HcalPedestalWidthsRcd,
                                      HcalGainsRcd,
                                      HcalGainWidthsRcd,
                                      HcalQIEDataRcd,
                                      HcalQIETypesRcd,
                                      HcalChannelQualityRcd,
                                      HcalZSThresholdsRcd,
                                      HcalRespCorrsRcd,
                                      HcalL1TriggerObjectsRcd,
                                      HcalTimeCorrsRcd,
                                      HcalLUTCorrsRcd,
                                      HcalPFCorrsRcd,
                                      HcalLutMetadataRcd,
                                      HcalSiPMParametersRcd,
                                      HcalTPChannelParametersRcd,
                                      HcalMCParamsRcd,
                                      HcalRecoParamsRcd,
                                      HcalElectronicsMapRcd,
                                      HcalFrontEndMapRcd,
                                      HcalSiPMCharacteristicsRcd,
                                      HcalTPParametersRcd>;

  // Helper functions and class for the tokens and work for HcalDbService
  template <typename ProductType>
  static void serviceSetData(HostType& host, const ProductType& item, std::false_type) {
    host.setData(&item);
  }
  template <typename ProductType>
  static void serviceSetData(HostType& host, const ProductType& item, std::true_type) {
    host.setData(&item, true);
  }
  template <typename ProductType, typename RecordType, const char* LABEL, typename EffectiveType>
  class ServiceTokenImpl {
  public:
    ServiceTokenImpl()
        : dumpName_(edm::typeDemangle(typeid(ProductType).name()).substr(4))  // remove leading "Hcal"
    {
      if constexpr (EffectiveType::value) {
        dumpName_ = "Effective" + dumpName_;
      }
    }
    void setConsumes(edm::ESConsumesCollector& cc) { token_ = cc.consumes(edm::ESInputTag{"", LABEL}); }
    void setupHcalDbService(HostType& host,
                            const RecordType& record,
                            const std::vector<std::string>& dumpRequest,
                            std::ostream* dumpStream) {
      const auto& item = record.get(token_);
      serviceSetData(host, item, EffectiveType{});

      if (std::find(dumpRequest.begin(), dumpRequest.end(), dumpName_) != dumpRequest.end()) {
        *dumpStream << "New HCAL " << dumpName_ << " set" << std::endl;
        HcalDbASCIIIO::dumpObject(*dumpStream, item);
      }
    }

  private:
    edm::ESGetToken<ProductType, RecordType> token_;
    std::string dumpName_;
  };

  template <typename ProductType, const char* LABEL, typename EffectiveType = std::false_type>
  struct ServiceToken {
    using Product = ProductType;
    static constexpr const char* label = LABEL;
    using Effective = EffectiveType;
  };

  template <typename RecordType, typename... TokenHolders>
  class TokensForServiceHolder {
  public:
    void setConsumes(edm::ESConsumesCollector& cc) {
      std::apply([&cc](auto&&... item) { ((item.setConsumes(cc)), ...); }, tokens_);
    }
    void setupHcalDbService(HostType& host,
                            const HcalDbRecord& record,
                            const std::vector<std::string>& dumpRequest,
                            std::ostream* dumpStream) {
      host.ifRecordChanges<RecordType>(record, [this, &host, &dumpRequest, &dumpStream](auto const& rec) {
        std::apply([&host, &rec, &dumpRequest, &dumpStream](
                       auto&&... item) { ((item.setupHcalDbService(host, rec, dumpRequest, dumpStream)), ...); },
                   tokens_);
      });
    }

  private:
    std::tuple<ServiceTokenImpl<typename TokenHolders::Product,
                                RecordType,
                                TokenHolders::label,
                                typename TokenHolders::Effective>...>
        tokens_;
  };

  // Helper class and functions for the individual products
  template <typename ProductT, typename RecordT>
  class TokenAndTopologyHolder {
  public:
    using Product = ProductT;
    using Record = RecordT;

    TokenAndTopologyHolder() = default;

    void setConsumes(edm::ESConsumesCollector&& cc, const edm::ESInputTag& tag) {
      token_ = cc.consumes(tag);
      topoToken_ = cc.consumes();
    }

    const auto& token() const { return token_; }

    const auto& topoToken() const { return topoToken_; }

  private:
    edm::ESGetToken<ProductT, RecordT> token_;
    edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> topoToken_;
  };

  template <typename ProductType, typename RecordType, size_t IND1, size_t IND2>
  std::unique_ptr<ProductType> produceWithTopology(RecordType const& record) {
    const auto& tokenHolder = std::get<IND2>(std::get<IND1>(tokenHolders_));
    auto item = record.getTransientHandle(tokenHolder.token());

    auto productWithTopology = std::make_unique<ProductType>(*item);
    const auto& topo = record.get(tokenHolder.topoToken());
    productWithTopology->setTopo(&topo);

    return productWithTopology;
  }

  template <size_t IND1, size_t IND2>
  void setupProduce(const char* label, const edm::ESInputTag& tag) {
    auto& holder = std::get<IND2>(std::get<IND1>(tokenHolders_));

    using HolderT = typename std::remove_reference<decltype(holder)>::type;
    using ProductT = typename HolderT::Product;
    using RecordT = typename HolderT::Record;

    holder.setConsumes(
        setWhatProduced(
            this, &HcalDbProducer::produceWithTopology<ProductT, RecordT, IND1, IND2>, edm::es::Label(label)),
        tag);
  }

  template <size_t IND1, size_t... IND2s>
  void setupProduceAllImpl(const char* label, const edm::ESInputTag& tag, std::index_sequence<IND2s...>) {
    ((setupProduce<IND1, IND2s>(label, tag)), ...);
  }

  template <size_t IND1>
  void setupProduceAll(const char* label, const edm::ESInputTag& tag) {
    setupProduceAllImpl<IND1>(
        label, tag, std::make_index_sequence<std::tuple_size_v<std::tuple_element_t<IND1, decltype(tokenHolders_)>>>{});
  }

  // ----------member data ---------------------------
  edm::ReusableObjectHolder<HostType> holder_;

  // Tokens for the "service" produce function
  constexpr static const char kWithTopoEff[] = "withTopoEff";
  constexpr static const char kWithTopo[] = "withTopo";
  constexpr static const char kEmpty[] = "";
  std::tuple<TokensForServiceHolder<HcalPedestalWidthsRcd,
                                    ServiceToken<HcalPedestalWidths, kWithTopoEff, std::true_type>,
                                    ServiceToken<HcalPedestalWidths, kWithTopo>>,
             TokensForServiceHolder<HcalPedestalsRcd,
                                    ServiceToken<HcalPedestals, kWithTopoEff, std::true_type>,
                                    ServiceToken<HcalPedestals, kWithTopo>>,
             TokensForServiceHolder<HcalRecoParamsRcd, ServiceToken<HcalRecoParams, kWithTopo>>,
             TokensForServiceHolder<HcalMCParamsRcd, ServiceToken<HcalMCParams, kWithTopo>>,
             TokensForServiceHolder<HcalLutMetadataRcd, ServiceToken<HcalLutMetadata, kWithTopo>>,
             TokensForServiceHolder<HcalTPParametersRcd, ServiceToken<HcalTPParameters, kEmpty>>,
             TokensForServiceHolder<HcalTPChannelParametersRcd, ServiceToken<HcalTPChannelParameters, kWithTopo>>,
             TokensForServiceHolder<HcalSiPMCharacteristicsRcd, ServiceToken<HcalSiPMCharacteristics, kEmpty>>,
             TokensForServiceHolder<HcalSiPMParametersRcd, ServiceToken<HcalSiPMParameters, kWithTopo>>,
             TokensForServiceHolder<HcalFrontEndMapRcd, ServiceToken<HcalFrontEndMap, kEmpty>>,
             TokensForServiceHolder<HcalElectronicsMapRcd, ServiceToken<HcalElectronicsMap, kEmpty>>,
             TokensForServiceHolder<HcalL1TriggerObjectsRcd, ServiceToken<HcalL1TriggerObjects, kWithTopo>>,
             TokensForServiceHolder<HcalZSThresholdsRcd, ServiceToken<HcalZSThresholds, kWithTopo>>,
             TokensForServiceHolder<HcalChannelQualityRcd, ServiceToken<HcalChannelQuality, kWithTopo>>,
             TokensForServiceHolder<HcalGainWidthsRcd, ServiceToken<HcalGainWidths, kWithTopo>>,
             TokensForServiceHolder<HcalQIETypesRcd, ServiceToken<HcalQIETypes, kWithTopo>>,
             TokensForServiceHolder<HcalQIEDataRcd, ServiceToken<HcalQIEData, kWithTopo>>,
             TokensForServiceHolder<HcalTimeCorrsRcd, ServiceToken<HcalTimeCorrs, kWithTopo>>,
             TokensForServiceHolder<HcalPFCorrsRcd, ServiceToken<HcalPFCorrs, kWithTopo>>,
             TokensForServiceHolder<HcalLUTCorrsRcd, ServiceToken<HcalLUTCorrs, kWithTopo>>,
             TokensForServiceHolder<HcalGainsRcd, ServiceToken<HcalGains, kWithTopo>>,
             TokensForServiceHolder<HcalRespCorrsRcd, ServiceToken<HcalRespCorrs, kWithTopo>>>
      tokensForService_;

  // Tokens for the produceWithTopology functions
  std::tuple<
      // First are withTopo
      std::tuple<TokenAndTopologyHolder<HcalPedestals, HcalPedestalsRcd>,
                 TokenAndTopologyHolder<HcalPedestalWidths, HcalPedestalWidthsRcd>,
                 TokenAndTopologyHolder<HcalGains, HcalGainsRcd>,
                 TokenAndTopologyHolder<HcalGainWidths, HcalGainWidthsRcd>,
                 TokenAndTopologyHolder<HcalQIEData, HcalQIEDataRcd>,
                 TokenAndTopologyHolder<HcalQIETypes, HcalQIETypesRcd>,
                 TokenAndTopologyHolder<HcalChannelQuality, HcalChannelQualityRcd>,
                 TokenAndTopologyHolder<HcalZSThresholds, HcalZSThresholdsRcd>,
                 TokenAndTopologyHolder<HcalRespCorrs, HcalRespCorrsRcd>,
                 TokenAndTopologyHolder<HcalL1TriggerObjects, HcalL1TriggerObjectsRcd>,
                 TokenAndTopologyHolder<HcalTimeCorrs, HcalTimeCorrsRcd>,
                 TokenAndTopologyHolder<HcalLUTCorrs, HcalLUTCorrsRcd>,
                 TokenAndTopologyHolder<HcalPFCorrs, HcalPFCorrsRcd>,
                 TokenAndTopologyHolder<HcalLutMetadata, HcalLutMetadataRcd>,
                 TokenAndTopologyHolder<HcalSiPMParameters, HcalSiPMParametersRcd>,
                 TokenAndTopologyHolder<HcalTPChannelParameters, HcalTPChannelParametersRcd>,
                 TokenAndTopologyHolder<HcalMCParams, HcalMCParamsRcd>,
                 TokenAndTopologyHolder<HcalRecoParams, HcalRecoParamsRcd>>,
      // Then withTopoEff
      std::tuple<TokenAndTopologyHolder<HcalPedestals, HcalPedestalsRcd>,
                 TokenAndTopologyHolder<HcalPedestalWidths, HcalPedestalWidthsRcd>>>
      tokenHolders_;

  std::vector<std::string> mDumpRequest;
  std::ostream* mDumpStream;
};

HcalDbProducer::HcalDbProducer(const edm::ParameterSet& fConfig) : ESProducer(), mDumpRequest(), mDumpStream(nullptr) {
  auto cc = setWhatProduced(this);
  std::apply([&cc](auto&&... item) { ((item.setConsumes(cc)), ...); }, tokensForService_);

  // Setup all withTopo produces functions and their consumes
  setupProduceAll<0>("withTopo", edm::ESInputTag{});
  // Setup all withTopoEff produces functions and their consumes
  setupProduceAll<1>("withTopoEff", edm::ESInputTag{"", "effective"});

  mDumpRequest = fConfig.getUntrackedParameter<std::vector<std::string>>("dump", std::vector<std::string>());
  if (!mDumpRequest.empty()) {
    std::string otputFile = fConfig.getUntrackedParameter<std::string>("file", "");
    mDumpStream = otputFile.empty() ? &std::cout : new std::ofstream(otputFile.c_str());
  }
}

HcalDbProducer::~HcalDbProducer() {
  if (mDumpStream != &std::cout)
    delete mDumpStream;
}

// ------------ method called to produce the data  ------------
std::shared_ptr<HcalDbService> HcalDbProducer::produce(const HcalDbRecord& record) {
  auto host = holder_.makeOrGet([]() { return new HostType; });

  std::apply([this, &host, &record](
                 auto&&... item) { ((item.setupHcalDbService(*host, record, mDumpRequest, mDumpStream)), ...); },
             tokensForService_);

  return host;
}

DEFINE_FWK_EVENTSETUP_MODULE(HcalDbProducer);
