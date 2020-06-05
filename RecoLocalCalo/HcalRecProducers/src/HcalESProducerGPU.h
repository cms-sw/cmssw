#ifndef RecoLocalCalo_HcalRecProducers_src_HcalESProducerGPU_h
#define RecoLocalCalo_HcalRecProducers_src_HcalESProducerGPU_h

#include <array>
#include <iostream>
#include <tuple>
#include <utility>

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESProductHost.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/eventsetuprecord_registration_macro.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ReusableObjectHolder.h"
#include "FWCore/Utilities/interface/typelookup.h"

template <typename Record, typename Target, typename Source>
class HcalESProducerGPU : public edm::ESProducer {
public:
  explicit HcalESProducerGPU(edm::ParameterSet const& ps) {
    auto const label = ps.getParameter<std::string>("label");
    std::string name = ps.getParameter<std::string>("ComponentName");
    auto cc = setWhatProduced(this, name);

    cc.setConsumes(token_, edm::ESInputTag{"", label});
  }

  std::unique_ptr<Target> produce(Record const& record) {
    // retrieve conditions in old format
    auto sourceProduct = record.getTransientHandle(token_);

    return std::make_unique<Target>(*sourceProduct);
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& confDesc) {
    edm::ParameterSetDescription desc;

    desc.add<std::string>("ComponentName", "");
    desc.add<std::string>("label", "")->setComment("Product Label");
    confDesc.addWithDefaultLabel(desc);
  }

private:
  edm::ESGetToken<Source, Record> token_;
};

namespace detail {
  // simple implementation of a type zipper over 2 tuples
  // here, the main requirement is the default constructor fro Gen template
  // which __does__ exist for ESGetToken

  template <template <typename, typename> class Gen, typename Tuple1, typename Tuple2>
  struct TypeZipper;

  template <template <typename, typename> class Gen, typename Tuple1, typename Tuple2, std::size_t... Is>
  auto TypeZipperImpl(Tuple1 const& t1, Tuple2 const& t2, std::index_sequence<Is...>) {
    return std::make_tuple(
        Gen<typename std::tuple_element<Is, Tuple1>::type, typename std::tuple_element<Is, Tuple2>::type>{}...);
  }

  template <template <typename, typename> class Gen, typename... Ts1, typename... Ts2>
  struct TypeZipper<Gen, std::tuple<Ts1...>, std::tuple<Ts2...>> {
    static_assert(sizeof...(Ts1) == sizeof...(Ts2));
    using type = typename std::decay<decltype(
        TypeZipperImpl<Gen>(std::tuple<Ts1...>{}, std::tuple<Ts2...>{}, std::index_sequence_for<Ts1...>{}))>::type;
  };

}  // namespace detail

template <typename CombinedRecord, typename Target, typename... Dependencies>
class HcalESProducerGPUWithDependencies;

template <template <typename...> typename CombinedRecord,
          typename... DepsRecords,
          typename Target,
          typename... Dependencies>
class HcalESProducerGPUWithDependencies<CombinedRecord<DepsRecords...>, Target, Dependencies...>
    : public edm::ESProducer {
public:
  static constexpr std::size_t nsources = sizeof...(Dependencies);
  static_assert(sizeof...(Dependencies) == sizeof...(DepsRecords));

  explicit HcalESProducerGPUWithDependencies(edm::ParameterSet const& ps) {
    std::vector<std::string> labels(nsources);
    for (std::size_t i = 0; i < labels.size(); i++)
      labels[i] = ps.getParameter<std::string>("label" + std::to_string(i));

    std::string name = ps.getParameter<std::string>("ComponentName");
    auto cc = setWhatProduced(this, name);
    WalkConsumes<nsources - 1, decltype(cc)>::iterate(cc, tokens_, labels);
  }

  std::unique_ptr<Target> produce(CombinedRecord<DepsRecords...> const& record) {
    auto handles = std::tuple<edm::ESTransientHandle<Dependencies>...>{};
    WalkAndCall<nsources - 1, edm::ESTransientHandle<Dependencies>...>::iterate(record, handles, tokens_);

    return std::apply([](auto const&... handles) { return std::make_unique<Target>((*handles)...); }, handles);
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& confDesc) {
    edm::ParameterSetDescription desc;

    desc.add<std::string>("ComponentName", "");
    for (std::size_t i = 0; i < nsources; i++)
      desc.add<std::string>("label" + std::to_string(i), "")->setComment("Product Label");
    confDesc.addWithDefaultLabel(desc);
  }

private:
  using TokenType =
      typename detail::TypeZipper<edm::ESGetToken, std::tuple<Dependencies...>, std::tuple<DepsRecords...>>::type;
  TokenType tokens_;

private:
  template <std::size_t N, typename CC>
  struct WalkConsumes {
    static void iterate(CC& cc, TokenType& tokens, std::vector<std::string> const& labels) {
      cc.setConsumes(std::get<N>(tokens), edm::ESInputTag{"", labels[N]});
      WalkConsumes<N - 1, CC>::iterate(cc, tokens, labels);
    }
  };

  template <typename CC>
  struct WalkConsumes<0, CC> {
    static void iterate(CC& cc, TokenType& tokens, std::vector<std::string> const& labels) {
      cc.setConsumes(std::get<0>(tokens), edm::ESInputTag{"", labels[0]});
    }
  };

  template <std::size_t N, typename... Types>
  struct WalkAndCall {
    static void iterate(CombinedRecord<DepsRecords...> const& containingRecord,
                        std::tuple<Types...>& ts,
                        TokenType const& tokens) {
      using Record = typename std::tuple_element<N, std::tuple<DepsRecords...>>::type;
      // get the right dependent record
      auto const& record = containingRecord.template getRecord<Record>();
      // assign the right element of the tuple
      //record.get(labels[N], std::get<N>(ts));
      std::get<N>(ts) = record.getTransientHandle(std::get<N>(tokens));
      // iterate
      WalkAndCall<N - 1, Types...>::iterate(containingRecord, ts, tokens);
    }
  };

  template <typename... Types>
  struct WalkAndCall<0, Types...> {
    static void iterate(CombinedRecord<DepsRecords...> const& containingRecord,
                        std::tuple<Types...>& ts,
                        TokenType const& tokens) {
      using Record = typename std::tuple_element<0, std::tuple<DepsRecords...>>::type;
      // get the right dependent record
      auto const& record = containingRecord.template getRecord<Record>();
      // assign the very first element of the tuple
      //record.get(labels[0], std::get<0>(ts));
      std::get<0>(ts) = record.getTransientHandle(std::get<0>(tokens));
    }
  };
};

#endif
