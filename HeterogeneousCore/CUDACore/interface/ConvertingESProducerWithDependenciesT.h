#ifndef HeterogeneousCore_CUDACore_interface_ConvertingESProducerWithDependenciesT_h
#define HeterogeneousCore_CUDACore_interface_ConvertingESProducerWithDependenciesT_h

#include <tuple>
#include <utility>

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/eventsetuprecord_registration_macro.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/typelookup.h"

namespace detail {
  // simple implementation of a type zipper over 2 tuples
  // here, the main requirement is the default constructor for Gen template
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
class ConvertingESProducerWithDependenciesT;

template <template <typename...> typename CombinedRecord,
          typename... DepsRecords,
          typename Target,
          typename... Dependencies>
class ConvertingESProducerWithDependenciesT<CombinedRecord<DepsRecords...>, Target, Dependencies...>
    : public edm::ESProducer {
public:
  static constexpr std::size_t nsources = sizeof...(Dependencies);
  static_assert(sizeof...(Dependencies) == sizeof...(DepsRecords));

  explicit ConvertingESProducerWithDependenciesT(edm::ParameterSet const& ps) {
    std::vector<edm::ESInputTag> tags(nsources);
    for (std::size_t i = 0; i < nsources; i++)
      tags[i] = edm::ESInputTag{"", ps.getParameter<std::string>("label" + std::to_string(i))};

    std::string const& name = ps.getParameter<std::string>("ComponentName");
    edm::ESConsumesCollectorT<CombinedRecord<DepsRecords...>> cc = setWhatProduced(this, name);
    WalkConsumes<nsources - 1>::iterate(cc, tokens_, tags);
  }

  std::unique_ptr<Target> produce(CombinedRecord<DepsRecords...> const& record) {
    auto handles = std::tuple<edm::ESHandle<Dependencies>...>{};
    WalkAndCall<nsources - 1, edm::ESHandle<Dependencies>...>::iterate(record, handles, tokens_);

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
  template <std::size_t N>
  struct WalkConsumes {
    static void iterate(edm::ESConsumesCollectorT<CombinedRecord<DepsRecords...>>& cc,
                        TokenType& tokens,
                        std::vector<edm::ESInputTag> const& tags) {
      if constexpr (N > 0)
        WalkConsumes<N - 1>::iterate(cc, tokens, tags);
      std::get<N>(tokens) = cc.consumes(tags[N]);
    }
  };

  template <std::size_t N, typename... Types>
  struct WalkAndCall {
    static void iterate(CombinedRecord<DepsRecords...> const& containingRecord,
                        std::tuple<Types...>& ts,
                        TokenType const& tokens) {
      using Record = typename std::tuple_element<N, std::tuple<DepsRecords...>>::type;
      if constexpr (N > 0)
        WalkAndCall<N - 1, Types...>::iterate(containingRecord, ts, tokens);
      // get the right dependent record
      auto const& record = containingRecord.template getRecord<Record>();
      // assign the right element of the tuple
      std::get<N>(ts) = record.getHandle(std::get<N>(tokens));
    }
  };
};

#endif  // HeterogeneousCore_CUDACore_interface_ConvertingESProducerWithDependenciesT_h
