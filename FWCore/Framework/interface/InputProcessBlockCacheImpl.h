#ifndef FWCore_Framework_InputProcessBlockCacheImpl_h
#define FWCore_Framework_InputProcessBlockCacheImpl_h

/** \class edm::impl::InputProcessBlockCacheImpl

\author W. David Dagenhart, created 18 February, 2021

*/

#include "DataFormats/Provenance/interface/ProvenanceFwd.h"
#include "FWCore/Common/interface/ProcessBlockHelperBase.h"
#include "FWCore/Framework/interface/CacheHandle.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"
#include "FWCore/Framework/interface/ProcessBlock.h"
#include "FWCore/Framework/interface/processBlockUtilities.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/ProductLabels.h"
#include "FWCore/Utilities/interface/TypeID.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace edm {

  class Event;

  namespace impl {

    template <typename M>
    constexpr std::size_t countTypeInParameterPack() {
      return 0;
    }

    template <typename M, typename V1, typename... Vs>
    constexpr std::size_t countTypeInParameterPack() {
      return std::is_same<M, V1>::value ? 1 + countTypeInParameterPack<M, Vs...>()
                                        : countTypeInParameterPack<M, Vs...>();
    }

    class InvalidCacheType {};

    template <typename W, typename U = InvalidCacheType, typename... Types>
    constexpr std::size_t indexInputProcessBlockCache() {
      if constexpr (std::is_same<W, U>::value) {
        return 0;
      } else {
        if constexpr (sizeof...(Types) > 0) {
          return 1 + indexInputProcessBlockCache<W, Types...>();
        } else {
          static_assert(sizeof...(Types) > 0,
                        "CacheType used with registerProcessBlockCacheFiller does not match any template parameters of "
                        "InputProcessBlockCache");
          return 0;
        }
      }
    }

    struct TokenInfo {
      EDGetToken token_;
      TypeID typeID_;
    };

    template <typename CacheType>
    class CacheFiller {
    public:
      std::function<std::shared_ptr<CacheType>(ProcessBlock const&, std::shared_ptr<CacheType> const&)> func_;
    };

    template <typename... CacheTypes>
    class InputProcessBlockCacheImpl {
    public:
      InputProcessBlockCacheImpl() = default;
      InputProcessBlockCacheImpl(InputProcessBlockCacheImpl const&) = delete;
      InputProcessBlockCacheImpl& operator=(InputProcessBlockCacheImpl const&) = delete;

      template <std::size_t I>
      typename std::enable_if<I == sizeof...(CacheTypes), void>::type fillTuple(std::tuple<CacheHandle<CacheTypes>...>&,
                                                                                Event const&) const {}

      template <std::size_t I>
          typename std::enable_if <
          I<sizeof...(CacheTypes), void>::type fillTuple(std::tuple<CacheHandle<CacheTypes>...>& cacheHandles,
                                                         Event const& event) const {
        unsigned int index = eventProcessBlockIndex(event, processNames_[I]);

        // If the branch associated with the token was passed to registerProcessBlockCacheFiller
        // was not in the input file, then the index will be invalid. Note the branch (including
        // its process name) is selected using the first input file. Also note that even if
        // the branch is present and this index is valid, the product might not be there.
        // The functor in the CacheFiller must deal with that case.
        if (index != ProcessBlockHelperBase::invalidCacheIndex()) {
          std::get<I>(cacheHandles) = CacheHandle(std::get<I>(caches_.at(index).cacheTuple_).get());
        }
        fillTuple<I + 1>(cacheHandles, event);
      }

      std::tuple<CacheHandle<CacheTypes>...> processBlockCaches(Event const& event) const {
        std::tuple<CacheHandle<CacheTypes>...> cacheHandles;
        // processNames will be empty if and only if registerProcessBlockCacheFiller
        // was never called by the module constructor
        if (!processNames_.empty()) {
          fillTuple<0>(cacheHandles, event);
        }
        return cacheHandles;
      }

      void selectInputProcessBlocks(ProductRegistry const& productRegistry,
                                    ProcessBlockHelperBase const& processBlockHelperBase,
                                    EDConsumerBase const& edConsumerBase) {
        unsigned int i = 0;
        for (auto const& tokenInfo : tokenInfos_) {
          if (!tokenInfo.token_.isUninitialized()) {
            ProductLabels productLabels;
            edConsumerBase.labelsForToken(tokenInfo.token_, productLabels);

            processNames_[i] = processBlockHelperBase.selectProcess(productRegistry, productLabels, tokenInfo.typeID_);
          }
          ++i;
        }
        tokenInfos_.clear();
      }

      template <std::size_t N>
      using CacheTypeT = typename std::tuple_element<N, std::tuple<CacheTypes...>>::type;

      template <std::size_t ICacheType, typename DataType, typename Func>
      void registerProcessBlockCacheFiller(EDGetTokenT<DataType> const& token, Func&& func) {
        registerProcessBlockCacheFiller<ICacheType, CacheTypeT<ICacheType>, DataType, Func>(token,
                                                                                            std::forward<Func>(func));
      }

      template <typename CacheType, typename DataType, typename Func>
      void registerProcessBlockCacheFiller(EDGetTokenT<DataType> const& token, Func&& func) {
        static_assert(countTypeInParameterPack<CacheType, CacheTypes...>() == 1u,
                      "If registerProcessBlockCacheFiller is called with a type template parameter\n"
                      "then that type must appear exactly once in the template parameters of InputProcessBlockCache");

        // Find the index into the parameter pack from the CacheType
        constexpr unsigned int I = indexInputProcessBlockCache<CacheType, CacheTypes...>();

        registerProcessBlockCacheFiller<I, CacheType, DataType, Func>(token, std::forward<Func>(func));
      }

      // This gets used for stream type modules where the InputProcessBlockCacheImpl
      // object is held by the adaptor. For stream modules, we use a registerProcessBlockCacheFiller
      // function defined in edm::stream::impl::InputProcessBlockCacheHolder then
      // move the information.
      void moveProcessBlockCacheFiller(std::vector<edm::impl::TokenInfo>& tokenInfos,
                                       std::tuple<edm::impl::CacheFiller<CacheTypes>...>& functors) {
        tokenInfos_ = std::move(tokenInfos);
        functors_ = std::move(functors);
        if (!tokenInfos_.empty()) {
          processNames_.resize(sizeof...(CacheTypes));
        }
      }

      // These are used to fill the CacheTuples
      // One CacheFiller for each CacheType

      class CacheTuple {
      public:
        std::tuple<std::shared_ptr<CacheTypes>...> cacheTuple_;
      };

      template <std::size_t I>
      typename std::enable_if<I == sizeof...(CacheTypes), void>::type fillCache(ProcessBlock const&,
                                                                                CacheTuple const&,
                                                                                CacheTuple&) {}

      template <std::size_t I>
          typename std::enable_if < I<sizeof...(CacheTypes), void>::type fillCache(ProcessBlock const& pb,
                                                                                   CacheTuple const& previousCacheTuple,
                                                                                   CacheTuple& cacheTuple) {
        if (pb.processName() == processNames_[I]) {
          auto const& previousSharedPtr = std::get<I>(previousCacheTuple.cacheTuple_);
          std::get<I>(cacheTuple.cacheTuple_) = std::get<I>(functors_).func_(pb, previousSharedPtr);
        }
        fillCache<I + 1>(pb, previousCacheTuple, cacheTuple);
      }

      void accessInputProcessBlock(ProcessBlock const& pb) {
        if (sizeof...(CacheTypes) > 0 && !processNames_.empty()) {
          CacheTuple cacheTuple;
          if (caches_.empty()) {
            CacheTuple firstCacheTuple;
            fillCache<0>(pb, firstCacheTuple, cacheTuple);
          } else {
            CacheTuple const& previousCacheTuple = caches_.back();
            fillCache<0>(pb, previousCacheTuple, cacheTuple);
          }
          caches_.push_back(std::move(cacheTuple));
        }
      }

      void clearCaches() { caches_.clear(); }
      auto cacheSize() const { return caches_.size(); }

    private:
      template <std::size_t ICacheType, typename CacheType, typename DataType, typename Func>
      void registerProcessBlockCacheFiller(EDGetTokenT<DataType> const& token, Func&& func) {
        static_assert(ICacheType < sizeof...(CacheTypes), "ICacheType out of range");
        processNames_.resize(sizeof...(CacheTypes));
        tokenInfos_.resize(sizeof...(CacheTypes));
        if (!tokenInfos_[ICacheType].token_.isUninitialized()) {
          throw Exception(errors::LogicError)
              << "registerProcessBlockCacheFiller should only be called once per cache type";
        }

        tokenInfos_[ICacheType] = TokenInfo{EDGetToken(token), TypeID(typeid(DataType))};

        std::get<ICacheType>(functors_).func_ = std::forward<Func>(func);
      }

      // ------------ Data members --------------------

      // This holds an entry per ProcessBlock
      std::vector<CacheTuple> caches_;

      // The following 3 data members have one element for each CacheType
      std::tuple<CacheFiller<CacheTypes>...> functors_;
      std::vector<std::string> processNames_;
      std::vector<TokenInfo> tokenInfos_;
    };

  }  // namespace impl
}  // namespace edm
#endif
