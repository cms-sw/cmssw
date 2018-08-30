#ifndef RecoEgamma_EgammaTools_MultiToken_H
#define RecoEgamma_EgammaTools_MultiToken_H

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

/*
 * This class is a wrapper around a vector of EDM tokens, of which at least one
 * is expected to yield a valid handle.
 *
 * The first time you call getValidHandle() or getHandle(), it will go over all
 * tokens and try to get a valid handle. If no token yields a valid handle, it
 * will either throw and exception or return the last non-valid handle.
 *
 * Once it found a valid handle, it will remember which token was used to get
 * it and therefore doesn't need to loop over all tokens from there on.
 *
 * Example use case: auto-detection of AOD vs. MiniAOD.
 *
 * Created by Jonas Rembser on August 3, 2018.
 */

#include <memory>

template <typename T>
class MultiTokenT {

  public:

    // Constructor which gets the input tags from a config to create the tokens
    template <typename ... Tags>
    MultiTokenT(edm::ConsumesCollector && cc, const edm::ParameterSet& pset, Tags && ... tags)
      : isMaster_(true)
    {
        for (auto&& tag : { tags... }) {
            tokens_.push_back(cc.mayConsume<T>(pset.getParameter<edm::InputTag>(tag)));
        }
        goodIndex_ = std::make_shared<int>(-1);
    }

    // Constructor which gets the input tags from a config to create the tokens plus master token
    template <typename S, typename ... Tags>
    MultiTokenT(MultiTokenT<S>& master, edm::ConsumesCollector && cc, const edm::ParameterSet& pset, Tags && ... tags)
      : isMaster_(false)
      , goodIndex_(master.getGoodTokenIndexPtr())
    {
        for (auto&& tag : { tags... }) {
            tokens_.push_back(cc.mayConsume<T>(pset.getParameter<edm::InputTag>(tag)));
        }
    }

    // Get a handle on the event data, non-valid handle is allowed
    edm::Handle<T> getHandle(const edm::Event& iEvent)
    {
        edm::Handle<T> handle;

        // If we already know which token works, take that one
        if (*goodIndex_ >= 0) {
            iEvent.getByToken(tokens_[*goodIndex_], handle);
            return handle;
        }

        if (!isMaster_) {
            throw cms::Exception("MultiTokenTException") <<
                "Trying to get a handle from a depending MultiToken before the master!";
        }

        // If not, set the good token index parallel to getting the handle
        handle = getInitialHandle(iEvent);

        if (*goodIndex_ == -1) {
            *goodIndex_ = tokens_.size() - 1;
        }
        return handle;
    }

    // Get a handle on the event data,
    // throw exception if no token yields a valid handle
    edm::Handle<T> getValidHandle(const edm::Event& iEvent)
    {
        edm::Handle<T> handle;

        // If we already know which token works, take that one
        if (*goodIndex_ >= 0) {
            iEvent.getByToken(tokens_[*goodIndex_], handle);
            if (!handle.isValid())
                throw cms::Exception("MultiTokenTException") <<
                    "Token gave valid handle in previously but not anymore!";
            return handle;
        }

        if (!isMaster_) {
            throw cms::Exception("MultiTokenTException") <<
                "Trying to get a handle from a depending MultiToken before the master!";
        }

        // If not, set the good token index parallel to getting the handle
        handle = getInitialHandle(iEvent);

        if (*goodIndex_ == -1) {
            throw cms::Exception("MultiTokenTException") << "Neither handle is valid!";
        }
        return handle;
    }

    // get the good token
    edm::EDGetTokenT<T> get(const edm::Event& iEvent)
    {
        // If we already know which token works, take that index
        if (*goodIndex_ >= 0)
            return tokens_[*goodIndex_];

        // If this is not a master MultiToken, just return what it got
        if (!isMaster_) {
            throw cms::Exception("MultiTokenTException") <<
                "Trying to get a handle from a depending MultiToken before the master!";
        }

        // Find which token is the good one by trying to get a handle
        edm::Handle<T> handle;
        for (auto token:tokens_ ) {
            iEvent.getByToken(token, handle);
            if (handle.isValid()) {
                return token;
            }
        }

        throw cms::Exception("MultiTokenTException") << "Neither token is valid!";
    }

    int getGoodTokenIndex() const
    {
        return *goodIndex_;
    }

    std::shared_ptr<int> getGoodTokenIndexPtr() const
    {
        return goodIndex_;
    }

  private:

    edm::Handle<T> getInitialHandle(const edm::Event& iEvent)
    {
        // Try to retrieve the collection from the event. If we fail to
        // retrieve the collection with one name, we next look for the one with
        // the other name and so on.
        edm::Handle<T> handle;
        for (size_t i = 0; i < tokens_.size(); ++i) {
            iEvent.getByToken(tokens_[i], handle);
            if (handle.isValid()) {
                *goodIndex_ = i;
                return handle;
            }
        }
        return handle;
    }

    const bool isMaster_;
    std::vector<edm::EDGetTokenT<T>> tokens_;
    std::shared_ptr<int> goodIndex_;
};

#endif
