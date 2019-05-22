#ifndef RecoEgamma_EgammaTools_MVAVariableHelper_H
#define RecoEgamma_EgammaTools_MVAVariableHelper_H

#include "DataFormats/Common/interface/ValueMap.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Event.h"

#include <unordered_map>
#include <vector>
#include <string>

template<class ParticleType>
class MVAVariableIndexMap {

  public:

    MVAVariableIndexMap();

    int getIndex(std::string const& name) const { return indexMap_.at(name); }

  private:

    const std::unordered_map<std::string, int> indexMap_;
};

template<class ParticleType>
class MVAVariableHelper {

  public:

    MVAVariableHelper(edm::ConsumesCollector && cc);

    const std::vector<float> getAuxVariables(edm::Ptr<ParticleType> const& particlePtr,
                                             const edm::Event& iEvent) const;

  private:

    static float getVariableFromValueMapToken(edm::Ptr<ParticleType> const& particlePtr,
                                       edm::EDGetToken const& token, edm::Event const& iEvent) {
        edm::Handle<edm::ValueMap<float>> handle;
        iEvent.getByToken(token, handle);
        return (*handle)[particlePtr];
    }

    static float getVariableFromDoubleToken(edm::EDGetToken const& token, const edm::Event& iEvent) {
        edm::Handle<double> handle;
        iEvent.getByToken(token, handle);
        return *handle;
    }

    const std::vector<edm::EDGetToken> tokens_;
};

#endif
