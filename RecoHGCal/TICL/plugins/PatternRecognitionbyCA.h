// Author: Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 09/2018
// Copyright CERN

#ifndef __RecoHGCal_TICL_PRbyCA_H__
#define __RecoHGCal_TICL_PRbyCA_H__
#include "RecoHGCal/TICL/interface/PatternRecognitionAlgoBase.h"


#include <iostream>


class PatternRecognitionbyCA final : public PatternRecognitionAlgoBase {
public:
    PatternRecognitionbyCA(const edm::ParameterSet& conf) : PatternRecognitionAlgoBase(conf) {

        // TODO get number of bins from configuration
        // eta min 1.5, max 3.0
        // phi min -pi, max +pi
        // bins of size 0.05 in eta/phi -> 30 bins in eta, 126 bins in phi
        histogram_.resize(nLayers_);
        auto nBins = nEtaBins_*nPhiBins_;
        for(uint i = 0; i < nLayers_; ++i)
        {
            histogram_[i].resize(nBins);
        }
    }
    

    void fillHistogram(const std::vector<reco::CaloCluster>& layerClusters,
            const std::vector<std::pair<unsigned int, float> >& mask);

    void makeDoublets(unsigned int deltaEta, unsigned int deltaPhi, unsigned int deltaLayers);

    void makeTracksters(
        const edm::Event& ev,
        const edm::EventSetup& es,
        const std::vector<reco::CaloCluster>& layerClusters,
        const std::vector<std::pair<unsigned int, float> >& mask, std::vector<Trackster>& result) const override;

private:

    std::vector< std::vector< std::vector<unsigned int> > > histogram_; // a histogram of layerClusters IDs per layer

    
    const unsigned int nEtaBins_ = 30;
    const unsigned int nPhiBins_ = 126;
    const unsigned int nLayers_ = 104;
};

#endif
