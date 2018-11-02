#include <vector>
#include "RecoHGCal/TICL/interface/TICLConstants.h"
#include "PatternRecognitionbyCA.h"


void PatternRecognitionbyCA::fillHistogram(const std::vector<reco::CaloCluster>& layerClusters,
            const std::vector<std::pair<unsigned int, float> >& mask)
{
    std::cout << "filling eta/phi histogram per Layer" << std::endl;
    for(auto& m : mask)
    {
        auto lcId = m.first;
        const auto& lc = layerClusters[lcId];
        //getting the layer Id from the detId of the first hit of the layerCluster
        const auto firstHitDetId = lc.hitsAndFractions()[0].first;
        int layer = rhtools_.getLayerWithOffset(firstHitDetId) + ticlConstants::maxNumberOfLayers*((rhtools_.zside(firstHitDetId)+1)>>1)-1;
        assert(layer>=0);
        auto etaBin = getEtaBin(lc.eta());
        auto phiBin = getPhiBin(lc.phi());
        histogram_[layer][globalBin(etaBin,phiBin)].push_back(lcId);
    }
}




void PatternRecognitionbyCA::makeTracksters(
      const edm::Event& ev,
      const edm::EventSetup& es,
      const std::vector<reco::CaloCluster>& layerClusters,
      const std::vector<std::pair<unsigned int, float> >& mask, std::vector<Trackster>& result) 
{
    rhtools_.getEventSetup(es);

    clearHistogram();
    std::cout << "making Tracksters with CA" << std::endl;

    fillHistogram(layerClusters,mask);
    theGraph_.makeAndConnectDoublets(histogram_, nEtaBins_, nPhiBins_, layerClusters, mask.size(), 2, 2, 0.9);
}
