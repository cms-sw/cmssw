#include "PatternRecognitionbyCA.h"
#include "RecoHGCal/TICL/interface/TICLConstants.h"
#include <vector>

void PatternRecognitionbyCA::fillHistogram(const std::vector<reco::CaloCluster>& layerClusters,
            const std::vector<std::pair<unsigned int, float> >& mask)
{
    const auto nClusters = layerClusters.size();
    for(uint i=0; i<nClusters; ++i)
    {
        
    }

}


void PatternRecognitionbyCA::makeTracksters(
      const edm::Event& ev,
      const edm::EventSetup& es,
      const std::vector<reco::CaloCluster>& layerClusters,
      const std::vector<std::pair<unsigned int, float> >& mask, std::vector<Trackster>& result) const
      {


          std::cout << "making Tracksters" << std::endl;
      }


