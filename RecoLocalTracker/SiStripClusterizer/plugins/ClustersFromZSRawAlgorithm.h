#ifndef RecoLocalTrackerSiStripClusterizerClustersFromZSRawAlgorithm_H
#define RecoLocalTrackerSiStripClusterizerClustersFromZSRawAlgorithm_H

#include "RecoLocalTracker/SiStripClusterizer/interface/StripClusterizerAlgorithm.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterTools.h"

// produce SiStripCluster direclty out of those in ZeroSupressed Raw data
class ClustersFromZSRawAlgorithm final : public StripClusterizerAlgorithm {
public:
  explicit ClustersFromZSRawAlgorithm(float clusterChargeCut) : m_clusterChargeCut(clusterChargeCut){}

  using Det = StripClusterizerAlgorithm::Det;

  template<typename OUT, bool NOISE_CUT=false>
  void clustersFromZS(uint8_t const * data, int offset, int lenght, uint16_t stripOffset,
                      Det const & det, OUT & out) const {

    constexpr int clusterNoiseCutFactor = 4;  // 10^2/5^2 : 10 because noise is store *10; 5 because os a 5 sigma S/N cut

    auto sti = siStripClusterTools::sensorThicknessInverse(det.detId);
    int is=0;
    uint16_t endStrip = 6*128+1; // max value 
    if (!out.empty()) endStrip = out.back().endStrip();
    // ZS clusters are limited to single APV: each channels has 2 APV; 
    while (is<lenght) {
       uint16_t firstStrip = stripOffset + data[(offset++) ^ 7];
       auto weight = det.weight(firstStrip);
       int clusSize = data[(offset++) ^ 7];
       bool extend = (firstStrip == endStrip);
       if(extend && firstStrip%128!=0) std::cout << "extend?? " << firstStrip <<' '<< firstStrip%128 <<' '<<firstStrip/128 <<' '<< clusSize << std::endl;
       endStrip = firstStrip+clusSize;
       is+=clusSize+2;
       int sum=0;
       int noise2=0;
       std::vector<uint8_t> adc(clusSize);
       for (int ic=0; ic<clusSize; ++ic) {
         auto ladc = data[(offset++) ^ 7];
         if constexpr (NOISE_CUT) {
           uint16_t strip = firstStrip+ic;
           int noise = det.rawNoise(strip);
           if (10*ladc<2*noise) ladc=0;
           else  noise2 += noise*noise;  // cannot overflow
         }
         sum += ladc; // no way it can overflow
         adc[ic]=ladc;
         if (adc[ic] < 254) {
           int charge = 0.5f+float(adc[ic])*weight;
           adc[ic] = (charge > 1022 ? 255 : (charge > 253 ? 254 : charge));
         }
       }
       if constexpr (NOISE_CUT) {
         if (0==sum) continue;
         // do not cut if extendable
         if (!extend && endStrip%128!=0  && clusterNoiseCutFactor*sum*sum < noise2) continue;
         // shall we remove leading&trailing zeros?
       }
       if (extend) out.back().extend(adc.begin(),adc.end());
       else if (endStrip%128==0 || sum*weight*sti > m_clusterChargeCut)
         out.push_back(std::move(SiStripCluster(firstStrip,std::move(adc))));
    }
  }
private:
  float m_clusterChargeCut = 1200.0f;

};
#endif
