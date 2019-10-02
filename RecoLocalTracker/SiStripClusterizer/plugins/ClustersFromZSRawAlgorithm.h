#ifndef RecoLocalTrackerSiStripClusterizerClustersFromZSRawAlgorithm_H
#define RecoLocalTrackerSiStripClusterizerClustersFromZSRawAlgorithm_H

#include "RecoLocalTracker/SiStripClusterizer/interface/StripClusterizerAlgorithm.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterTools.h"

// produce SiStripCluster direclty out of those in ZeroSupressed Raw data
class ClustersFromZSRawAlgorithm final : public StripClusterizerAlgorithm {
public:
  explicit ClustersFromZSRawAlgorithm(float clusterChargeCut) : m_clusterChargeCut(clusterChargeCut){}

  using Det = StripClusterizerAlgorithm::Det;

  template<typename OUT, bool WIDE_CLUS = false, bool NOISE_CUT=false>
  void clustersFromZS(uint8_t const * data, int offset, int lenght, uint16_t stripOffset,
                      Det const & det, OUT & out) const {

    constexpr int clusterNoiseCutFactor = 4;  // 10^2/5^2 : 10 because noise is store *10; 5 because os a 5 sigma S/N cut

    auto sti = siStripClusterTools::sensorThicknessInverse(det.detId);
    int is=0;
    uint16_t endStrip = 6*128+1; // max value 
    if (!out.empty()) endStrip = out.back().endStrip();
    // ZS clusters are limited to single APV: each channels has 2 APV; 
    while (is<lenght) {
       uint16_t firstFedStrip = stripOffset + data[(offset++) ^ 7];
       uint16_t firstStrip = firstFedStrip;
       auto weight = det.weight(firstStrip);
       int clusFedSize = data[(offset++) ^ 7];
       bool extend = (firstStrip == endStrip);
       endStrip = firstStrip+clusFedSize;
       is+=clusFedSize+2;
       int sumRaw=0;
       // int sumCharge=0;
       int noise2=0;
       int clusSize=0;
       std::vector<uint8_t> adc(clusFedSize);
       auto saveCluster = [&]() {
           // save cluster
           adc.resize(clusSize);
           if constexpr (NOISE_CUT) {
             // do not cut if extendable
             if (!extend && endStrip%128!=0  && clusterNoiseCutFactor*sumRaw*sumRaw < noise2) { 
               // loop may continue
               sumRaw=0;
               // sumCharge=0;
               noise2=0;
               clusSize=0;
               return;
             }
           }
           if constexpr (WIDE_CLUS) {
             // if large remove tails
             if (clusSize>5) {
               auto peak = std::max_element(adc.begin(),adc.end());
               auto lc=peak+2;
               for (;lc<adc.end(); ++lc) if (*(lc-1) < *(lc)) break;
               auto fc=peak-1;
               for (;fc>adc.begin(); --fc) if (*(fc) < *(fc-1)) break;
               if(lc<adc.end()) adc.erase(lc,adc.end());
               if(fc>adc.begin()) { 
                 firstStrip+=(fc-adc.begin()); 
                 adc.erase(adc.begin(),fc); 
                 extend=false;
               }
               clusSize=adc.size();
             } // end
           }  // end WC
           //
           endStrip = firstStrip+clusSize;
           if (extend) out.back().extend(adc.begin(),adc.end());
           else if (endStrip%128==0 || sumRaw*weight*sti > m_clusterChargeCut)
             out.push_back(std::move(SiStripCluster(firstStrip,std::move(adc))));
           // loop may continue
           sumRaw=0;
           // sumCharge=0;
           noise2=0;
           clusSize=0;
       }; // end save cluster

      // loop over strips
      for (int ic=0; ic<clusFedSize; ++ic) {
        auto ladc = data[(offset++) ^ 7];
        if (0==ladc) {
          if (clusSize>0) {
             saveCluster();
             adc = std::vector<uint8_t>(clusFedSize-ic);
          }
          firstStrip = firstFedStrip + ic+1;
          extend=false;
          continue;
        }
        if constexpr (NOISE_CUT) {
          uint16_t strip = firstFedStrip+ic;
          int noise = det.rawNoise(strip);
          if (5*ladc<noise) ladc=0;
          else  noise2 += noise*noise;  // cannot overflow
        }
        sumRaw += ladc; // no way it can overflow
        if (ladc < 254) {
          int charge = 0.5f+float(ladc)*weight;
          ladc = (charge > 1022 ? 255 : (charge > 253 ? 254 : charge));
        }
        adc[clusSize++]=ladc;
      } // end loop over strips
      // save last (hopefully only) cluster
      if (clusSize>0) saveCluster();
    }  // end while
  }
private:
  float m_clusterChargeCut = 1200.0f;

};
#endif
