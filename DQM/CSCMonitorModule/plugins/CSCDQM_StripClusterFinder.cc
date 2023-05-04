
#include "CSCDQM_StripClusterFinder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace cscdqm {

  StripClusterFinder::StripClusterFinder(int l, int s, int cf, int st, bool ME11) {
    //
    // Options
    //
    //	fOpt = new CalibOptions();
    LayerNmb = l;
    TimeSliceNmb = s;
    StripNmb = cf * 16;
    isME11 = ME11;
    if (cf == 7) {
      is7DCFEBs = true;
      isME11 = true;
    } else {
      is7DCFEBs = false;
    }
  }
  void StripClusterFinder::DoAction(int LayerId, float* Cathodes) {
    int TimeId, StripId;
    MEStripClusters.clear();
    StripClusterFitData PulseHeightMapTMP;

    thePulseHeightMap.clear();

    // fill
    //===========================================================================

    for (StripId = 0; StripId < StripNmb; StripId++) {
      for (TimeId = 0; TimeId < TimeSliceNmb; TimeId++) {
        PulseHeightMapTMP.height_[TimeId] = *(Cathodes + StripNmb * (TimeSliceNmb * LayerId + TimeId) + StripId);
      }
      PulseHeightMapTMP.bx_ = 0;
      PulseHeightMapTMP.channel_ = StripId;  // was StripId
      thePulseHeightMap.push_back(PulseHeightMapTMP);
    }
    sort(thePulseHeightMap.begin(), thePulseHeightMap.end(), Sort());
    //===========================================================================

    if (thePulseHeightMap.empty())
      return;
    SearchMax(LayerId);
    SearchBorders();
    Match();
    RefindMax();
    /*
  int val;
  for(i=0;i<MEStripClusters.size();i++){
    val=MEStripClusters[i].LFTBNDStrip;
    MEStripClusters[i].LFTBNDStrip=thePulseHeightMap[val].channel_;
    val=MEStripClusters[i].IRTBNDStrip;
    MEStripClusters[i].IRTBNDStrip=thePulseHeightMap[val].channel_;
    for(j=0;j<MEStripClusters[i].localMax.size();j++){
      val=MEStripClusters[i].localMax[j].Strip;
      MEStripClusters[i].localMax[j].Strip=thePulseHeightMap[val].channel_;
    }
  }
  */

    float sumstrip;
    float sumtime;
    float sumheight;

    for (uint32_t i = 0; i < MEStripClusters.size(); i++) {
      MEStripClusters[i].ClusterPulseMapHeight.clear();
      for (uint32_t j = 0; j < thePulseHeightMap.size(); j++) {
        if (thePulseHeightMap[j].channel_ >= MEStripClusters[i].LFTBNDStrip &&
            thePulseHeightMap[j].channel_ <= MEStripClusters[i].IRTBNDStrip)
          MEStripClusters[i].ClusterPulseMapHeight.push_back(thePulseHeightMap[j]);
      }
      sumstrip = 0;
      sumtime = 0;
      sumheight = 0;
      for (uint32_t k = 0; k < MEStripClusters[i].ClusterPulseMapHeight.size(); k++) {
        for (int l = 0; l < 16; l++) {
          sumstrip += MEStripClusters[i].ClusterPulseMapHeight[k].height_[l] *
                      (MEStripClusters[i].ClusterPulseMapHeight[k].channel_ + 1);
          sumtime += MEStripClusters[i].ClusterPulseMapHeight[k].height_[l] * (l + 1);
          sumheight += MEStripClusters[i].ClusterPulseMapHeight[k].height_[l];
        }
      }
      if (sumheight) {
        MEStripClusters[i].Mean[0] = sumstrip / sumheight;
        MEStripClusters[i].Mean[1] = sumtime / sumheight;
      }
    }
    //  printClusters();
    return;
  }

  void StripClusterFinder::SearchMax(int32_t layerId) {
    StripCluster tmpCluster;
    for (uint32_t i = 1; i < (thePulseHeightMap.size() - 1); i++) {
      if (isME11 && (thePulseHeightMap[i].channel_ == 63 || thePulseHeightMap[i].channel_ == 64))
        continue;
      for (uint32_t j = 1; j < 15; j++) {
        if (thePulseHeightMap[i].height_[j] > thePulseHeightMap[i - 1].height_[j] &&
            thePulseHeightMap[i].height_[j] > thePulseHeightMap[i + 1].height_[j] &&
            thePulseHeightMap[i].height_[j] > thePulseHeightMap[i].height_[j - 1] &&
            thePulseHeightMap[i].height_[j] > thePulseHeightMap[i].height_[j + 1] &&
            thePulseHeightMap[i].height_[j] > thePulseHeightMap[i - 1].height_[j - 1] &&
            thePulseHeightMap[i].height_[j] > thePulseHeightMap[i - 1].height_[j + 1] &&
            thePulseHeightMap[i].height_[j] > thePulseHeightMap[i + 1].height_[j - 1] &&
            thePulseHeightMap[i].height_[j] > thePulseHeightMap[i + 1].height_[j + 1]) {
          tmpCluster.localMax.clear();
          localMaxTMP.Strip = i;
          localMaxTMP.Time = j;
          tmpCluster.localMax.push_back(localMaxTMP);
          tmpCluster.LayerId = layerId;
          tmpCluster.LFTBNDTime = -100;
          tmpCluster.LFTBNDStrip = -100;
          tmpCluster.IRTBNDTime = -100;
          tmpCluster.IRTBNDStrip = -100;
          MEStripClusters.push_back(tmpCluster);
        }
      }
    }
    return;
  }
  void StripClusterFinder::SearchBorders(void) {
    uint32_t iS, iT, iL, jL, iR, jR;

    //              SEARCHING PARAMETERS OF THE CLASTERS (LEFT DOWN & RIGHT UP)

    for (uint32_t i = 0; i < MEStripClusters.size(); i++) {
      if (MEStripClusters[i].localMax.empty()) {
        edm::LogWarning("NoLocalMax") << "Cluster " << i << " has no local Maxima";
        continue;
      }
      iS = MEStripClusters[i].localMax[0].Strip;
      iT = MEStripClusters[i].localMax[0].Time;
      //              LEFT DOWN
      // strip
      MEStripClusters[i].LFTBNDStrip = 0;
      for (iL = iS - 1; iL > 0; iL--) {
        if (isME11 && (thePulseHeightMap[iL].channel_ == 64)) {
          MEStripClusters[i].LFTBNDStrip = iL;
          break;
        }
        if (thePulseHeightMap[iL].height_[iT] == 0.) {
          MEStripClusters[i].LFTBNDStrip = iL + 1;
          break;
        }
      }
      //time
      MEStripClusters[i].LFTBNDTime = 0;
      for (jL = iT - 1; jL > 0; jL--) {
        if (thePulseHeightMap[iS].height_[jL] == 0.) {
          MEStripClusters[i].LFTBNDTime = jL + 1;
          break;
        }
      }
      //              RIGHT UP
      //strip
      MEStripClusters[i].IRTBNDStrip = thePulseHeightMap.size() - 1;
      for (iR = iS + 1; iR < thePulseHeightMap.size(); iR++) {
        if (isME11 && (thePulseHeightMap[iR].channel_ == 63)) {
          MEStripClusters[i].IRTBNDStrip = iR;
          break;
        }
        if (thePulseHeightMap[iR].height_[iT] == 0.) {
          MEStripClusters[i].IRTBNDStrip = iR - 1;
          break;
        }
      }
      //time
      MEStripClusters[i].IRTBNDTime = 15;
      for (jR = iT + 1; jR < 16; jR++) {
        if (thePulseHeightMap[iS].height_[jR] == 0.) {
          MEStripClusters[i].IRTBNDTime = jR - 1;
          break;
        }
      }
    }
    return;
  }

  void StripClusterFinder::Match(void) {
    //              MATCHING THE OVERLAPING CLASTERS
    bool find2match = true;
    do {
      find2match = FindAndMatch();
    } while (find2match);

    return;
  }

  bool StripClusterFinder::FindAndMatch(void) {
    // Find clusters to match
    for (uint32_t ic1 = 0; ic1 < MEStripClusters.size(); ic1++) {
      C1 c1;
      c1.IC1MIN = MEStripClusters[ic1].LFTBNDStrip;
      c1.IC1MAX = MEStripClusters[ic1].IRTBNDStrip;
      c1.JC1MIN = MEStripClusters[ic1].LFTBNDTime;
      c1.JC1MAX = MEStripClusters[ic1].IRTBNDTime;
      for (uint32_t ic2 = ic1 + 1; ic2 < MEStripClusters.size(); ic2++) {
        C2 c2;
        c2.IC2MIN = MEStripClusters[ic2].LFTBNDStrip;
        c2.IC2MAX = MEStripClusters[ic2].IRTBNDStrip;
        c2.JC2MIN = MEStripClusters[ic2].LFTBNDTime;
        c2.JC2MAX = MEStripClusters[ic2].IRTBNDTime;
        if ((c2.IC2MIN >= c1.IC1MIN && c2.IC2MIN <= c1.IC1MAX && c2.JC2MIN >= c1.JC1MIN && c2.JC2MIN <= c1.JC1MAX) ||
            (c2.IC2MIN >= c1.IC1MIN && c2.IC2MIN <= c1.IC1MAX && c2.JC2MAX >= c1.JC1MIN && c2.JC2MAX <= c1.JC1MAX) ||
            (c2.IC2MAX >= c1.IC1MIN && c2.IC2MAX <= c1.IC1MAX && c2.JC2MIN >= c1.JC1MIN && c2.JC2MIN <= c1.JC1MAX) ||
            (c2.IC2MAX >= c1.IC1MIN && c2.IC2MAX <= c1.IC1MAX && c2.JC2MAX >= c1.JC1MIN && c2.JC2MAX <= c1.JC1MAX)) {
          KillCluster(ic1, ic2, c1, c2);
          return true;
        } else {
          if ((c1.IC1MIN >= c2.IC2MIN && c1.IC1MIN <= c2.IC2MAX && c1.JC1MIN >= c2.JC2MIN && c1.JC1MIN <= c2.JC2MAX) ||
              (c1.IC1MIN >= c2.IC2MIN && c1.IC1MIN <= c2.IC2MAX && c1.JC1MAX >= c2.JC2MIN && c1.JC1MAX <= c2.JC2MAX) ||
              (c1.IC1MAX >= c2.IC2MIN && c1.IC1MAX <= c2.IC2MAX && c1.JC1MIN >= c2.JC2MIN && c1.JC1MIN <= c2.JC2MAX) ||
              (c1.IC1MAX >= c2.IC2MIN && c1.IC1MAX <= c2.IC2MAX && c1.JC1MAX >= c2.JC2MIN && c1.JC1MAX <= c2.JC2MAX)) {
            KillCluster(ic1, ic2, c1, c2);
            return true;
          }
        }
      }
    }
    return false;
  }
  void StripClusterFinder::KillCluster(uint32_t ic1, uint32_t ic2, C1 const& c1, C2 const& c2) {
    // Match Clusters and kill one of clusters.
    if (c1.IC1MIN < c2.IC2MIN)
      MEStripClusters[ic1].LFTBNDStrip = c1.IC1MIN;
    else
      MEStripClusters[ic1].LFTBNDStrip = c2.IC2MIN;
    if (c1.JC1MIN < c2.JC2MIN)
      MEStripClusters[ic1].LFTBNDTime = c1.JC1MIN;
    else
      MEStripClusters[ic1].LFTBNDTime = c2.JC2MIN;
    if (c1.IC1MAX > c2.IC2MAX)
      MEStripClusters[ic1].IRTBNDStrip = c1.IC1MAX;
    else
      MEStripClusters[ic1].IRTBNDStrip = c2.IC2MAX;
    if (c1.JC1MAX > c2.JC2MAX)
      MEStripClusters[ic1].IRTBNDTime = c1.JC1MAX;
    else
      MEStripClusters[ic1].IRTBNDTime = c2.JC2MAX;

    MEStripClusters.erase(MEStripClusters.begin() + ic2);
    return;
  }
  void StripClusterFinder::RefindMax(void) {
    //             SEARCHING EXTREMUMS IN THE CLUSTERS

    for (uint32_t i = 0; i < MEStripClusters.size(); i++) {
      MEStripClusters[i].localMax.clear();
      int iLS = MEStripClusters[i].LFTBNDStrip;
      int iRS = MEStripClusters[i].IRTBNDStrip;
      int iLT = MEStripClusters[i].LFTBNDTime;
      int iRT = MEStripClusters[i].IRTBNDTime;

      for (int iS = iLS; iS <= iRS; iS++) {
        if (isME11 && (thePulseHeightMap[iS].channel_ == 63 || thePulseHeightMap[iS].channel_ == 64))
          continue;
        for (int jT = iLT; jT <= iRT; jT++) {
          if (iS == 0 || jT == 0 || (!is7DCFEBs && (iS == 79)) || (is7DCFEBs && (iS == 111)) || jT == 7)
            continue;
          if (thePulseHeightMap[iS].height_[jT] > thePulseHeightMap[iS - 1].height_[jT] &&
              thePulseHeightMap[iS].height_[jT] > thePulseHeightMap[iS + 1].height_[jT] &&
              thePulseHeightMap[iS].height_[jT] > thePulseHeightMap[iS].height_[jT - 1] &&
              thePulseHeightMap[iS].height_[jT] > thePulseHeightMap[iS].height_[jT + 1] &&
              thePulseHeightMap[iS].height_[jT] > thePulseHeightMap[iS - 1].height_[jT - 1] &&
              thePulseHeightMap[iS].height_[jT] > thePulseHeightMap[iS - 1].height_[jT + 1] &&
              thePulseHeightMap[iS].height_[jT] > thePulseHeightMap[iS + 1].height_[jT - 1] &&
              thePulseHeightMap[iS].height_[jT] > thePulseHeightMap[iS + 1].height_[jT + 1]) {
            localMaxTMP.Strip = iS;
            localMaxTMP.Time = jT;
            MEStripClusters[i].localMax.push_back(localMaxTMP);
          }
        }
      }
      // kill local maximums rellated to noise, maximums with pulse height less then 10% of Global max of clust.
      //fing Global Max
      float GlobalMax = 0;
      if (!MEStripClusters[i].localMax.empty()) {
        //std::cout << "Cluster: " << i << " Number of local maximums before erase: "
        //		<< MEStripClusters[i].localMax.size() << std::endl;
        for (uint32_t j = 0; j < MEStripClusters[i].localMax.size(); j++) {
          int iS = MEStripClusters[i].localMax[j].Strip;
          int jT = MEStripClusters[i].localMax[j].Time;
          /*
	  std::cout << "Current Max:" 
	  << " " << iS
	  << " " << jT
	  << " " << thePulseHeightMap[iS].height_[jT] << std::endl;
	*/
          if (thePulseHeightMap[iS].height_[jT] > GlobalMax)
            GlobalMax = thePulseHeightMap[iS].height_[jT];
        }
        GlobalMax = (float)(GlobalMax / 10.);
        //erase noise localMaximums
        bool Erased;
        do {
          Erased = false;
          for (uint32_t j = 0; j < MEStripClusters[i].localMax.size(); j++) {
            int iS = MEStripClusters[i].localMax[j].Strip;
            int jT = MEStripClusters[i].localMax[j].Time;
            if (thePulseHeightMap[iS].height_[jT] < GlobalMax) {
              MEStripClusters[i].localMax.erase(MEStripClusters[i].localMax.begin() + j);
              Erased = true;
              break;
            }
          }
        } while (Erased);

        //debug outs
        //std::cout << "Cluster: " << i << " Number of local maximums: "
        //	<< MEStripClusters[i].localMax.size() << std::endl;
        /*
      for(j=0;j<MEStripClusters[i].localMax.size();j++){
	iS=MEStripClusters[i].localMax[j].Strip;
	jT=MEStripClusters[i].localMax[j].Time;
	std::cout << "Local Max: " << j << " Strip: " << iS << " Time: " << jT 
		  << " Height: " << thePulseHeightMap[iS].height_[jT] 
		  << " Cut Value: " << GlobalMax << std::endl;
      }
      */
      }
    }
    return;
  }
  void StripClusterFinder::printClusters(void) {
    int iS, jT;
    std::cout << "====================================================================" << std::endl;
    std::cout << "debug information from StripClusterFinder" << std::endl;
    for (uint32_t i = 0; i < MEStripClusters.size(); i++) {
      if (MEStripClusters[i].localMax.empty())
        continue;
      std::cout << " Cluster: " << i + 1 << " Number of local Maximums " << MEStripClusters[i].localMax.size()
                << std::endl;
      for (uint32_t j = 0; j < MEStripClusters[i].localMax.size(); j++) {
        iS = MEStripClusters[i].localMax[j].Strip;
        jT = MEStripClusters[i].localMax[j].Time;

        //      std::cout << "Local Max: " << j << " Strip: " << iS << " Time: " << jT << std::endl;
        for (uint32_t k = 0; k < MEStripClusters[i].ClusterPulseMapHeight.size(); k++) {
          if (MEStripClusters[i].ClusterPulseMapHeight[k].channel_ == iS)
            std::cout << "Local Max: " << j + 1 << " Strip: " << iS + 1 << " Time: " << jT + 1
                      << " Height: " << MEStripClusters[i].ClusterPulseMapHeight[k].height_[jT] << std::endl;
        }
      }
      for (uint32_t k = 0; k < MEStripClusters[i].ClusterPulseMapHeight.size(); k++) {
        std::cout << "Strip: " << MEStripClusters[i].ClusterPulseMapHeight[k].channel_ + 1;
        for (int l = 0; l < 16; l++)
          std::cout << " " << MEStripClusters[i].ClusterPulseMapHeight[k].height_[l];
        std::cout << std::endl;
      }

      std::cout << " Left  Top    corner strip: " << MEStripClusters[i].LFTBNDStrip + 1 << " "
                << " time: " << MEStripClusters[i].LFTBNDTime + 1 << std::endl;
      std::cout << " Right Bottom corner strip: " << MEStripClusters[i].IRTBNDStrip + 1 << " "
                << " time: " << MEStripClusters[i].IRTBNDTime + 1 << std::endl;
    }
    std::cout << "======================================================================" << std::endl;
    return;
  }
  bool StripClusterFinder::Sort::operator()(const StripClusterFitData& a, const StripClusterFitData& b) const {
    return a.channel_ < b.channel_;
  }

}  // namespace cscdqm
