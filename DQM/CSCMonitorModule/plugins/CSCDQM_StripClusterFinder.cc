
#include "CSCDQM_StripClusterFinder.h"

namespace cscdqm {

  StripClusterFinder::StripClusterFinder(int l, int s, int cf, int st) {
    //
    // Options
    //
    //	fOpt = new CalibOptions();
    LayerNmb = l;
    TimeSliceNmb = s;
    StripNmb = cf * 16;
  }
  void StripClusterFinder::DoAction(int LayerId, float* Cathodes) {
    int TimeId, StripId;
    this->LId = LayerId;
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
    SearchMax();
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

    for (i = 0; i < MEStripClusters.size(); i++) {
      MEStripClusters[i].ClusterPulseMapHeight.clear();
      for (j = 0; j < thePulseHeightMap.size(); j++) {
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

  void StripClusterFinder::SearchMax(void) {
    StripCluster tmpCluster;
    for (i = 1; i < (thePulseHeightMap.size() - 1); i++) {
      if (thePulseHeightMap[i].channel_ == 63 || thePulseHeightMap[i].channel_ == 64)
        continue;
      for (j = 1; j < 15; j++) {
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
          tmpCluster.LayerId = this->LId;
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

    for (i = 0; i < MEStripClusters.size(); i++) {
      if (MEStripClusters[i].localMax.empty()) {
        std::cout << "!!!Warning Cluster has'nt local Maxima" << std::endl;
        continue;
      }
      iS = MEStripClusters[i].localMax[0].Strip;
      iT = MEStripClusters[i].localMax[0].Time;
      //              LEFT DOWN
      // strip
      MEStripClusters[i].LFTBNDStrip = 0;
      for (iL = iS - 1; iL > 0; iL--) {
        if (thePulseHeightMap[iL].channel_ == 64) {
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
        if (thePulseHeightMap[iR].channel_ == 63) {
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
    icstart = 0;  //!!!???
    for (ic1 = icstart; ic1 < MEStripClusters.size(); ic1++) {
      IC1MIN = MEStripClusters[ic1].LFTBNDStrip;
      IC1MAX = MEStripClusters[ic1].IRTBNDStrip;
      JC1MIN = MEStripClusters[ic1].LFTBNDTime;
      JC1MAX = MEStripClusters[ic1].IRTBNDTime;
      for (ic2 = ic1 + 1; ic2 < MEStripClusters.size(); ic2++) {
        IC2MIN = MEStripClusters[ic2].LFTBNDStrip;
        IC2MAX = MEStripClusters[ic2].IRTBNDStrip;
        JC2MIN = MEStripClusters[ic2].LFTBNDTime;
        JC2MAX = MEStripClusters[ic2].IRTBNDTime;
        if ((IC2MIN >= IC1MIN && IC2MIN <= IC1MAX && JC2MIN >= JC1MIN && JC2MIN <= JC1MAX) ||
            (IC2MIN >= IC1MIN && IC2MIN <= IC1MAX && JC2MAX >= JC1MIN && JC2MAX <= JC1MAX) ||
            (IC2MAX >= IC1MIN && IC2MAX <= IC1MAX && JC2MIN >= JC1MIN && JC2MIN <= JC1MAX) ||
            (IC2MAX >= IC1MIN && IC2MAX <= IC1MAX && JC2MAX >= JC1MIN && JC2MAX <= JC1MAX)) {
          KillCluster();
          return true;
        } else {
          if ((IC1MIN >= IC2MIN && IC1MIN <= IC2MAX && JC1MIN >= JC2MIN && JC1MIN <= JC2MAX) ||
              (IC1MIN >= IC2MIN && IC1MIN <= IC2MAX && JC1MAX >= JC2MIN && JC1MAX <= JC2MAX) ||
              (IC1MAX >= IC2MIN && IC1MAX <= IC2MAX && JC1MIN >= JC2MIN && JC1MIN <= JC2MAX) ||
              (IC1MAX >= IC2MIN && IC1MAX <= IC2MAX && JC1MAX >= JC2MIN && JC1MAX <= JC2MAX)) {
            KillCluster();
            return true;
          }
        }
      }
    }
    return false;
  }
  void StripClusterFinder::KillCluster(void) {
    // Match Clusters and kill one of clusters.
    if (IC1MIN < IC2MIN)
      MEStripClusters[ic1].LFTBNDStrip = IC1MIN;
    else
      MEStripClusters[ic1].LFTBNDStrip = IC2MIN;
    if (JC1MIN < JC2MIN)
      MEStripClusters[ic1].LFTBNDTime = JC1MIN;
    else
      MEStripClusters[ic1].LFTBNDTime = JC2MIN;
    if (IC1MAX > IC2MAX)
      MEStripClusters[ic1].IRTBNDStrip = IC1MAX;
    else
      MEStripClusters[ic1].IRTBNDStrip = IC2MAX;
    if (JC1MAX > JC2MAX)
      MEStripClusters[ic1].IRTBNDTime = JC1MAX;
    else
      MEStripClusters[ic1].IRTBNDTime = JC2MAX;

    MEStripClusters.erase(MEStripClusters.begin() + ic2);
    icstart = ic1;

    return;
  }
  void StripClusterFinder::RefindMax(void) {
    int iLS, iRS, iLT, iRT;
    int iS, jT;
    int ilocal;
    float GlobalMax;
    bool Erased;
    //             SEARCHING EXTREMUMS IN THE CLASTERS

    for (i = 0; i < MEStripClusters.size(); i++) {
      MEStripClusters[i].localMax.clear();
      ilocal = 0;
      iLS = MEStripClusters[i].LFTBNDStrip;
      iRS = MEStripClusters[i].IRTBNDStrip;
      iLT = MEStripClusters[i].LFTBNDTime;
      iRT = MEStripClusters[i].IRTBNDTime;

      /*
    for(iS=iLS+1;iS<=iRS-1;iS++){ 
      for(jT=iLT+1;jT<=iRT-1;jT++){
    */
      for (iS = iLS; iS <= iRS; iS++) {
        if (thePulseHeightMap[iS].channel_ == 63 || thePulseHeightMap[iS].channel_ == 64)
          continue;
        for (jT = iLT; jT <= iRT; jT++) {
          if (iS == 0 || jT == 0 || iS == 79 || jT == 7)
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
            ilocal++;
          }
        }
      }
      // kill local maximums rellated to noise, maximums with pulse height less then 10% of Global max of clust.
      //fing Global Max
      GlobalMax = 0;
      if (!MEStripClusters[i].localMax.empty()) {
        //std::cout << "Cluster: " << i << " Number of local maximums before erase: "
        //		<< MEStripClusters[i].localMax.size() << std::endl;
        for (j = 0; j < MEStripClusters[i].localMax.size(); j++) {
          iS = MEStripClusters[i].localMax[j].Strip;
          jT = MEStripClusters[i].localMax[j].Time;
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
        do {
          Erased = false;
          for (j = 0; j < MEStripClusters[i].localMax.size(); j++) {
            iS = MEStripClusters[i].localMax[j].Strip;
            jT = MEStripClusters[i].localMax[j].Time;
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
    for (i = 0; i < MEStripClusters.size(); i++) {
      if (MEStripClusters[i].localMax.empty())
        continue;
      std::cout << " Cluster: " << i + 1 << " Number of local Maximums " << MEStripClusters[i].localMax.size()
                << std::endl;
      for (j = 0; j < MEStripClusters[i].localMax.size(); j++) {
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
