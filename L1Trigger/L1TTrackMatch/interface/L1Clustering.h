#ifndef L1Trigger_L1TTrackMatch_L1Clustering_HH
#define L1Trigger_L1TTrackMatch_L1Clustering_HH
#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>

//Each individual box in the eta and phi dimension.
//  Also used to store final cluster data for each zbin.
struct EtaPhiBin {
  float pTtot = 0;
  int numtracks = 0;
  int numttrks = 0;
  int numtdtrks = 0;
  int numttdtrks = 0;
  bool used = false;
  float phi;  //average phi value (halfway b/t min and max)
  float eta;  //average eta value
  std::vector<unsigned int> trackidx;
};

//store important information for plots
struct MaxZBin {
  int znum;    //Numbered from 0 to nzbins (16, 32, or 64) in order
  int nclust;  //number of clusters in this bin
  float zbincenter;
  std::vector<EtaPhiBin> clusters;  //list of all the clusters in this bin
  float ht;                         //sum of all cluster pTs--only the zbin with the maximum ht is stored
};

inline std::vector<EtaPhiBin> L1_clustering(EtaPhiBin *phislice, int etaBins_, float etaStep_) {
  std::vector<EtaPhiBin> clusters;
  // Find eta-phibin with maxpT, make center of cluster, add neighbors if not already used
  int nclust = 0;

  // get tracks in eta bins in increasing eta order
  for (int etabin = 0; etabin < etaBins_; ++etabin) {
    float my_pt = 0, previousbin_pt = 0;  //, nextbin_pt=0, next2bin_pt=0;
    float nextbin_pt = 0, nextbin2_pt = 0;

    // skip (already) used tracks
    if (phislice[etabin].used)
      continue;
    my_pt = phislice[etabin].pTtot;
    if (my_pt == 0)
      continue;
    //get previous bin pT
    if (etabin > 0 && !phislice[etabin - 1].used)
      previousbin_pt = phislice[etabin - 1].pTtot;

    // get next bins pt
    if (etabin < etaBins_ - 1 && !phislice[etabin + 1].used) {
      nextbin_pt = phislice[etabin + 1].pTtot;
      if (etabin < etaBins_ - 2 && !phislice[etabin + 2].used) {
        nextbin2_pt = phislice[etabin + 2].pTtot;
      }
    }
    // check if pT of current cluster is higher than neighbors
    if (my_pt < previousbin_pt || my_pt <= nextbin_pt) {
      // if unused pT in the left neighbor, spit it out as a cluster
      if (previousbin_pt > 0) {
        clusters.push_back(phislice[etabin - 1]);
        phislice[etabin - 1].used = true;
        nclust++;
      }
      continue;  //if it is not the local max pT skip
    }
    // here reach only unused local max clusters
    clusters.push_back(phislice[etabin]);
    phislice[etabin].used = true;  //if current bin a cluster
    if (previousbin_pt > 0) {
      clusters[nclust].pTtot += previousbin_pt;
      clusters[nclust].numtracks += phislice[etabin - 1].numtracks;
      clusters[nclust].numtdtrks += phislice[etabin - 1].numtdtrks;
      for (unsigned int itrk = 0; itrk < phislice[etabin - 1].trackidx.size(); itrk++)
        clusters[nclust].trackidx.push_back(phislice[etabin - 1].trackidx[itrk]);
    }

    if (my_pt >= nextbin2_pt && nextbin_pt > 0) {
      clusters[nclust].pTtot += nextbin_pt;
      clusters[nclust].numtracks += phislice[etabin + 1].numtracks;
      clusters[nclust].numtdtrks += phislice[etabin + 1].numtdtrks;
      for (unsigned int itrk = 0; itrk < phislice[etabin + 1].trackidx.size(); itrk++)
        clusters[nclust].trackidx.push_back(phislice[etabin + 1].trackidx[itrk]);
      phislice[etabin + 1].used = true;
    }

    nclust++;

  }  // for each etabin

  // Merge close-by clusters
  for (int m = 0; m < nclust - 1; ++m) {
    if (std::abs(clusters[m + 1].eta - clusters[m].eta) < 1.5 * etaStep_) {
      if (clusters[m + 1].pTtot > clusters[m].pTtot) {
        clusters[m].eta = clusters[m + 1].eta;
      }
      clusters[m].pTtot += clusters[m + 1].pTtot;
      clusters[m].numtracks += clusters[m + 1].numtracks;  // total ntrk
      clusters[m].numtdtrks += clusters[m + 1].numtdtrks;  // total ndisp
      for (unsigned int itrk = 0; itrk < clusters[m + 1].trackidx.size(); itrk++)
        clusters[m].trackidx.push_back(clusters[m + 1].trackidx[itrk]);

      // if remove the merged cluster - all the others must be closer to 0
      for (int m1 = m + 1; m1 < nclust - 1; ++m1) {
        clusters[m1] = clusters[m1 + 1];
        //clusters.erase(clusters.begin()+m1);
      }
      //  clusters[m1] = clusters[m1 + 1];
      clusters.erase(clusters.begin() + nclust);
      nclust--;
      m = -1;
    }  // end if clusters neighbor in eta
  }    // end for (m) loop

  return clusters;
}

inline void Fill_L2Cluster(EtaPhiBin &bin, float pt, int ntrk, int ndtrk, std::vector<unsigned int> trkidx) {
  bin.pTtot += pt;
  bin.numtracks += ntrk;
  bin.numtdtrks += ndtrk;
  for (unsigned int itrk = 0; itrk < trkidx.size(); itrk++)
    bin.trackidx.push_back(trkidx[itrk]);
}

inline float DPhi(float phi1, float phi2) {
  float x = phi1 - phi2;
  if (x >= M_PI)
    x -= 2 * M_PI;
  if (x < -1 * M_PI)
    x += 2 * M_PI;
  return x;
}

inline std::vector<EtaPhiBin> L2_clustering(std::vector<std::vector<EtaPhiBin>> &L1clusters,
                                            int phiBins_,
                                            float phiStep_,
                                            float etaStep_) {
  std::vector<EtaPhiBin> clusters;
  for (int phibin = 0; phibin < phiBins_; ++phibin) {  //Find eta-phibin with highest pT
    if (L1clusters[phibin].empty())
      continue;

    // sort L1 clusters max -> min
    sort(L1clusters[phibin].begin(), L1clusters[phibin].end(), [](struct EtaPhiBin &a, struct EtaPhiBin &b) {
      return a.pTtot > b.pTtot;
    });
    for (unsigned int imax = 0; imax < L1clusters[phibin].size(); ++imax) {
      if (L1clusters[phibin][imax].used)
        continue;
      float pt_current = L1clusters[phibin][imax].pTtot;  //current cluster (pt0)
      float pt_next = 0;                                  // next phi bin (pt1)
      float pt_next2 = 0;                                 // next to next phi bin2 (pt2)
      int trk1 = 0;
      int trk2 = 0;
      int tdtrk1 = 0;
      int tdtrk2 = 0;
      std::vector<unsigned int> trkidx1;
      std::vector<unsigned int> trkidx2;
      clusters.push_back(L1clusters[phibin][imax]);

      L1clusters[phibin][imax].used = true;

      // if we are in the last phi bin, dont check phi+1 phi+2
      if (phibin == phiBins_ - 1)
        continue;
      std::vector<unsigned int> used_already;  //keep phi+1 clusters that have been used
      for (unsigned int icluster = 0; icluster < L1clusters[phibin + 1].size(); ++icluster) {
        if (L1clusters[phibin + 1][icluster].used)
          continue;
        if (std::abs(L1clusters[phibin + 1][icluster].eta - L1clusters[phibin][imax].eta) > 1.5 * etaStep_)
          continue;
        pt_next += L1clusters[phibin + 1][icluster].pTtot;
        trk1 += L1clusters[phibin + 1][icluster].numtracks;
        tdtrk1 += L1clusters[phibin + 1][icluster].numtdtrks;
        for (unsigned int itrk = 0; itrk < L1clusters[phibin + 1][icluster].trackidx.size(); itrk++)
          trkidx1.push_back(L1clusters[phibin + 1][icluster].trackidx[itrk]);
        used_already.push_back(icluster);
      }

      if (pt_next < pt_current) {  // if pt1<pt1, merge both clusters
        Fill_L2Cluster(clusters[clusters.size() - 1], pt_next, trk1, tdtrk1, trkidx1);
        for (unsigned int iused : used_already)
          L1clusters[phibin + 1][iused].used = true;
        continue;
      }
      // if phi = next to last bin there is no "next to next"
      if (phibin == phiBins_ - 2) {
        Fill_L2Cluster(clusters[clusters.size() - 1], pt_next, trk1, tdtrk1, trkidx1);
        clusters[clusters.size() - 1].phi = L1clusters[phibin + 1][used_already[0]].phi;
        for (unsigned int iused : used_already)
          L1clusters[phibin + 1][iused].used = true;
        continue;
      }
      std::vector<int> used_already2;  //keep used clusters in phi+2
      for (unsigned int icluster = 0; icluster < L1clusters[phibin + 2].size(); ++icluster) {
        if (L1clusters[phibin + 2][icluster].used)
          continue;
        if (std::abs(L1clusters[phibin + 2][icluster].eta - L1clusters[phibin][imax].eta) > 1.5 * etaStep_)
          continue;
        pt_next2 += L1clusters[phibin + 2][icluster].pTtot;
        trk2 += L1clusters[phibin + 2][icluster].numtracks;
        tdtrk2 += L1clusters[phibin + 2][icluster].numtdtrks;
        for (unsigned int itrk = 0; itrk < L1clusters[phibin + 2][icluster].trackidx.size(); itrk++)
          trkidx2.push_back(L1clusters[phibin + 2][icluster].trackidx[itrk]);
        used_already2.push_back(icluster);
      }
      if (pt_next2 < pt_next) {
        std::vector<unsigned int> trkidx_both;
        trkidx_both.reserve(trkidx1.size() + trkidx2.size());
        trkidx_both.insert(trkidx_both.end(), trkidx1.begin(), trkidx1.end());
        trkidx_both.insert(trkidx_both.end(), trkidx2.begin(), trkidx2.end());
        Fill_L2Cluster(clusters[clusters.size() - 1], pt_next + pt_next2, trk1 + trk2, tdtrk1 + tdtrk2, trkidx_both);
        clusters[clusters.size() - 1].phi = L1clusters[phibin + 1][used_already[0]].phi;
        for (unsigned int iused : used_already)
          L1clusters[phibin + 1][iused].used = true;
        for (unsigned int iused : used_already2)
          L1clusters[phibin + 2][iused].used = true;
      }
    }  // for each L1 cluster
  }    // for each phibin

  int nclust = clusters.size();

  // merge close-by clusters
  for (int m = 0; m < nclust - 1; ++m) {
    for (int n = m + 1; n < nclust; ++n) {
      if (clusters[n].eta != clusters[m].eta)
        continue;
      if (std::abs(DPhi(clusters[n].phi, clusters[m].phi)) > 1.5 * phiStep_)
        continue;

      if (clusters[n].pTtot > clusters[m].pTtot)
        clusters[m].phi = clusters[n].phi;

      clusters[m].pTtot += clusters[n].pTtot;
      clusters[m].numtracks += clusters[n].numtracks;
      clusters[m].numtdtrks += clusters[n].numtdtrks;
      for (unsigned int itrk = 0; itrk < clusters[n].trackidx.size(); itrk++)
        clusters[m].trackidx.push_back(clusters[n].trackidx[itrk]);
      for (int m1 = n; m1 < nclust - 1; ++m1)
        clusters[m1] = clusters[m1 + 1];
      clusters.erase(clusters.begin() + nclust);

      nclust--;
    }  // end of n-loop
  }    // end of m-loop
  return clusters;
}
#endif
