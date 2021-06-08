#include "L1Trigger/VertexFinder/interface/VertexFinder.h"

using namespace std;

namespace l1tVertexFinder {

  void VertexFinder::computeAndSetVertexParameters(RecoVertex<>& vertex,
                                                   const std::vector<float>& bin_centers,
                                                   const std::vector<unsigned int>& counts) {
    double pt = 0.;
    double z0 = 0.;
    double z0width = 0.;
    bool highPt = false;
    double highestPt = 0.;
    unsigned int numHighPtTracks = 0;

    float SumZ = 0.;
    float z0square = 0.;
    float trackPt = 0.;

    std::vector<double> bin_pt(bin_centers.size(), 0.0);
    unsigned int ibin = 0;
    unsigned int itrack = 0;

    for (const L1Track* track : vertex.tracks()) {
      itrack++;
      trackPt = track->pt();
      if (trackPt > settings_->vx_TrackMaxPt()) {
        highPt = true;
        numHighPtTracks++;
        highestPt = (trackPt > highestPt) ? trackPt : highestPt;
        if (settings_->vx_TrackMaxPtBehavior() == 0)
          continue;  // ignore this track
        else if (settings_->vx_TrackMaxPtBehavior() == 1)
          trackPt = settings_->vx_TrackMaxPt();  // saturate
      }

      pt += std::pow(trackPt, settings_->vx_weightedmean());
      if (bin_centers.empty() && counts.empty()) {
        SumZ += track->z0() * std::pow(trackPt, settings_->vx_weightedmean());
        z0square += track->z0() * track->z0();
      } else {
        bin_pt[ibin] += std::pow(trackPt, settings_->vx_weightedmean());
        if (itrack == counts[ibin]) {
          SumZ += bin_centers[ibin] * bin_pt[ibin];
          z0square += bin_centers[ibin] * bin_centers[ibin];
          itrack = 0;
          ibin++;
        }
      }
    }

    z0 = SumZ / ((settings_->vx_weightedmean() > 0) ? pt : vertex.numTracks());
    z0square /= vertex.numTracks();
    z0width = sqrt(std::abs(z0 * z0 - z0square));

    vertex.setParameters(pt, z0, z0width, highPt, numHighPtTracks, highestPt);
  }

  void VertexFinder::GapClustering() {
    sort(fitTracks_.begin(), fitTracks_.end(), SortTracksByZ0());
    iterations_ = 0;
    RecoVertex Vertex;
    for (unsigned int i = 0; i < fitTracks_.size(); ++i) {
      Vertex.insert(&fitTracks_[i]);
      iterations_++;
      if ((i + 1 < fitTracks_.size() and fitTracks_[i + 1].z0() - fitTracks_[i].z0() > settings_->vx_distance()) or
          i == fitTracks_.size() - 1) {
        if (Vertex.numTracks() >= settings_->vx_minTracks()) {
          computeAndSetVertexParameters(Vertex, {}, {});
          vertices_.push_back(Vertex);
        }
        Vertex.clear();
      }
    }
  }

  float VertexFinder::maxDistance(RecoVertex<> cluster0, RecoVertex<> cluster1) {
    float distance = 0;
    for (const L1Track* track0 : cluster0.tracks()) {
      for (const L1Track* track1 : cluster1.tracks()) {
        if (std::abs(track0->z0() - track1->z0()) > distance) {
          distance = std::abs(track0->z0() - track1->z0());
        }
      }
    }

    return distance;
  }

  float VertexFinder::minDistance(RecoVertex<> cluster0, RecoVertex<> cluster1) {
    float distance = 9999;
    for (const L1Track* track0 : cluster0.tracks()) {
      for (const L1Track* track1 : cluster1.tracks()) {
        if (std::abs(track0->z0() - track1->z0()) < distance) {
          distance = std::abs(track0->z0() - track1->z0());
        }
      }
    }

    return distance;
  }

  float VertexFinder::meanDistance(RecoVertex<> cluster0, RecoVertex<> cluster1) {
    float distanceSum = 0;

    for (const L1Track* track0 : cluster0.tracks()) {
      for (const L1Track* track1 : cluster1.tracks()) {
        distanceSum += std::abs(track0->z0() - track1->z0());
      }
    }

    float distance = distanceSum / (cluster0.numTracks() * cluster1.numTracks());
    return distance;
  }

  float VertexFinder::centralDistance(RecoVertex<> cluster0, RecoVertex<> cluster1) {
    computeAndSetVertexParameters(cluster0, {}, {});
    computeAndSetVertexParameters(cluster1, {}, {});
    float distance = std::abs(cluster0.z0() - cluster1.z0());
    return distance;
  }

  void VertexFinder::agglomerativeHierarchicalClustering() {
    iterations_ = 0;

    sort(fitTracks_.begin(), fitTracks_.end(), SortTracksByZ0());

    std::vector<RecoVertex<>> vClusters;
    vClusters.resize(fitTracks_.size());

    for (unsigned int i = 0; i < fitTracks_.size(); ++i) {
      vClusters[i].insert(&fitTracks_[i]);
      // iterations_++;
    }

    while (true) {
      float MinimumScore = 9999;

      unsigned int clusterId0 = 0;
      unsigned int clusterId1 = 0;
      for (unsigned int iClust = 0; iClust < vClusters.size() - 1; iClust++) {
        iterations_++;

        float M = 0;
        if (settings_->vx_distanceType() == 0)
          M = maxDistance(vClusters[iClust], vClusters[iClust + 1]);
        else if (settings_->vx_distanceType() == 1)
          M = minDistance(vClusters[iClust], vClusters[iClust + 1]);
        else if (settings_->vx_distanceType() == 2)
          M = meanDistance(vClusters[iClust], vClusters[iClust + 1]);
        else
          M = centralDistance(vClusters[iClust], vClusters[iClust + 1]);

        if (M < MinimumScore) {
          MinimumScore = M;
          clusterId0 = iClust;
          clusterId1 = iClust + 1;
        }
      }
      if (MinimumScore > settings_->vx_distance() or vClusters[clusterId1].tracks().empty())
        break;
      for (const L1Track* track : vClusters[clusterId0].tracks()) {
        vClusters[clusterId1].insert(track);
      }
      vClusters.erase(vClusters.begin() + clusterId0);
    }

    for (RecoVertex clust : vClusters) {
      if (clust.numTracks() >= settings_->vx_minTracks()) {
        computeAndSetVertexParameters(clust, {}, {});
        vertices_.push_back(clust);
      }
    }
  }

  void VertexFinder::DBSCAN() {
    // std::vector<RecoVertex> vClusters;
    std::vector<unsigned int> visited;
    std::vector<unsigned int> saved;

    sort(fitTracks_.begin(), fitTracks_.end(), SortTracksByPt());
    iterations_ = 0;

    for (unsigned int i = 0; i < fitTracks_.size(); ++i) {
      if (find(visited.begin(), visited.end(), i) != visited.end())
        continue;

      // if(fitTracks_[i]->pt()>10.){
      visited.push_back(i);
      std::set<unsigned int> neighbourTrackIds;
      unsigned int numDensityTracks = 0;
      if (fitTracks_[i].pt() > settings_->vx_dbscan_pt())
        numDensityTracks++;
      else
        continue;
      for (unsigned int k = 0; k < fitTracks_.size(); ++k) {
        iterations_++;
        if (k != i and std::abs(fitTracks_[k].z0() - fitTracks_[i].z0()) < settings_->vx_distance()) {
          neighbourTrackIds.insert(k);
          if (fitTracks_[k].pt() > settings_->vx_dbscan_pt()) {
            numDensityTracks++;
          }
        }
      }

      if (numDensityTracks < settings_->vx_dbscan_mintracks()) {
        // mark track as noise
      } else {
        RecoVertex vertex;
        vertex.insert(&fitTracks_[i]);
        saved.push_back(i);
        for (unsigned int id : neighbourTrackIds) {
          if (find(visited.begin(), visited.end(), id) == visited.end()) {
            visited.push_back(id);
            std::vector<unsigned int> neighbourTrackIds2;
            for (unsigned int k = 0; k < fitTracks_.size(); ++k) {
              iterations_++;
              if (std::abs(fitTracks_[k].z0() - fitTracks_[id].z0()) < settings_->vx_distance()) {
                neighbourTrackIds2.push_back(k);
              }
            }

            // if (neighbourTrackIds2.size() >= settings_->vx_minTracks()) {
            for (unsigned int id2 : neighbourTrackIds2) {
              neighbourTrackIds.insert(id2);
            }
            // }
          }
          if (find(saved.begin(), saved.end(), id) == saved.end())
            vertex.insert(&fitTracks_[id]);
        }
        computeAndSetVertexParameters(vertex, {}, {});
        if (vertex.numTracks() >= settings_->vx_minTracks())
          vertices_.push_back(vertex);
      }
      // }
    }
  }

  void VertexFinder::PVR() {
    bool start = true;
    FitTrackCollection discardedTracks, acceptedTracks;
    iterations_ = 0;
    for (const L1Track& track : fitTracks_) {
      acceptedTracks.push_back(track);
    }

    while (discardedTracks.size() >= settings_->vx_minTracks() or start == true) {
      start = false;
      bool removing = true;
      discardedTracks.clear();
      while (removing) {
        float oldDistance = 0.;

        if (settings_->debug() > 2)
          edm::LogInfo("VertexFinder") << "PVR::AcceptedTracks " << acceptedTracks.size();

        float z0start = 0;
        for (const L1Track& track : acceptedTracks) {
          z0start += track.z0();
          iterations_++;
        }

        z0start /= acceptedTracks.size();
        if (settings_->debug() > 2)
          edm::LogInfo("VertexFinder") << "PVR::z0 vertex " << z0start;
        FitTrackCollection::iterator badTrackIt = acceptedTracks.end();
        removing = false;

        for (FitTrackCollection::iterator it = acceptedTracks.begin(); it < acceptedTracks.end(); ++it) {
          const L1Track* track = &*it;
          iterations_++;
          if (std::abs(track->z0() - z0start) > settings_->vx_distance() and
              std::abs(track->z0() - z0start) > oldDistance) {
            badTrackIt = it;
            oldDistance = std::abs(track->z0() - z0start);
            removing = true;
          }
        }

        if (removing) {
          const L1Track badTrack = *badTrackIt;
          if (settings_->debug() > 2)
            edm::LogInfo("VertexFinder") << "PVR::Removing track " << badTrack.z0() << " at distance " << oldDistance;
          discardedTracks.push_back(badTrack);
          acceptedTracks.erase(badTrackIt);
        }
      }

      if (acceptedTracks.size() >= settings_->vx_minTracks()) {
        RecoVertex vertex;
        for (const L1Track& track : acceptedTracks) {
          vertex.insert(&track);
        }
        computeAndSetVertexParameters(vertex, {}, {});
        vertices_.push_back(vertex);
      }
      if (settings_->debug() > 2)
        edm::LogInfo("VertexFinder") << "PVR::DiscardedTracks size " << discardedTracks.size();
      acceptedTracks.clear();
      acceptedTracks = discardedTracks;
    }
  }

  void VertexFinder::adaptiveVertexReconstruction() {
    bool start = true;
    iterations_ = 0;
    FitTrackCollection discardedTracks, acceptedTracks, discardedTracks2;

    for (const L1Track& track : fitTracks_) {
      discardedTracks.push_back(track);
    }

    while (discardedTracks.size() >= settings_->vx_minTracks() or start == true) {
      start = false;
      discardedTracks2.clear();
      FitTrackCollection::iterator it = discardedTracks.begin();
      const L1Track track = *it;
      acceptedTracks.push_back(track);
      float z0sum = track.z0();

      for (FitTrackCollection::iterator it2 = discardedTracks.begin(); it2 < discardedTracks.end(); ++it2) {
        if (it2 != it) {
          const L1Track secondTrack = *it2;
          // Calculate new vertex z0 adding this track
          z0sum += secondTrack.z0();
          float z0vertex = z0sum / (acceptedTracks.size() + 1);
          // Calculate chi2 of new vertex
          float chi2 = 0.;
          float dof = 0.;
          for (const L1Track& accTrack : acceptedTracks) {
            iterations_++;
            float Residual = accTrack.z0() - z0vertex;
            if (std::abs(accTrack.eta()) < 1.2)
              Residual /= 0.1812;  // Assumed z0 resolution
            else if (std::abs(accTrack.eta()) >= 1.2 && std::abs(accTrack.eta()) < 1.6)
              Residual /= 0.2912;
            else if (std::abs(accTrack.eta()) >= 1.6 && std::abs(accTrack.eta()) < 2.)
              Residual /= 0.4628;
            else
              Residual /= 0.65;

            chi2 += Residual * Residual;
            dof = (acceptedTracks.size() + 1) * 2 - 1;
          }
          if (chi2 / dof < settings_->vx_chi2cut()) {
            acceptedTracks.push_back(secondTrack);
          } else {
            discardedTracks2.push_back(secondTrack);
            z0sum -= secondTrack.z0();
          }
        }
      }

      if (acceptedTracks.size() >= settings_->vx_minTracks()) {
        RecoVertex vertex;
        for (const L1Track& track : acceptedTracks) {
          vertex.insert(&track);
        }
        computeAndSetVertexParameters(vertex, {}, {});
        vertices_.push_back(vertex);
      }

      acceptedTracks.clear();
      discardedTracks.clear();
      discardedTracks = discardedTracks2;
    }
  }

  void VertexFinder::HPV() {
    iterations_ = 0;
    sort(fitTracks_.begin(), fitTracks_.end(), SortTracksByPt());

    RecoVertex vertex;
    bool first = true;
    float z = 99.;
    for (const L1Track& track : fitTracks_) {
      if (track.pt() < 50.) {
        if (first) {
          first = false;
          z = track.z0();
          vertex.insert(&track);
        } else {
          if (std::abs(track.z0() - z) < settings_->vx_distance())
            vertex.insert(&track);
        }
      }
    }

    computeAndSetVertexParameters(vertex, {}, {});
    vertex.setZ0(z);
    vertices_.push_back(vertex);
  }

  void VertexFinder::Kmeans() {
    unsigned int NumberOfClusters = settings_->vx_kmeans_nclusters();

    vertices_.resize(NumberOfClusters);
    float ClusterSeparation = 30. / NumberOfClusters;

    for (unsigned int i = 0; i < NumberOfClusters; ++i) {
      float ClusterCentre = -15. + ClusterSeparation * (i + 0.5);
      vertices_[i].setZ0(ClusterCentre);
    }
    unsigned int iterations = 0;
    // Initialise Clusters
    while (iterations < settings_->vx_kmeans_iterations()) {
      for (unsigned int i = 0; i < NumberOfClusters; ++i) {
        vertices_[i].clear();
      }

      for (const L1Track& track : fitTracks_) {
        float distance = 9999;
        if (iterations == settings_->vx_kmeans_iterations() - 3)
          distance = settings_->vx_distance() * 2;
        if (iterations > settings_->vx_kmeans_iterations() - 3)
          distance = settings_->vx_distance();
        unsigned int ClusterId;
        bool NA = true;
        for (unsigned int id = 0; id < NumberOfClusters; ++id) {
          if (std::abs(track.z0() - vertices_[id].z0()) < distance) {
            distance = std::abs(track.z0() - vertices_[id].z0());
            ClusterId = id;
            NA = false;
          }
        }
        if (!NA) {
          vertices_[ClusterId].insert(&track);
        }
      }
      for (unsigned int i = 0; i < NumberOfClusters; ++i) {
        if (vertices_[i].numTracks() >= settings_->vx_minTracks())
          computeAndSetVertexParameters(vertices_[i], {}, {});
      }
      iterations++;
    }
  }

  void VertexFinder::findPrimaryVertex() {
    double vertexPt = 0;
    pv_index_ = 0;

    for (unsigned int i = 0; i < vertices_.size(); ++i) {
      if (vertices_[i].pt() > vertexPt) {
        vertexPt = vertices_[i].pt();
        pv_index_ = i;
      }
    }
  }

  void VertexFinder::associatePrimaryVertex(double trueZ0) {
    double distance = 999.;
    for (unsigned int id = 0; id < vertices_.size(); ++id) {
      if (std::abs(trueZ0 - vertices_[id].z0()) < distance) {
        distance = std::abs(trueZ0 - vertices_[id].z0());
        pv_index_ = id;
      }
    }
  }

  void VertexFinder::fastHistoLooseAssociation() {
    float vxPt = 0.;
    RecoVertex leading_vertex;

    for (float z = settings_->vx_histogram_min(); z < settings_->vx_histogram_max();
         z += settings_->vx_histogram_binwidth()) {
      RecoVertex vertex;
      for (const L1Track& track : fitTracks_) {
        if (std::abs(z - track.z0()) < settings_->vx_width()) {
          vertex.insert(&track);
        }
      }
      computeAndSetVertexParameters(vertex, {}, {});
      vertex.setZ0(z);
      if (vertex.pt() > vxPt) {
        leading_vertex = vertex;
        vxPt = vertex.pt();
      }
    }

    vertices_.emplace_back(leading_vertex);
    pv_index_ = 0;  // by default fastHistoLooseAssociation algorithm finds only hard PV
  }                 // end of fastHistoLooseAssociation

  void VertexFinder::fastHisto(const TrackerTopology* tTopo) {
    // Create the histogram
    int nbins =
        std::ceil((settings_->vx_histogram_max() - settings_->vx_histogram_min()) / settings_->vx_histogram_binwidth());
    std::vector<RecoVertex<>> hist(nbins);
    std::vector<RecoVertex<>> sums(nbins - settings_->vx_windowSize());
    std::vector<float> bounds(nbins + 1);
    strided_iota(std::begin(bounds),
                 std::next(std::begin(bounds), nbins + 1),
                 settings_->vx_histogram_min(),
                 settings_->vx_histogram_binwidth());

    // Loop over the tracks and fill the histogram
    for (const L1Track& track : fitTracks_) {
      if ((track.z0() < settings_->vx_histogram_min()) || (track.z0() > settings_->vx_histogram_max()))
        continue;
      if (track.getTTTrackPtr()->chi2() > settings_->vx_TrackMaxChi2())
        continue;
      if (track.pt() < settings_->vx_TrackMinPt())
        continue;

      // Get the number of stubs and the number of stubs in PS layers
      float nPS = 0., nstubs = 0;

      // Get pointers to stubs associated to the L1 track
      const auto& theStubs = track.getTTTrackPtr()->getStubRefs();
      if (theStubs.empty()) {
        edm::LogWarning("VertexFinder") << "fastHisto::Could not retrieve the vector of stubs.";
        continue;
      }

      // Loop over the stubs
      for (const auto& stub : theStubs) {
        nstubs++;
        bool isPS = false;
        DetId detId(stub->getDetId());
        if (detId.det() == DetId::Detector::Tracker) {
          if (detId.subdetId() == StripSubdetector::TOB && tTopo->tobLayer(detId) <= 3)
            isPS = true;
          else if (detId.subdetId() == StripSubdetector::TID && tTopo->tidRing(detId) <= 9)
            isPS = true;
        }
        if (isPS)
          nPS++;
      }  // End loop over stubs
      if (nstubs < settings_->vx_NStubMin())
        continue;
      if (nPS < settings_->vx_NStubPSMin())
        continue;

      // Quality cuts, may need to be re-optimized
      int trk_nstub = (int)track.getTTTrackPtr()->getStubRefs().size();
      float chi2dof = track.getTTTrackPtr()->chi2() / (2 * trk_nstub - 4);

      if (settings_->vx_DoPtComp()) {
        float trk_consistency = track.getTTTrackPtr()->stubPtConsistency();
        if (trk_nstub == 4) {
          if (std::abs(track.eta()) < 2.2 && trk_consistency > 10)
            continue;
          else if (std::abs(track.eta()) > 2.2 && chi2dof > 5.0)
            continue;
        }
      }
      if (settings_->vx_DoTightChi2()) {
        if (track.pt() > 10.0 && chi2dof > 5.0)
          continue;
      }

      // Assign the track to the correct vertex
      auto upper_bound = std::lower_bound(bounds.begin(), bounds.end(), track.z0());
      int index = std::distance(bounds.begin(), upper_bound) - 1;
      hist.at(index).insert(&track);
    }  // end loop over tracks

    // Compute the sums
    // sliding windows ... sum_i_i+(w-1) where i in (0,nbins-w) and w is the window size
    std::vector<float> bin_centers(settings_->vx_windowSize(), 0.0);
    std::vector<unsigned int> counts(settings_->vx_windowSize(), 0);
    for (unsigned int i = 0; i < sums.size(); i++) {
      for (unsigned int j = 0; j < settings_->vx_windowSize(); j++) {
        bin_centers[j] = settings_->vx_histogram_min() + ((i + j) * settings_->vx_histogram_binwidth()) +
                         (0.5 * settings_->vx_histogram_binwidth());
        counts[j] = hist.at(i + j).numTracks();
        sums.at(i) += hist.at(i + j);
      }
      computeAndSetVertexParameters(sums.at(i), bin_centers, counts);
    }

    // Find the maxima of the sums
    float sigma_max = -999;
    int imax = -999;
    std::vector<int> found;
    found.reserve(settings_->vx_nvtx());
    for (unsigned int ivtx = 0; ivtx < settings_->vx_nvtx(); ivtx++) {
      sigma_max = -999;
      imax = -999;
      for (unsigned int i = 0; i < sums.size(); i++) {
        // Skip this window if it will already be returned
        if (find(found.begin(), found.end(), i) != found.end())
          continue;
        if (sums.at(i).pt() > sigma_max) {
          sigma_max = sums.at(i).pt();
          imax = i;
        }
      }

      found.push_back(imax);
      vertices_.emplace_back(sums.at(imax));
    }
    pv_index_ = 0;
  }  // end of fastHisto

}  // namespace l1tVertexFinder
