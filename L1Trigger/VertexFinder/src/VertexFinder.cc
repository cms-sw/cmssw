#include "L1Trigger/VertexFinder/interface/VertexFinder.h"

using namespace std;

namespace l1tVertexFinder {

  void VertexFinder::computeAndSetVertexParameters(RecoVertex<>& vertex,
                                                   const std::vector<float>& bin_centers,
                                                   const std::vector<unsigned int>& counts) {
    double pt = 0.;
    double z0 = -999.;
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

      // Skip the bins with no tracks
      while (ibin < counts.size() && counts[ibin] == 0)
        ibin++;

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

    RecoVertexCollection vClusters;
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
    if (settings_->vx_precision() == Precision::Emulation) {
      pv_index_ = std::distance(verticesEmulation_.begin(),
                                std::max_element(verticesEmulation_.begin(),
                                                 verticesEmulation_.end(),
                                                 [](const l1t::VertexWord& vertex0, const l1t::VertexWord& vertex1) {
                                                   return (vertex0.pt() < vertex1.pt());
                                                 }));
    } else {
      pv_index_ = std::distance(
          vertices_.begin(),
          std::max_element(
              vertices_.begin(), vertices_.end(), [](const RecoVertex<>& vertex0, const RecoVertex<>& vertex1) {
                return (vertex0.pt() < vertex1.pt());
              }));
    }
  }

  // Possible Formatting Codes: https://misc.flogisoft.com/bash/tip_colors_and_formatting
  template <class data_type, typename stream_type>
  void VertexFinder::printHistogram(stream_type& stream,
                                    std::vector<data_type> data,
                                    int width,
                                    int minimum,
                                    int maximum,
                                    std::string title,
                                    std::string color) {
    int tableSize = data.size();

    if (maximum == -1) {
      maximum = float(*std::max_element(std::begin(data), std::end(data))) * 1.05;
    } else if (maximum <= minimum) {
      maximum = float(*std::max_element(std::begin(data), std::end(data))) * 1.05;
      minimum = float(*std::min_element(std::begin(data), std::end(data)));
    }

    if (minimum < 0) {
      minimum *= 1.05;
    } else {
      minimum = 0;
    }

    std::vector<std::string> intervals(tableSize, "");
    std::vector<std::string> values(tableSize, "");
    char buffer[128];
    int intervalswidth = 0, valueswidth = 0, tmpwidth = 0;
    for (int i = 0; i < tableSize; i++) {
      //Format the bin labels
      tmpwidth = sprintf(buffer, "[%-.5g, %-.5g)", float(i), float(i + 1));
      intervals[i] = buffer;
      if (i == (tableSize - 1)) {
        intervals[i][intervals[i].size() - 1] = ']';
      }
      if (tmpwidth > intervalswidth)
        intervalswidth = tmpwidth;

      //Format the values
      tmpwidth = sprintf(buffer, "%-.5g", float(data[i]));
      values[i] = buffer;
      if (tmpwidth > valueswidth)
        valueswidth = tmpwidth;
    }

    sprintf(buffer, "%-.5g", float(minimum));
    std::string minimumtext = buffer;
    sprintf(buffer, "%-.5g", float(maximum));
    std::string maximumtext = buffer;

    int plotwidth =
        std::max(int(minimumtext.size() + maximumtext.size()), width - (intervalswidth + 1 + valueswidth + 1 + 2));
    std::string scale =
        minimumtext + std::string(plotwidth + 2 - minimumtext.size() - maximumtext.size(), ' ') + maximumtext;

    float norm = float(plotwidth) / float(maximum - minimum);
    int zero = std::round((0.0 - minimum) * norm);
    std::vector<char> line(plotwidth, '-');

    if ((minimum != 0) && (0 <= zero) && (zero < plotwidth)) {
      line[zero] = '+';
    }
    std::string capstone =
        std::string(intervalswidth + 1 + valueswidth + 1, ' ') + "+" + std::string(line.begin(), line.end()) + "+";

    std::vector<std::string> out;
    if (!title.empty()) {
      out.push_back(title);
      out.push_back(std::string(title.size(), '='));
    }
    out.push_back(std::string(intervalswidth + valueswidth + 2, ' ') + scale);
    out.push_back(capstone);
    for (int i = 0; i < tableSize; i++) {
      std::string interval = intervals[i];
      std::string value = values[i];
      data_type x = data[i];
      std::fill_n(line.begin(), plotwidth, ' ');

      int pos = std::round((float(x) - minimum) * norm);
      if (x < 0) {
        std::fill_n(line.begin() + pos, zero - pos, '*');
      } else {
        std::fill_n(line.begin() + zero, pos - zero, '*');
      }

      if ((minimum != 0) && (0 <= zero) && (zero < plotwidth)) {
        line[zero] = '|';
      }

      sprintf(buffer,
              "%-*s %-*s |%s|",
              intervalswidth,
              interval.c_str(),
              valueswidth,
              value.c_str(),
              std::string(line.begin(), line.end()).c_str());
      out.push_back(buffer);
    }
    out.push_back(capstone);
    if (!color.empty())
      stream << color;
    for (const auto& o : out) {
      stream << o << "\n";
    }
    if (!color.empty())
      stream << "\e[0m";
    stream << "\n";
  }

  void VertexFinder::sortVerticesInPt() {
    if (settings_->vx_precision() == Precision::Emulation) {
      std::sort(
          verticesEmulation_.begin(),
          verticesEmulation_.end(),
          [](const l1t::VertexWord& vertex0, const l1t::VertexWord& vertex1) { return (vertex0.pt() > vertex1.pt()); });
    } else {
      std::sort(vertices_.begin(), vertices_.end(), [](const RecoVertex<>& vertex0, const RecoVertex<>& vertex1) {
        return (vertex0.pt() > vertex1.pt());
      });
    }
  }

  void VertexFinder::sortVerticesInZ0() {
    if (settings_->vx_precision() == Precision::Emulation) {
      std::sort(
          verticesEmulation_.begin(),
          verticesEmulation_.end(),
          [](const l1t::VertexWord& vertex0, const l1t::VertexWord& vertex1) { return (vertex0.z0() < vertex1.z0()); });
    } else {
      std::sort(vertices_.begin(), vertices_.end(), [](const RecoVertex<>& vertex0, const RecoVertex<>& vertex1) {
        return (vertex0.z0() < vertex1.z0());
      });
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
    std::vector<RecoVertex<>> sums(nbins - settings_->vx_windowSize() + 1);
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
      // The values are ordered with bounds [lower, upper)
      // Values below bounds.begin() return 0 as the index (underflow)
      // Values above bounds.end() will return the index of the last bin (overflow)
      auto upper_bound = std::upper_bound(bounds.begin(), bounds.end(), track.z0());
      int index = std::distance(bounds.begin(), upper_bound) - 1;
      if (index == -1)
        index = 0;
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

    if (settings_->debug() >= 1) {
      edm::LogInfo log("VertexProducer");
      log << "fastHisto::Checking the output parameters ... \n";
      std::vector<double> tmp;
      std::transform(std::begin(sums), std::end(sums), std::back_inserter(tmp), [](const RecoVertex<>& v) -> double {
        return v.pt();
      });
      printHistogram<double, edm::LogInfo>(log, tmp, 80, 0, -1, "fastHisto::sums", "\e[92m");
      for (unsigned int i = 0; i < found.size(); i++) {
        log << "RecoVertex " << i << ": bin index = " << found[i] << "\tsumPt = " << sums.at(imax).pt()
            << "\tz0 = " << sums.at(imax).z0();
      }
    }
  }  // end of fastHisto

  void VertexFinder::fastHistoEmulation() {
    // Relevant constants for the track word
    enum TrackBitWidths {
      kZ0Size = 12,             // Width of z-position (40cm / 0.1)
      kZ0MagSize = 5,           // Width of z-position magnitude (signed)
      kPtSize = 14,             // Width of pt
      kPtMagSize = 9,           // Width of pt magnitude (unsigned)
      kReducedPrecisionPt = 7,  // Width of the reduced precision, integer only, pt
    };

    enum HistogramBitWidths {
      kBinSize = 8,                     // Width of a single bin in z
      kBinFixedSize = 8,                // Width of a single z0 bin in fixed point representation
      kBinFixedMagSize = 5,             // Width (magnitude) of a single z0 bin in fixed point representation
      kSlidingSumSize = 11,             // Width of the sum of a window of bins
      kInverseSize = 14,                // Width of the inverse sum
      kInverseMagSize = 1,              // Width of the inverse sum magnitude (unsigned)
      kWeightedSlidingSumSize = 20,     // Width of the pT weighted sliding sum
      kWeightedSlidingSumMagSize = 10,  // Width of the pT weighted sliding sum magnitude (signed)
      kWindowSize = 3,                  // Number of bins in the window used to sum histogram bins
      kSumPtLinkSize = 9,  // Number of bits used to represent the sum of track pts in a single bin from a single link

      kSumPtWindowBits = BitsToRepresent(HistogramBitWidths::kWindowSize * (1 << HistogramBitWidths::kSumPtLinkSize)),
      // Number of bits to represent the untruncated sum of track pts in a single bin from a single link
      kSumPtUntruncatedLinkSize = TrackBitWidths::kPtSize + 2,
      kSumPtUntruncatedLinkMagSize = TrackBitWidths::kPtMagSize + 2,
    };

    static constexpr unsigned int kTableSize =
        ((1 << HistogramBitWidths::kSumPtLinkSize) - 1) * HistogramBitWidths::kWindowSize;

    typedef ap_ufixed<TrackBitWidths::kPtSize, TrackBitWidths::kPtMagSize, AP_RND_CONV, AP_SAT> pt_t;
    // Same size as TTTrack_TrackWord::z0_t, but now taking into account the sign bit (i.e. 2's complement)
    typedef ap_int<TrackBitWidths::kZ0Size> z0_t;
    // 7 bits chosen to represent values between [0,127]
    // This is the next highest power of 2 value to our chosen track pt saturation value (100)
    typedef ap_ufixed<TrackBitWidths::kReducedPrecisionPt, TrackBitWidths::kReducedPrecisionPt, AP_RND_INF, AP_SAT>
        track_pt_fixed_t;
    // Histogram bin index
    typedef ap_uint<HistogramBitWidths::kBinSize> histbin_t;
    // Histogram bin in fixed point representation, before truncation
    typedef ap_ufixed<HistogramBitWidths::kBinFixedSize, HistogramBitWidths::kBinFixedMagSize, AP_RND_INF, AP_SAT>
        histbin_fixed_t;
    // This type is slightly arbitrary, but 2 bits larger than untruncated track pt to store sums in histogram bins
    // with truncation just before vertex-finding
    typedef ap_ufixed<HistogramBitWidths::kSumPtUntruncatedLinkSize,
                      HistogramBitWidths::kSumPtUntruncatedLinkMagSize,
                      AP_RND_INF,
                      AP_SAT>
        histbin_pt_sum_fixed_t;
    // This value is slightly arbitrary, but small enough that the windows sums aren't too big.
    typedef ap_ufixed<HistogramBitWidths::kSumPtLinkSize, HistogramBitWidths::kSumPtLinkSize, AP_RND_INF, AP_SAT>
        link_pt_sum_fixed_t;
    // Enough bits to store HistogramBitWidths::kWindowSize * (2**HistogramBitWidths::kSumPtLinkSize)
    typedef ap_ufixed<HistogramBitWidths::kSumPtWindowBits, HistogramBitWidths::kSumPtWindowBits, AP_RND_INF, AP_SAT>
        window_pt_sum_fixed_t;
    // pt weighted sum of bins in window
    typedef ap_fixed<HistogramBitWidths::kWeightedSlidingSumSize,
                     HistogramBitWidths::kWeightedSlidingSumMagSize,
                     AP_RND_INF,
                     AP_SAT>
        zsliding_t;
    // Sum of histogram bins in window
    typedef ap_uint<HistogramBitWidths::kSlidingSumSize> slidingsum_t;
    // Inverse of sum of bins in a given window
    typedef ap_ufixed<HistogramBitWidths::kInverseSize, HistogramBitWidths::kInverseMagSize, AP_RND_INF, AP_SAT>
        inverse_t;

    auto track_quality_check = [&](const track_pt_fixed_t& pt) -> bool {
      // Track quality cuts
      if (pt.to_double() < settings_->vx_TrackMinPt())
        return false;
      return true;
    };

    auto fetch_bin = [&](const z0_t& z0, int nbins) -> std::pair<histbin_t, bool> {
      // Increase the the number of bits in the word to allow for additional dynamic range
      ap_int<TrackBitWidths::kZ0Size + 1> z0_13 = z0;
      // Add a number equal to half of the range in z0, meaning that the range is now [0, 2*z0_max]
      ap_int<TrackBitWidths::kZ0Size + 1> absz0_13 = z0_13 + (1 << (TrackBitWidths::kZ0Size - 1));
      // Shift the bits down to truncate the dynamic range to the most significant HistogramBitWidths::kBinFixedSize bits
      ap_int<TrackBitWidths::kZ0Size + 1> absz0_13_reduced =
          absz0_13 >> (TrackBitWidths::kZ0Size - HistogramBitWidths::kBinFixedSize);
      // Put the relevant bits into the histbin_t container
      histbin_t bin = absz0_13_reduced.range(HistogramBitWidths::kBinFixedSize - 1, 0);

      if (settings_->debug() > 2) {
        edm::LogInfo("VertexProducer")
            << "fastHistoEmulation::fetchBin() Checking the mapping from z0 to bin index ... \n"
            << "histbin_fixed_t(1.0 / settings_->vx_histogram_binwidth()) = "
            << histbin_fixed_t(1.0 / settings_->vx_histogram_binwidth()) << "\n"
            << "histbin_t(std::floor(nbins / 2) = " << histbin_t(std::floor(nbins / 2.)) << "\n"
            << "z0 = " << z0 << "\n"
            << "bin = " << bin;
      }
      bool valid = true;
      if (bin < 0) {
        return std::make_pair(0, false);
      } else if (bin > (nbins - 1)) {
        return std::make_pair(0, false);
      }
      return std::make_pair(bin, valid);
    };

    // Replace with https://stackoverflow.com/questions/13313980/populate-an-array-using-constexpr-at-compile-time ?
    auto init_inversion_table = [&]() -> std::vector<inverse_t> {
      std::vector<inverse_t> table_out(kTableSize, 0.);
      for (unsigned int ii = 0; ii < kTableSize; ii++) {
        // Compute lookup table function. This matches the format of the GTT HLS code.
        // Biased generation f(x) = 1 / (x + 1) is inverted by g(y) = inversion(x - 1) = 1 / (x - 1 + 1) = 1 / y
        table_out.at(ii) = (1.0 / (ii + 1));
      }
      return table_out;
    };

    auto inversion = [&](slidingsum_t& data_den) -> inverse_t {
      std::vector<inverse_t> inversion_table = init_inversion_table();

      // Index into the lookup table based on data
      int index;
      if (data_den < 0)
        data_den = 0;
      if (data_den > (kTableSize - 1))
        data_den = kTableSize - 1;
      index = data_den;
      return inversion_table.at(index);
    };

    auto bin_center = [&](zsliding_t iz, int nbins) -> l1t::VertexWord::vtxz0_t {
      zsliding_t z = iz - histbin_t(std::floor(nbins / 2.));
      std::unique_ptr<edm::LogInfo> log;
      if (settings_->debug() >= 1) {
        log = std::make_unique<edm::LogInfo>("VertexProducer");
        *log << "bin_center information ...\n"
             << "iz = " << iz << "\n"
             << "histbin_t(std::floor(nbins / 2.)) = " << histbin_t(std::floor(nbins / 2.)) << "\n"
             << "binwidth = " << zsliding_t(settings_->vx_histogram_binwidth()) << "\n"
             << "z = " << z << "\n"
             << "zsliding_t(z * zsliding_t(binwidth)) = " << std::setprecision(7)
             << l1t::VertexWord::vtxz0_t(z * zsliding_t(settings_->vx_histogram_binwidth()));
      }
      return l1t::VertexWord::vtxz0_t(z * zsliding_t(settings_->vx_histogram_binwidth()));
    };

    auto weighted_position = [&](histbin_t b_max,
                                 const std::vector<link_pt_sum_fixed_t>& binpt,
                                 slidingsum_t maximums,
                                 int nbins) -> zsliding_t {
      zsliding_t zvtx_sliding = 0;
      slidingsum_t zvtx_sliding_sum = 0;
      inverse_t inv = 0;

      std::unique_ptr<edm::LogInfo> log;
      if (settings_->debug() >= 1) {
        log = std::make_unique<edm::LogInfo>("VertexProducer");
        *log << "Progression of weighted_position() ...\n"
             << "zvtx_sliding_sum = ";
      }

      // Find the weighted position within the window in index space (width = 1)
      for (ap_uint<BitsToRepresent(HistogramBitWidths::kWindowSize)> w = 0; w < HistogramBitWidths::kWindowSize; ++w) {
        zvtx_sliding_sum += (binpt.at(w) * w);
        if (settings_->debug() >= 1) {
          *log << "(" << w << " * " << binpt.at(w) << ")";
          if (w < HistogramBitWidths::kWindowSize - 1) {
            *log << " + ";
          }
        }
      }

      if (settings_->debug() >= 1) {
        *log << " = " << zvtx_sliding_sum << "\n";
      }

      if (maximums != 0) {
        //match F/W inversion_lut offset (inversion[x] = 1 / (x + 1); inversion[x - 1] = 1 / x;), for consistency
        slidingsum_t offsetmaximums = maximums - 1;
        inv = inversion(offsetmaximums);
        zvtx_sliding = zvtx_sliding_sum * inv;
      } else {
        zvtx_sliding = (settings_->vx_windowSize() / 2.0) + (((int(settings_->vx_windowSize()) % 2) != 0) ? 0.5 : 0.0);
      }
      if (settings_->debug() >= 1) {
        *log << "inversion(" << maximums << ") = " << inv << "\nzvtx_sliding = " << zvtx_sliding << "\n";
      }

      // Add the starting index plus half an index to shift the z position to its weighted position (still in inxex space) within all of the bins
      zvtx_sliding += b_max;
      zvtx_sliding += ap_ufixed<1, 0>(0.5);
      if (settings_->debug() >= 1) {
        *log << "b_max = " << b_max << "\n";
        *log << "zvtx_sliding + b_max + 0.5 = " << zvtx_sliding << "\n";
      }

      // Shift the z position from index space into z [cm] space
      zvtx_sliding = bin_center(zvtx_sliding, nbins);
      if (settings_->debug() >= 1) {
        *log << "bin_center(zvtx_sliding + b_max + 0.5, nbins) = " << std::setprecision(7) << zvtx_sliding;
        log.reset();
      }
      return zvtx_sliding;
    };

    // Create the histogram
    unsigned int nbins = std::round((settings_->vx_histogram_max() - settings_->vx_histogram_min()) /
                                    settings_->vx_histogram_binwidth());
    unsigned int nsums = nbins - settings_->vx_windowSize() + 1;
    std::vector<link_pt_sum_fixed_t> hist(nbins, 0);
    std::vector<histbin_pt_sum_fixed_t> hist_untruncated(nbins, 0);

    // Loop over the tracks and fill the histogram
    if (settings_->debug() > 2) {
      edm::LogInfo("VertexProducer") << "fastHistoEmulation::Processing " << fitTracks_.size() << " tracks";
    }
    for (const L1Track& track : fitTracks_) {
      // Get the track pt and z0
      // Convert them to an appropriate data format
      // Truncation and saturation taken care of by the data type specification, now delayed to end of histogramming
      pt_t tkpt = 0;
      tkpt.V = track.getTTTrackPtr()->getTrackWord()(TTTrack_TrackWord::TrackBitLocations::kRinvMSB - 1,
                                                     TTTrack_TrackWord::TrackBitLocations::kRinvLSB);
      z0_t tkZ0 = track.getTTTrackPtr()->getZ0Word();

      if ((settings_->vx_DoQualityCuts() && track_quality_check(tkpt)) || (!settings_->vx_DoQualityCuts())) {
        //
        // Check bin validity of bin found for the current track
        //
        std::pair<histbin_t, bool> bin = fetch_bin(tkZ0, nbins);
        assert(bin.first >= 0 && bin.first < nbins);

        //
        // If the bin is valid then sum the tracks
        //
        if (settings_->debug() > 2) {
          edm::LogInfo("VertexProducer") << "fastHistoEmulation::Checking the track word ... \n"
                                         << "track word = " << track.getTTTrackPtr()->getTrackWord().to_string(2)
                                         << "\n"
                                         << "tkZ0 = " << tkZ0.to_double() << "(" << tkZ0.to_string(2)
                                         << ")\ttkpt = " << tkpt.to_double() << "(" << tkpt.to_string(2)
                                         << ")\tbin = " << bin.first.to_int() << "\n"
                                         << "pt sum in bin " << bin.first.to_int()
                                         << " BEFORE adding track = " << hist_untruncated.at(bin.first).to_double();
        }
        if (bin.second) {
          hist_untruncated.at(bin.first) = hist_untruncated.at(bin.first) + tkpt;
        }
        if (settings_->debug() > 2) {
          edm::LogInfo("VertexProducer") << "fastHistoEmulation::\npt sum in bin " << bin.first.to_int()
                                         << " AFTER adding track = " << hist_untruncated.at(bin.first).to_double();
        }
      } else {
        if (settings_->debug() > 2) {
          edm::LogInfo("VertexProducer") << "fastHistoEmulation::Did not add the following track ... \n"
                                         << "track word = " << track.getTTTrackPtr()->getTrackWord().to_string(2)
                                         << "\n"
                                         << "tkZ0 = " << tkZ0.to_double() << "(" << tkZ0.to_string(2)
                                         << ")\ttkpt = " << tkpt.to_double() << "(" << tkpt.to_string(2) << ")";
        }
      }
    }  // end loop over tracks

    // HLS histogramming used to truncate track pt before adding, using
    // track_pt_fixed_t pt_tmp = tkpt;
    // Now, truncation should happen after histograms are filled but prior to the vertex-finding part of the algo
    for (unsigned int hb = 0; hb < hist.size(); ++hb) {
      link_pt_sum_fixed_t bin_trunc = hist_untruncated.at(hb).range(
          HistogramBitWidths::kSumPtUntruncatedLinkSize - 1,
          HistogramBitWidths::kSumPtUntruncatedLinkSize - HistogramBitWidths::kSumPtUntruncatedLinkMagSize);
      hist.at(hb) = bin_trunc;
      if (settings_->debug() > 2) {
        edm::LogInfo("VertexProducer") << "fastHistoEmulation::truncating histogram bin pt once filling is complete \n"
                                       << "hist_untruncated.at(" << hb << ") = " << hist_untruncated.at(hb).to_double()
                                       << "(" << hist_untruncated.at(hb).to_string(2)
                                       << ")\tbin_trunc = " << bin_trunc.to_double() << "(" << bin_trunc.to_string(2)
                                       << ")\n\thist.at(" << hb << ") = " << hist.at(hb).to_double() << "("
                                       << hist.at(hb).to_string(2) << ")";
      }
    }

    // Loop through all bins, taking into account the fact that the last bin is nbins-window_width+1,
    // and compute the sums using sliding windows ... sum_i_i+(w-1) where i in (0,nbins-w) and w is the window size
    std::vector<window_pt_sum_fixed_t> hist_window_sums(nsums, 0);
    for (unsigned int b = 0; b < nsums; ++b) {
      for (unsigned int w = 0; w < HistogramBitWidths::kWindowSize; ++w) {
        unsigned int index = b + w;
        hist_window_sums.at(b) += hist.at(index);
      }
    }

    // Find the top N vertices
    std::vector<int> found;
    found.reserve(settings_->vx_nvtx());
    for (unsigned int ivtx = 0; ivtx < settings_->vx_nvtx(); ivtx++) {
      histbin_t b_max = 0;
      window_pt_sum_fixed_t max_pt = 0;
      zsliding_t zvtx_sliding = -999;
      std::vector<link_pt_sum_fixed_t> binpt_max(HistogramBitWidths::kWindowSize, 0);

      // Find the maxima of the sums
      for (unsigned int i = 0; i < hist_window_sums.size(); i++) {
        // Skip this window if it will already be returned
        if (find(found.begin(), found.end(), i) != found.end())
          continue;
        if (hist_window_sums.at(i) > max_pt) {
          b_max = i;
          max_pt = hist_window_sums.at(b_max);
          std::copy(std::begin(hist) + b_max,
                    std::begin(hist) + b_max + HistogramBitWidths::kWindowSize,
                    std::begin(binpt_max));

          // Find the weighted position only for the highest sum pt window
          zvtx_sliding = weighted_position(b_max, binpt_max, max_pt, nbins);
        }
      }
      if (settings_->debug() >= 1) {
        edm::LogInfo log("VertexProducer");
        log << "fastHistoEmulation::Checking the output parameters ... \n";
        if (found.empty()) {
          printHistogram<link_pt_sum_fixed_t, edm::LogInfo>(log, hist, 80, 0, -1, "fastHistoEmulation::hist", "\e[92m");
          printHistogram<window_pt_sum_fixed_t, edm::LogInfo>(
              log, hist_window_sums, 80, 0, -1, "fastHistoEmulation::hist_window_sums", "\e[92m");
        }
        printHistogram<link_pt_sum_fixed_t, edm::LogInfo>(
            log, binpt_max, 80, 0, -1, "fastHistoEmulation::binpt_max", "\e[92m");
        log << "bin index (not a VertexWord parameter) = " << b_max << "\n"
            << "sumPt = " << max_pt.to_double() << "\n"
            << "z0 = " << zvtx_sliding.to_double();
      }
      found.push_back(b_max);
      verticesEmulation_.emplace_back(l1t::VertexWord::vtxvalid_t(1),
                                      l1t::VertexWord::vtxz0_t(zvtx_sliding),
                                      l1t::VertexWord::vtxmultiplicity_t(0),
                                      l1t::VertexWord::vtxsumpt_t(max_pt),
                                      l1t::VertexWord::vtxquality_t(0),
                                      l1t::VertexWord::vtxinversemult_t(0),
                                      l1t::VertexWord::vtxunassigned_t(0));
    }
    pv_index_ = 0;
  }  // end of fastHistoEmulation

}  // namespace l1tVertexFinder
