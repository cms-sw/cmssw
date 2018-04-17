
#include "L1Trigger/VertexFinder/interface/VertexFinder.h"


#include "L1Trigger/VertexFinder/interface/AlgoSettings.h"



using namespace std;

namespace l1tVertexFinder {

void VertexFinder::GapClustering()
{

  sort(fitTracks_.begin(), fitTracks_.end(), SortTracksByZ0());
  iterations_ = 0;
  RecoVertex Vertex;
  for (unsigned int i = 0; i < fitTracks_.size(); ++i) {
    Vertex.insert(fitTracks_[i]);
    iterations_++;
    if ((i + 1 < fitTracks_.size() and fitTracks_[i + 1]->z0() - fitTracks_[i]->z0() > settings_->vx_distance()) or i == fitTracks_.size() - 1) {
      if (Vertex.numTracks() >= settings_->vx_minTracks()) {
        Vertex.computeParameters(settings_->vx_weightedmean());
        vertices_.push_back(Vertex);
      }
      Vertex.clear();
    }
  }
}

float VertexFinder::MaxDistance(RecoVertex cluster0, RecoVertex cluster1)
{
  float distance = 0;
  for (const L1Track* track0 : cluster0.tracks()) {
    for (const L1Track* track1 : cluster1.tracks()) {
      if (fabs(track0->z0() - track1->z0()) > distance) {
        distance = fabs(track0->z0() - track1->z0());
      }
    }
  }

  return distance;
}

float VertexFinder::MinDistance(RecoVertex cluster0, RecoVertex cluster1)
{
  float distance = 9999;
  for (const L1Track* track0 : cluster0.tracks()) {
    for (const L1Track* track1 : cluster1.tracks()) {
      if (fabs(track0->z0() - track1->z0()) < distance) {
        distance = fabs(track0->z0() - track1->z0());
      }
    }
  }

  return distance;
}

float VertexFinder::MeanDistance(RecoVertex cluster0, RecoVertex cluster1)
{

  float distanceSum = 0;

  for (const L1Track* track0 : cluster0.tracks()) {
    for (const L1Track* track1 : cluster1.tracks()) {
      distanceSum += fabs(track0->z0() - track1->z0());
    }
  }

  float distance = distanceSum / (cluster0.numTracks() * cluster1.numTracks());
  return distance;
}

float VertexFinder::CentralDistance(RecoVertex cluster0, RecoVertex cluster1)
{
  cluster0.computeParameters(settings_->vx_weightedmean());
  cluster1.computeParameters(settings_->vx_weightedmean());

  float distance = fabs(cluster0.z0() - cluster1.z0());
  return distance;
}

void VertexFinder::AgglomerativeHierarchicalClustering()
{
  iterations_ = 0;

  sort(fitTracks_.begin(), fitTracks_.end(), SortTracksByZ0());

  std::vector<RecoVertex> vClusters;
  vClusters.resize(fitTracks_.size());

  for (unsigned int i = 0; i < fitTracks_.size(); ++i) {
    vClusters[i].insert(fitTracks_[i]);
    // iterations_++;
  }

  while (1) {
    float MinimumScore = 9999;

    unsigned int clusterId0 = 0;
    unsigned int clusterId1 = 0;
    for (unsigned int iClust = 0; iClust < vClusters.size() - 1; iClust++) {
      iterations_++;

      float M = 0;
      if (settings_->vx_distanceType() == 0)
        M = MaxDistance(vClusters[iClust], vClusters[iClust + 1]);
      else if (settings_->vx_distanceType() == 1)
        M = MinDistance(vClusters[iClust], vClusters[iClust + 1]);
      else if (settings_->vx_distanceType() == 2)
        M = MeanDistance(vClusters[iClust], vClusters[iClust + 1]);
      else
        M = CentralDistance(vClusters[iClust], vClusters[iClust + 1]);

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
      clust.computeParameters(settings_->vx_weightedmean());
      vertices_.push_back(clust);
    }
  }
}

void VertexFinder::DBSCAN()
{
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
    if (fitTracks_[i]->pt() > settings_->vx_dbscan_pt())
      numDensityTracks++;
    for (unsigned int k = 0; k < fitTracks_.size(); ++k) {
      iterations_++;
      if (k != i and (fabs(fitTracks_[k]->z0() - fitTracks_[i]->z0()) < settings_->vx_distance() or (fabs(fitTracks_[i]->eta()) > 1.5 and fabs(fitTracks_[k]->z0() - fitTracks_[i]->z0()) < 0.5))) {
        neighbourTrackIds.insert(k);
        if (fitTracks_[k]->pt() > settings_->vx_dbscan_pt()) {
          numDensityTracks++;
        }
      }
    }

    if (numDensityTracks < settings_->vx_dbscan_mintracks()) {
      // mark track as noise
    }
    else {
      RecoVertex vertex;
      vertex.insert(fitTracks_[i]);
      saved.push_back(i);
      for (unsigned int id : neighbourTrackIds) {
        if (find(visited.begin(), visited.end(), id) == visited.end()) {
          visited.push_back(id);
          std::vector<unsigned int> neighbourTrackIds2;
          for (unsigned int k = 0; k < fitTracks_.size(); ++k) {
            iterations_++;
            if (fabs(fitTracks_[k]->z0() - fitTracks_[id]->z0()) < settings_->vx_distance() or (fabs(fitTracks_[id]->eta()) > 1.5 and fabs(fitTracks_[k]->z0() - fitTracks_[id]->z0()) < 0.5)) {
              neighbourTrackIds2.push_back(k);
            }
          }

          if (neighbourTrackIds2.size() >= settings_->vx_minTracks()) {
            for (unsigned int id2 : neighbourTrackIds2) {
              neighbourTrackIds.insert(id2);
            }
          }
        }
        if (find(saved.begin(), saved.end(), id) == saved.end())
          vertex.insert(fitTracks_[id]);
      }
      vertex.computeParameters(settings_->vx_weightedmean());
      if (vertex.numTracks() >= settings_->vx_minTracks())
        vertices_.push_back(vertex);
    }
    // }
  }
}

void VertexFinder::PVR()
{
  bool start = true;
  FitTrackCollection discardedTracks, acceptedTracks;
  iterations_ = 0;
  for (const L1Track* track : fitTracks_) {
    acceptedTracks.push_back(track);
  }


  while (discardedTracks.size() >= settings_->vx_minTracks() or start == true) {
    start = false;
    bool removing = true;
    discardedTracks.clear();
    while (removing) {
      float oldDistance = 0.;

      if (settings_->debug() == 7)
        cout << "acceptedTracks " << acceptedTracks.size() << endl;

      float z0start = 0;
      for (const L1Track* track : acceptedTracks) {
        z0start += track->z0();
        iterations_++;
      }

      z0start /= acceptedTracks.size();
      if (settings_->debug() == 7)
        cout << "z0 vertex " << z0start << endl;
      FitTrackCollection::iterator badTrackIt = acceptedTracks.end();
      removing = false;

      for (FitTrackCollection::iterator it = acceptedTracks.begin(); it < acceptedTracks.end(); ++it) {
        const L1Track* track = *it;
        iterations_++;
        if (fabs(track->z0() - z0start) > settings_->vx_distance() and fabs(track->z0() - z0start) > oldDistance) {
          badTrackIt = it;
          oldDistance = fabs(track->z0() - z0start);
          removing = true;
        }
      }

      if (removing) {
        const L1Track* badTrack = *badTrackIt;
        if (settings_->debug() == 7)
          cout << "removing track " << badTrack->z0() << " at distance " << oldDistance << endl;
        discardedTracks.push_back(badTrack);
        acceptedTracks.erase(badTrackIt);
      }
    }

    if (acceptedTracks.size() >= settings_->vx_minTracks()) {
      RecoVertex vertex;
      for (const L1Track* track : acceptedTracks) {
        vertex.insert(track);
      }
      vertex.computeParameters(settings_->vx_weightedmean());
      vertices_.push_back(vertex);
    }
    if (settings_->debug() == 7)
      cout << "discardedTracks size " << discardedTracks.size() << endl;
    acceptedTracks.clear();
    acceptedTracks = discardedTracks;
  }
}

void VertexFinder::AdaptiveVertexReconstruction()
{
  bool start = true;
  iterations_ = 0;
  FitTrackCollection discardedTracks, acceptedTracks, discardedTracks2;

  for (const L1Track* track : fitTracks_) {
    discardedTracks.push_back(track);
  }


  while (discardedTracks.size() >= settings_->vx_minTracks() or start == true) {
    start = false;
    discardedTracks2.clear();
    FitTrackCollection::iterator it = discardedTracks.begin();
    const L1Track* track = *it;
    acceptedTracks.push_back(track);
    float z0sum = track->z0();

    for (FitTrackCollection::iterator it2 = discardedTracks.begin(); it2 < discardedTracks.end(); ++it2) {
      if (it2 != it) {
        const L1Track* secondTrack = *it2;
        // Calculate new vertex z0 adding this track
        z0sum += secondTrack->z0();
        float z0vertex = z0sum / (acceptedTracks.size() + 1);
        // Calculate chi2 of new vertex
        float chi2 = 0.;
        float dof = 0.;
        for (const L1Track* accTrack : acceptedTracks) {
          iterations_++;
          float Residual = accTrack->z0() - z0vertex;
          if (fabs(accTrack->eta()) < 1.2)
            Residual /= 0.1812; // Assumed z0 resolution
          else if (fabs(accTrack->eta()) >= 1.2 && fabs(accTrack->eta()) < 1.6)
            Residual /= 0.2912;
          else if (fabs(accTrack->eta()) >= 1.6 && fabs(accTrack->eta()) < 2.)
            Residual /= 0.4628;
          else
            Residual /= 0.65;

          chi2 += Residual * Residual;
          dof = (acceptedTracks.size() + 1) * 2 - 1;
        }
        if (chi2 / dof < settings_->vx_chi2cut()) {
          acceptedTracks.push_back(secondTrack);
        }
        else {
          discardedTracks2.push_back(secondTrack);
          z0sum -= secondTrack->z0();
        }
      }
    }

    if (acceptedTracks.size() >= settings_->vx_minTracks()) {
      RecoVertex vertex;
      for (const L1Track* track : acceptedTracks) {
        vertex.insert(track);
      }
      vertex.computeParameters(settings_->vx_weightedmean());
      vertices_.push_back(vertex);
    }

    acceptedTracks.clear();
    discardedTracks.clear();
    discardedTracks = discardedTracks2;
  }
}

void VertexFinder::HPV()
{
  iterations_ = 0;
  sort(fitTracks_.begin(), fitTracks_.end(), SortTracksByPt());

  RecoVertex vertex;
  bool first = true;
  float z = 99.;
  for (const L1Track* track : fitTracks_) {
    if (track->pt() < 50.) {
      if (first) {
        first = false;
        z = track->z0();
        vertex.insert(track);
      }
      else {
        if (fabs(track->z0() - z) < settings_->vx_distance())
          vertex.insert(track);
      }
    }
  }

  vertex.computeParameters(settings_->vx_weightedmean());

  vertex.setZ(z);
  vertices_.push_back(vertex);
}

void VertexFinder::Kmeans()
{
  unsigned int NumberOfClusters = settings_->vx_kmeans_nclusters();

  vertices_.resize(NumberOfClusters);
  float ClusterSeparation = 30. / NumberOfClusters;

  for (unsigned int i = 0; i < NumberOfClusters; ++i) {
    float ClusterCentre = -15. + ClusterSeparation * (i + 0.5);
    vertices_[i].setZ(ClusterCentre);
  }
  unsigned int iterations = 0;
  // Initialise Clusters
  while (iterations < settings_->vx_kmeans_iterations()) {
    for (unsigned int i = 0; i < NumberOfClusters; ++i) {
      vertices_[i].clear();
    }

    for (const L1Track* track : fitTracks_) {
      float distance = 9999;
      if (iterations == settings_->vx_kmeans_iterations() - 3)
        distance = settings_->vx_distance() * 2;
      if (iterations > settings_->vx_kmeans_iterations() - 3)
        distance = settings_->vx_distance();
      unsigned int ClusterId;
      bool NA = true;
      // cout << "iteration "<< iterations << endl;
      // cout << "track z0 "<< track->z0() << endl;
      for (unsigned int id = 0; id < NumberOfClusters; ++id) {
        if (fabs(track->z0() - vertices_[id].z0()) < distance) {
          // cout << "vertex id "<< id << " z0 " << vertices_[id].z0() << endl;
          distance = fabs(track->z0() - vertices_[id].z0());
          ClusterId = id;
          NA = false;
        }
      }
      if (!NA) {
        vertices_[ClusterId].insert(track);
        // cout << "track in cluster "<< ClusterId << endl;
      }
    }
    // cout << "iteration "<< iterations << endl;
    for (unsigned int i = 0; i < NumberOfClusters; ++i) {
      if (vertices_[i].numTracks() >= settings_->vx_minTracks())
        vertices_[i].computeParameters(settings_->vx_weightedmean());
    }
    iterations++;
  }
}

void VertexFinder::FindPrimaryVertex()
{
  double vertexPt = 0;
  pv_index_ = 0;

  for (unsigned int i = 0; i < vertices_.size(); ++i) {
    if (vertices_[i].pT() > vertexPt) {
      vertexPt = vertices_[i].pT();
      pv_index_ = i;
    }
  }
}

void VertexFinder::AssociatePrimaryVertex(double trueZ0)
{

  double distance = 999.;
  for (unsigned int id = 0; id < vertices_.size(); ++id) {
    if (fabs(trueZ0 - vertices_[id].z0()) < distance) {
      distance = fabs(trueZ0 - vertices_[id].z0());
      pv_index_ = id;
    }
  }
}


void VertexFinder::TDRalgorithm()
{
  float vxPt = 0.;

  for (float z = -14.95; z < 15.; z += 0.1) {
    RecoVertex vertex;
    FitTrackCollection tracks;

    for (const L1Track* track : fitTracks_) {
      if (fabs(z - track->z0()) < settings_->tdr_vx_width()) {
        vertex.insert(track);
      }
      else {
        tracks.push_back(track);
      }
    }
    vertex.computeParameters(settings_->vx_weightedmean());
    // cout << "TDR pt "<< vertex.pT() << endl;
    vertex.setZ(z);
    if (vertex.pT() > vxPt) {
      tdr_vertex_ = vertex;
      tdr_pileup_tracks_ = tracks;
      vxPt = vertex.pT();
    }
  }
} // end of TDRalgorithm

} // end ns l1tVertexFinder
