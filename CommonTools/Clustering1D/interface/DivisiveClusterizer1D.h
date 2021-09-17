#ifndef _DivisiveClusterizer1D_H_
#define _DivisiveClusterizer1D_H_

#include "CommonTools/Clustering1D/interface/Clusterizer1D.h"
#include "CommonTools/Clustering1D/interface/Cluster1DMerger.h"
#include "CommonTools/Clustering1D/interface/Cluster1DCleaner.h"
#include "CommonTools/Clustering1D/interface/TrivialWeightEstimator.h"
#include "CommonTools/Clustering1D/interface/Clusterizer1DCommons.h"

/**
 * Find the modes with a simple divisive method.
 */
template <class T>
class DivisiveClusterizer1D : public Clusterizer1D<T> {
public:
  /**
     *  \param zoffset maximum distance between track position and position of its cluster
     *         (depending on useError its either weighted or physical distance)
     *  \param ntkmin Minimum number of tracks required to form a cluster.
     *  \param useError physical distances or weighted distances.
     *  \param zsep Maximum distance between two adjacent tracks that belong
     *         to the same initial cluster.
     *  \param wei Compute the cluster "center" with an unweighted or a weighted
     *         average of the tracks. Weighted means weighted with the error
     *         of the data point.
     */
  DivisiveClusterizer1D(float zoffset = 5., int ntkmin = 5, bool useError = true, float zsep = 0.05, bool wei = true);

  ~DivisiveClusterizer1D();

  std::pair<std::vector<Cluster1D<T> >, std::vector<const T*> > operator()(const std::vector<Cluster1D<T> >&) const;
  virtual DivisiveClusterizer1D* clone() const;

private:
  //methods
  void findCandidates(const std::vector<Cluster1D<T> >&,
                      std::vector<Cluster1D<T> >&,
                      std::vector<Cluster1D<T> >&) const;
  std::vector<Cluster1D<T> > makeCluster1Ds(std::vector<Cluster1D<T> >&, std::vector<Cluster1D<T> >&) const;
  void insertTracks(std::vector<Cluster1D<T> >&, std::vector<Cluster1D<T> >&) const;
  std::vector<const T*> takeTracks(const std::vector<Cluster1D<T> >&) const;
  Cluster1D<T> mergeCluster1Ds(std::vector<Cluster1D<T> >&) const;
  //data members
  // std::vector<Cluster1D<T> > theCluster1Ds;
  // std::vector<Cluster1D<T> > theTotalDiscardedTracks;
  //  std::vector<Cluster1D<T> > theDiscardedTracks;
  Cluster1DMerger<T>* theMerger;
  Cluster1DCleaner<T>* theCleaner;
  float theZOffSet;
  float theZSeparation;
  unsigned theNTkMin;
  bool theWei;
  bool theUseError;
};

/*
 *        implementation 
 *
 */

template <class T>
DivisiveClusterizer1D<T>::DivisiveClusterizer1D(float zoffset, int ntkmin, bool useError, float zsep, bool wei)
    : theZOffSet(zoffset), theZSeparation(zsep), theNTkMin(ntkmin), theWei(wei), theUseError(useError) {
  //  theDiscardedTracks.clear();
  // theTotalDiscardedTracks.clear();
  //  theCluster1Ds.clear();
  TrivialWeightEstimator<T> weightEstimator;
  theMerger = new Cluster1DMerger<T>(weightEstimator);
  theCleaner = new Cluster1DCleaner<T>(theZOffSet, theUseError);
}

template <class T>
DivisiveClusterizer1D<T>::~DivisiveClusterizer1D() {
  delete theMerger;
  delete theCleaner;
}

template <class T>
std::pair<std::vector<Cluster1D<T> >, std::vector<const T*> > DivisiveClusterizer1D<T>::operator()(
    const std::vector<Cluster1D<T> >& input) const {
  std::vector<Cluster1D<T> > discardedCluster1Ds;
  std::vector<Cluster1D<T> > output;
  findCandidates(input, output, discardedCluster1Ds);
  return std::pair<std::vector<Cluster1D<T> >, std::vector<const T*> >(output, takeTracks(discardedCluster1Ds));
}

template <class T>
DivisiveClusterizer1D<T>* DivisiveClusterizer1D<T>::clone() const {
  return new DivisiveClusterizer1D<T>(*this);
}

template <class T>
void DivisiveClusterizer1D<T>::findCandidates(const std::vector<Cluster1D<T> >& inputo,
                                              std::vector<Cluster1D<T> >& finalCluster1Ds,
                                              std::vector<Cluster1D<T> >& totDiscardedTracks) const {
  using namespace Clusterizer1DCommons;

  std::vector<Cluster1D<T> > input = inputo;
  std::vector<Cluster1D<T> > discardedTracks;
  if (input.size() < theNTkMin) {
    insertTracks(input, totDiscardedTracks);
    return;
  }
  sort(input.begin(), input.end(), ComparePairs<T>());
  int ncount = 0;
  std::vector<Cluster1D<T> > partOfPTracks;
  partOfPTracks.push_back(input.front());
  for (typename std::vector<Cluster1D<T> >::const_iterator ic = (input.begin()) + 1; ic != input.end(); ic++) {
    ncount++;
    if (fabs((*ic).position().value() - (*(ic - 1)).position().value()) < (double)theZSeparation) {
      partOfPTracks.push_back((*ic));
    } else {
      if (partOfPTracks.size() >= theNTkMin) {
        std::vector<Cluster1D<T> > clusters = makeCluster1Ds(partOfPTracks, discardedTracks);
        for (typename std::vector<Cluster1D<T> >::const_iterator iclus = clusters.begin(); iclus != clusters.end();
             iclus++) {
          finalCluster1Ds.push_back(*iclus);
        }
        insertTracks(discardedTracks, totDiscardedTracks);
      } else {
        insertTracks(partOfPTracks, totDiscardedTracks);
      }
      partOfPTracks.clear();
      partOfPTracks.push_back((*ic));
    }
  }
  if (partOfPTracks.size() >= theNTkMin) {
    std::vector<Cluster1D<T> > clusters = makeCluster1Ds(partOfPTracks, discardedTracks);
    for (typename std::vector<Cluster1D<T> >::const_iterator iclus = clusters.begin(); iclus != clusters.end();
         iclus++) {
      finalCluster1Ds.push_back(*iclus);
    }
    insertTracks(discardedTracks, totDiscardedTracks);
  } else {
    insertTracks(partOfPTracks, totDiscardedTracks);
  }

  sort(finalCluster1Ds.begin(), finalCluster1Ds.end(), ComparePairs<T>());
  // reverse(theCluster1Ds.begin(), theCluster1Ds.end());

  return;
}

template <class T>
std::vector<Cluster1D<T> > DivisiveClusterizer1D<T>::makeCluster1Ds(std::vector<Cluster1D<T> >& clusters,
                                                                    std::vector<Cluster1D<T> >& discardedTracks) const {
  std::vector<Cluster1D<T> > finalCluster1Ds;
  discardedTracks.clear();
  std::vector<Cluster1D<T> > pvClu0 = clusters;
  std::vector<Cluster1D<T> > pvCluNew = pvClu0;
  bool stop = false;
  while (!stop) {
    int nDiscardedAtIteration = 100;
    while (nDiscardedAtIteration != 0) {
      pvCluNew = theCleaner->clusters(pvClu0);
      std::vector<Cluster1D<T> > tracksAtIteration = theCleaner->discardedCluster1Ds();
      nDiscardedAtIteration = tracksAtIteration.size();
      if (nDiscardedAtIteration != 0) {
        insertTracks(tracksAtIteration, discardedTracks);
        pvClu0 = pvCluNew;
      }
    }  // while nDiscardedAtIteration
    unsigned ntkclus = pvCluNew.size();
    unsigned ndiscard = discardedTracks.size();

    if (ntkclus >= theNTkMin) {
      //save the cluster
      finalCluster1Ds.push_back(mergeCluster1Ds(pvCluNew));
      if (ndiscard >= theNTkMin) {  //make a new cluster and reset
        pvClu0 = discardedTracks;
        discardedTracks.clear();
      } else {  //out of loop
        stop = true;
      }
    } else {
      insertTracks(pvCluNew, discardedTracks);
      stop = true;
    }
  }  // while stop
  return finalCluster1Ds;
}

template <class T>
void DivisiveClusterizer1D<T>::insertTracks(std::vector<Cluster1D<T> >& clusou,
                                            std::vector<Cluster1D<T> >& cludest) const {
  if (clusou.empty())
    return;
  for (typename std::vector<Cluster1D<T> >::const_iterator iclu = clusou.begin(); iclu != clusou.end(); iclu++) {
    cludest.push_back(*iclu);
  }
  /*
    for ( typename std::vector< Cluster1D<T> >::const_iterator iclu = clu.begin(); 
    iclu != clu.end(); iclu++){
      if (total) {
        theTotalDiscardedTracks.push_back(*iclu);
      }else { 
        theDiscardedTracks.push_back(*iclu);
      }
    }
    */
  return;
}

template <class T>
std::vector<const T*> DivisiveClusterizer1D<T>::takeTracks(const std::vector<Cluster1D<T> >& clu) const {
  std::vector<const T*> tracks;
  for (typename std::vector<Cluster1D<T> >::const_iterator iclu = clu.begin(); iclu != clu.end(); iclu++) {
    std::vector<const T*> clutks = iclu->tracks();
    for (typename std::vector<const T*>::const_iterator i = clutks.begin(); i != clutks.end(); ++i) {
      tracks.push_back(*i);
    }
  }
  return tracks;
}

template <class T>
Cluster1D<T> DivisiveClusterizer1D<T>::mergeCluster1Ds(std::vector<Cluster1D<T> >& clusters) const {
  Cluster1D<T> result = clusters.front();
  for (typename std::vector<Cluster1D<T> >::iterator iclu = (clusters.begin()) + 1; iclu != clusters.end(); iclu++) {
    Cluster1D<T> old = result;
    result = (*theMerger)(old, *iclu);
  }
  return result;
}

#endif
