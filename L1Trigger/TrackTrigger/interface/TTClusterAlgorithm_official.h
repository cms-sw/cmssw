/*! \class   TTClusterAlgorithm_official
 *  \brief   Class for "official" algorithm to be used
 *           in TTClusterBuilder
 *  \details 2D clusters: make 1D and then attach them to each other
 *           if their pixels are close to each other, CW cut at the end
 *           After moving from SimDataFormats to DataFormats,
 *           the template structure of the class was maintained
 *           in order to accomodate any types other than PixelDigis
 *           in case there is such a need in the future.
 *
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 12
 *
 */

#ifndef L1_TRACK_TRIGGER_CLUSTER_ALGO_official_H
#define L1_TRACK_TRIGGER_CLUSTER_ALGO_official_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "L1Trigger/TrackTrigger/interface/TTClusterAlgorithm.h"
#include "L1Trigger/TrackTrigger/interface/TTClusterAlgorithmRecord.h"

#include <memory>
#include <string>
#include <map>

template <typename T>
class TTClusterAlgorithm_official : public TTClusterAlgorithm<T> {
private:
  /// Data members
  int mWidthCut;  /// Cluster max width

  /// Function to compare clusters and sort them by row
  static bool CompareClusters(const T& a, const T& b);

public:
  /// Constructor

  TTClusterAlgorithm_official(int aWidthCut) : TTClusterAlgorithm<T>(__func__) { mWidthCut = aWidthCut; }

  /// Destructor
  ~TTClusterAlgorithm_official() override {}

  /// Clustering operations
  void Cluster(std::vector<std::vector<T> >& output, const std::vector<T>& input, bool isPS) const override;

};  /// Close class

/*! \brief   Implementation of methods
 *  \details Here, in the header file, the methods which do not depend
 *           on the specific type <T> that can fit the template.
 *           Other methods, with type-specific features, are implemented
 *           in the source file.
 */

/// Function to compare clusters and sort them by row
template <>
bool TTClusterAlgorithm_official<Ref_Phase2TrackerDigi_>::CompareClusters(const Ref_Phase2TrackerDigi_& a,
                                                                          const Ref_Phase2TrackerDigi_& b);

/// Clustering operations
template <>
void TTClusterAlgorithm_official<Ref_Phase2TrackerDigi_>::Cluster(
    std::vector<std::vector<Ref_Phase2TrackerDigi_> >& output,
    const std::vector<Ref_Phase2TrackerDigi_>& input,
    bool isPS) const;

/*! \class   ES_TTClusterAlgorithm_official
 *  \brief   Class to declare the algorithm to the framework
 *
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 12
 *
 */

template <typename T>
class ES_TTClusterAlgorithm_official : public edm::ESProducer {
private:
  /// Data members
  int mWidthCut;

public:
  /// Constructor
  ES_TTClusterAlgorithm_official(const edm::ParameterSet& p) : mWidthCut(p.getParameter<int>("WidthCut")) {
    setWhatProduced(this);
  }

  /// Destructor
  ~ES_TTClusterAlgorithm_official() override {}

  /// Implement the producer
  std::unique_ptr<TTClusterAlgorithm<T> > produce(const TTClusterAlgorithmRecord& record) {
    TTClusterAlgorithm<T>* TTClusterAlgo = new TTClusterAlgorithm_official<T>(mWidthCut);

    return std::unique_ptr<TTClusterAlgorithm<T> >(TTClusterAlgo);
  }

};  /// Close class

#endif
