/*! \class   TTClusterAlgorithm
 *  \brief   Base class for any algorithm to be used
 *           in TTClusterBuilder
 *  \details After moving from SimDataFormats to DataFormats,
 *           the template structure of the class was maintained
 *           in order to accomodate any types other than PixelDigis
 *           in case there is such a need in the future.
 *
 *  \author Andrew W. Rose
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 12
 *
 */

#ifndef L1_TRACK_TRIGGER_CLUSTER_ALGO_BASE_H
#define L1_TRACK_TRIGGER_CLUSTER_ALGO_BASE_H

#include <sstream>
#include <map>
#include <string>
#include "classNameFinder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

template <typename T>
class TTClusterAlgorithm {
protected:
  /// Data members
  std::string className_;

public:
  /// Constructors

  TTClusterAlgorithm(std::string fName) { className_ = classNameFinder<T>(fName); }

  /// Destructor
  virtual ~TTClusterAlgorithm() {}

  /// Clustering operations
  /// Overloaded method (***) to preserve the interface of all the algorithms but official
  virtual void Cluster(std::vector<std::vector<T> > &output, const std::vector<T> &input, bool module) const {
    Cluster(output, input);
  }

  /// Basic version common to all the algorithms but official
  virtual void Cluster(std::vector<std::vector<T> > &output, const std::vector<T> &input) const { output.clear(); }

  /// NOTE
  /// When calling TTClusterAlgoHandle->Cluster( output, input, module )
  /// in L1TkClusterBuilder, this will go in the following way
  /// * case official
  /// it will go with the overloaded method (***) which has its
  /// specific implementation in TTClusterAlgorithm_official.h
  /// * case "everything else"
  /// the overloaded method will call the basic one
  /// it is the basic one which has its specific implementation
  /// in the various TTClusterAlgorithm_*.h
  /// This way, all the existing interfaces will be
  /// unchanged, but the new algorithm will use the new interface

  /// Algorithm name
  virtual std::string AlgorithmName() const { return className_; }

};  /// Close class

#endif
