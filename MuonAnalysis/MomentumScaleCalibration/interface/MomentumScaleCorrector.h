/**
 * MomentumScaleCorrector class
 * Author M. De Mattia - 18/11/2008
 */

#ifndef MomentumScaleCorrector_h
#define MomentumScaleCorrector_h

#include <fstream>
#include <sstream>
#include "MuonAnalysis/MomentumScaleCalibration/interface/Functions.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "CondFormats/MomentumScaleCalibrationObjects/interface/MuScleFitScale.h"

/**
 * This is used to have a common set of functions for the specialized templates to use.
 * The constructor receives the name identifying the parameters for the correction function.
 * It reads the parameters from a txt file in data/.
 *
 * It handles multiple iterations. It is also possible to use different functions in
 * different iterations.
 *
 * ATTENTION: it is important that iterations numbers in the txt file start from 0.
 */
class MomentumScaleCorrector
{
 public:
  /**
   * The constructor takes a string identifying the parameters to read. It
   * parses the txt file containing the parameters, extracts the index of the
   * correction function and saves the corresponding pointer. It then fills the
   * vector of parameters.
   */
  MomentumScaleCorrector( TString identifier )
  {
    identifier.Prepend("MuonAnalysis/MomentumScaleCalibration/data/");
    identifier.Append(".txt");
    edm::FileInPath fileWithFullPath(identifier.Data());
    readParameters( fileWithFullPath.fullPath() );
  }
  /**
   * This constructor is used when reading parameters from the db.
   * It receives a pointer to an object of type MuScleFitScale containing
   * the parameters and the functions identifiers.
   */
  MomentumScaleCorrector( const MuScleFitScale * scaleObject )
  {
    scaleFunctionId_ = scaleObject->identifiers;
    parScaleVec_ = scaleObject->parameters;
    // Needed for the tests in convertToArrays
    iterationNum_ = scaleFunctionId_.size()-1;
    vector<int>::const_iterator id = scaleFunctionId_.begin();
    for( ; id != scaleFunctionId_.end(); ++id ) {
      scaleFunctionVec_.push_back( scaleFunctionArray[*id] );
    }
    // Fill the arrays that will be used when calling the correction function.
    convertToArrays();
  }

  ~MomentumScaleCorrector() {
    if( parScaleArray_ != 0 ) {
      for( unsigned int i=0; i<parScaleVec_.size(); ++i ) {
        delete[] parScaleArray_[i];
      }
      delete parScaleArray_;
    }
    delete[] scaleFunction_;
  }
  /// Method to do the corrections. It is templated to work with all the track types.
  template <class U>
  double operator()( const U & track ) {

    // Loop on all the functions and apply them iteratively on the pt corrected by the previous function.
    double pt = track.pt();
    for( int i=0; i<iterationNum_; ++i ) {
      // return ( scaleFunction_->scale( track.pt(), track.eta(), track.phi(), track.charge(), parScale_) );
      pt = ( scaleFunction_[i]->scale( pt, track.eta(), track.phi(), track.charge(), parScaleArray_[i]) );
    }
    return pt;
  }
  /// Return the vectors of parameters
  vector<vector<double> > parameters() const {
    return parScaleVec_;
  }
  /// Return the vector of function identifiers
  vector<int> identifiers() const {
    return scaleFunctionId_;
  }
 protected:
  /// Parser of the parameters file
  void readParameters( TString fileName );
  /// Convert vectors to arrays for faster random access.
  void convertToArrays();
  scaleFunctionBase<double * > ** scaleFunction_;
  vector<scaleFunctionBase<double * > * > scaleFunctionVec_;
  vector<int> scaleFunctionId_;
  vector<vector<double> > parScaleVec_;
  // We will use the array for the function calls because it is faster than the vector for random access.
  double ** parScaleArray_;
  int iterationNum_;
};

#endif // MomentumScaleCorrector_h
