/**
 * MomentumScaleCorrector class
 * Author M. De Mattia - 18/11/2008
 */

#ifndef MomentumScaleCorrector_h
#define MomentumScaleCorrector_h

#include <fstream>
#include <sstream>
#include "MuonAnalysis/MomentumScaleCalibration/interface/BaseFunction.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/Functions.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

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
class MomentumScaleCorrector : public BaseFunction
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
   * It receives a pointer to an object of type MuScleFitDBobject containing
   * the parameters and the functions identifiers.
   */
  MomentumScaleCorrector( const MuScleFitDBobject * dbObject ) : BaseFunction( dbObject )
  {
    std::vector<int>::const_iterator id = functionId_.begin();
    for( ; id != functionId_.end(); ++id ) {
      scaleFunctionVec_.push_back( scaleFunctionService( *id ) );
    }
    // Fill the arrays that will be used when calling the correction function.
    convertToArrays(scaleFunction_, scaleFunctionVec_);
  }

  ~MomentumScaleCorrector() {
    if( parArray_ != nullptr ) {
      for( unsigned int i=0; i<functionId_.size(); ++i ) {
        delete[] parArray_[i];
        delete scaleFunction_[i];
      }
      delete[] parArray_;
      delete[] scaleFunction_;
    }
  }

  /// Returns a pointer to the selected function
  scaleFunctionBase<double * > * function(const int i) { return scaleFunction_[i]; }


  /// Method to do the corrections. It is templated to work with all the track types.
  template <class U>
  double operator()( const U & track ) {

    // Loop on all the functions and apply them iteratively on the pt corrected by the previous function.
    double pt = track.pt();
    for( int i=0; i<=iterationNum_; ++i ) {
      // return ( scaleFunction_->scale( track.pt(), track.eta(), track.phi(), track.charge(), parScale_) );
      pt = ( scaleFunction_[i]->scale( pt, track.eta(), track.phi(), track.charge(), parArray_[i]) );
    }
    return pt;
  }

  /// Alternative method that can be used with lorentzVectors.
  template <class U>
  double correct( const U & lorentzVector ) {

    // Loop on all the functions and apply them iteratively on the pt corrected by the previous function.
    double pt = lorentzVector.Pt();
    for( int i=0; i<=iterationNum_; ++i ) {
      pt = ( scaleFunction_[i]->scale( pt, lorentzVector.Eta(), lorentzVector.Phi(), 1, parArray_[i]) );
    }
    return pt;
  }

 protected:
  /// Parser of the parameters file
  void readParameters( TString fileName );

  scaleFunctionBase<double * > ** scaleFunction_;
  std::vector<scaleFunctionBase<double * > * > scaleFunctionVec_;
};

#endif // MomentumScaleCorrector_h
