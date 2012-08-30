/**
 * \class ResolutionFunction
 * Class for the resolution function. It can be built from local file or from db.
 */

#ifndef ResolutionFunction_h
#define ResolutionFunction_h

#include <fstream>
#include <sstream>
#include "MuonAnalysis/MomentumScaleCalibration/interface/BaseFunction.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/Functions.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

class ResolutionFunction : public BaseFunction
{
public:
  /**
   * The constructor takes a string identifying the parameters to read. It
   * parses the txt file containing the parameters, extracts the index of the
   * correction function and saves the corresponding pointer. It then fills the
   * vector of parameters.
   */
  ResolutionFunction( TString identifier )
  {
    identifier.Prepend("MuonAnalysis/MomentumScaleCalibration/data/");
    identifier.Append(".txt");
    edm::FileInPath fileWithFullPath(identifier.Data());
    readParameters( fileWithFullPath.fullPath() );

    std::vector<int>::const_iterator idIt = functionId_.begin();
    for( ; idIt != functionId_.end(); ++idIt ) std::cout << "idIt = " << *idIt << std::endl;
  }
  /**
   * This constructor is used when reading parameters from the db.
   * It receives a pointer to an object of type MuScleFitDBobject containing
   * the parameters and the functions identifiers.
   * The object is the same for all the functions.
   */
  ResolutionFunction( const MuScleFitDBobject * dbObject ) : BaseFunction( dbObject )
  {
    std::vector<int>::const_iterator id = functionId_.begin();
    for( ; id != functionId_.end(); ++id ) {
      resolutionFunctionVec_.push_back( resolutionFunctionService( *id ) );
    }
    // Fill the arrays that will be used when calling the correction function.
    convertToArrays(resolutionFunction_, resolutionFunctionVec_);
  }

  ~ResolutionFunction() {
    if( parArray_ != 0 ) {
      for( unsigned int i=0; i<functionId_.size(); ++i ) {
        delete[] parArray_[i];
        delete resolutionFunction_[i];
      }
      delete[] parArray_;
      delete[] resolutionFunction_;
    }
  }

  /// The second, optional, parameter is the iteration number
  template <class U>
  double sigmaPt(const U & track, const int i = 0) {
    if( i > iterationNum_ || i < 0 ) {
      std::cout << "Error: wrong iteration number, there are " << iterationNum_ << "iterations, ther first one is 0" << std::endl;
      exit(1);
    }
    return resolutionFunction_[i]->sigmaPt(track.pt(), track.eta(), parArray_[i]);
  }
  /// The second, optional, parameter is the iteration number
  template <class U>
  double sigmaCotgTh(const U & track, const int i = 0) {
    if( i > iterationNum_ || i < 0 ) {
      std::cout << "Error: wrong iteration number, there are " << iterationNum_ << "iterations, ther first one is 0" << std::endl;
      exit(1);
    }
    return resolutionFunction_[i]->sigmaCotgTh(track.pt(), track.eta(), parArray_[i]);
  }
  /// The second, optional, parameter is the iteration number
  template <class U>
  double sigmaPhi(const U & track, const int i = 0) {
    if( i > iterationNum_ || i < 0 ) {
      std::cout << "Error: wrong iteration number, there are " << iterationNum_ << "iterations, ther first one is 0" << std::endl;
      exit(1);
    }
    return resolutionFunction_[i]->sigmaPhi(track.pt(), track.eta(), parArray_[i]);
  }
  /// Get the ith resolution function
  resolutionFunctionBase<double * > * function( const unsigned int i )
  {
    if( resolutionFunctionVec_.size() > i ) return resolutionFunction_[i];
    else return 0;
  }
protected:
  /// Parser of the parameters file
  void readParameters( TString fileName );

  resolutionFunctionBase<double * > ** resolutionFunction_;
  std::vector<resolutionFunctionBase<double * > * > resolutionFunctionVec_;
};

#endif // ResolutionFunction_h
