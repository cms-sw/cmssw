/**
 * \class BackgroundFunction
 * Class for the resolution function. It can be built from local file or from db.
 */

#ifndef BackgroundFunction_h
#define BackgroundFunction_h

#include <fstream>
#include <sstream>
#include "MuonAnalysis/MomentumScaleCalibration/interface/BaseFunction.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/Functions.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

class BackgroundFunction : public BaseFunction
{
public:
  /**
   * The constructor takes a string identifying the parameters to read. It
   * parses the txt file containing the parameters, extracts the index of the
   * correction function and saves the corresponding pointer. It then fills the
   * vector of parameters.
   */
  BackgroundFunction( TString identifier )
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
  BackgroundFunction( const MuScleFitDBobject * dbObject ) : BaseFunction( dbObject )
  {
    std::vector<int>::const_iterator id = functionId_.begin();
    for( ; id != functionId_.end(); ++id ) {
      // TODO: fix the values for the lower and upper limits
      backgroundFunctionVec_.push_back( backgroundFunctionService( *id , 0., 200. ) );
    }
    // Fill the arrays that will be used when calling the correction function.
    convertToArrays(backgroundFunction_, backgroundFunctionVec_);
  }

  ~BackgroundFunction() {
    if( parArray_ != 0 ) {
      for( unsigned int i=0; i<functionId_.size(); ++i ) {
        delete[] parArray_[i];
        delete backgroundFunction_[i];
      }
      delete[] parArray_;
      delete[] backgroundFunction_;
    }
  }
  /// Get the ith background function
  backgroundFunctionBase * function( const unsigned int i )
  {
    if( backgroundFunctionVec_.size() > i ) return backgroundFunction_[i];
    else return 0;
  }

protected:
  /// Parser of the parameters file
  void readParameters( TString fileName );

  backgroundFunctionBase ** backgroundFunction_;
  std::vector<backgroundFunctionBase * > backgroundFunctionVec_;
};

#endif // BackgroundFunction_h
