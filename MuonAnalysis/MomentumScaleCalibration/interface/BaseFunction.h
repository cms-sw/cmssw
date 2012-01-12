/**
 * \class BaseFunction
 * This class is used as base from scale, resolution and background functions.
 */

#ifndef BaseFunction_h
#define BaseFunction_h

#include <iostream>
#include <vector>
#include <cstdlib>
#include "CondFormats/RecoMuonObjects/interface/MuScleFitDBobject.h"

class BaseFunction
{
public:
  BaseFunction() {}

  /// Constructor when receiving database parameters
  BaseFunction( const MuScleFitDBobject * dbObject )
  {
    functionId_ = dbObject->identifiers;
    parVecVec_ = dbObject->parameters;
    // Needed for the tests in convertToArrays
    iterationNum_ = functionId_.size()-1;
  }

  /// Return the vector of function identifiers
  std::vector<int> identifiers() const {
    return functionId_;
  }
  /// Return the vector of parameters
  std::vector<double> parameters() const {
    return parVecVec_;
  }
  /// Return the vector of fit quality values
  std::vector<double> fitQuality() const {
    return parVecVec_;
  }

protected:
  /// Convert vectors to arrays for faster random access. The first pointer is replaced, thus it is taken by reference.
  template<class T>
  void convertToArrays(T **& function_, const std::vector<T*> & functionVec_);

  std::vector<int> functionId_;
  std::vector<double> parVecVec_;
  std::vector<double> fitQuality_;
  // We will use the array for the function calls because it is faster than the vector for random access.
  double ** parArray_;
  double ** fitQualityArray_;
  int iterationNum_;
};

template <class T>
void BaseFunction::convertToArrays(T **& function_, const std::vector<T*> & functionVec_)
{
  // Check for consistency of number of passed parameters and number of required parameters.
  int totParNums = 0;
  typename std::vector<T*>::const_iterator funcIt = functionVec_.begin();
  for( ; funcIt != functionVec_.end(); ++funcIt ) {
    totParNums += (*funcIt)->parNum();
  }
  int parVecVecSize = parVecVec_.size();
  int functionVecSize = functionVec_.size();
  if( functionVecSize != iterationNum_+1 ) {
    std::cout << "Error: inconsistent number of functions("<<functionVecSize<<") and iterations("<<iterationNum_+1<<")" << std::endl;
    exit(1);
  }
  else if( totParNums != parVecVecSize ) {
    std::cout << "Error: inconsistent total number of requested parameters("<<totParNums<<") and parameters read("<<parVecVecSize<<")" << std::endl;
    exit(1);
  }
//   else if( parVecVecSize != functionVecSize ) {
//     std::cout << "Error: inconsistent number of functions("<<functionVecSize<<") and parameter sets("<<parVecVecSize<<")" << std::endl;
//     exit(1);
//   }
//   else if( parVecVecSize != iterationNum_+1 ) {
//     std::cout << "Error: inconsistent number of parameter sets("<<parVecVecSize<<") and iterations("<<iterationNum_+1<<")" << std::endl;
//     exit(1);
//   }
  // parArray_ = new double*[parVecVecSize];

  parArray_ = new double*[functionVecSize];

//  std::vector<double>::const_iterator parVec = parVecVec_.begin();
  // iterationNum_ starts from 0.
  function_ = new T*[functionVecSize];
  typename std::vector<T * >::const_iterator func = functionVec_.begin();
  std::vector<double>::const_iterator parVec = parVecVec_.begin();

  int iterationCounter = 0;
  for( ; func != functionVec_.end(); ++func, ++iterationCounter ) {

    // Loop on the parameters size for each function and create corresponding parameter arrays
    int parNum = (*func)->parNum();
    parArray_[iterationCounter] = new double[parNum];
    for( int par = 0; par < parNum; ++par ) {
      parArray_[iterationCounter][par] = *parVec;
      ++parVec;
    }

//     parArray_[iterationCounter] = new double[parVec->size()];
//     std::vector<double>::const_iterator par = parVec->begin();
//     int parNum = 0;
//     for ( ; par != parVec->end(); ++par, ++parNum ) {
//       parArray_[iterationCounter][parNum] = *par;
//       // std::cout << "parameter["<<parNum<<"] = " << parArray_[iterationCounter][parNum] << std::endl;
//     }
//     // return make_pair(parameters, parameterErrors);

    function_[iterationCounter] = *func;
  }
}

#endif // BaseFunction_h
