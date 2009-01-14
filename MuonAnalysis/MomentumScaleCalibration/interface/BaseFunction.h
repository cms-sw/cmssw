/**
 * \class BaseFunction
 * This class is used as base from scale, resolution and background functions.
 */

#ifndef BaseFunction_h
#define BaseFunction_h

#include <iostream>
#include <vector>
#include "CondFormats/MomentumScaleCalibrationObjects/interface/MuScleFitDBobject.h"

using namespace std;

class BaseFunction
{
public:
  BaseFunction() {}

  /// Constructor when receiving database parameters
  BaseFunction( const MuScleFitDBobject * dbObject )
  {
    functionId_ = dbObject->identifiers;
    vector<int>::const_iterator id = functionId_.begin();
    parVecVec_ = dbObject->parameters;
    // Needed for the tests in convertToArrays
    iterationNum_ = functionId_.size()-1;
  }

  /// Return the vectors of parameters
  vector<vector<double> > parameters() const {
    return parVecVec_;
  }
  /// Return the vector of function identifiers
  vector<int> identifiers() const {
    return functionId_;
  }
protected:
  /// Convert vectors to arrays for faster random access. The first pointer is replaced, thus it is taken by reference.
  template<class T>
  void convertToArrays(T **& function_, const vector<T*> & functionVec_);

  vector<int> functionId_;
  vector<vector<double> > parVecVec_;
  // We will use the array for the function calls because it is faster than the vector for random access.
  double ** parArray_;
  int iterationNum_;
};

template <class T>
void BaseFunction::convertToArrays(T **& function_, const vector<T*> & functionVec_)
{
  int parVecVecSize = parVecVec_.size();
  int functionVecSize = functionVec_.size();
  if( parVecVecSize != functionVecSize ) {
    cout << "Error: inconsistent number of functions("<<functionVecSize<<") and parameter sets("<<parVecVecSize<<")" << endl;
    exit(1);
  }
  else if( parVecVecSize != iterationNum_+1 ) {
    cout << "Error: inconsistent number of parameter sets("<<parVecVecSize<<") and iterations("<<iterationNum_+1<<")" << endl;
    exit(1);
  }
  else if( functionVecSize != iterationNum_+1 ) {
    cout << "Error: inconsistent number of functions("<<functionVecSize<<") and iterations("<<iterationNum_+1<<")" << endl;
    exit(1);
  }
  parArray_ = new double*[parVecVecSize];
  vector<vector<double> >::const_iterator parVec = parVecVec_.begin();
  // iterationNum_ starts from 0.
  function_ = new T*[parVecVecSize];
  typename vector<T * >::const_iterator func = functionVec_.begin();

  int iterationCounter = 0;
  for( ; parVec != parVecVec_.end(); ++parVec, ++func, ++iterationCounter ) {

    parArray_[iterationCounter] = new double[parVec->size()];
    vector<double>::const_iterator par = parVec->begin();
    int parNum = 0;
    for ( ; par != parVec->end(); ++par, ++parNum ) {
      parArray_[iterationCounter][parNum] = *par;
      // cout << "parameter["<<parNum<<"] = " << parArray_[iterationCounter][parNum] << endl;
    }
    // return make_pair(parameters, parameterErrors);

    function_[iterationCounter] = *func;
  }
}

#endif // BaseFunction_h
