#include "MuonAnalysis/MomentumScaleCalibration/interface/MomentumScaleCorrector.h"

using namespace std;

void MomentumScaleCorrector::readParameters( TString fileName )
{
  iterationNum_ = 0;
  parScaleArray_ = 0;
  // vector<double> parameterErrors;

  // Read the parameters file
  ifstream parametersFile(fileName.Data());
  string line;

  string iteration("Iteration ");
  // Loop on the file lines
  while (parametersFile) {
    getline( parametersFile, line );
    unsigned int lineInt = line.find("value");

    // if( line.find(iteration) != string::npos ) {
    unsigned int iterationSubStr = line.find(iteration);

    // Take the iteration number
    if( iterationSubStr != string::npos ) {

      int scaleFunctionNum = 0;
      // This can be used when dealing with multiple iterations

      // cout << "line = " << line << endl;
      stringstream sLine(line);
      string num;
      int wordCounter = 0;
      // Warning: this strongly depends on the parameters file structure.
      while( sLine >> num ) {
        ++wordCounter;
        //         cout << "num["<<wordCounter<<"] = " << num << endl;
        if( wordCounter == 9 ) {
          stringstream in(num);
          in >> scaleFunctionNum;
        }
        if( wordCounter == 13 ) {
          stringstream in(num);
          in >> iterationNum_;
        }
      }
      // cout << "iteration number = " << iterationNum_ << endl;
      // cout << "scale function number = " << scaleFunctionNum << endl;

      // Create a new vector to hold the parameters for this iteration
      vector<double> parScale;
      parScaleVec_.push_back(parScale);

      // Set the scaleFunction
      // scaleFunction_ = scaleFunctionArrayForVec[scaleFunctionNum];
      // scaleFunction_ = scaleFunctionArray[scaleFunctionNum];
      scaleFunctionId_.push_back(scaleFunctionNum);
      scaleFunctionVec_.push_back( scaleFunctionArray[scaleFunctionNum] );
    }
    // Take the parameters for the current iteration
    if ( (lineInt != string::npos) ) {
      int subStr1 = line.find("value");
      stringstream paramStr;
      double param = 0;
      // Even if all the rest of the line is taken, the following
      // convertion to a double will stop at the end of the first number.
      paramStr << line.substr(subStr1+5);
      paramStr >> param;
      // Fill the last vector of parameters, which corresponds to this iteration.
      parScaleVec_.back().push_back(param);
      // cout << "param = " << param << endl;

      // This is to extract parameter errors
      // int subStr2 = line.find("+-");
      // stringstream parErrorStr;
      // double parError = 0;
      // parErrorStr << line.substr(subStr2+1);
      // parErrorStr >> parError;
      // parameterErrors.push_back(parError);
      // cout << "parError = " << parError << endl;
    }
  }

  convertToArrays();
}

void MomentumScaleCorrector::convertToArrays()
{
  int parScaleVecSize = parScaleVec_.size();
  int scaleFunctionVecSize = scaleFunctionVec_.size();
  if( parScaleVecSize != scaleFunctionVecSize ) {
    cout << "Error: inconsistent number of functions("<<scaleFunctionVecSize<<") and parameter sets("<<parScaleVecSize<<")" << endl;
    exit(1);
  }
  else if( parScaleVecSize != iterationNum_+1 ) {
    cout << "Error: inconsistent number of parameter sets("<<parScaleVecSize<<") and iterations("<<iterationNum_+1<<")" << endl;
    exit(1);
  }
  else if( scaleFunctionVecSize != iterationNum_+1 ) {
    cout << "Error: inconsistent number of functions("<<scaleFunctionVecSize<<") and iterations("<<iterationNum_+1<<")" << endl;
    exit(1);
  }
  parScaleArray_ = new double*[parScaleVec_.size()];
  vector<vector<double> >::const_iterator parScale = parScaleVec_.begin();

  scaleFunction_ = new scaleFunctionBase<double * >*[iterationNum_];
  vector<scaleFunctionBase<double * > * >::const_iterator scaleFunc = scaleFunctionVec_.begin();

  int iterationCounter = 0;
  for( ; parScale != parScaleVec_.end(); ++parScale, ++scaleFunc, ++iterationCounter ) {

    parScaleArray_[iterationCounter] = new double[parScale->size()];
    vector<double>::const_iterator par = parScale->begin();
    int parNum = 0;
    for ( ; par != parScale->end(); ++par, ++parNum ) {
      parScaleArray_[iterationCounter][parNum] = *par;
      // cout << "parameter["<<parNum<<"] = " << parScaleArray_[iterationCounter][parNum] << endl;
    }
    // return make_pair(parameters, parameterErrors);

    scaleFunction_[iterationCounter] = *scaleFunc;
    // cout << "scaleFunction pointer = " << scaleFunction_[iterationCounter] << endl;
  }
}
