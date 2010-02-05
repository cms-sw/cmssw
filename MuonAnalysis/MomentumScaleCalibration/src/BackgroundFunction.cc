#include "MuonAnalysis/MomentumScaleCalibration/interface/BackgroundFunction.h"

using namespace std;

void BackgroundFunction::readParameters( TString fileName )
{
  iterationNum_ = 0;
  parArray_ = 0;
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

      int functionNum = 0;
      // This can be used when dealing with multiple iterations

      // cout << "line = " << line << endl;
      stringstream sLine(line);
      string num;
      int wordCounter = 0;
      // Warning: this strongly depends on the parameters file structure.
      while( sLine >> num ) {
        ++wordCounter;
        //         cout << "num["<<wordCounter<<"] = " << num << endl;
        if( wordCounter == 10 ) {
          stringstream in(num);
          in >> functionNum;
        }
        if( wordCounter == 13 ) {
          stringstream in(num);
          in >> iterationNum_;
        }
      }
      // cout << "iteration number = " << iterationNum_ << endl;
      // cout << "scale function number = " << scaleFunctionNum << endl;

//       // Create a new vector to hold the parameters for this iteration
//       vector<double> parVec;
//       parVecVec_.push_back(parVec);

      // Set the scaleFunction
      // scaleFunction_ = scaleFunctionArrayForVec[scaleFunctionNum];
      // scaleFunction_ = scaleFunctionArray[scaleFunctionNum];
      functionId_.push_back(functionNum);
      // scaleFunctionVec_.push_back( scaleFunctionArray[scaleFunctionNum] );
      backgroundFunctionVec_.push_back( backgroundFunctionService( functionNum ) );
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
//       // Fill the last vector of parameters, which corresponds to this iteration.
//       parVecVec_.back().push_back(param);
      parVecVec_.push_back(param);
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

  convertToArrays( backgroundFunction_, backgroundFunctionVec_ );
}
