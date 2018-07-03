#include "MuonAnalysis/MomentumScaleCalibration/interface/MomentumScaleCorrector.h"

void MomentumScaleCorrector::readParameters( TString fileName )
{
  iterationNum_ = 0;
  parArray_ = nullptr;
  // std::vector<double> parameterErrors;

  // Read the parameters file
  std::ifstream parametersFile(fileName.Data());

  if( !parametersFile.is_open() ) {
    std::cout << "Error: file " << fileName << " not found. Aborting." << std::endl;
    abort();
  }
  std::string line;

  std::string iteration("Iteration ");
  // Loop on the file lines
  while (parametersFile) {
    getline( parametersFile, line );
    size_t lineInt = line.find("value");

    // if( line.find(iteration) != std::string::npos ) {
    size_t iterationSubStr = line.find(iteration);

    // Take the iteration number
    if( iterationSubStr != std::string::npos ) {

      int scaleFunctionNum = 0;
      // This can be used when dealing with multiple iterations

      // std::cout << "line = " << line << std::endl;
      std::stringstream sLine(line);
      std::string num;
      int wordCounter = 0;
      // Warning: this strongly depends on the parameters file structure.
      while( sLine >> num ) {
        ++wordCounter;
        //         std::cout << "num["<<wordCounter<<"] = " << num << std::endl;
        if( wordCounter == 9 ) {
	  std::stringstream in(num);
          in >> scaleFunctionNum;
        }
        if( wordCounter == 13 ) {
	  std::stringstream in(num);
          in >> iterationNum_;
        }
      }
      // std::cout << "iteration number = " << iterationNum_ << std::endl;
      // std::cout << "scale function number = " << scaleFunctionNum << std::endl;

      // Create a new vector to hold the parameters for this iteration
//       std::vector<double> parScale;
//       parVecVec_.push_back(parScale);

      // Set the scaleFunction
      // scaleFunction_ = scaleFunctionArrayForVec[scaleFunctionNum];
      // scaleFunction_ = scaleFunctionArray[scaleFunctionNum];
      functionId_.push_back(scaleFunctionNum);
      // scaleFunctionVec_.push_back( scaleFunctionArray[scaleFunctionNum] );
      scaleFunctionVec_.push_back( scaleFunctionService( scaleFunctionNum ) );
    }
    // Take the parameters for the current iteration
    if ( (lineInt != std::string::npos) ) {
      size_t subStr1 = line.find("value");
      std::stringstream paramStr;
      double param = 0;
      // Even if all the rest of the line is taken, the following
      // conversion to a double will stop at the end of the first number.
      paramStr << line.substr(subStr1+5);
      paramStr >> param;
//       // Fill the last vector of parameters, which corresponds to this iteration.
//       parVecVec_.back().push_back(param);
      parVecVec_.push_back(param);
      // std::cout << "param = " << param << std::endl;

      // This is to extract parameter errors
      // size_t subStr2 = line.find("+-");
      // std::stringstream parErrorStr;
      // double parError = 0;
      // parErrorStr << line.substr(subStr2+1);
      // parErrorStr >> parError;
      // parameterErrors.push_back(parError);
      // std::cout << "parError = " << parError << std::endl;
    }
  }

  convertToArrays( scaleFunction_, scaleFunctionVec_ );
}
