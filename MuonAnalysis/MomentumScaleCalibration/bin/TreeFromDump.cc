#include <memory>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <stdlib.h>

#include <TH1F.h>
#include <TROOT.h>
#include <TFile.h>
#include <TSystem.h>

#include "FWCore/FWLite/interface/AutoLibraryLoader.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/RootTreeHandler.h"

/**
 * Builds a tree from the dump in the TreeDump.txt file produced by the TreeDump macro.
 */

// Useful function to convert 4-vector coordinates
// -----------------------------------------------
lorentzVector fromPtEtaPhiToPxPyPz( const double* ptEtaPhiE )
{
  double muMass = 0.105658;
  double px = ptEtaPhiE[0]*cos(ptEtaPhiE[2]);
  double py = ptEtaPhiE[0]*sin(ptEtaPhiE[2]);
  double tmp = 2*atan(exp(-ptEtaPhiE[1]));
  double pz = ptEtaPhiE[0]*cos(tmp)/sin(tmp);
  double E  = sqrt(px*px+py*py+pz*pz+muMass*muMass);

  return lorentzVector(px,py,pz,E);
}

int main(int argc, char* argv[]) 
{

  if( argc != 3 ) {
    std::cout << "Please provide the name of the file and if there is generator information (0 is false)" << std::endl;
    exit(1);
  }
  std::string fileName(argv[1]);
  std::stringstream ss;
  ss << argv[2];
  bool genInfo = false;
  ss >> genInfo;
  std::cout << "Reading tree dump with genInfo = " << genInfo << std::endl;

  // load framework libraries
  gSystem->Load( "libFWCoreFWLite" );
  AutoLibraryLoader::enable();
  
  // MuonPairVector pairVector;
  std::vector<MuonPair> pairVector;
  std::vector<GenMuonPair> genPairVector;

  // Create the RootTreeHandler to save the events in the root tree
  RootTreeHandler treeHandler;

  std::ifstream inputFile;
  inputFile.open(fileName.c_str());

  std::string line;
  double value[6];
  double genValue[6];
  // Read the information from a txt file
  while( !inputFile.eof() ) {
    getline(inputFile, line);
    if( line != "" ) {
      // std::cout << "line = " << line << std::endl;
      std::stringstream ss(line);
      for( int i=0; i<6; ++i ) {
	ss >> value[i];
	// std::cout << "value["<<i<<"] = " << value[i] << std::endl;
      }
      pairVector.push_back(MuonPair(fromPtEtaPhiToPxPyPz(value), fromPtEtaPhiToPxPyPz(&(value[3])), MuScleFitEvent(0,0,0,0,0)) );
      if( genInfo ) {
	for( int i=0; i<6; ++i ) {
	  ss >> genValue[i];
	  // std::cout << "genValue["<<i<<"] = " << genValue[i] << std::endl;
	}
	genPairVector.push_back(GenMuonPair(fromPtEtaPhiToPxPyPz(genValue), fromPtEtaPhiToPxPyPz(&(genValue[3])), 0));
      }
    }
  }
  inputFile.close();
  
  if( (pairVector.size() != genPairVector.size()) && genInfo ) {
    std::cout << "Error: the size of pairVector and genPairVector is different" << std::endl;
  }

  if( genInfo ) {
    treeHandler.writeTree("TreeFromDump.root", &pairVector, 0, &genPairVector);
    std::cout << "Filling tree with genInfo" << std::endl;
  }
  else {
    treeHandler.writeTree("TreeFromDump.root", &pairVector);
    std::cout << "Filling tree" << std::endl;
  }
  // close input file
  inputFile.close();

  return 0;
}
