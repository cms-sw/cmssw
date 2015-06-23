#include <stdlib.h>

#include <TH1F.h>
#include <TROOT.h>
#include <TFile.h>
#include <TSystem.h>
#include <sstream>
#include <fstream>

#include "FWCore/FWLite/interface/FWLiteEnabler.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/RootTreeHandler.h"

/**
 * Dumps the content of a tree to a local TreeDump.txt file. <br>
 * The txt file contains one pair per row and the values are: <br>
 * - for genInfo = 0 <br>
 * pt1 eta1 phi1 pt2 eta2 phi2 <br>
 * - for genInfo != 0 <br>
 * pt1 eta1 phi1 pt2 eta2 phi2 genPt1 genEta1 genPhi1 genPt2 genEta2 genPhi2.
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
    std::cout << "Please provide the name of the file (with file: or rfio: as needed) and if there is generator information (0 is false)" << std::endl;
    exit(1);
  }
  std::string fileName(argv[1]);
  if( fileName.find("file:") != 0 && fileName.find("rfio:") != 0 ) {
    std::cout << "Please provide the name of the file with file: or rfio: as needed" << std::endl;
    exit(1);
  }
  std::stringstream ss;
  ss << argv[2];
  bool genInfo = false;
  ss >> genInfo;
  std::cout << "Dumping tree with genInfo = " << genInfo << std::endl;

  // load framework libraries
  gSystem->Load( "libFWCoreFWLite" );
  FWLiteEnabler::enable();

  // open input file (can be located on castor)
  TFile* inFile = TFile::Open(fileName.c_str());

  // MuonPairVector pairVector;
  MuonPairVector pairVector;
  MuonPairVector genPairVector;

  // Create the RootTreeHandler to save the events in the root tree
  RootTreeHandler treeHandler;
  // treeHandler.readTree(-1, fileName, &pairVector, &genPairVector);
  std::vector<std::pair<unsigned int, unsigned long long> > evtRun;
  treeHandler.readTree(-1, fileName, &pairVector, -20, &evtRun, &genPairVector);

  if( (pairVector.size() != genPairVector.size()) && genInfo ) {
    std::cout << "Error: the size of pairVector and genPairVector is different" << std::endl;
  }

  std::ofstream outputFile;
  outputFile.open("TreeDump.txt");

  MuonPairVector::const_iterator it = pairVector.begin();
  MuonPairVector::const_iterator genIt = genPairVector.begin();
  std::vector<std::pair<unsigned int, unsigned long long> >::iterator evtRunIt = evtRun.begin();
  for( ; it != pairVector.end(); ++it, ++genIt, ++evtRunIt ) {
    // Write the information to a txt file
    outputFile << it->first.pt()  << " " << it->first.eta()  << " " << it->first.phi()  << " "
               << it->second.pt() << " " << it->second.eta() << " " << it->second.phi() << " ";
    if( genInfo ) {
      outputFile << genIt->first.pt()  << " " << genIt->first.eta()  << " " << genIt->first.phi()  << " "
		 << genIt->second.pt() << " " << genIt->second.eta() << " " << genIt->second.phi() << " ";
    }
    outputFile << " " << evtRunIt->first << " " << evtRunIt->second;
    outputFile << std::endl;
  }

  // size_t namePos = fileName.find_last_of("/");
  // treeHandler.writeTree(("tree_"+fileName.substr(namePos+1, fileName.size())).c_str(), &pairVector);

  // close input and output files
  inFile->Close();
  outputFile.close();

  return 0;
}
