#include <memory>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>

#include <TH1F.h>
#include <TROOT.h>
#include <TFile.h>
#include <TSystem.h>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "FWCore/FWLite/interface/FWLiteEnabler.h"
#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "PhysicsTools/FWLite/interface/TFileService.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/RootTreeHandler.h"

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

  // lorentzVector corrMu(px,py,pz,E);
  // To fix memory leaks, this is to be substituted with
  // std::unique_ptr<lorentzVector> corrMu(new lorentzVector(px, py, pz, E));

  return lorentzVector(px,py,pz,E);
}

int main(int argc, char* argv[]) 
{

  if( argc != 2 ) {
    std::cout << "Please provide the name of the file with file: or rfio: as needed" << std::endl;
    exit(1);
  }
  std::string fileName(argv[1]);
  if( fileName.find("file:") != 0 && fileName.find("rfio:") != 0 ) {
    std::cout << "Please provide the name of the file with file: or rfio: as needed" << std::endl;
    exit(1);
  }

  // ----------------------------------------------------------------------
  // First Part:
  //
  //  * enable FWLite 
  //  * book the histograms of interest 
  //  * open the input file
  // ----------------------------------------------------------------------

  // load framework libraries
  gSystem->Load( "libFWCoreFWLite" );
  FWLiteEnabler::enable();
  
  // book a set of histograms
  fwlite::TFileService fs = fwlite::TFileService("analyzeBasics.root");
  TFileDirectory theDir = fs.mkdir("analyzeBasic");
  TH1F* muonPt_  = theDir.make<TH1F>("muonPt", "pt",    100,  0.,300.);
  TH1F* muonEta_ = theDir.make<TH1F>("muonEta","eta",   100, -3.,  3.);
  TH1F* muonPhi_ = theDir.make<TH1F>("muonPhi","phi",   100, -5.,  5.);  
  
  // open input file (can be located on castor)
  TFile* inFile = TFile::Open(fileName.c_str());

//   TFile* inFile = TFile::Open("rfio:/castor/cern.ch/user/f/fabozzi/36XSkimData/run_139791-140159/NtupleLoose_139791-140159_v2.root");
//   TFile* inFile = TFile::Open("rfio:/castor/cern.ch/user/f/fabozzi/36XSkimData/run_140160-140182/NtupleLoose_140160-140182.root");
//   TFile* inFile = TFile::Open("rfio:/castor/cern.ch/user/f/fabozzi/36XSkimData/run_140183-140399/NtupleLoose_140183-140399.root");
//   TFile* inFile = TFile::Open("rfio:/castor/cern.ch/user/d/degrutto/36XSkimData/run_140440-141961/NtupleLoose_140440-141961.root");
//   TFile* inFile = TFile::Open("rfio:/castor/cern.ch/user/d/degrutto/36XSkimData/run_142035-142664/NtupleLoose_142035-142664.root");


  // ----------------------------------------------------------------------
  // Second Part: 
  //
  //  * loop the events in the input file 
  //  * receive the collections of interest via fwlite::Handle
  //  * fill the histograms
  //  * after the loop close the input file
  // ----------------------------------------------------------------------

  // Create the RootTreeHandler to save the events in the root tree
  RootTreeHandler treeHandler;
  // MuonPairVector pairVector;
  std::vector<MuonPair> pairVector;

  // loop the events
  unsigned int iEvent=0;
  fwlite::Event ev(inFile);
  for(ev.toBegin(); !ev.atEnd(); ++ev, ++iEvent){
    
    // simple event counter
    if(iEvent>0 && iEvent%100==0){
      std::cout << "  processing event: " << iEvent << std::endl;
    }

    // Handle to the muon collection
    fwlite::Handle<std::vector<float> > muon1pt;
    fwlite::Handle<std::vector<float> > muon1eta;
    fwlite::Handle<std::vector<float> > muon1phi;
    fwlite::Handle<std::vector<float> > muon2pt;
    fwlite::Handle<std::vector<float> > muon2eta;
    fwlite::Handle<std::vector<float> > muon2phi;
    muon1pt.getByLabel(ev, "goodZToMuMuEdmNtupleLoose", "zGoldenDau1Pt");
    muon1eta.getByLabel(ev, "goodZToMuMuEdmNtupleLoose", "zGoldenDau1Eta");
    muon1phi.getByLabel(ev, "goodZToMuMuEdmNtupleLoose", "zGoldenDau1Phi");
    muon2pt.getByLabel(ev, "goodZToMuMuEdmNtupleLoose", "zGoldenDau2Pt");
    muon2eta.getByLabel(ev, "goodZToMuMuEdmNtupleLoose", "zGoldenDau2Eta");
    muon2phi.getByLabel(ev, "goodZToMuMuEdmNtupleLoose", "zGoldenDau2Phi");

    if( !muon1pt.isValid() ) continue;
    if( !muon1eta.isValid() ) continue;
    if( !muon1phi.isValid() ) continue;
    if( !muon2pt.isValid() ) continue;
    if( !muon2eta.isValid() ) continue;
    if( !muon2phi.isValid() ) continue;
    // std::cout << "muon1pt = " << muon1pt->size() << std::endl;

    // loop muon collection and fill histograms
    if( muon1pt->size() != muon2pt->size() ) {
      std::cout << "Error: size of muon1 and muon2 is different. Skipping event" << std::endl;
      continue;
    }
    for(unsigned i=0; i<muon1pt->size(); ++i){
      muonPt_->Fill( (*muon1pt)[i] );
      muonEta_->Fill( (*muon1eta)[i] );
      muonPhi_->Fill( (*muon1phi)[i] );
      muonPt_->Fill( (*muon2pt)[i] );
      muonEta_->Fill( (*muon2eta)[i] );
      muonPhi_->Fill( (*muon2phi)[i] );

      double muon1[3] = {(*muon1pt)[i], (*muon1eta)[i], (*muon1phi)[i]};
      double muon2[3] = {(*muon2pt)[i], (*muon2eta)[i], (*muon2phi)[i]};

      // pairVector.push_back( std::make_pair( fromPtEtaPhiToPxPyPz(muon1), fromPtEtaPhiToPxPyPz(muon2) ) );
      pairVector.push_back( MuonPair(fromPtEtaPhiToPxPyPz(muon1), fromPtEtaPhiToPxPyPz(muon2), MuScleFitEvent(0,0,0,0,0,0)) );
    }
  }
  size_t namePos = fileName.find_last_of("/");
  treeHandler.writeTree(("tree_"+fileName.substr(namePos+1, fileName.size())).c_str(), &pairVector);

  // close input file
  inFile->Close();

  // ----------------------------------------------------------------------
  // Third Part: 
  //
  //  * never forget to free the memory of objects you created
  // ----------------------------------------------------------------------

  // in this example there is nothing to do 
  
  // that's it!
  return 0;
}
