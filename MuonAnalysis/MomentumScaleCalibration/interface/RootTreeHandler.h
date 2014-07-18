#include <iostream>

#include <TFile.h>
#include <TTree.h>

#include <MuonAnalysis/MomentumScaleCalibration/interface/MuonPair.h>
#include <MuonAnalysis/MomentumScaleCalibration/interface/GenMuonPair.h>
#include <MuonAnalysis/MomentumScaleCalibration/interface/MuScleFitProvenance.h>
#include <TH1F.h>
#include <stdlib.h>
#include <vector>

typedef std::vector<std::pair<lorentzVector,lorentzVector> > MuonPairVector;

/**
 * This class can be used to save all the muon pairs (and gen muon pairs if any) to a root tree. <br>
 * The writeTree method gets the name of the file to store the tree and the savedPair (and possibly genPair)
 * vector of muon pairs. <br>
 * Likewise, the readTree method takes the same arguments. It reads back from the file with the given name the
 * pairs and stores them in the given savedPair (and genPair) vector.
 */

class RootTreeHandler
{
public:
  // void writeTree( const TString & fileName, const MuonPairVector * savedPair, const int muonType = 0,
  //                 const MuonPairVector * genPair = 0, const bool saveAll = false )
  void writeTree( const TString & fileName, const std::vector<MuonPair> * savedPair, const int muonType = 0,
		  const std::vector<GenMuonPair> * genPair = 0, const bool saveAll = false )
  {
    lorentzVector emptyLorentzVector(0,0,0,0);
    TFile * f1 = new TFile(fileName, "RECREATE");
    TTree * tree = new TTree("T", "Muon pairs");
    MuonPair * muonPair = new MuonPair;
    GenMuonPair * genMuonPair = new GenMuonPair;
    // MuonPair * genMuonPair = new MuonPair;
    tree->Branch("event", "MuonPair", &muonPair);
    if( genPair != 0 ) {
      tree->Branch("genEvent", "GenMuonPair", &genMuonPair);
      // tree->Branch("genEvent", "MuonPair", &genMuonPair);
      
      if( savedPair->size() != genPair->size() ) {
	std::cout << "Error: savedPair size ("
	<< savedPair->size() <<") and genPair size ("
	<< genPair->size() <<") are different. This is severe and I will not write the tree." << std::endl;
	exit(1);
      }
    }
    std::cout << "savedPair->size() is "<<savedPair->size()<< std::endl;
    std::vector<MuonPair>::const_iterator muonPairIt = savedPair->begin();
    unsigned int iev = 0;
    for( ; muonPairIt != savedPair->end(); ++muonPairIt, ++iev ) {

      if( saveAll || ( (muonPairIt->mu1.p4() != emptyLorentzVector) && (muonPairIt->mu2.p4() != emptyLorentzVector) ) ) {

	// muonPair->setPair(muonType, std::make_pair(muonPairIt->first, muonPairIt->second));
	muonPair->copy(*muonPairIt);

	// if( genPair != 0 && genPair->size() != 0 ) {
	//   genMuonPair->setPair(muonId, std::make_pair((*genPair)[iev].first, (*genPair)[iev].second));
	//   genMuonPair->mu1 = ((*genPair)[iev].first);
	//   genMuonPair->mu2 = ((*genPair)[iev].second);
        // }
	if( genPair != 0 ) {
	  genMuonPair->copy((*genPair)[iev]);
	}

	tree->Fill();
      }
      // // Tree filled. Clear the map for the next event.
      // muonPair->muonPairs.clear();
    }

    // Save provenance information in the TFile
    TH1F muonTypeHisto("MuonType", "MuonType", 40, -20, 20);
    muonTypeHisto.Fill(muonType);
    muonTypeHisto.Write();
    MuScleFitProvenance provenance(muonType);
    provenance.Write();

    f1->Write();
    f1->Close();
  }

  // void readTree( const int maxEvents, const TString & fileName, MuonPairVector * savedPair,
  //           	    const int muonType, MuonPairVector * genPair = 0 )
  void readTree( const int maxEvents, const TString & fileName, MuonPairVector * savedPair,
		 const int muonType, std::vector<std::pair<int, int>  > * evtRun, MuonPairVector * genPair = 0 )
  {
    TFile * file = TFile::Open(fileName, "READ");
    if( file->IsOpen() ) {
      TTree * tree = (TTree*)file->Get("T");
      MuonPair * muonPair = 0;
      GenMuonPair * genMuonPair = 0;
      // MuonPair * genMuonPair = 0;
      tree->SetBranchAddress("event",&muonPair);
      if( genPair != 0 ) {
        tree->SetBranchAddress("genEvent",&genMuonPair);
      }

      Long64_t nentries = tree->GetEntries();
      if( (maxEvents != -1) && (nentries > maxEvents) ) nentries = maxEvents;
      for( Long64_t i=0; i<nentries; ++i ) {
        tree->GetEntry(i);
	//std::cout << "Reco muon1, pt = " << muonPair->mu1 << "; Reco muon2, pt = " << muonPair->mu2 << std::endl;
        savedPair->push_back(std::make_pair(muonPair->mu1.p4(), muonPair->mu2.p4()));
	evtRun->push_back(std::make_pair(muonPair->event.event(), muonPair->event.run()));
        // savedPair->push_back(muonPair->getPair(muonType));
        if( genPair != 0 ) {
          genPair->push_back(std::make_pair(genMuonPair->mu1.p4(), genMuonPair->mu2.p4()));
	  //std::cout << "Gen muon1, pt = " << genMuonPair->mu1 << "; Gen muon2, pt = " << genMuonPair->mu2 << std::endl;
          // genPair->push_back(genMuonPair->getPair(muonId));
        }
      }
    }
    else {
      std::cout << "ERROR: no file " << fileName << " found. Please, correct the file name or specify an empty field in the InputRootTreeFileName parameter to read events from the edm source." << std::endl;
      exit(1);
    }
    file->Close();
  }

  /// Used to read the external trees
  void readTree( const int maxEvents, const TString & fileName, std::vector<MuonPair> * savedPair,
		 const int muonType, std::vector<GenMuonPair> * genPair = 0 )
  {
    TFile * file = TFile::Open(fileName, "READ");
    if( file->IsOpen() ) {
      TTree * tree = (TTree*)file->Get("T");
      MuonPair * muonPair = 0;
      GenMuonPair * genMuonPair = 0;
      tree->SetBranchAddress("event",&muonPair);
      if( genPair != 0 ) {
        tree->SetBranchAddress("genEvent",&genMuonPair);
      }

      Long64_t nentries = tree->GetEntries();
      if( (maxEvents != -1) && (nentries > maxEvents) ) nentries = maxEvents;
      for( Long64_t i=0; i<nentries; ++i ) {
        tree->GetEntry(i);
        savedPair->push_back(*muonPair);
        if( genPair != 0 ) {
          genPair->push_back(*genMuonPair);
        }
      }
    }
    else {
      std::cout << "ERROR: no file " << fileName << " found. Please, correct the file name or specify an empty field in the InputRootTreeFileName parameter to read events from the edm source." << std::endl;
      exit(1);
    }
    file->Close();
  }

};
