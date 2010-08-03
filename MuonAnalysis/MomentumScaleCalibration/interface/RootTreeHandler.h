#include <iostream>

#include <TFile.h>
#include <TTree.h>

#include <MuonAnalysis/MomentumScaleCalibration/interface/MuonPair.h>

typedef std::vector<std::pair<lorentzVector,lorentzVector> > MuonPairVector;

/**
 * This class can be used to save all the muon pair (and gen muon pairs if any) to a root tree. <br>
 * The writeTree method gets the name of the file to store the tree and the savedPair (and possibly genPair)
 * vector of muon pairs. <br>
 * Likewise, the readTree method takes the same arguments. It reads back from the file with the given name the
 * pairs and stores them in the given savedPair (and genPair) vector.
 */

class RootTreeHandler
{
public:
  void writeTree( const TString & fileName, const MuonPairVector * savedPair, const MuonPairVector * genPair = 0, const bool saveAll = false )
  {
    lorentzVector emptyLorentzVector(0,0,0,0);
    TFile * f1 = new TFile(fileName, "RECREATE");
    TTree * tree = new TTree("T", "Muon pairs");
    MuonPair * muonPair = new MuonPair;
    MuonPair * genMuonPair = new MuonPair;
    tree->Branch("event", "MuonPair", &muonPair);
    if( genPair != 0 ) {
      tree->Branch("genEvent", "MuonPair", &genMuonPair);
    }

    MuonPairVector::const_iterator muonPairIt = savedPair->begin();
    unsigned int iev = 0;
    for( ; muonPairIt != savedPair->end(); ++muonPairIt, ++iev ) {

      if( saveAll || ( (muonPairIt->first != emptyLorentzVector) && (muonPairIt->second != emptyLorentzVector) ) ) {
	muonPair->mu1 = (muonPairIt->first);
	muonPair->mu2 = (muonPairIt->second);

	if( genPair != 0 && genPair->size() != 0 ) {
	  genMuonPair->mu1 = ((*genPair)[iev].first);
	  genMuonPair->mu2 = ((*genPair)[iev].second);
	}

	tree->Fill();
      }
    }
    f1->Write();
    f1->Close();
  }

  void readTree( const int maxEvents, const TString & fileName, MuonPairVector * savedPair, MuonPairVector * genPair = 0 )
  {
    TFile * file = new TFile(fileName, "READ");
    if( file->IsOpen() ) {
      TTree * tree = (TTree*)file->Get("T");
      MuonPair * muonPair = 0;
      MuonPair * genMuonPair = 0;
      tree->SetBranchAddress("event",&muonPair);
      if( genPair != 0 ) {
        tree->SetBranchAddress("genEvent",&genMuonPair);
      }

      Long64_t nentries = tree->GetEntries();
      if( (maxEvents != -1) && (nentries > maxEvents) ) nentries = maxEvents;
      for( Long64_t i=0; i<nentries; ++i ) {
        tree->GetEntry(i);
        savedPair->push_back(std::make_pair(muonPair->mu1, muonPair->mu2));
        if( genPair != 0 ) {
          genPair->push_back(std::make_pair(genMuonPair->mu1, genMuonPair->mu2));
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
