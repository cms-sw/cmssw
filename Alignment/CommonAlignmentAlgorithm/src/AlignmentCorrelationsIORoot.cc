#include "TTree.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// this class's header
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentCorrelationsIORoot.h"

// ----------------------------------------------------------------------------
// constructor

AlignmentCorrelationsIORoot::AlignmentCorrelationsIORoot()
{
  treename = "AlignmentCorrelations";
  treetxt = "Correlations";
}

// ----------------------------------------------------------------------------

void AlignmentCorrelationsIORoot::createBranches(void) 
{
  tree->Branch("Ali1Id",    &Ali1Id,    "Ali1Id/i");
  tree->Branch("Ali2Id",    &Ali2Id,    "Ali2Id/i");
  tree->Branch("Ali1ObjId", &Ali1ObjId, "Ali1ObjId/I");
  tree->Branch("Ali2ObjId", &Ali2ObjId, "Ali2ObjId/I");
  tree->Branch("corSize",   &corSize,   "corSize/I");
  tree->Branch("CorMatrix", &CorMatrix, "CorMatrix[corSize]/D");
}

// ----------------------------------------------------------------------------

void AlignmentCorrelationsIORoot::setBranchAddresses(void) 
{
  tree->SetBranchAddress("corSize",   &corSize);
  tree->SetBranchAddress("Ali1Id",    &Ali1Id);
  tree->SetBranchAddress("Ali2Id",    &Ali2Id);
  tree->SetBranchAddress("Ali1ObjId", &Ali1ObjId);
  tree->SetBranchAddress("Ali2ObjId", &Ali2ObjId);	 
  tree->SetBranchAddress("CorMatrix", &CorMatrix);
}

// ----------------------------------------------------------------------------

int AlignmentCorrelationsIORoot::write(const align::Correlations& cor, bool validCheck)
{
  int icount=0;

  for(align::Correlations::const_iterator it=cor.begin();
    it!=cor.end();it++) {
    AlgebraicMatrix mat=(*it).second;
	std::pair<Alignable*,Alignable*> Pair = (*it).first;
    Alignable* ali1 = Pair.first;
    Alignable* ali2 = Pair.second;
    if( (ali1->alignmentParameters()->isValid()
	 && ali2->alignmentParameters()->isValid()) || !(validCheck)) {
      Ali1ObjId = ali1->alignableObjectId(); 
      Ali2ObjId = ali2->alignableObjectId(); 
      Ali1Id = ali1->id();
      Ali2Id = ali2->id();
      int maxColumn = mat.num_row();
      corSize = maxColumn*maxColumn;
      for(int row = 0;row<maxColumn;row++)	  
		for(int col = 0;col<maxColumn;col++)
		  CorMatrix[row+col*maxColumn] =mat[row][col]; 
      tree->Fill();
      icount++;
    }
  }
  edm::LogInfo("AlignmentCorrelationsIORoot") << "Writing correlations: all,written: " 
                                              << cor.size() << "," << icount;
  return 0;
}

// ----------------------------------------------------------------------------
// read correlations for those alignables in vector given as argument

align::Correlations 
AlignmentCorrelationsIORoot::read(const align::Alignables& alivec, int& ierr)
{
  align::Correlations theMap;

  // create ID map for all Alignables in alivec
  align::Alignables::const_iterator it1;
  std::map< std::pair<unsigned int,int>, Alignable* > idAlis;
  for( it1=alivec.begin();it1!=alivec.end();it1++ )
    idAlis[std::make_pair((*it1)->id(),(*it1)->alignableObjectId())] = (*it1);

  std::map<std::pair<unsigned int,int>,Alignable*>::const_iterator aliSearch1;  
  std::map<std::pair<unsigned int,int>,Alignable*>::const_iterator aliSearch2;  
  int nfound=0;
  double maxEntry = tree->GetEntries();
  for( int entry = 0;entry<maxEntry;entry++ ) 
	{
	  tree->GetEntry(entry);
	  aliSearch1 = idAlis.find(std::make_pair(Ali1Id,Ali1ObjId));
	  aliSearch2 = idAlis.find(std::make_pair(Ali2Id,Ali2ObjId));
	  if (aliSearch1!=idAlis.end()
		  && aliSearch2!=idAlis.end()) 
		{
		  // Alignables for this pair found
		  nfound++;
		  Alignable* myAli1 = (*aliSearch1).second;
		  Alignable* myAli2 = (*aliSearch2).second;
		  // FIXME: instead of nParMax in the next few lines one should probably
		  //        use something like sqrt(corSize) - but take care of rounding!
		  //        I have no time to test... :-( GF
		  AlgebraicMatrix  mat(nParMax,nParMax);
		  for(int row = 0;row<nParMax;row++) 
			for(int col = 0;col<nParMax;col++) 
			  mat[row][col] = CorMatrix[row+col*nParMax];
		  theMap[ std::make_pair(myAli1,myAli2) ] = mat;
		}
	}

  edm::LogInfo("AlignmentCorrelationsIORoot") << "Read correlations: all,read: " 
                                              << alivec.size() << "," << nfound;

  ierr=0;
  return theMap;
}
