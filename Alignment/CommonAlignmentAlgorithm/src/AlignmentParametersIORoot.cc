// this class's header
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParametersIORoot.h"

#include "Alignment/CommonAlignment/interface/Alignable.h" 
#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"
#include "Alignment/CommonAlignmentParametrization/interface/AlignmentParametersFactory.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TTree.h"


// ----------------------------------------------------------------------------
// constructor
AlignmentParametersIORoot::AlignmentParametersIORoot()
{
  treename = "AlignmentParameters";
  treetxt = "Alignment Parameters";
}


// ----------------------------------------------------------------------------
void AlignmentParametersIORoot::createBranches(void) 
{
  tree->Branch("parSize",   &theCovRang,   "CovRang/I");
  tree->Branch("Id",        &theId,        "Id/i");
  tree->Branch("paramType", &theParamType, "paramType/I");
  tree->Branch("Par",       &thePar,       "Par[CovRang]/D");
  tree->Branch("covarSize", &theCovarRang, "CovarRang/I");
  tree->Branch("Cov",       &theCov,       "Cov[CovarRang]/D");
  tree->Branch("ObjId",     &theObjId,     "ObjId/I");
  tree->Branch("HieraLevel",&theHieraLevel,"HieraLevel/I");
}


// ----------------------------------------------------------------------------
void AlignmentParametersIORoot::setBranchAddresses(void) 
{
  tree->SetBranchAddress("parSize",   &theCovRang);
  tree->SetBranchAddress("covarSize", &theCovarRang);
  tree->SetBranchAddress("Id",        &theId);
  tree->SetBranchAddress("Par",       &thePar);
  tree->SetBranchAddress("paramType", &theParamType);
  tree->SetBranchAddress("Cov",       &theCov);
  tree->SetBranchAddress("ObjId",     &theObjId);
  tree->SetBranchAddress("HieraLevel",&theHieraLevel);
}


// ----------------------------------------------------------------------------
int AlignmentParametersIORoot::writeOne(Alignable* ali)
{
  const AlignmentParameters* ap =ali->alignmentParameters();
  const AlgebraicVector& params = ap->parameters();
  const AlgebraicSymMatrix& cov = ap->covariance();

  theCovRang   = params.num_row();
  theCovarRang = theCovRang*(theCovRang+1)/2;
  int count=0;
  for(int row=0;row<theCovRang;row++){
    thePar[row]=params[row];
    for(int col=0;col<theCovRang;col++){
      if(row-1<col) { theCov[count] = cov[row][col]; count++; }
    }
  }

  theId = ali->id();
  theParamType = ap->type();
  theObjId = ali->alignableObjectId();
  theHieraLevel = ap->hierarchyLevel();

  tree->Fill();
  return 0;
}


// ----------------------------------------------------------------------------
AlignmentParameters* AlignmentParametersIORoot::readOne( Alignable* ali, int& ierr )
{
  
  if( tree->GetEntryWithIndex( ali->id(), ali->alignableObjectId() ) > 0 ) 
  {
    int covsize = theCovRang;
    int count=0;
    AlgebraicVector par(covsize, 0);
    AlgebraicSymMatrix cov(covsize, 0);
    for(int row=0;row<covsize;row++) 
    {
      par[row]=thePar[row];
      for(int col=0; col < covsize;col++) {
	if(row-1<col) {cov[row][col]=theCov[count];count++;}
      }
    }

    using namespace AlignmentParametersFactory;
    ParametersType parType = parametersType(theParamType);
    AlignmentParameters* alipar1; 
    if ( ali->alignmentParameters() )
    {
      const std::vector<bool>& sel = ali->alignmentParameters()->selector();
      alipar1 = createParameters(ali, parType, sel); 
    } else {
      const std::vector<bool> sel( theCovRang, true );
      alipar1 = createParameters(ali, parType, sel); 
    }
    AlignmentParameters* alipar = alipar1->clone(par,cov);
    alipar->setValid(true); 
    ierr=0;
    delete alipar1;
    return alipar;
  }

  ierr=-1;
  return(nullptr);
}


int AlignmentParametersIORoot::close()
{
  if ( bWrite )
  {
    int nIndices = tree->BuildIndex( "Id", "ObjId" );
    edm::LogInfo( "Alignment" ) << "@SUB=AlignmentParametersIORoot::setBranchAddresses"
				<< "number of indexed entries: " << nIndices;
  }

  return closeRoot();
}
