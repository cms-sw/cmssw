/**
 * \file MillePedeVariablesIORoot.cc
 *
 *  \author    : Gero Flucke
 *  date       : November 2006
 *  $Revision: 1.7 $
 *  $Date: 2011/09/15 12:20:09 $
 *  (last update by $Author: mussgill $)
 */

// this class's header
#include "Alignment/MillePedeAlignmentAlgorithm/interface/MillePedeVariablesIORoot.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"
#include "Alignment/MillePedeAlignmentAlgorithm/interface/MillePedeVariables.h"

#include "TTree.h"

// -------------------------------------------------------------------------------------------------
MillePedeVariablesIORoot::MillePedeVariablesIORoot() :
  myId(0), myObjId(0), myNumPar(0),
  myHitsX(0), myHitsY(0), myLabel(0)
{
  treename = "MillePedeUser";
  treetxt = "MillePede User Variables";
  for (unsigned int i=0;i<kMaxNumPar;i++) {
    myIsValid[i] = 0;
    myDiffBefore[i] = 0.;
    myGlobalCor[i] = 0.;
    myPreSigma[i] = 0.;
    myParameter[i] = 0.;
    mySigma[i] = 0.;
  }
}

// -------------------------------------------------------------------------------------------------
void MillePedeVariablesIORoot::writeMillePedeVariables
(const std::vector<Alignable*> &alivec, const char *filename, int iter, bool validCheck, int &ierr)
{
  ierr = 0;

  int iret = this->open(filename, iter, true); 
  if (iret != 0) {
    ierr = -1; 
  } else {
    iret = this->write(alivec, validCheck);
    tree->BuildIndex("Id", "ObjId");
    if (iret != 0) {
      ierr = -2;
    } else {
      iret = this->close();
      if (iret != 0) {
        ierr = -3;
      }
    }
  }
  
  return;
}

// -------------------------------------------------------------------------------------------------
std::vector<AlignmentUserVariables*> MillePedeVariablesIORoot::readMillePedeVariables
(const std::vector<Alignable*> &alivec, const char *filename, int iter, int &ierr)
{
  std::vector<AlignmentUserVariables*> result;
  ierr = 0;
  int iret = this->open(filename, iter, false);
  if (iret != 0) {
    ierr = -1;
  } else {
    result = this->read(alivec, iret);
    if (iret != 0) {
      ierr = -2;
    } else {
      iret = this->close();
      if (iret != 0) {
        ierr = -3;
      }
    }
  }

  return result;
}

// -------------------------------------------------------------------------------------------------
int MillePedeVariablesIORoot::writeOne(Alignable* ali)
{
  if (!ali || !ali->alignmentParameters() 
      || !dynamic_cast<MillePedeVariables*>(ali->alignmentParameters()->userVariables())) {
    edm::LogError("Alignment") << "@SUB=MillePedeVariablesIORoot::writeOne"
                               << "No MillePedeVariables found!"; 
    return -1;
  }

  const MillePedeVariables *mpVar = 
    static_cast<MillePedeVariables*>(ali->alignmentParameters()->userVariables());
  myNumPar = mpVar->size();
  if (myNumPar >= kMaxNumPar) {
    edm::LogError("Alignment") << "@SUB=MillePedeVariablesIORoot::writeOne"
                               << "Ignoring parameters " << static_cast<int>(kMaxNumPar) << " to " << myNumPar-1;
    myNumPar = kMaxNumPar;
  }

  for (unsigned int iPar = 0; iPar < myNumPar; ++iPar) {
    myIsValid[iPar]    = mpVar->isValid()[iPar];
    myDiffBefore[iPar] = mpVar->diffBefore()[iPar];
    myGlobalCor[iPar]  = mpVar->globalCor()[iPar];
    myPreSigma[iPar]   = mpVar->preSigma()[iPar];
    myParameter[iPar]  = mpVar->parameter()[iPar];
    mySigma[iPar]      = mpVar->sigma()[iPar];
  }
  myHitsX = mpVar->hitsX();
  myHitsY = mpVar->hitsY();
  myLabel = mpVar->label();

  myId = ali->id();
  myObjId = ali->alignableObjectId();

  tree->Fill();

  return 0;
}

// -------------------------------------------------------------------------------------------------
AlignmentUserVariables* MillePedeVariablesIORoot::readOne(Alignable *ali, int &ierr)
{
  ierr = 0;

  if (tree->GetEntryWithIndex(ali->id(), ali->alignableObjectId()) < 0) {
    edm::LogError("Alignment") << "@SUB=MillePedeVariablesIORoot::readOne"
                               << "No index for id/type = (" << ali->id() << "/"
                               << ali->alignableObjectId() << ") found!";
    ierr = 1;
    return 0;
  }

  MillePedeVariables *mpVar = new MillePedeVariables(myNumPar, myLabel);
  for (unsigned int iPar = 0; iPar < myNumPar; ++iPar) {
    mpVar->isValid()[iPar]    = myIsValid[iPar];
    mpVar->diffBefore()[iPar] = myDiffBefore[iPar];
    mpVar->globalCor()[iPar]  = myGlobalCor[iPar];
    mpVar->preSigma()[iPar]   = myPreSigma[iPar];
    mpVar->parameter()[iPar]  = myParameter[iPar];
    mpVar->sigma()[iPar]      = mySigma[iPar];
  }
  mpVar->setHitsX(myHitsX);
  mpVar->setHitsY(myHitsY);
  
  return mpVar;
}

// -------------------------------------------------------------------------------------------------
void MillePedeVariablesIORoot::createBranches() 
{
  tree->Branch("Id",        &myId,        "Id/i");
  tree->Branch("ObjId",     &myObjId,     "ObjId/I");
  tree->Branch("NumPar",    &myNumPar,    "NumPar/i");
  tree->Branch("IsValid",    myIsValid,   "IsValid[NumPar]/b");
  tree->Branch("DiffBefore", myDiffBefore,"DiffBefore[NumPar]/F");
  tree->Branch("GlobalCor",  myGlobalCor, "GlobalCor[NumPar]/F");
  tree->Branch("PreSigma",   myPreSigma,  "PreSigma[NumPar]/F");
  tree->Branch("Par",        myParameter, "Par[NumPar]/F"); // name as in AlignmentParametersIORoot
  tree->Branch("Sigma",      mySigma,     "Sigma[NumPar]/F");
  tree->Branch("HitsX",     &myHitsX,     "HitsX/i");
  tree->Branch("HitsY",     &myHitsY,     "HitsY/i");
  tree->Branch("Label",     &myLabel,     "Label/i");
}

// -------------------------------------------------------------------------------------------------
void MillePedeVariablesIORoot::setBranchAddresses() 
{
  tree->SetBranchAddress("Id",        &myId);
  tree->SetBranchAddress("ObjId",     &myObjId);
  tree->SetBranchAddress("NumPar",    &myNumPar);
  tree->SetBranchAddress("IsValid",    myIsValid);
  tree->SetBranchAddress("DiffBefore", myDiffBefore);
  tree->SetBranchAddress("GlobalCor",  myGlobalCor);
  tree->SetBranchAddress("PreSigma",   myPreSigma);
  tree->SetBranchAddress("Par",        myParameter);
  tree->SetBranchAddress("Sigma",      mySigma);
  tree->SetBranchAddress("HitsX",     &myHitsX);
  tree->SetBranchAddress("HitsY",     &myHitsY);
  tree->SetBranchAddress("Label",     &myLabel);
}
