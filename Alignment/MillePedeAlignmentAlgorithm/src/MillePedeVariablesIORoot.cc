/**
 * \file MillePedeVariablesIORoot.cc
 *
 *  \author    : Gero Flucke
 *  date       : November 2006
 *  $Revision: 1.2 $
 *  $Date: 2006/11/07 10:45:09 $
 *  (last update by $Author: flucke $)
 */

// this class's header
#include "Alignment/MillePedeAlignmentAlgorithm/interface/MillePedeVariablesIORoot.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/TrackerAlignment/interface/TrackerAlignableId.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"
#include "Alignment/MillePedeAlignmentAlgorithm/interface/MillePedeVariables.h"


// -------------------------------------------------------------------------------------------------
MillePedeVariablesIORoot::MillePedeVariablesIORoot()
{
  treename = "MillePedeUser";
  treetxt = "MillePede User Variables";
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
                               << "no MillePedeVariables found!"; 
    return -1;
  }

  const MillePedeVariables *mpVar = 
    dynamic_cast<MillePedeVariables*>(ali->alignmentParameters()->userVariables());
  myNumPar = mpVar->size();
  if (myNumPar >= kMaxNumPar) {
    edm::LogError("Alignment") << "@SUB=MillePedeVariablesIORoot::writeOne"
                               << "ignoring parameters " << kMaxNumPar << " to " << myNumPar-1;
    myNumPar = kMaxNumPar;
  }

  for (unsigned int iPar = 0; iPar < myNumPar; ++iPar) {
    myIsValid[iPar]    = mpVar->isValid()[iPar];
    myDiffBefore[iPar] = mpVar->diffBefore()[iPar];
    myGlobalCor[iPar]  = mpVar->globalCor()[iPar];
    myPreSigma[iPar]   = mpVar->preSigma()[iPar];
  }

  const TrackerAlignableId ID;
  const TrackerAlignableId::UniqueId detType = ID.alignableUniqueId(ali); 
  myId = detType.first;
  myObjId = detType.second;

  tree->Fill();

  return 0;
}

// -------------------------------------------------------------------------------------------------
AlignmentUserVariables* MillePedeVariablesIORoot::readOne(Alignable *ali, int &ierr)
{
  ierr = 0;

  const TrackerAlignableId ID;
  const TrackerAlignableId::UniqueId detType = ID.alignableUniqueId(ali); 
  
  if (tree->GetEntryWithIndex(detType.first, detType.second) < 0) {
    edm::LogError("Alignment") << "@SUB=MillePedeVariablesIORoot::readOne"
                               << "no index for detType = (" << detType.first << "/"
                               << detType.second << ") found!";
    ierr = 1;
    return 0;
  }

  MillePedeVariables *mpVar = new MillePedeVariables(myNumPar);
  for (unsigned int iPar = 0; iPar < myNumPar; ++iPar) {
    mpVar->isValid()[iPar]    = myIsValid[iPar];
    mpVar->diffBefore()[iPar] = myDiffBefore[iPar];
    mpVar->globalCor()[iPar]  = myGlobalCor[iPar];
    mpVar->preSigma()[iPar]   = myPreSigma[iPar];
  }
  
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
}
