/**
 * \file AlignmentUserVariablesIO
 *
 *  $Revision: 1.4 $
 *  $Date: 2007/10/08 14:38:16 $
 *  $Author: cklae $ (at least last update...)
 */

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// this class's header
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentUserVariablesIO.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"

//-----------------------------------------------------------------------------
// write many user variables

int 
AlignmentUserVariablesIO::write(const align::Alignables& alivec, 
  bool validCheck) 
{
  int icount=0;
  for(align::Alignables::const_iterator it=alivec.begin();
    it!=alivec.end(); it++) {
    if ((*it)->alignmentParameters()->isValid() || !(validCheck)) {
      icount++;
      int iret=writeOne(*it);
      if (iret!=0) return iret;
    }
  }
  edm::LogInfo("Alignment") << "@SUB=AlignmentUserVariablesIO::write"
                            << "Write variables all,written: " << alivec.size() <<","<< icount;
  return 0;
}

//-----------------------------------------------------------------------------
// read many user variables

std::vector<AlignmentUserVariables*> 
AlignmentUserVariablesIO::read(const align::Alignables& alivec, int& ierr) 
{
  std::vector<AlignmentUserVariables*> retvec;
  ierr=0;
  int ierr2;
  int icount=0;
  int icount2=0;
  for(align::Alignables::const_iterator it=alivec.begin();
    it!=alivec.end(); it++) {
    AlignmentUserVariables* ad=readOne(*it, ierr2); // should create with new!
    if (ierr2==0) { 
      retvec.push_back(ad); icount++; 
      if (ad!=0) icount2++;
    }
  }
  edm::LogInfo("Alignment") << "@SUB=AlignmentUserVariablesIO::read"
                            << "Read variables all,read,valid: " << alivec.size() <<","
                            << icount <<","<< icount2;
  return retvec;
}
