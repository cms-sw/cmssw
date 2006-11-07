#include "FWCore/MessageLogger/interface/MessageLogger.h"

// this class's header
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentUserVariablesIO.h"

//-----------------------------------------------------------------------------
// destructor (deletes read in user variables)
 
AlignmentUserVariablesIO::~AlignmentUserVariablesIO()
{
  LogDebug("Alignment") <<"@SUB=AlignmentUserVariablesIO::~AlignmentUserVariablesIO";
  //int i=0;
  //for (vector<AlignmentUserVariables*>::iterator 
  //   it=theReadUserVariables.begin(); it!=theReadUserVariables.end(); it++) {
    //cout <<"now deleting read uservar " << i++ << endl;
    // delete *it;
  //}

}

//-----------------------------------------------------------------------------
// write many user variables

int 
AlignmentUserVariablesIO::write(const std::vector<Alignable*>& alivec, 
  bool validCheck) 
{
  int icount=0;
  for(std::vector<Alignable*>::const_iterator it=alivec.begin();
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
AlignmentUserVariablesIO::read(const std::vector<Alignable*>& alivec, int& ierr) 
{
  std::vector<AlignmentUserVariables*> retvec;
  ierr=0;
  int ierr2;
  int icount=0;
  int icount2=0;
  for(std::vector<Alignable*>::const_iterator it=alivec.begin();
    it!=alivec.end(); it++) {
    AlignmentUserVariables* ad=readOne(*it, ierr2);
    if (ierr2==0) { 
      retvec.push_back(ad); icount++; 
      if (ad!=0) icount2++;
      //cout <<"read uvar " << icount2 << endl;
    }
    // for destructor
    if (ad!=0) theReadUserVariables.push_back(ad);

  }
  edm::LogInfo("Alignment") << "@SUB=AlignmentUserVariablesIO::read"
                            << "Read variables all,read,valid: " << alivec.size() <<","
                            << icount <<","<< icount2;
  return retvec;
}
