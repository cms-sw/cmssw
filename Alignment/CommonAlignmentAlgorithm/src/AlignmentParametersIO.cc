#include "FWCore/MessageLogger/interface/MessageLogger.h"

// this class's header
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParametersIO.h"

//-----------------------------------------------------------------------------
// write many parameters
int 
AlignmentParametersIO::write(const std::vector<Alignable*>& alivec, 
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
  edm::LogInfo("AlignmentParametersIO::write") << "Write parameters all,written: " 
											   << alivec.size() <<","<< icount;
  return 0;

}

//-----------------------------------------------------------------------------
// read many parameters

std::vector<AlignmentParameters*> 
AlignmentParametersIO::read(const std::vector<Alignable*>& alivec, int& ierr) 
{
  std::vector<AlignmentParameters*> retvec;
  int ierr2;
  int icount=0;
  for(std::vector<Alignable*>::const_iterator it=alivec.begin();
    it!=alivec.end(); it++) {
    AlignmentParameters* ad=readOne(*it, ierr2);
    if (ad!=0 && ierr2==0) { retvec.push_back(ad); icount++; }
  }
  edm::LogInfo("AlignmentParametersIO::write") << "Read parameters all,read: " 
											   << alivec.size() <<","<< icount;
  return retvec;
}
