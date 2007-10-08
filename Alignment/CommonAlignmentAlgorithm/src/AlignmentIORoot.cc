#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParametersIORoot.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentCorrelationsIORoot.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignableDataIORoot.h"

// this class's header
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentIORoot.h"

// ----------------------------------------------------------------------------
// write alignment parameters 

void 
AlignmentIORoot::writeAlignmentParameters(const align::Alignables& alivec, 
                                          const char* filename, int iter, 
                                          bool validCheck, int& ierr)
{
  AlignmentParametersIORoot theIO;
  ierr=0;
  int iret;
  iret = theIO.open(filename,iter,true);
  if (iret!=0) { ierr=-1; return;}
  iret = theIO.write(alivec,validCheck);
  if (iret!=0) { ierr=-2; return;}
  iret = theIO.close();
  if (iret!=0) { ierr=-3; return;}
}

// ----------------------------------------------------------------------------
// read alignment parameters 
align::Parameters 
AlignmentIORoot::readAlignmentParameters(const align::Alignables& alivec, 
                                         const char* filename, int iter, int& ierr)
{
  align::Parameters result;

  AlignmentParametersIORoot theIO;
  ierr=0;
  int iret;
  iret = theIO.open(filename,iter,false);
  if (iret!=0) { ierr=-1; return result;}
  result = theIO.read(alivec,iret);
  if (iret!=0) { ierr=-2; return result;}
  iret = theIO.close();
  if (iret!=0) { ierr=-3; return result;}

  return result;
}

// ----------------------------------------------------------------------------
// write alignment parameters 
void
AlignmentIORoot::writeOrigRigidBodyAlignmentParameters
(const align::Alignables& alivec, const char* filename, int iter, bool validCheck, int& ierr)
{
  AlignmentParametersIORoot theIO;
  ierr = 0;
  int iret = theIO.open(filename, iter, true);
  if (iret != 0) { ierr = -1; return;}
  iret = theIO.writeOrigRigidBody(alivec, validCheck);
  if (iret != 0) { ierr = -2; return;}
  iret = theIO.close();
  if (iret != 0) { ierr = -3; return;}
}


// ----------------------------------------------------------------------------
// write correlations

void 
AlignmentIORoot::writeCorrelations (const align::Correlations& cormap, 
                                    const char* filename, int iter, bool validCheck, int& ierr)
{
  AlignmentCorrelationsIORoot theIO;
  ierr=0;
  int iret;
  iret = theIO.open(filename,iter,true);
  if (iret!=0) { ierr=-1; return;}
  iret = theIO.write(cormap,validCheck);
  if (iret!=0) { ierr=-2; return;}
  iret = theIO.close();
  if (iret!=0) { ierr=-3; return;}
}

// ----------------------------------------------------------------------------
// read correlations

align::Correlations 
AlignmentIORoot::readCorrelations (const align::Alignables& alivec, const char* filename, 
                                   int iter, int& ierr)
{   
  align::Correlations result;

  AlignmentCorrelationsIORoot theIO;
  ierr=0;
  int iret;
  iret = theIO.open(filename,iter,false);
  if (iret!=0) { ierr=-1; return result;}
  result = theIO.read(alivec,iret);
  if (iret!=0) { ierr=-2; return result;}
  iret = theIO.close();
  if (iret!=0) { ierr=-3; return result;}

  return result;
}


// ----------------------------------------------------------------------------
// write absolute position of alignable

void AlignmentIORoot::writeAlignableAbsolutePositions ( const align::Alignables& alivec, 
                                                        const char* filename, int iter, 
                                                        bool validCheck, int& ierr)
{
  AlignableDataIORoot theIO(AlignableDataIORoot::Abs);
  ierr=0;
  int iret;
  iret = theIO.open(filename,iter,true);
  if (iret!=0) { ierr=-1; return;}
  iret = theIO.writeAbsPos(alivec,validCheck);
  if (iret!=0) { ierr=-2; return;}
  iret = theIO.close();
  if (iret!=0) { ierr=-3; return;}
}

// ----------------------------------------------------------------------------
// read absolute position of alignable

AlignablePositions 
AlignmentIORoot::readAlignableAbsolutePositions (const align::Alignables& alivec, 
                                                 const char* filename, int iter, int& ierr)
{
  AlignablePositions result;

  AlignableDataIORoot theIO(AlignableDataIORoot::Abs);
  ierr=0;
  int iret;
  iret = theIO.open(filename,iter,false);
  if (iret!=0) { ierr=-1; return result;}
  result = theIO.readAbsPos(alivec,iret);
  if (iret!=0) { ierr=-2; return result;}
  iret = theIO.close();
  if (iret!=0) { ierr=-3; return result;}

  return result;
}

// ----------------------------------------------------------------------------
// write original position of alignable

void AlignmentIORoot::writeAlignableOriginalPositions ( const align::Alignables& alivec, 
                                                        const char* filename, int iter, 
                                                        bool validCheck, int& ierr)
{
  AlignableDataIORoot theIO(AlignableDataIORoot::Org);
  ierr=0;
  int iret;
  iret = theIO.open(filename,iter,true);
  if (iret!=0) { ierr=-1; return;}
  iret = theIO.writeOrgPos(alivec,validCheck);
  if (iret!=0) { ierr=-2; return;}
  iret = theIO.close();
  if (iret!=0) { ierr=-3; return;}
}

// ----------------------------------------------------------------------------
// read original position of alignable

AlignablePositions 
AlignmentIORoot::readAlignableOriginalPositions (const align::Alignables& alivec, 
                                                 const char* filename, int iter, int& ierr)
{
  AlignablePositions result;

  AlignableDataIORoot theIO(AlignableDataIORoot::Org);
  ierr=0;
  int iret;
  iret = theIO.open(filename,iter,false);
  if (iret!=0) { ierr=-1; return result;}
  result = theIO.readOrgPos(alivec,iret);
  if (iret!=0) { ierr=-2; return result;}
  iret = theIO.close();
  if (iret!=0) { ierr=-3; return result;}

  return result;
}

// ----------------------------------------------------------------------------
// write relative position of alignable

void AlignmentIORoot::writeAlignableRelativePositions( const align::Alignables& alivec,
                                                       const char* filename,
                                                       int iter, bool validCheck, int& ierr)
{
  AlignableDataIORoot theIO(AlignableDataIORoot::Rel);
  ierr=0;
  int iret;
  iret = theIO.open(filename,iter,true);
  if (iret!=0) { ierr=-1; return;}
  iret = theIO.writeRelPos(alivec,validCheck);
  if (iret!=0) { ierr=-2; return;}
  iret = theIO.close();
  if (iret!=0) { ierr=-3; return;}
}

// ----------------------------------------------------------------------------
// read relative position of alignable

AlignableShifts 
AlignmentIORoot::readAlignableRelativePositions (const align::Alignables& alivec, 
                                                 const char* filename, int iter, int& ierr)
{
  AlignableShifts result;

  AlignableDataIORoot theIO(AlignableDataIORoot::Rel);
  ierr=0;
  int iret;
  iret = theIO.open(filename,iter,false);
  if (iret!=0) { ierr=-1; return result;}
  result = theIO.readRelPos(alivec,iret);
  if (iret!=0) { ierr=-2; return result;}
  iret = theIO.close();
  if (iret!=0) { ierr=-3; return result;}

  return result;
}






