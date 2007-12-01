#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "Alignment/TrackerAlignment/interface/TrackerAlignableId.h"

// this class's header
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignableDataIO.h"

// ----------------------------------------------------------------------------
AlignableAbsData AlignableDataIO::readAbsPos(Alignable* ali, int& ierr)
{ 
  return readAbsRaw(ali,ierr);
}


// ----------------------------------------------------------------------------
AlignableAbsData AlignableDataIO::readOrgPos(Alignable* ali, int& ierr)
{ 
  return readAbsRaw(ali,ierr);
}


// ----------------------------------------------------------------------------
AlignableRelData AlignableDataIO::readRelPos(Alignable* ali, int& ierr)
{ 
  return readRelRaw(ali,ierr);
}


// ----------------------------------------------------------------------------
int AlignableDataIO::writeAbsPos(Alignable* ali, bool validCheck)
{
  
  if ( !(validCheck) || ali->alignmentParameters()->isValid() ) 
    {
      // position in global frame
      align::PositionType pos = ali->surface().position();
      // global rotation
      align::RotationType rot = ali->surface().rotation();
      // write
      TrackerAlignableId converter;
      return writeAbsRaw( 
			 AlignableAbsData( pos,rot,
					   converter.alignableId(ali),
					   converter.alignableTypeId(ali) )
			 );
    }

  return 1;
}


// ----------------------------------------------------------------------------
int AlignableDataIO::writeRelPos(Alignable* ali, bool validCheck)
{
  if ( !(validCheck) || ali->alignmentParameters()->isValid() ) 
    {
      // rel. shift in global frame
      align::GlobalVector pos = ali->displacement();
      // rel. rotation in global frame
      align::RotationType rot = ali->rotation();
      // write
      TrackerAlignableId converter;
      return writeRelRaw(AlignableRelData(pos,rot,converter.alignableId(ali),
					  converter.alignableTypeId(ali)));
    }

  return 1;
}


// ----------------------------------------------------------------------------
int AlignableDataIO::writeOrgPos(Alignable* ali, bool validCheck)
{
  if ( !(validCheck) || ali->alignmentParameters()->isValid() ) 
    {
      // orig position
      align::PositionType pos = ali->globalPosition() - ali->displacement();
      // orig rotation
      align::RotationType rot = ali->globalRotation() * ali->rotation().transposed();
      // write
      TrackerAlignableId converter;
      return writeAbsRaw(AlignableAbsData(pos,rot,converter.alignableId(ali),
					  converter.alignableTypeId(ali)));
    }

  return 1;
}


// ----------------------------------------------------------------------------
int AlignableDataIO::writeAbsPos(const std::vector<Alignable*>& alivec, 
				 bool validCheck)
{

  int icount=0;
  for( std::vector<Alignable*>::const_iterator it=alivec.begin();
       it!=alivec.end(); it++ ) 
    {
      int iret = writeAbsPos(*it,validCheck);
      if (iret==0) icount++;
    }
  LogDebug("WriteAbsPos") << "all,written: " << alivec.size() <<","<< icount;

  return 0;

}


// ----------------------------------------------------------------------------
AlignablePositions 
AlignableDataIO::readAbsPos(const std::vector<Alignable*>& alivec, int& ierr) 
{
 
  AlignablePositions retvec;
  int ierr2=0;
  ierr=0;
  for( std::vector<Alignable*>::const_iterator it=alivec.begin();
       it!=alivec.end(); it++ ) 
    {
      AlignableAbsData ad=readAbsPos(*it, ierr2);
      if (ierr2==0) retvec.push_back(ad);
    }
  
  LogDebug("ReadAbsPos") << "all,written: " << alivec.size() <<"," << retvec.size();

  return retvec;

}


// ----------------------------------------------------------------------------
int AlignableDataIO::writeOrgPos( const std::vector<Alignable*>& alivec, 
				  bool validCheck )
{

  int icount=0;
  for( std::vector<Alignable*>::const_iterator it=alivec.begin();
       it!=alivec.end(); it++ ) 
    {
      int iret=writeOrgPos(*it,validCheck);
      if (iret==0) icount++;
    }
  
  LogDebug("WriteOrgPos") << "all,written: " << alivec.size() <<"," << icount;
  return 0;

}


// ----------------------------------------------------------------------------
AlignablePositions 
AlignableDataIO::readOrgPos(const std::vector<Alignable*>& alivec, int& ierr) 
{

  AlignablePositions retvec;
  int ierr2=0;
  ierr=0;
  for( std::vector<Alignable*>::const_iterator it=alivec.begin();
       it!=alivec.end(); it++ ) 
    {
      AlignableAbsData ad=readOrgPos(*it, ierr2);
      if (ierr2==0) retvec.push_back(ad);
    }

  LogDebug("ReadOrgPos") << "all,read: " << alivec.size() <<", "<< retvec.size();

  return retvec;

}


// ----------------------------------------------------------------------------
int AlignableDataIO::writeRelPos(const std::vector<Alignable*>& alivec, 
				 bool validCheck )
{

  int icount=0;
  for( std::vector<Alignable*>::const_iterator it=alivec.begin();
       it!=alivec.end(); it++ ) {
    int iret=writeRelPos(*it,validCheck);
    if (iret==0) icount++;
  }
  LogDebug("WriteRelPos") << "all,written: " << alivec.size() <<", "<< icount;
  return 0;

}


// ----------------------------------------------------------------------------
AlignableShifts 
AlignableDataIO::readRelPos(const std::vector<Alignable*>& alivec, int& ierr) 
{

  AlignableShifts retvec;
  int ierr2=0;
  ierr=0;
  for( std::vector<Alignable*>::const_iterator it=alivec.begin();
       it!=alivec.end(); it++ ) 
    {
      AlignableRelData ad=readRelPos(*it, ierr2);
      if (ierr2==0) retvec.push_back(ad);
    }
  LogDebug("ReadRelPos") << "all,read: " << alivec.size() <<", "<< retvec.size();

  return retvec;

}
