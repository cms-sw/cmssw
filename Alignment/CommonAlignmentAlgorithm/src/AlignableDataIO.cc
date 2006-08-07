#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/CommonAlignmentParametrization/interface/AlignmentTransformations.h"
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
  
  if ( ali->alignmentParameters()->isValid() || !(validCheck) ) 
	{
	  // position in global frame
	  GlobalPoint pos = ali->surface().position();
	  // global rotation
	  Surface::RotationType rot = ali->surface().rotation();
	  // write
	  TrackerAlignableId converter;
	  return writeAbsRaw( 
						 AlignableAbsData( pos,rot,
										   converter.alignableId(ali),
										   converter.alignableTypeId(ali) )
						 );
	}
  else 
	return 1;
}


// ----------------------------------------------------------------------------
int AlignableDataIO::writeRelPos(Alignable* ali, bool validCheck)
{
  if ( ali->alignmentParameters()->isValid() || !(validCheck) ) 
	{
	  // rel. shift in global frame
	  GlobalVector pos = ali->displacement();
	  // rel. rotation in global frame
	  Surface::RotationType rot = ali->rotation();
	  // write
	  TrackerAlignableId converter;
	  return writeRelRaw(AlignableRelData(pos,rot,converter.alignableId(ali),
										  converter.alignableTypeId(ali)));
	}
  else 
	return 1;
}


// ----------------------------------------------------------------------------
int AlignableDataIO::writeOrgPos(Alignable* ali, bool validCheck)
{
  if ( ali->alignmentParameters()->isValid() || !(validCheck) ) 
	{
	  // misalignment shift/rotation
	  GlobalVector ashift = ali->displacement();
	  Surface::RotationType arot = ali->rotation();
	  // global position/rotation
	  GlobalPoint cpos = ali->surface().position();
	  Surface::RotationType crot = ali->surface().rotation();
	  // orig position
	  GlobalPoint pos(cpos.x()-ashift.x(),
					  cpos.y()-ashift.y(),
					  cpos.z()-ashift.z());
	  // orig rotation
	  int ierr;
	  AlignmentTransformations alignTransform;
	  Surface::RotationType rot 
		= crot*alignTransform.rotationType( alignTransform.algebraicMatrix(arot).inverse(ierr) );
	  // write
	  TrackerAlignableId converter;
	  return writeAbsRaw(AlignableAbsData(pos,rot,converter.alignableId(ali),
										  converter.alignableTypeId(ali)));
  }
  else 
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
  edm::LogInfo("WriteAbsPos") << "all,written: " << alivec.size() <<","<< icount;

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
  
  edm::LogInfo("ReadAbsPos") << "all,written: " << alivec.size() <<"," << retvec.size();

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
  
  edm::LogInfo("WriteOrgPos") << "all,written: " << alivec.size() <<"," << icount;
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

  edm::LogInfo("ReadOrgPos") << "all,read: " << alivec.size() <<", "<< retvec.size();

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
  edm::LogInfo("WriteRelPos") << "all,written: " << alivec.size() <<", "<< icount;
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
  edm::LogInfo("ReadRelPos") << "all,read: " << alivec.size() <<", "<< retvec.size();

  return retvec;

}

