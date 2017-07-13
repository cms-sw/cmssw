#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CommonTopologies/interface/SurfaceDeformation.h"

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
      // if a unit: store surface deformation (little kind of hack)...
      std::vector<double> pars;
      if (ali->alignableObjectId() == align::AlignableDetUnit) { // only detunits have them
        std::vector<std::pair<int,SurfaceDeformation*> > result;
        if (1 == ali->surfaceDeformationIdPairs(result)) { // might not have any...
          pars = result[0].second->parameters();
        }
      }

      // write
      return writeAbsRaw( 
			 AlignableAbsData( pos,rot,
					   ali->id(),
					   ali->alignableObjectId(),
                                           pars)
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
      const align::GlobalVector& pos = ali->displacement();
      // rel. rotation in global frame
      align::RotationType rot = ali->rotation();
      // FIXME: should add something to store changes of surface deformations...
      std::vector<double> pars;
      // write
      return writeRelRaw(AlignableRelData(pos,rot,ali->id(),
					  ali->alignableObjectId(), pars));
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
      // FIXME: should add something to store changes of surface deformations...
      std::vector<double> pars;
      // write
      return writeAbsRaw(AlignableAbsData(pos,rot,ali->id(),
					  ali->alignableObjectId(), pars));
    }

  return 1;
}


// ----------------------------------------------------------------------------
int AlignableDataIO::writeAbsPos(const align::Alignables& alivec, 
				 bool validCheck)
{

  int icount=0;
  for( align::Alignables::const_iterator it=alivec.begin();
       it!=alivec.end(); ++it ) 
    {
      int iret = writeAbsPos(*it,validCheck);
      if (iret==0) icount++;
    }
  LogDebug("WriteAbsPos") << "all,written: " << alivec.size() <<","<< icount;

  return 0;

}


// ----------------------------------------------------------------------------
AlignablePositions 
AlignableDataIO::readAbsPos(const align::Alignables& alivec, int& ierr) 
{
 
  AlignablePositions retvec;
  int ierr2=0;
  ierr=0;
  for( align::Alignables::const_iterator it=alivec.begin();
       it!=alivec.end(); ++it ) 
    {
      AlignableAbsData ad=readAbsPos(*it, ierr2);
      if (ierr2==0) retvec.push_back(ad);
    }
  
  LogDebug("ReadAbsPos") << "all,written: " << alivec.size() <<"," << retvec.size();

  return retvec;

}


// ----------------------------------------------------------------------------
int AlignableDataIO::writeOrgPos( const align::Alignables& alivec, 
				  bool validCheck )
{

  int icount=0;
  for( align::Alignables::const_iterator it=alivec.begin();
       it!=alivec.end(); ++it ) 
    {
      int iret=writeOrgPos(*it,validCheck);
      if (iret==0) icount++;
    }
  
  LogDebug("WriteOrgPos") << "all,written: " << alivec.size() <<"," << icount;
  return 0;

}


// ----------------------------------------------------------------------------
AlignablePositions 
AlignableDataIO::readOrgPos(const align::Alignables& alivec, int& ierr) 
{

  AlignablePositions retvec;
  int ierr2=0;
  ierr=0;
  for( align::Alignables::const_iterator it=alivec.begin();
       it!=alivec.end(); ++it ) 
    {
      AlignableAbsData ad=readOrgPos(*it, ierr2);
      if (ierr2==0) retvec.push_back(ad);
    }

  LogDebug("ReadOrgPos") << "all,read: " << alivec.size() <<", "<< retvec.size();

  return retvec;

}


// ----------------------------------------------------------------------------
int AlignableDataIO::writeRelPos(const align::Alignables& alivec, 
				 bool validCheck )
{

  int icount=0;
  for( align::Alignables::const_iterator it=alivec.begin();
       it!=alivec.end(); ++it ) {
    int iret=writeRelPos(*it,validCheck);
    if (iret==0) icount++;
  }
  LogDebug("WriteRelPos") << "all,written: " << alivec.size() <<", "<< icount;
  return 0;

}


// ----------------------------------------------------------------------------
AlignableShifts 
AlignableDataIO::readRelPos(const align::Alignables& alivec, int& ierr) 
{

  AlignableShifts retvec;
  int ierr2=0;
  ierr=0;
  for( align::Alignables::const_iterator it=alivec.begin();
       it!=alivec.end(); ++it ) 
    {
      AlignableRelData ad=readRelPos(*it, ierr2);
      if (ierr2==0) retvec.push_back(ad);
    }
  LogDebug("ReadRelPos") << "all,read: " << alivec.size() <<", "<< retvec.size();

  return retvec;

}
