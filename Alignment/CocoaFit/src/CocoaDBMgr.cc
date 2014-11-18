#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "CondFormats/Alignment/interface/AlignTransform.h"
#include "CondFormats/Alignment/interface/AlignTransformErrorExtended.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h" 
#include "FWCore/Framework/interface/ESHandle.h"

#include "Alignment/CocoaFit/interface/CocoaDBMgr.h"
#include "CondFormats/OptAlignObjects/interface/OpticalAlignMeasurementInfo.h" 
#include "CondFormats/DataRecord/interface/OpticalAlignmentsRcd.h" 
#include "CondFormats/AlignmentRecord/interface/DTAlignmentRcd.h" 
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "Alignment/CocoaUtilities/interface/ALIUtils.h"
#include "Alignment/CocoaModel/interface/Model.h"
#include "Alignment/CocoaFit/interface/Fit.h"
#include "Alignment/CocoaModel/interface/Entry.h"
#include "Alignment/CocoaUtilities/interface/ALIFileOut.h"
#include "Alignment/CocoaModel/interface/CocoaDaqReaderRoot.h"
#include "Alignment/CocoaModel/interface/OpticalObject.h"
#include "Alignment/CocoaUtilities/interface/GlobalOptionMgr.h"

#include "CondFormats/OptAlignObjects/interface/OpticalAlignments.h"
#include "CondFormats/OptAlignObjects/interface/OpticalAlignInfo.h"
#include "CondFormats/OptAlignObjects/interface/OpticalAlignMeasurements.h"

#include "CondCore/CondDB/interface/Serialization.h"

CocoaDBMgr* CocoaDBMgr::instance = 0;

//----------------------------------------------------------------------
CocoaDBMgr* CocoaDBMgr::getInstance()
{
  if(!instance) {
    instance = new CocoaDBMgr;
  }
  return instance;
}

//----------------------------------------------------------------------
CocoaDBMgr::CocoaDBMgr()
{
}

//-----------------------------------------------------------------------
bool CocoaDBMgr::DumpCocoaResults()
{
  edm::Service<cond::service::PoolDBOutputService> myDbService;

  GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
  int nrcd;

  cond::Time_t appendTime = Fit::nEvent+1;
  if(gomgr->GlobalOptions()["writeDBOptAlign"] > 0 ) {

    //----- Build OpticalAlignments
    OpticalAlignments* optalign = BuildOpticalAlignments();
    
    //--- Dump OpticalAlignments  
    nrcd = optalign->opticalAlignments_.size();
    if( !myDbService.isAvailable() ){
      throw cms::Exception("CocoaDBMgr::DumpCocoaResults DB not available");
    }
    //    try{
    if ( myDbService->isNewTagRequest( "OpticalAlignmentsRcd" ) ) {
      std::cout << " new OA to DB "  << "begin " << myDbService->beginOfTime() << " current " << myDbService->currentTime() << " end " << myDbService->endOfTime() << std::endl;
      myDbService->createNewIOV<OpticalAlignments>(optalign,
						   myDbService->beginOfTime(),
						   myDbService->endOfTime(),
						   //						   myDbService->endOfTime(),
						   "OpticalAlignmentsRcd");
    } else {
      std::cout << " old OA to DB " << " current " << myDbService->currentTime() << " end " << myDbService->endOfTime() << std::endl;
      myDbService->appendSinceTime<OpticalAlignments>( optalign,
						       //		      myDbService->endOfTime(),
						       appendTime,
						       //						       myDbService->currentTime(),
						      "OpticalAlignmentsRcd");
    }
  
    
    /*    }catch(const cond::Exception& er) {
	  std::cout<<er.what()<<std::endl;
	  }catch(const std::exception& er){
	  std::cout<<"caught std::exception "<<er.what()<<std::endl;
	  }catch(...){
	  std::cout<<"Funny error"<<std::endl;
	  } */
    
    if(ALIUtils::debug >= 2) std::cout << "OpticalAlignmentsRcd WRITTEN TO DB : "<< nrcd << std::endl;
  }

  if( gomgr->GlobalOptions()["writeDBAlign"] > 0) {

    // Build DT alignments and errors
    std::pair< Alignments*,AlignmentErrorsExtended*> dtali = BuildAlignments(1);
    Alignments*      dt_Alignments = dtali.first;
    AlignmentErrorsExtended* dt_AlignmentErrors = dtali.second;
    
    // Dump DT alignments and errors
    nrcd = dt_Alignments->m_align.size();
    if ( myDbService->isNewTagRequest( "DTAlignmentRcd" ) ) {
      myDbService->createNewIOV<Alignments>(&(*dt_Alignments),
					    myDbService->beginOfTime(),
					    myDbService->endOfTime(),
					    "DTAlignmentRcd");
    } else {
      myDbService->appendSinceTime<Alignments>( &(*dt_Alignments),
						       appendTime,
					       //					       myDbService->currentTime(),
					       "DTAlignmentRcd");
    }
    if(ALIUtils::debug >= 2) std::cout << "DTAlignmentRcd WRITTEN TO DB : "<< nrcd << std::endl;
    
    nrcd = dt_AlignmentErrors->m_alignError.size();
    if ( myDbService->isNewTagRequest( "DTAlignmentErrorExtendedRcd" ) ) {
      myDbService->createNewIOV<AlignmentErrorsExtended>(&(*dt_AlignmentErrors),
						 myDbService->beginOfTime(),
						 myDbService->endOfTime(),
						 "DTAlignmentErrorExtendedRcd");
    } else {
      myDbService->appendSinceTime<AlignmentErrorsExtended>( &(*dt_AlignmentErrors),
						       appendTime,
						    "DTAlignmentErrorExtendedRcd");
    }
    if(ALIUtils::debug >= 2) std::cout << "DTAlignmentErrorExtendedRcd WRITTEN TO DB : "<< nrcd << std::endl;
    
    // Build CSC alignments and errors
    std::pair< Alignments*,AlignmentErrorsExtended*> cscali = BuildAlignments(0);
    Alignments*      csc_Alignments = cscali.first;
    AlignmentErrorsExtended* csc_AlignmentErrors = cscali.second;
    
    // Dump CSC alignments and errors
    nrcd = csc_Alignments->m_align.size();
    if ( myDbService->isNewTagRequest( "CSCAlignmentRcd" ) ) {
      myDbService->createNewIOV<Alignments>(&(*csc_Alignments),
					    myDbService->beginOfTime(),
					    myDbService->endOfTime(),
					    "CSCAlignmentRcd");
    } else {
      myDbService->appendSinceTime<Alignments>( &(*csc_Alignments),
						       appendTime,
					       "CSCAlignmentRcd");
    }
    if(ALIUtils::debug >= 2) std::cout << "CSCAlignmentRcd WRITTEN TO DB : "<< nrcd << std::endl;
    
    nrcd = csc_AlignmentErrors->m_alignError.size();
    if ( myDbService->isNewTagRequest( "CSCAlignmentErrorExtendedRcd" ) ) {
      myDbService->createNewIOV<AlignmentErrorsExtended>(&(*csc_AlignmentErrors),
						 myDbService->beginOfTime(),
						 myDbService->endOfTime(),
						 "CSCAlignmentErrorExtendedRcd");
    } else {
      myDbService->appendSinceTime<AlignmentErrorsExtended>( &(*csc_AlignmentErrors),
						       appendTime,
						    "CSCAlignmentErrorExtendedRcd");
    }
    if(ALIUtils::debug >= 2) std::cout << "CSCAlignmentErrorExtendedRcd WRITTEN TO DB : "<< nrcd << std::endl;
    
    //? gives unreadable error???  std::cout << "@@@@ OPTICALALIGNMENTS WRITTEN TO DB " << *optalign << std::endl;
    
    return TRUE;
  }

  return TRUE;
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
OpticalAlignInfo CocoaDBMgr::GetOptAlignInfoFromOptO( OpticalObject* opto )
{
  std::cout << " CocoaDBMgr::GetOptAlignInfoFromOptO " << opto->name() << std::endl;
  OpticalAlignInfo data;
  data.ID_=opto->getCmsswID();
  data.type_=opto->type();
  data.name_=opto->name();

  //----- Centre in local coordinates
  CLHEP::Hep3Vector centreLocal = opto->centreGlob() - opto->parent()->centreGlob();
  CLHEP::HepRotation parentRmGlobInv = inverseOf( opto->parent()->rmGlob() );
  centreLocal = parentRmGlobInv * centreLocal;

  const std::vector< Entry* > theCoordinateEntryVector = opto->CoordinateEntryList();
  std::cout << " CocoaDBMgr::GetOptAlignInfoFromOptO starting coord " <<std::endl;

  data.x_.value_= centreLocal.x() / 100.; // in cm
  std::cout << " matrix " << Fit::GetAtWAMatrix() << std::endl;
  std::cout << " matrix " << Fit::GetAtWAMatrix()->Mat() << " " << theCoordinateEntryVector[0]->fitPos() << std::endl;
  data.x_.error_= GetEntryError( theCoordinateEntryVector[0] ) / 100.; // in cm

  data.y_.value_= centreLocal.y() / 100.; // in cm
  std::cout << " matrix " << Fit::GetAtWAMatrix()->Mat()  << " " << theCoordinateEntryVector[1]->fitPos() << std::endl;
  data.y_.error_= GetEntryError( theCoordinateEntryVector[1] ) / 100.; // in cm

  data.z_.value_= centreLocal.z() / 100.; // in cm
  std::cout << " matrix " << Fit::GetAtWAMatrix()->Mat()  << " " << theCoordinateEntryVector[2]->fitPos() << std::endl;
  data.z_.error_= GetEntryError( theCoordinateEntryVector[2] ) / 100.; // in cm

  //----- angles in local coordinates
  std::vector<double> anglocal = opto->getLocalRotationAngles( theCoordinateEntryVector );

  data.angx_.value_= anglocal[0] *180./M_PI; // in deg
  std::cout << " matrix " << Fit::GetAtWAMatrix()->Mat() << theCoordinateEntryVector[3]->fitPos() << std::endl;
  data.angx_.error_= GetEntryError( theCoordinateEntryVector[3] ) * 180./M_PI; // in deg;

  data.angy_.value_= anglocal[1] * 180./M_PI; // in deg
  std::cout << " matrix " << Fit::GetAtWAMatrix()->Mat() << theCoordinateEntryVector[4]->fitPos() << std::endl;
  data.angy_.error_= GetEntryError( theCoordinateEntryVector[4] ) * 180./M_PI; // in deg;;

  data.angz_.value_= anglocal[2] *180./M_PI; // in deg
  std::cout << " matrix " << Fit::GetAtWAMatrix()->Mat() << theCoordinateEntryVector[5]->fitPos() << std::endl;
  data.angz_.error_= GetEntryError( theCoordinateEntryVector[5] ) * 180./M_PI; // in deg;

  
  const std::vector< Entry* > theExtraEntryVector = opto->ExtraEntryList();  std::cout << " CocoaDBMgr::GetOptAlignInfoFromOptO starting entry " << std::endl;

  std::vector< Entry* >::const_iterator ite;
  for( ite = theExtraEntryVector.begin(); ite != theExtraEntryVector.end(); ite++ ) {
    OpticalAlignParam extraEntry; 
    extraEntry.name_ = (*ite)->name();
    extraEntry.dim_type_ = (*ite)->type();
    extraEntry.value_ = (*ite)->value();
    extraEntry.error_ = (*ite)->sigma();
    extraEntry.quality_ = (*ite)->quality();
    data.extraEntries_.push_back( extraEntry );
  std::cout << " CocoaDBMgr::GetOptAlignInfoFromOptO done extra entry " << extraEntry.name_ << std::endl;

  }

  return data;
}


//-----------------------------------------------------------------------
double CocoaDBMgr::GetEntryError( const Entry* entry )
{
  if( entry->quality() > 0 ) {
    return sqrt(Fit::GetAtWAMatrix()->Mat()->me[entry->fitPos()][entry->fitPos()]);
  } else { //entry not fitted, return original error
    return entry->sigma();
  }
}


//-----------------------------------------------------------------------
double CocoaDBMgr::GetEntryError( const Entry* entry1, const Entry* entry2 )
{
  if( entry1 == entry2 ) return GetEntryError( entry1 );

  if( entry1->quality() > 0 && entry2->quality() > 0 ) {
    return sqrt(Fit::GetAtWAMatrix()->Mat()->me[entry1->fitPos()][entry2->fitPos()]);
  } else { //entries not fitted, correlation is 0
    return 0.;
  }
}


//-----------------------------------------------------------------------
OpticalAlignments* CocoaDBMgr::BuildOpticalAlignments()
{
  OpticalAlignments* optalign= new OpticalAlignments;

  static std::vector< OpticalObject* > optolist = Model::OptOList();
  static std::vector< OpticalObject* >::const_iterator ite;
  for(ite = optolist.begin(); ite != optolist.end(); ite++ ){
    if( (*ite)->type() == "system" ) continue;    
    OpticalAlignInfo data = GetOptAlignInfoFromOptO( *ite );
    optalign->opticalAlignments_.push_back(data);
    if(ALIUtils::debug >= 5) {
      std::cout << "@@@@ OPTALIGNINFO TO BE WRITTEN TO DB " 
		<< data 
		<< std::endl;  
    }
  }
  return optalign;
}


//-----------------------------------------------------------------------
std::pair< Alignments*,AlignmentErrorsExtended*> CocoaDBMgr::BuildAlignments(bool bDT)
{
  Alignments*      alignments = new Alignments;
  AlignmentErrorsExtended* alignmentErrors = new AlignmentErrorsExtended;

  //read 
  static std::vector< OpticalObject* > optolist = Model::OptOList();
  static std::vector< OpticalObject* >::const_iterator ite;
  for(ite = optolist.begin(); ite != optolist.end(); ite++ ){
    if( (*ite)->type() == "system" ) continue; 
      std::cout << "CocoaDBMgr::BuildAlignments getCmsswID " << (*ite) << std::endl;
      std::cout << "CocoaDBMgr::BuildAlignments getCmsswID " << (*ite)->getCmsswID()  << std::endl;
    //check CMSSW ID
    if( (*ite)->getCmsswID() > 0 ) { //put the numbers of DT or CSC objects 
      std::cout << " cal fill alignments " << std::endl;
      alignments->m_align.push_back( *(GetAlignInfoFromOptO( *ite )));
      std::cout << " fill alignments " << std::endl;
      //      AlignTransformErrorExtended* err = 
      //GetAlignInfoErrorFromOptO( *ite );
      alignmentErrors->m_alignError.push_back(*(GetAlignInfoErrorFromOptO( *ite )));
      std::cout << "CocoaDBMgr::BuildAlignments add alignmentError " <<  alignmentErrors->m_alignError.size() << std::endl;
    }
  }

  if(ALIUtils::debug >= 4) std::cout << "CocoaDBMgr::BuildAlignments end with n alignment " << alignments->m_align.size() << " n alignmentError " << alignmentErrors->m_alignError.size() << std::endl;
  return std::pair< Alignments*,AlignmentErrorsExtended*>(alignments,alignmentErrors);
}

  
//-----------------------------------------------------------------------
AlignTransform* CocoaDBMgr::GetAlignInfoFromOptO( OpticalObject* opto )
{
  if(ALIUtils::debug >= 3) std::cout << "@@@ CocoaDBMgr::GetAlignInfoFromOptO " << opto->name() << std::endl;

  AlignTransform::Translation trans = opto->centreGlob();
  AlignTransform::Rotation rot = opto->rmGlob();
  align::ID cmsswID = opto->getCmsswID();

  std::cout << "@@@ CocoaDBMgr::GetAlignInfoFromOptO buildalign" << opto->name() << std::endl;
  AlignTransform* align = new AlignTransform( trans, rot, cmsswID );
  
  std::cout << "@@@ CocoaDBMgr::GetAlignInfoFromOptO alig built " << opto->name() << std::endl;

  return align;
  //  return dd;
}

//-----------------------------------------------------------------------
AlignTransformErrorExtended* CocoaDBMgr::GetAlignInfoErrorFromOptO( OpticalObject* opto )
{
  if(ALIUtils::debug >= 3) std::cout << "@@@ CocoaDBMgr::GetAlignInfoErrorFromOptO " << opto->name() << std::endl;

  align::ID cmsswID = opto->getCmsswID();

 GlobalError gerr(1.,
		  0.,
		  1.,
		  0.,
		  0.,
		  1.);
 //double(dx*dx),  0., double(dy*dy),     0., 0., double(dz*dz) ) ;
  CLHEP::HepSymMatrix errms = asHepMatrix(gerr.matrix());
  AlignTransformErrorExtended* alignError = new AlignTransformErrorExtended( errms, cmsswID );
  return alignError;

  CLHEP::HepMatrix errm(3,3);
  const std::vector< Entry* > theCoordinateEntryVector = opto->CoordinateEntryList();
std::cout << "@@@ CocoaDBMgr::GetAlignInfoFromOptOfill errm " << opto->name() << std::endl;
  errm(0,0) = GetEntryError( theCoordinateEntryVector[0] ) / 100.; // in cm
  errm(1,1) = GetEntryError( theCoordinateEntryVector[1] ) / 100.; // in cm
  errm(2,2) = GetEntryError( theCoordinateEntryVector[2] ) / 100.; // in cm
  errm(0,1) = GetEntryError( theCoordinateEntryVector[0], theCoordinateEntryVector[1] ) / 100.; // in cm
  errm(0,2) = GetEntryError( theCoordinateEntryVector[0], theCoordinateEntryVector[2] ) / 100.; // in cm
  errm(1,2) = GetEntryError( theCoordinateEntryVector[1], theCoordinateEntryVector[2] ) / 100.; // in cm
  //   errm(1,0) = errm(0,1);
  // errm(2,0) = errm(0,2);
  // errm(2,1) = errm(1,2);

std::cout << "@@@ CocoaDBMgr::GetAlignInfoFromOptO errm filled" << opto->name() << std::endl;
//  CLHEP::HepSymMatrix errms(3);
//  errms.assign(errm); 
  
std::cout << "@@@ CocoaDBMgr::GetAlignInfoFromOptO errms filled " << opto->name() << std::endl;
//  AlignTransformErrorExtended* alignError = new AlignTransformErrorExtended( errms, cmsswID );
//  AlignTransformErrorExtended* alignError = 0;
  
  std::cout << alignError << "@@@ CocoaDBMgr::GetAlignInfoFromOptO error built " << opto->name() << std::endl;
  //t  return alignError;
  return (AlignTransformErrorExtended*)(0);
}


