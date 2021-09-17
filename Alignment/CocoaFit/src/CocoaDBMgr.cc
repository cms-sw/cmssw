#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "CondFormats/Alignment/interface/AlignTransform.h"
#include "CondFormats/Alignment/interface/AlignTransformErrorExtended.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include <DD4hep/DD4hepUnits.h>
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "Alignment/CocoaFit/interface/CocoaDBMgr.h"
#include "CondFormats/OptAlignObjects/interface/OpticalAlignMeasurementInfo.h"
#include "CondFormats/DataRecord/interface/OpticalAlignmentsRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentRcd.h"
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

CocoaDBMgr* CocoaDBMgr::instance = nullptr;

//----------------------------------------------------------------------
CocoaDBMgr* CocoaDBMgr::getInstance() {
  if (!instance) {
    instance = new CocoaDBMgr;
  }
  return instance;
}

//----------------------------------------------------------------------
CocoaDBMgr::CocoaDBMgr() {}

//-----------------------------------------------------------------------
bool CocoaDBMgr::DumpCocoaResults() {
  edm::Service<cond::service::PoolDBOutputService> myDbService;

  GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
  int nrcd;

  cond::Time_t appendTime = Fit::nEvent + 1;
  if (gomgr->GlobalOptions()["writeDBOptAlign"] > 0) {
    //----- Build OpticalAlignments
    OpticalAlignments* optalign = BuildOpticalAlignments();

    //--- Dump OpticalAlignments
    nrcd = optalign->opticalAlignments_.size();
    if (!myDbService.isAvailable()) {
      throw cms::Exception("CocoaDBMgr::DumpCocoaResults DB not available");
    }
    //    try{
    if (myDbService->isNewTagRequest("OpticalAlignmentsRcd")) {
      std::cout << " new OA to DB "
                << "begin " << myDbService->beginOfTime() << " current " << myDbService->currentTime() << " end "
                << myDbService->endOfTime() << std::endl;
      myDbService->createNewIOV<OpticalAlignments>(
          optalign,
          myDbService->beginOfTime(),
          myDbService->endOfTime(),
          //						   myDbService->endOfTime(),
          "OpticalAlignmentsRcd");
    } else {
      std::cout << " old OA to DB "
                << " current " << myDbService->currentTime() << " end " << myDbService->endOfTime() << std::endl;
      myDbService->appendSinceTime<OpticalAlignments>(
          optalign,
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

    if (ALIUtils::debug >= 2)
      std::cout << "OpticalAlignmentsRcd WRITTEN TO DB : " << nrcd << std::endl;
  }

  if (gomgr->GlobalOptions()["writeDBAlign"] > 0) {
    // Build DT alignments and errors
    std::pair<Alignments*, AlignmentErrorsExtended*> dtali = BuildAlignments(true);
    Alignments* dt_Alignments = dtali.first;
    AlignmentErrorsExtended* dt_AlignmentErrors = dtali.second;

    // Dump DT alignments and errors
    nrcd = dt_Alignments->m_align.size();
    if (myDbService->isNewTagRequest("DTAlignmentRcd")) {
      myDbService->createNewIOV<Alignments>(
          &(*dt_Alignments), myDbService->beginOfTime(), myDbService->endOfTime(), "DTAlignmentRcd");
    } else {
      myDbService->appendSinceTime<Alignments>(&(*dt_Alignments),
                                               appendTime,
                                               //					       myDbService->currentTime(),
                                               "DTAlignmentRcd");
    }
    if (ALIUtils::debug >= 2)
      std::cout << "DTAlignmentRcd WRITTEN TO DB : " << nrcd << std::endl;

    nrcd = dt_AlignmentErrors->m_alignError.size();
    if (myDbService->isNewTagRequest("DTAlignmentErrorExtendedRcd")) {
      myDbService->createNewIOV<AlignmentErrorsExtended>(
          &(*dt_AlignmentErrors), myDbService->beginOfTime(), myDbService->endOfTime(), "DTAlignmentErrorExtendedRcd");
    } else {
      myDbService->appendSinceTime<AlignmentErrorsExtended>(
          &(*dt_AlignmentErrors), appendTime, "DTAlignmentErrorExtendedRcd");
    }
    if (ALIUtils::debug >= 2)
      std::cout << "DTAlignmentErrorExtendedRcd WRITTEN TO DB : " << nrcd << std::endl;

    // Build CSC alignments and errors
    std::pair<Alignments*, AlignmentErrorsExtended*> cscali = BuildAlignments(false);
    Alignments* csc_Alignments = cscali.first;
    AlignmentErrorsExtended* csc_AlignmentErrors = cscali.second;

    // Dump CSC alignments and errors
    nrcd = csc_Alignments->m_align.size();
    if (myDbService->isNewTagRequest("CSCAlignmentRcd")) {
      myDbService->createNewIOV<Alignments>(
          &(*csc_Alignments), myDbService->beginOfTime(), myDbService->endOfTime(), "CSCAlignmentRcd");
    } else {
      myDbService->appendSinceTime<Alignments>(&(*csc_Alignments), appendTime, "CSCAlignmentRcd");
    }
    if (ALIUtils::debug >= 2)
      std::cout << "CSCAlignmentRcd WRITTEN TO DB : " << nrcd << std::endl;

    nrcd = csc_AlignmentErrors->m_alignError.size();
    if (myDbService->isNewTagRequest("CSCAlignmentErrorExtendedRcd")) {
      myDbService->createNewIOV<AlignmentErrorsExtended>(&(*csc_AlignmentErrors),
                                                         myDbService->beginOfTime(),
                                                         myDbService->endOfTime(),
                                                         "CSCAlignmentErrorExtendedRcd");
    } else {
      myDbService->appendSinceTime<AlignmentErrorsExtended>(
          &(*csc_AlignmentErrors), appendTime, "CSCAlignmentErrorExtendedRcd");
    }
    if (ALIUtils::debug >= 2)
      std::cout << "CSCAlignmentErrorExtendedRcd WRITTEN TO DB : " << nrcd << std::endl;

    //? gives unreadable error???  std::cout << "@@@@ OPTICALALIGNMENTS WRITTEN TO DB " << *optalign << std::endl;

    return TRUE;
  }

  return TRUE;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
OpticalAlignInfo CocoaDBMgr::GetOptAlignInfoFromOptO(OpticalObject* opto) {
  LogDebug("Alignment") << " CocoaDBMgr::GetOptAlignInfoFromOptO " << opto->name();
  OpticalAlignInfo data;
  data.ID_ = opto->getCmsswID();
  data.type_ = opto->type();
  data.name_ = opto->name();

  //----- Centre in local coordinates
  CLHEP::Hep3Vector centreLocal = opto->centreGlob() - opto->parent()->centreGlob();
  CLHEP::HepRotation parentRmGlobInv = inverseOf(opto->parent()->rmGlob());
  centreLocal = parentRmGlobInv * centreLocal;

  const std::vector<Entry*>& theCoordinateEntryVector = opto->CoordinateEntryList();
  LogDebug("Alignment") << " CocoaDBMgr::GetOptAlignInfoFromOptO starting coord ";
  if (theCoordinateEntryVector.size() == 6) {
    const Entry* const translationX = theCoordinateEntryVector.at(0);
    OpticalAlignParam translationXDataForDB;
    translationXDataForDB.name_ = translationX->name();
    translationXDataForDB.dim_type_ = translationX->type();
    translationXDataForDB.value_ = centreLocal.x() * dd4hep::m;              // m in COCOA, dd4hep unit in DB
    translationXDataForDB.error_ = GetEntryError(translationX) * dd4hep::m;  // m in COCOA, dd4hep unit in DB
    translationXDataForDB.quality_ = translationX->quality();
    data.x_ = translationXDataForDB;

    const Entry* const translationY = theCoordinateEntryVector.at(1);
    OpticalAlignParam translationYDataForDB;
    translationYDataForDB.name_ = translationY->name();
    translationYDataForDB.dim_type_ = translationY->type();
    translationYDataForDB.value_ = centreLocal.y() * dd4hep::m;              // m in COCOA, dd4hep unit in DB
    translationYDataForDB.error_ = GetEntryError(translationY) * dd4hep::m;  // m in COCOA, dd4hep unit in DB
    translationYDataForDB.quality_ = translationY->quality();
    data.y_ = translationYDataForDB;

    const Entry* const translationZ = theCoordinateEntryVector.at(2);
    OpticalAlignParam translationZDataForDB;
    translationZDataForDB.name_ = translationZ->name();
    translationZDataForDB.dim_type_ = translationZ->type();
    translationZDataForDB.value_ = centreLocal.z() * dd4hep::m;              // m in COCOA, dd4hep unit in DB
    translationZDataForDB.error_ = GetEntryError(translationZ) * dd4hep::m;  // m in COCOA, dd4hep unit in DB
    translationZDataForDB.quality_ = translationZ->quality();
    data.z_ = translationZDataForDB;

    //----- angles in local coordinates
    std::vector<double> anglocal = opto->getLocalRotationAngles(theCoordinateEntryVector);
    if (anglocal.size() == 3) {
      const Entry* const rotationX = theCoordinateEntryVector.at(3);
      OpticalAlignParam rotationXDataForDB;
      rotationXDataForDB.name_ = rotationX->name();
      rotationXDataForDB.dim_type_ = rotationX->type();
      rotationXDataForDB.value_ = anglocal.at(0);
      rotationXDataForDB.error_ = GetEntryError(rotationX);
      rotationXDataForDB.quality_ = rotationX->quality();
      data.angx_ = rotationXDataForDB;

      const Entry* const rotationY = theCoordinateEntryVector.at(4);
      OpticalAlignParam rotationYDataForDB;
      rotationYDataForDB.name_ = rotationY->name();
      rotationYDataForDB.dim_type_ = rotationY->type();
      rotationYDataForDB.value_ = anglocal.at(1);
      rotationYDataForDB.error_ = GetEntryError(rotationY);
      rotationYDataForDB.quality_ = rotationY->quality();
      data.angy_ = rotationYDataForDB;

      const Entry* const rotationZ = theCoordinateEntryVector.at(5);
      OpticalAlignParam rotationZDataForDB;
      rotationZDataForDB.name_ = rotationZ->name();
      rotationZDataForDB.dim_type_ = rotationZ->type();
      rotationZDataForDB.value_ = anglocal.at(2);
      rotationZDataForDB.error_ = GetEntryError(rotationZ);
      rotationZDataForDB.quality_ = rotationZ->quality();
      data.angz_ = rotationZDataForDB;
    }
  }

  std::cout << " CocoaDBMgr::GetOptAlignInfoFromOptO starting entry " << std::endl;
  for (const auto& myDBExtraEntry : opto->ExtraEntryList()) {
    OpticalAlignParam extraEntry;
    extraEntry.name_ = myDBExtraEntry->name();
    extraEntry.dim_type_ = myDBExtraEntry->type();
    extraEntry.value_ = myDBExtraEntry->value();
    extraEntry.error_ = myDBExtraEntry->sigma();
    if (extraEntry.dim_type_ == "centre" || extraEntry.dim_type_ == "length") {
      extraEntry.value_ *= dd4hep::m;  // m in COCOA, dd4hep unit in DB
      extraEntry.error_ *= dd4hep::m;  // m in COCOA, dd4hep unit in DB
    }
    extraEntry.quality_ = myDBExtraEntry->quality();
    data.extraEntries_.emplace_back(extraEntry);
    std::cout << " CocoaDBMgr::GetOptAlignInfoFromOptO done extra entry " << extraEntry.name_ << std::endl;
  }

  return data;
}

//-----------------------------------------------------------------------
double CocoaDBMgr::GetEntryError(const Entry* entry) {
  if (entry->quality() > 0) {
    return sqrt(Fit::GetAtWAMatrix()->Mat()->me[entry->fitPos()][entry->fitPos()]);
  } else {  //entry not fitted, return original error
    return entry->sigma();
  }
}

//-----------------------------------------------------------------------
double CocoaDBMgr::GetEntryError(const Entry* entry1, const Entry* entry2) {
  if (entry1 == entry2)
    return GetEntryError(entry1);

  if (entry1->quality() > 0 && entry2->quality() > 0) {
    return sqrt(Fit::GetAtWAMatrix()->Mat()->me[entry1->fitPos()][entry2->fitPos()]);
  } else {  //entries not fitted, correlation is 0
    return 0.;
  }
}

//-----------------------------------------------------------------------
OpticalAlignments* CocoaDBMgr::BuildOpticalAlignments() {
  OpticalAlignments* optalign = new OpticalAlignments;

  static std::vector<OpticalObject*> optolist = Model::OptOList();
  static std::vector<OpticalObject*>::const_iterator ite;
  for (ite = optolist.begin(); ite != optolist.end(); ++ite) {
    if ((*ite)->type() == "system")
      continue;
    OpticalAlignInfo data = GetOptAlignInfoFromOptO(*ite);
    optalign->opticalAlignments_.push_back(data);
    if (ALIUtils::debug >= 5) {
      std::cout << "@@@@ OPTALIGNINFO TO BE WRITTEN TO DB " << data << std::endl;
    }
  }
  return optalign;
}

//-----------------------------------------------------------------------
std::pair<Alignments*, AlignmentErrorsExtended*> CocoaDBMgr::BuildAlignments(bool bDT) {
  Alignments* alignments = new Alignments;
  AlignmentErrorsExtended* alignmentErrors = new AlignmentErrorsExtended;

  //read
  static std::vector<OpticalObject*> optolist = Model::OptOList();
  static std::vector<OpticalObject*>::const_iterator ite;
  for (ite = optolist.begin(); ite != optolist.end(); ++ite) {
    if ((*ite)->type() == "system")
      continue;
    std::cout << "CocoaDBMgr::BuildAlignments getCmsswID " << (*ite) << std::endl;
    std::cout << "CocoaDBMgr::BuildAlignments getCmsswID " << (*ite)->getCmsswID() << std::endl;
    //check CMSSW ID
    if ((*ite)->getCmsswID() > 0) {  //put the numbers of DT or CSC objects
      std::cout << " cal fill alignments " << std::endl;
      alignments->m_align.push_back(*(GetAlignInfoFromOptO(*ite)));
      std::cout << " fill alignments " << std::endl;
      //      AlignTransformErrorExtended* err =
      //GetAlignInfoErrorFromOptO( *ite );
      alignmentErrors->m_alignError.push_back(*(GetAlignInfoErrorFromOptO(*ite)));
      std::cout << "CocoaDBMgr::BuildAlignments add alignmentError " << alignmentErrors->m_alignError.size()
                << std::endl;
    }
  }

  if (ALIUtils::debug >= 4)
    std::cout << "CocoaDBMgr::BuildAlignments end with n alignment " << alignments->m_align.size()
              << " n alignmentError " << alignmentErrors->m_alignError.size() << std::endl;
  return std::pair<Alignments*, AlignmentErrorsExtended*>(alignments, alignmentErrors);
}

//-----------------------------------------------------------------------
AlignTransform* CocoaDBMgr::GetAlignInfoFromOptO(OpticalObject* opto) {
  if (ALIUtils::debug >= 3)
    std::cout << "@@@ CocoaDBMgr::GetAlignInfoFromOptO " << opto->name() << std::endl;

  const AlignTransform::Translation& trans = opto->centreGlob();
  const AlignTransform::Rotation& rot = opto->rmGlob();
  align::ID cmsswID = opto->getCmsswID();

  std::cout << "@@@ CocoaDBMgr::GetAlignInfoFromOptO buildalign" << opto->name() << std::endl;
  AlignTransform* align = new AlignTransform(trans, rot, cmsswID);

  std::cout << "@@@ CocoaDBMgr::GetAlignInfoFromOptO alig built " << opto->name() << std::endl;

  return align;
  //  return dd;
}

//-----------------------------------------------------------------------
AlignTransformErrorExtended* CocoaDBMgr::GetAlignInfoErrorFromOptO(OpticalObject* opto) {
  if (ALIUtils::debug >= 3)
    std::cout << "@@@ CocoaDBMgr::GetAlignInfoErrorFromOptO " << opto->name() << std::endl;

  align::ID cmsswID = opto->getCmsswID();

  GlobalError gerr(1., 0., 1., 0., 0., 1.);
  //double(dx*dx),  0., double(dy*dy),     0., 0., double(dz*dz) ) ;
  CLHEP::HepSymMatrix errms = asHepMatrix(gerr.matrix());
  AlignTransformErrorExtended* alignError = new AlignTransformErrorExtended(errms, cmsswID);
  return alignError;

  CLHEP::HepMatrix errm(3, 3);
  const std::vector<Entry*>& theCoordinateEntryVector = opto->CoordinateEntryList();
  std::cout << "@@@ CocoaDBMgr::GetAlignInfoFromOptOfill errm " << opto->name() << std::endl;
  errm(0, 0) = GetEntryError(theCoordinateEntryVector[0]) * dd4hep::m;  // m in COCOA, dd4hep unit in DB
  errm(1, 1) = GetEntryError(theCoordinateEntryVector[1]) * dd4hep::m;  // m in COCOA, dd4hep unit in DB
  errm(2, 2) = GetEntryError(theCoordinateEntryVector[2]) * dd4hep::m;  // m in COCOA, dd4hep unit in DB
  errm(0, 1) = GetEntryError(theCoordinateEntryVector[0], theCoordinateEntryVector[1]) *
               dd4hep::m;  // m in COCOA, dd4hep unit in DB
  errm(0, 2) = GetEntryError(theCoordinateEntryVector[0], theCoordinateEntryVector[2]) *
               dd4hep::m;  // m in COCOA, dd4hep unit in DB
  errm(1, 2) = GetEntryError(theCoordinateEntryVector[1], theCoordinateEntryVector[2]) *
               dd4hep::m;  // m in COCOA, dd4hep unit in DB
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
  return (AlignTransformErrorExtended*)nullptr;
}
