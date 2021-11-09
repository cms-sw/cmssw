#ifndef Alignment_MuonAlignment_MuonAlignment_H
#define Alignment_MuonAlignment_MuonAlignment_H

/** \class MuonAlignment
 *  The MuonAlignment helper class for alignment jobs
 *
 *  $Date: 2011/06/07 19:28:47 $
 *  $Revision: 1.14 $
 *  \author Andre Sznajder - UERJ(Brazil)
 */

#include <map>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Alignment/MuonAlignment/interface/AlignableMuon.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "Alignment/CommonAlignment/interface/AlignableNavigator.h"
#include "Alignment/MuonAlignment/interface/MuonAlignmentInputMethod.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

class MuonAlignment {
public:
  MuonAlignment(const DTGeometry* dtGeometry, const CSCGeometry* cscGeometry, const GEMGeometry* gemGeometry);

  MuonAlignment(const edm::EventSetup& iSetup, const MuonAlignmentInputMethod& input);

  ~MuonAlignment() {
    delete theAlignableMuon;
    delete theAlignableNavigator;
  }

  AlignableMuon* getAlignableMuon() { return theAlignableMuon; }

  AlignableNavigator* getAlignableNavigator() { return theAlignableNavigator; }

  void moveAlignableLocalCoord(DetId&, align::Scalars&, align::Scalars&);
  void moveAlignableGlobalCoord(DetId&, align::Scalars&, align::Scalars&);

  void recursiveList(const align::Alignables& alignables, align::Alignables& theList);
  void recursiveMap(const align::Alignables& alignables, std::map<align::ID, Alignable*>& theMap);
  void recursiveStructureMap(const align::Alignables& alignables,
                             std::map<std::pair<align::StructureType, align::ID>, Alignable*>& theMap);

  void copyAlignmentToSurvey(double shiftErr, double angleErr);
  void fillGapsInSurvey(double shiftErr, double angleErr);
  void copySurveyToAlignment();

  void writeXML(const edm::ParameterSet& iConfig,
                const DTGeometry* dtGeometryXML,
                const CSCGeometry* cscGeometryXML,
                const GEMGeometry* gemGeometryXML);

  void saveDTSurveyToDB();
  void saveCSCSurveyToDB();
  void saveSurveyToDB();

  void saveDTtoDB();
  void saveCSCtoDB();
  void saveGEMtoDB();
  void saveToDB();

private:
  void init();
  void recursiveCopySurveyToAlignment(Alignable* alignable);

  std::string theDTAlignRecordName, theDTErrorRecordName;
  std::string theCSCAlignRecordName, theCSCErrorRecordName;
  std::string theGEMAlignRecordName, theGEMErrorRecordName;
  std::string theDTSurveyRecordName, theDTSurveyErrorRecordName;
  std::string theCSCSurveyRecordName, theCSCSurveyErrorRecordName;

  const DTGeometry* dtGeometry_;
  const CSCGeometry* cscGeometry_;
  const GEMGeometry* gemGeometry_;

  align::Scalars displacements;

  align::Scalars rotations;

  AlignableMuon* theAlignableMuon;

  AlignableNavigator* theAlignableNavigator;
};

#endif  //MuonAlignment_H
