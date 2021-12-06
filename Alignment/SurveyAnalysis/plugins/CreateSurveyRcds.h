#ifndef Alignment_SurveyAnalysis_CreateSurveyRcds_h
#define Alignment_SurveyAnalysis_CreateSurveyRcds_h

/** \class CreateSurveyRcds
 *
 *  Class to create Survey[Error]Rcd for alignment with survey constraint
 *
 *  $Date: 2012/06/13 09:22:26 $
 *  $Revision: 1.3 $
 *  \author Chung Khim Lae
 */
// user include files

#include "Alignment/SurveyAnalysis/interface/SurveyInputBase.h"
#include "Alignment/SurveyAnalysis/interface/SurveyInputTextReader.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "CondFormats/GeometryObjects/interface/PTrackerParameters.h"
#include "Geometry/Records/interface/PTrackerParametersRcd.h"

#include "CondFormats/GeometryObjects/interface/PTrackerAdditionalParametersPerDet.h"
#include "Geometry/Records/interface/PTrackerAdditionalParametersPerDetRcd.h"

#include "Alignment/CommonAlignment/interface/SurveyDet.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"

#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentErrorExtendedRcd.h"

class AlignableSurface;
class Alignments;

class CreateSurveyRcds : public SurveyInputBase {
public:
  CreateSurveyRcds(const edm::ParameterSet&);

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  /// module which modifies the geometry
  void setGeometry(Alignable*);
  /// module which creates/inserts the survey errors
  void setSurveyErrors(Alignable*);

  /// default values for assembly precision
  AlgebraicVector getStructurePlacements(int, int);

  /// default values for survey uncertainty
  AlgebraicVector getStructureErrors(int, int);

  // es tokens
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
  const edm::ESGetToken<GeometricDet, IdealGeometryRecord> geomDetToken_;
  const edm::ESGetToken<PTrackerParameters, PTrackerParametersRcd> ptpToken_;
  const edm::ESGetToken<PTrackerAdditionalParametersPerDet, PTrackerAdditionalParametersPerDetRcd> ptitpToken_;
  const edm::ESGetToken<Alignments, TrackerAlignmentRcd> aliToken_;
  const edm::ESGetToken<AlignmentErrorsExtended, TrackerAlignmentErrorExtendedRcd> aliErrToken_;

  std::string m_inputGeom;
  double m_inputSimpleMis;
  bool m_generatedRandom;
  bool m_generatedSimple;

  SurveyInputTextReader::MapType uIdMap;

  std::string textFileName;
};

#endif
