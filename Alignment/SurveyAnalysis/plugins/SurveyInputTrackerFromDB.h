#ifndef Alignment_SurveyAnalysis_SurveyInputTrackerFromDB_h
#define Alignment_SurveyAnalysis_SurveyInputTrackerFromDB_h

/** \class SurveyInputTrackerFromDB
 *
 *  Class to read ideal tracker from DB.
 *
 *  $Date: 2012/06/13 09:22:26 $
 *  $Revision: 1.4 $
 *  \author Chung Khim Lae
 */

#include "Alignment/SurveyAnalysis/interface/SurveyInputBase.h"
#include "Alignment/SurveyAnalysis/interface/SurveyInputTextReader.h"

namespace edm {
  class ParameterSet;
}

class SurveyInputTrackerFromDB : public SurveyInputBase {
public:
  SurveyInputTrackerFromDB(const edm::ParameterSet&);

  /// Read ideal tracker geometry from DB
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  SurveyInputTextReader::MapType uIdMap;

  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
  const edm::ESGetToken<GeometricDet, IdealGeometryRecord> geomDetToken_;
  const edm::ESGetToken<PTrackerParameters, PTrackerParametersRcd> ptpToken_;

  std::string textFileName;

  /// Add survey info to an alignable
  void addSurveyInfo(Alignable*);
};

#endif
