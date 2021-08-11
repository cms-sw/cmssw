#ifndef SurveyDataConverter_h
#define SurveyDataConverter_h

// Package:    SurveyDataConverter
// Class:      SurveyDataConverter
//
/**\class SurveyDataConverter SurveyDataConverter.h Alignment/SurveyDataConverter/interface/SurveyDataConverter.h

 Description: Reads survey corrections and applies them to the geometry.

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Roberto Covarelli
//         Created:  Fri Jul 14 18:05:49 CEST 2006
// $Id: SurveyDataConverter.h,v 1.2 2007/07/03 15:51:26 covarell Exp $
//
//

#include "Alignment/SurveyAnalysis/interface/SurveyDataReader.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Alignment/TrackerAlignment/interface/TrackerAlignment.h"

class SurveyDataConverter : public edm::one::EDAnalyzer<> {
  typedef SurveyDataReader::MapType MapType;
  typedef SurveyDataReader::PairType PairType;
  typedef SurveyDataReader::MapTypeOr MapTypeOr;
  typedef SurveyDataReader::PairTypeOr PairTypeOr;

public:
  explicit SurveyDataConverter(const edm::ParameterSet& iConfig);
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  void endJob() override{};

private:
  static const int NFILES = 2;

  // void applyAllSurveyInfo( align::Alignables alignables,
  //			   const MapType map );

  void applyCoarseSurveyInfo(TrackerAlignment& tr_align);

  void applyFineSurveyInfo(TrackerAlignment& tr_align, const MapType& map);

  void applyAPEs(TrackerAlignment& tr_align);

  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> ttrackerGeometryToken;
  edm::ParameterSet theParameterSet;
  edm::ParameterSet MisalignScenario;

  //private data members
  // AlignableTracker* theAlignableTracker;
  bool applyfineinfo, applycoarseinfo, adderrors;
};

#endif
