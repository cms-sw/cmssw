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
// $Id: SurveyDataConverter.h,v 1.1 2007/04/04 10:00:58 covarell Exp $
//
//


// System include files
#include <memory>
#include "boost/shared_ptr.hpp"

// Framework
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
///*
#include <vector>
// Geometry
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
// Alignment
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/TrackerAlignment/interface/TrackerAlignment.h"
//*/

static const int NFILES = 2;

class Alignable;
class AlignableTracker;
class SurveyDataConverter : public edm::EDAnalyzer
{

  typedef unsigned int DetIdType;
  typedef std::map<DetIdType, std::vector<float> >  MapType;
  typedef std::pair<DetIdType,std::vector<float> > PairType;
  typedef std::map< std::vector<int>, std::vector<float> > MapTypeOr;
  typedef std::pair< std::vector<int>, std::vector<float> > PairTypeOr;
  typedef Surface::RotationType RotationType;
	
public:
  explicit SurveyDataConverter(const edm::ParameterSet& iConfig);
  ~SurveyDataConverter() {};
	
  virtual void analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup);
  virtual void endJob() {};

private:

  typedef std::vector<Alignable*> Alignables;
  
  // void applyAllSurveyInfo( std::vector<Alignable*> alignables, 
  //			   const MapType map );
  
  void applyCoarseSurveyInfo(TrackerAlignment& tr_align);
  
  void applyFineSurveyInfo(TrackerAlignment& tr_align, const MapType map);
  
  void applyAPEs( TrackerAlignment& tr_align );
  
  edm::ParameterSet theParameterSet;
  edm::ParameterSet MisalignScenario;
  
  //private data members
  // AlignableTracker* theAlignableTracker;
  bool applyfineinfo, applycoarseinfo, adderrors;

};

#endif
