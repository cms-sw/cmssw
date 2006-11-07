#ifndef CocoaAnalyser_HH
#define CocoaAnalyser_HH
//-*- C++ -*-
//
// Package:    Alignment/CocoaApplication
// Class:      CocoaAnalyzer
// 
/*

 Description: test access to the OpticalAlignMeasurements via OpticalAlignMeasurementsGeneratedSource
    This also should demonstrate access to a geometry via the XMLIdealGeometryESSource
    for use in THE COCOA analyzer.

 Implementation:
     Iterate over retrieved alignments.
*/
//

#include <stdexcept>
#include <string>
#include <iostream>
#include <map>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DetectorDescription/Core/interface/DDSpecifics.h"

class Event;
class EventSetup;
class Entry;
//#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/OptAlignObjects/interface/OAQuality.h"
#include "CondFormats/OptAlignObjects/interface/OpticalAlignments.h"
#include "CondFormats/OptAlignObjects/interface/OpticalAlignInfo.h"
#include "CondFormats/OptAlignObjects/interface/OpticalAlignMeasurements.h"
#include "CondFormats/OptAlignObjects/interface/OpticalAlignMeasurementInfo.h"

#include "CondFormats/DataRecord/interface/OpticalAlignmentsRcd.h"

class DDFilteredView;
class DDCompactView;
class DDSpecifics;
class OpticalObject;

#include "Geometry/Records/interface/IdealGeometryRecord.h"

using namespace std;

class CocoaAnalyzer : public edm::EDAnalyzer
{
 public:
  
  explicit  CocoaAnalyzer(edm::ParameterSet const& p);
  explicit  CocoaAnalyzer(int i) { }
  virtual ~ CocoaAnalyzer() { }
  
  virtual void beginJob(const edm::EventSetup& c);
  virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  // see note on endJob() at the bottom of the file.
  // virtual void endJob() ;

 private:
  void ReadXMLFile( const edm::EventSetup& evts );
  std::vector<OpticalAlignInfo> ReadCalibrationDB( const edm::EventSetup& evts );

  void CorrectOptAlignments( std::vector<OpticalAlignInfo>& oaListCalib );
  OpticalAlignInfo* FindOpticalAlignInfoXML( OpticalAlignInfo oaInfo );
  bool CorrectOaParam( OpticalAlignParam* oaParamXML, OpticalAlignParam oaParamDB );

  void RunCocoa();

  bool DumpCocoaResults(); 
  double GetEntryError( const Entry* entry );
 
  OpticalAlignInfo GetOptAlignInfoFromOptO( OpticalObject* opto );
  double myFetchDbl(const DDsvalues_type& dvst, 
	        		      const std::string& spName,
				    const size_t& vecInd );
  std::string myFetchString(const DDsvalues_type& dvst, 
				      const std::string& spName,
				    const size_t& vecInd );

 private:
  OpticalAlignments oaList_;
  OpticalAlignMeasurements measList_;
  std::string theCocoaDaqRootFileName;
};

#endif
