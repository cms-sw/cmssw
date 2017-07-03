#ifndef CalibTracker_SiPixelDynamicInefficiencyDB_SiPixelDynamicInefficiencyDB_h
#define CalibTracker_SiPixelDynamicInefficiencyDB_SiPixelDynamicInefficiencyDB_h

#include <map>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/DetId/interface/DetId.h"

class SiPixelDynamicInefficiencyDB : public edm::EDAnalyzer
{
 public:
  
  explicit SiPixelDynamicInefficiencyDB(const edm::ParameterSet& conf);
  
  ~SiPixelDynamicInefficiencyDB() override;
  
  void beginJob() override;
  
  void endJob() override; 
  
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;
  
  
 private:

  edm::ParameterSet conf_;
  std::string recordName_;

  typedef std::vector< edm::ParameterSet > Parameters;
  Parameters thePixelGeomFactors_;
  Parameters theColGeomFactors_;
  Parameters theChipGeomFactors_;
  Parameters thePUEfficiency_;
  double theInstLumiScaleFactor_;
};


#endif
