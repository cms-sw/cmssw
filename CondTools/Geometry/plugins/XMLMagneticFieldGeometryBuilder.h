#ifndef CondToolsGeometry_XMLMagneticFieldGeometryBuilder_h
#define CondToolsGeometry_XMLMagneticFieldGeometryBuilder_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>

class XMLMagneticFieldGeometryBuilder : public edm::EDAnalyzer
{
public:
  explicit XMLMagneticFieldGeometryBuilder( const edm::ParameterSet& );
  ~XMLMagneticFieldGeometryBuilder( void );
  
  virtual void beginJob( void );
  virtual void analyze( const edm::Event&, const edm::EventSetup& ) {}
  virtual void endJob( void ) {}
  
private:
  std::string fname;
  bool zip;
};

#endif
