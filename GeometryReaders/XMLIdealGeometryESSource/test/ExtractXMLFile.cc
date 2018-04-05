// -*- C++ -*-
//
// Package:    ExtractXMLFile
// Class:      ExtractXMLFile
// 
/**\class ExtractXMLFile ExtractXMLFile.cc test/ExtractXMLFile/src/ExtractXMLFile.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Tommaso Boccali
//         Created:  Tue Jul 26 08:47:57 CEST 2005
//
//


// system include files
#include <memory>
#include <iostream>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/OfflineDBLoader/interface/GeometryInfoDump.h"

#include "CondFormats/Common/interface/FileBlob.h"
#include "Geometry/Records/interface/GeometryFileRcd.h"
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Core/interface/DDRoot.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

class ExtractXMLFile : public edm::one::EDAnalyzer<> {
public:
  explicit ExtractXMLFile( const edm::ParameterSet& );
  ~ExtractXMLFile() override;

  void beginJob() override {} 
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void endJob() override {}

private:
  std::string label_;
  std::string fname_;
};

ExtractXMLFile::ExtractXMLFile( const edm::ParameterSet& iConfig ) :
  label_(iConfig.getUntrackedParameter<std::string>("label","")),
  fname_(iConfig.getUntrackedParameter<std::string>("fname",""))
{
}

ExtractXMLFile::~ExtractXMLFile()
{
}

void
ExtractXMLFile::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
   using namespace edm;

   std::cout << "Here I am " << std::endl;
   edm::ESHandle<FileBlob> gdd;
   iSetup.get<GeometryFileRcd>().get(label_, gdd);
   std::ofstream f(fname_.c_str());
   (*gdd).write(f);
   std::cout << "finished" << std::endl;
}

DEFINE_FWK_MODULE(ExtractXMLFile);
