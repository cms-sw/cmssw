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
// $Id: ExtractXMLFile.cc,v 1.3 2010/07/19 16:16:15 case Exp $
//
//


// system include files
#include <memory>
#include <iostream>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

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

//#include "DataSvc/RefException.h"
//#include "CoralBase/Exception.h"

//
// class decleration
//

class ExtractXMLFile : public edm::EDAnalyzer {
public:
  explicit ExtractXMLFile( const edm::ParameterSet& );
  ~ExtractXMLFile();

  
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
private:
  // ----------member data ---------------------------
  std::string label_;
  std::string fname_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
ExtractXMLFile::ExtractXMLFile( const edm::ParameterSet& iConfig ) :
  label_(iConfig.getUntrackedParameter<std::string>("label","")),
  fname_(iConfig.getUntrackedParameter<std::string>("fname",""))
{
}


ExtractXMLFile::~ExtractXMLFile()
{
 
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
ExtractXMLFile::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
   using namespace edm;

   std::cout << "Here I am " << std::endl;
   edm::ESHandle<FileBlob> gdd;
   iSetup.get<GeometryFileRcd>().get(label_, gdd);
   //std::vector<unsigned char>* tb = (*gdd).getUncompressedBlob();
   std::ofstream f(fname_.c_str());
   (*gdd).write(f);
   //f.close();
   std::cout << "finished" << std::endl;
}


//define this as a plug-in
DEFINE_FWK_MODULE(ExtractXMLFile);
