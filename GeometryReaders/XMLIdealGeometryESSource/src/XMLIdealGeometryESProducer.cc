// -*- C++ -*-
//
// Package:    XMLIdealGeometryESProducer
// Class:      XMLIdealGeometryESProducer
// 
/**\class XMLIdealGeometryESProducer XMLIdealGeometryESProducer.h GeometryReaders/XMLIdealGeometryESProducer/src/XMLIdealGeometryESProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Mike Case
//         Created:  Fri Jan 16 01:45:49 CET 2009
// $Id: XMLIdealGeometryESProducer.cc,v 1.9 2010/01/26 21:49:28 case Exp $
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDRoot.h"
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/GeometryFileRcd.h"
#include "CondFormats/GeometryObjects/interface/GeometryFile.h"
#include "Geometry/Records/interface/GeometryFileRcd.h"

#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSpecifics.h"
#include "DetectorDescription/Base/interface/DDRotationMatrix.h"

#include "DetectorDescription/Core/src/Material.h"
#include "DetectorDescription/Core/src/Solid.h"
#include "DetectorDescription/Core/src/LogicalPart.h"
#include "DetectorDescription/Core/src/Specific.h"

//
// class decleration
//

class XMLIdealGeometryESProducer : public edm::ESProducer {
public:
  XMLIdealGeometryESProducer(const edm::ParameterSet&);
  ~XMLIdealGeometryESProducer();
  
  typedef std::auto_ptr<DDCompactView> ReturnType;
  
  ReturnType produce(const IdealGeometryRecord&);
private:
  // ----------member data ---------------------------
  std::string rootDDName_; // this must be the form namespace:name
  std::string label_;
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
XMLIdealGeometryESProducer::XMLIdealGeometryESProducer(const edm::ParameterSet& iConfig)
  :   rootDDName_(iConfig.getParameter<std::string>("rootDDName")),
      label_(iConfig.getUntrackedParameter<std::string>("label","GeometryExtended"))
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced(this);
   //now do what ever other initialization is needed
}


XMLIdealGeometryESProducer::~XMLIdealGeometryESProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
XMLIdealGeometryESProducer::ReturnType
XMLIdealGeometryESProducer::produce(const IdealGeometryRecord& iRecord)
{
   using namespace edm::es;

   edm::ESTransientHandle<GeometryFile> gdd;
   iRecord.getRecord<GeometryFileRcd>().get( label_, gdd );

   DDName ddName(rootDDName_);
   DDLogicalPart rootNode(ddName);
   DDRootDef::instance().set(rootNode);
   ReturnType returnValue(new DDCompactView(rootNode));
   DDLParser parser(*returnValue);
   parser.getDDLSAX2FileHandler()->setUserNS(true);
   parser.clearFiles();
   
   std::vector<unsigned char>* tb = (*gdd).getUncompressedBlob();
   
   parser.parse(*tb, tb->size()); 
   
   delete tb;
   
   std::cout << "In XMLIdealGeometryESProducer::produce" << std::endl;
   returnValue->lockdown();

   return returnValue ;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(XMLIdealGeometryESProducer);
