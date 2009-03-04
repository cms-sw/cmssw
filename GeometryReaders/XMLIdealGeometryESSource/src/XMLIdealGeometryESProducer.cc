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
// $Id$
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDRoot.h"
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/GeometryFileRcd.h"
#include "CondFormats/GeometryObjects/interface/GeometryFile.h"
#include "Geometry/Records/interface/GeometryFileRcd.h"



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
  //  std::string label_;
  std::string rootDDName_; // this must be the form namespace:name
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
  :   rootDDName_(iConfig.getParameter<std::string>("rootDDName"))
  //  :   label_(iConfig.getUntrackedParameter<std::string>("label","")),
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

   edm::ESHandle<GeometryFile> gdd;
   iRecord.getRecord<GeometryFileRcd>().get( "", gdd );

   DDLParser * parser = DDLParser::instance();
   DDRootDef::instance().set(DDName(rootDDName_));

   std::vector<unsigned char>* tb = (*gdd).getUncompressedBlob();

   parser->parse(*tb, tb->size()); 


   delete tb;
   
   std::cout << std::endl;

   std::auto_ptr<DDCompactView> returnValue(new DDCompactView(DDLogicalPart(DDName(rootDDName_))));


   DDCompactView globalOne;
   returnValue->writeableGraph() = globalOne.graph();

   return returnValue ;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(XMLIdealGeometryESProducer);
