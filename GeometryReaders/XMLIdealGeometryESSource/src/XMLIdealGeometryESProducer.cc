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
// $Id: XMLIdealGeometryESProducer.cc,v 1.2 2009/08/12 01:00:28 case Exp $
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

#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSpecifics.h"
#include "DetectorDescription/Base/interface/DDRotationMatrix.h"


namespace DDI {
  class Material;
  class Solid;
  class LogicalPart;
  class Specific;
}

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
    // 2009-07-09 memory patch
    // for copying and protecting DD Store's after parsing is complete.
    DDI::Store<DDName, DDI::Material*>::registry_type matStore_;
    DDI::Store<DDName, DDI::Solid*>::registry_type solidStore_;
    DDI::Store<DDName, DDI::LogicalPart*>::registry_type lpStore_;
    DDI::Store<DDName, DDI::Specific*>::registry_type specStore_;
    DDI::Store<DDName, DDRotationMatrix*>::registry_type rotStore_;    

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
   parser->getDDLSAX2FileHandler()->setUserNS(true);
   // 2009-07-09 memory patch
   parser->clearFiles();
   //std::cout <<"got in produce"<<std::endl;
   DDName ddName(rootDDName_);
   //std::cout <<"ddName \""<<ddName<<"\""<<std::endl;
   DDLogicalPart rootNode(ddName);
   //std::cout <<"made the DDLogicalPart"<<std::endl;
   DDRootDef::instance().set(rootNode);
   
   
   std::vector<unsigned char>* tb = (*gdd).getUncompressedBlob();
   
   parser->parse(*tb, tb->size()); 
   
   delete tb;
   
   std::cout << std::endl;
  // at this point we should have a valid store of DDObjects and we will move these
  // to the local storage area using swaps with the existing Singleton<Store...>'s
  // NOTE TO SELF:  this is similar to the below explicit copy of the graph of the
  // DDCompactView at this point with this XMLIdealGeometryESSource so as not to
  // share the Store's anymore.
  // 2009-07-09 memory patch
   DDMaterial::StoreT::instance().swap(matStore_);
   DDSolid::StoreT::instance().swap(solidStore_);
   DDLogicalPart::StoreT::instance().swap(lpStore_);
   DDSpecifics::StoreT::instance().swap(specStore_);
   DDRotation::StoreT::instance().swap(rotStore_);

   DDMaterial::StoreT::instance().setReadOnly(true);
   DDSolid::StoreT::instance().setReadOnly(true);
   DDLogicalPart::StoreT::instance().setReadOnly(true);
   DDSpecifics::StoreT::instance().setReadOnly(true);
   DDRotation::StoreT::instance().setReadOnly(true);

   std::auto_ptr<DDCompactView> returnValue(new DDCompactView(rootNode));

   //copy the graph from the global one
   DDCompactView globalOne;
   returnValue->swap(globalOne);
   
   return returnValue ;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(XMLIdealGeometryESProducer);
