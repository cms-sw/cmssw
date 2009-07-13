/***************************************************************************
                          testDoc.cpp  -  description
                             -------------------
    Author               : Michael Case
    email                : case@ucdhep.ucdavis.edu

    Last Updated         : Jan 9, 2004
 ***************************************************************************/

#include <string>
#include <vector>
#include <map>
#include <iostream>
//#include <cstdlib>

// includes for CMSSW main
#include <boost/shared_ptr.hpp>
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/PluginManager/interface/ProblemTracker.h"
#include "FWCore/Utilities/interface/Presence.h"
#include "FWCore/PluginManager/interface/PresenceFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/MakeParameterSets.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
// end includes for CMSSW main

#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Parser/interface/DDLConfiguration.h"
#include "DetectorDescription/Core/interface/DDMap.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DetectorDescription/Core/interface/DDVector.h"
#include "DetectorDescription/Core/interface/DDNumeric.h"
#include "DetectorDescription/Core/interface/DDString.h"
#include "DetectorDescription/Algorithm/src/AlgoInit.h"
#include "DetectorDescription/Core/src/DDCheck.h"
#include "DetectorDescription/Core/src/DDCheckMaterials.cc"
#include "DetectorDescription/Base/interface/DDException.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDExpandedNode.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/Core/interface/DDRoot.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Parser/src/StrX.h"

class DDLTestDoc : public DDLDocumentProvider {

 public:

  DDLTestDoc();
  virtual ~DDLTestDoc();

  /// Return a list of files as a vector of strings.
  virtual const std::vector < std::string >&  getFileList(void) const;

  /// Return a list of urls as a vector of strings.
  virtual const std::vector < std::string >&  getURLList(void) const;

  /// Print out the list of files.
  virtual void dumpFileList(void) const;

  /// Return whether Validation should be on or off and where the DDL SchemaLocation is.
  virtual bool doValidation() const;

  /// Return the designation for where to look for the schema.
  std::string getSchemaLocation() const;

  /// ReadConfig
  virtual int readConfig(const std:: string& filename);

  void push_back(std::string fileName, std::string url = std::string("./"));

  void setSchemaLocation(std::string path = std::string("../../DDSchema"));

  void setValidation(bool val);

  void clear();

  // test routines
//    void testRotations();

//    void testMaterials();

//    void testSolids();

//    void testLogicalParts();

//    void testPosParts();

//    void testAlgoPosParts();

//    void testAlgorithm();

 protected:

 private:
  std::vector < std::string > fnames_;
  std::vector < std::string > urls_;
  std::string schemaLoc_;
  bool validate_;
};


namespace std{} using namespace std;
namespace xercesc_2_7{} using namespace xercesc_2_7;

//--------------------------------------------------------------------------
//  DDLTestDoc:  Default constructor and destructor.
//--------------------------------------------------------------------------
DDLTestDoc::~DDLTestDoc()
{ 
}

DDLTestDoc::DDLTestDoc() : validate_(true)
{ 
  schemaLoc_ = "http://www.cern.ch/cms/DDL ../../Schema/DDLSchema.xsd";
}

const vector<string>&  DDLTestDoc::getFileList(void) const
{
  return fnames_;
}

const vector<string>&  DDLTestDoc::getURLList(void) const
{
  return urls_;
}

void DDLTestDoc::push_back( string fileName, string url) 
{
  fnames_.push_back(fileName);
  urls_.push_back(url);
}

void DDLTestDoc::setValidation (bool val) { validate_= val; }

bool DDLTestDoc::doValidation() const { return validate_; }

void DDLTestDoc::setSchemaLocation(string path) { schemaLoc_ = path; }

string DDLTestDoc::getSchemaLocation() const { cout << schemaLoc_ << endl; return schemaLoc_; }

void DDLTestDoc::dumpFileList(void) const {
  cout << "File List:" << endl;
  vector<string> vst = getFileList();
  cout << "  number of files=" << vst.size() << endl;
  for (vector<string>::const_iterator it = vst.begin(); it != vst.end(); ++it)
    cout << *it << endl;
}

void DDLTestDoc::clear()
{
  fnames_.clear();
  urls_.clear();
}

//-----------------------------------------------------------------------
//  Here the Xerces parser is used to process the content of the 
//  configuration file.
//  FIX:  Right now, each config file passed to this will simply increase the 
//  size of the list of files.  So if this default DDLDocumentProvider is
//  called repeatedly (i.e. the same instance of it) then the file list MAY
//  repeat files.  It is the Parser which checks for maintains a list of real
//  files.
//-----------------------------------------------------------------------
int DDLTestDoc::readConfig(const std::string& filename)
{
  DCOUT('P', "DDLConfiguration::ReadConfig(): started");
  std::cout << "readConfig" << std::endl;
  //  configFileName_ = filename;

  // Set the parser to use the handler for the configuration file.
  // This makes sure the Parser is initialized and gets a handle to it.
  DDLParser * parser = DDLParser::instance();
  DDLSAX2Handler* errHandler;
  DDLSAX2ConfigHandler * sch;

  sch = new DDLSAX2ConfigHandler;
  errHandler = new DDLSAX2Handler;

  parser->getXMLParser()->setContentHandler(sch);
  parser->getXMLParser()->setErrorHandler(errHandler);

  try {
    parser->getXMLParser()->parse(filename.c_str());
  }
  catch (const XMLException& toCatch) {
    std::cout << "\nXMLException: parsing '" << filename << "'\n"
	 << "Exception message is: \n"
	 << std::string(StrX(toCatch.getMessage()).localForm()) << "\n" ;
    return -1;
  }
  catch (...)
    {
      std::cout << "\nUnexpected exception during parsing: '" << filename << "'\n";
      return 4;
    }

  fnames_ = sch->getFileNames();
  urls_ = sch->getURLs();
   std::cout << "there are " << fnames_.size() << " files." << std::endl;
   for (size_t i = 0; i < fnames_.size(); ++i)
     //std::cout << "url=" << sch->getURLs()[i] << " file=" << sch->getFileNames()[i] << std::endl;
     std::cout << "url=" << urls_[i] << " file=" << fnames_[i] << std::endl;
  return 0;
}


void testRotations() {     

  cout << "--------------- Parser testing Rotations --------------" << endl;
  cout << "z30 should be a rotation of 30 degrees around the z axis:" << endl;
  cout << DDRotation(DDName( "z30", "testRotations")) << endl;
  cout << endl;
  cout << "z30x20 should be a rotation 30 degrees around z, then 20 degrees around x:" << endl;
  cout << DDRotation(DDName( "z30x20", "testRotations")) << endl;
  cout << endl;
  cout << "x90y45 should be a rotation 90 degrees around x, then 45 degrees around y:" << endl;
  cout << DDRotation(DDName( "x90y45", "testRotations")) << endl;
  cout << endl;
  cout << "x90y90 should be a rotation 90 degrees around x, then 90 degrees around y:" << endl;
  cout << DDRotation(DDName( "x90y90", "testRotations")) << endl;
  cout << endl;
  cout << "x90y135 should be a rotation 90 degrees around x, then 135 degrees around y:" << endl;
  cout << DDRotation(DDName( "x90y135", "testRotations")) << endl;
  cout << endl;
  cout << "x90y180 should be a rotation 90 degrees around x, then 180 degrees around y:" << endl;
  cout << DDRotation(DDName( "x90y180", "testRotations")) << endl;
  cout << endl;
  cout << "x90 should be a rotation of 90 degrees around the x axis:" << endl;
  cout << DDRotation(DDName( "x90", "testRotations")) << endl;
  cout << endl;
  cout << "cmsimIdentity makes the identity rotation matrix using the cmsim method (phi and theta of each axis):" << endl;
  cout << DDRotation(DDName("cmsimIdentity", "testRotations")) << endl;
  cout << endl;
  cout << "newrotIdentity makes the identity rotation matrix by rotating 0 degrees around the z axis:" << endl;
  cout << DDRotation(DDName("newrotIdentity", "testRotations")) << endl;
  cout << endl;
  cout << "180R should be a REFLECTION rotation.  It is defined using the cmsim way:" << endl;
  cout << DDRotation(DDName("180R", "testRotations")) << endl;
  cout << endl;
}

void testMaterials() { 
  cout << "--------------- Parser testing Materials --------------" << endl;
  cout << "There should be 4 Elementary Materials: Nitrogen," << endl;
  cout << "Oxygen,Argon and Hydrogen.  There should be one composite" << endl;
  cout << " material:Air, made up of those 4 components." << endl;
  cout << DDMaterial(DDName("Nitrogen", "testMaterials")) << endl;
  cout << endl;
  cout << DDMaterial(DDName("Oxygen", "testMaterials")) << endl;
  cout << endl;
  cout << DDMaterial(DDName("Argon", "testMaterials")) << endl;
  cout << endl;
  cout << DDMaterial(DDName("Hydrogen", "testMaterials")) << endl;
  cout << endl;
  cout << DDMaterial(DDName("Air", "testMaterials")) << endl;
  cout << endl;
}

void testSolids() { 
  cout << "--------------- Parser testing Solids --------------" << endl;
  cout << "trap1 is a trapezoid:" << endl;
  cout << DDSolid(DDName("trap1", "testSolids")) << endl;
  cout << endl;
  cout << "trap2 is a trapezoid with more symmetry:" << endl;
  cout << DDSolid(DDName("trap2", "testSolids")) << endl;
  cout << endl;
  cout << "ptrap1 is a pseudo Trapezoid with atMinusZ=false:" << endl;
  cout << DDSolid(DDName("ptrap1", "testSolids")) << endl;
  cout << endl;
  cout << "ptrap2 is a psuedo Trapezoid with atMinusZ=true:" << endl;
  cout << DDSolid(DDName("ptrap2", "testSolids")) << endl;
  cout << endl;
  cout << "box1 is a Box:" << endl;
  cout << DDSolid(DDName("box1", "testSolids")) << endl;
  cout << endl;
  cout << "cone1 is a conical section (full 360 degree):" << endl;
  cout << DDSolid(DDName("cone1", "testSolids")) << endl;
  cout << endl;
  cout << "cone2 is a conical section (full 360 degree) which is actually a tube:" << endl;
  cout << DDSolid(DDName("cone2", "testSolids")) << endl;
  cout << endl;
  cout << "cone2hole is a conical section (20 degree):" << endl;
  cout << DDSolid(DDName("cone2hole", "testSolids")) << endl;
  cout << endl;
  cout << "pczsect is a polycone defined using z-sections:" << endl;
  cout << DDSolid(DDName("pczsect", "testSolids")) << endl;
  cout << endl;
  cout << "pcrz is a polycone defined using r & z points:" << endl;
  cout << DDSolid(DDName("pcrz", "testSolids")) << endl;
  cout << endl;
  cout << "phzsect is a polyhedra defined using z-sections:" << endl;
  cout << DDSolid(DDName("phzsect", "testSolids")) << endl;
  cout << endl;
  cout << "phrz is a polyhedra defined using r & z points:" << endl;
  cout << DDSolid(DDName("phrz", "testSolids")) << endl;
  cout << endl;
  cout << "trd1 is a \"special\" trapezoid declaration with fewer" ;
  cout << " parameters (Trd1):" << endl;
  cout << DDSolid(DDName("trd1", "testSolids")) << endl;
  cout << endl;
  cout << "trd2 is a \"special\" trapezoid declaration with fewer" ;
  cout << " parameters (Trd1):" << endl;
  cout << DDSolid(DDName("trd2", "testSolids")) << endl;
  cout << endl;
  cout << "tube1 is a tube:" << endl;
  cout << DDSolid(DDName("tube1", "testSolids")) << endl;
  cout << endl;
  cout << "tube2 is a tubs(Tube Section):" << endl;
  cout << DDSolid(DDName("tube2", "testSolids")) << endl;
  cout << endl;
  cout << "trunctubs1 is a trunctubs(Cut or truncated Tube Section):" << endl;
  cout << DDSolid(DDName("trunctubs1", "testSolids")) << endl;
  cout << endl;
  cout << "momma is a shapeless solid, a way of \"grouping\" things:" << endl;
  cout << DDSolid(DDName("momma", "testSolids")) << endl;
  cout << endl;
  cout << "MotherOfAllBoxes is a box and is the root's solid:" << endl;
  cout << DDSolid(DDName("MotherOfAllBoxes", "testSolids")) << endl;
  cout << endl;
  cout << "trd2mirror is a ReflectionSolid of trd2:" << endl;
  cout << DDSolid(DDName("trd2mirror", "testSolids")) << endl;
  cout << endl;
  cout << "subsolid is a subtraction solid, cone2-cone2hole:" << endl;
  cout << DDSolid(DDName("subsolid", "testSolids")) << endl;
  cout << endl;
  cout << "unionsolid is a union of pcrz and cone1:" << endl;
  cout << DDSolid(DDName("unionsolid", "testSolids")) << endl;
  cout << endl;
  cout << "intsolid is an Intersection(Solid) of cone1 and cone2:" << endl;
  cout << DDSolid(DDName("intsolid", "testSolids")) << endl;
  cout << endl;
}

void testLogicalParts() { 
  cout << "--------------- Parser testing LogicalParts --------------" << endl;
  cout << "LogicalPart trap1:" << endl;
  cout << DDLogicalPart(DDName("trap1", "testLogicalParts")) << endl;
  cout << endl;
  cout << "LogicalPart trap2:" << endl;
  cout << DDLogicalPart(DDName("trap2", "testLogicalParts")) << endl;
  cout << endl;
  cout << "LogicalPart ptrap1:" << endl;
  cout << DDLogicalPart(DDName("ptrap1", "testLogicalParts")) << endl;
  cout << endl;
  cout << "LogicalPart ptrap2:" << endl;
  cout << DDLogicalPart(DDName("ptrap2", "testLogicalParts")) << endl;
  cout << endl;
  cout << "LogicalPart box1:" << endl;
  cout << DDLogicalPart(DDName("box1", "testLogicalParts")) << endl;
  cout << endl;
  cout << "LogicalPart cone1:" << endl;
  cout << DDLogicalPart(DDName("cone1", "testLogicalParts")) << endl;
  cout << endl;
  cout << "LogicalPart cone2:" << endl;
  cout << DDLogicalPart(DDName("cone2", "testLogicalParts")) << endl;
  cout << endl;
  cout << "LogicalPart cone2hole:" << endl;
  cout << DDLogicalPart(DDName("cone2hole", "testLogicalParts")) << endl;
  cout << endl;
  cout << "LogicalPart pczsect:" << endl;
  cout << DDLogicalPart(DDName("pczsect", "testLogicalParts")) << endl;
  cout << endl;
  cout << "LogicalPart pcrz:" << endl;
  cout << DDLogicalPart(DDName("pcrz", "testLogicalParts")) << endl;
  cout << endl;
  cout << "LogicalPart phzsect:" << endl;
  cout << DDLogicalPart(DDName("phzsect", "testLogicalParts")) << endl;
  cout << endl;
  cout << "LogicalPart phrz:" << endl;
  cout << DDLogicalPart(DDName("phrz", "testLogicalParts")) << endl;
  cout << endl;
  cout << "LogicalPart trd1:" ;
  cout << DDLogicalPart(DDName("trd1", "testLogicalParts")) << endl;
  cout << endl;
  cout << "LogicalPart trd2:" << endl;
  cout << DDLogicalPart(DDName("trd2", "testLogicalParts")) << endl;
  cout << endl;
  cout << "LogicalPart tube1:" << endl;
  cout << DDLogicalPart(DDName("tube1", "testLogicalParts")) << endl;
  cout << endl;
  cout << "LogicalPart tube2:" << endl;
  cout << DDLogicalPart(DDName("tube2", "testLogicalParts")) << endl;
  cout << endl;
  cout << "LogicalPart trunctubs1:" << endl;
  cout << DDLogicalPart(DDName("trunctubs1", "testLogicalParts")) << endl;
  cout << endl;
  cout << "LogicalPart momma:" << endl;
  cout << DDLogicalPart(DDName("momma", "testLogicalParts")) << endl;
  cout << endl;
  cout << "LogicalPart MotherOfAllBoxes:" << endl;
  cout << DDLogicalPart(DDName("MotherOfAllBoxes", "testLogicalParts")) << endl;
  cout << endl;
  cout << "LogicalPart torus:" << endl;
  cout << DDLogicalPart(DDName("torus", "testLogicalParts")) << endl;
  cout << endl;
  cout << "LogicalPart trd2mirror:" << endl;
  cout << DDLogicalPart(DDName("trd2mirror", "testLogicalParts")) << endl;
  cout << endl;
  cout << "LogicalPart subsolid:" << endl;
  cout << DDLogicalPart(DDName("subsolid", "testLogicalParts")) << endl;
  cout << endl;
  cout << "LogicalPart unionsolid:" << endl;
  cout << DDLogicalPart(DDName("unionsolid", "testLogicalParts")) << endl;
  cout << endl;
  cout << "LogicalPart intsolid:" << endl;
  cout << DDLogicalPart(DDName("intsolid", "testLogicalParts")) << endl;
  cout << endl;
}

void testPosParts() { 


}

void testAlgoPosParts() { 

}

void testAlgorithm() { 

}

int main(int argc, char *argv[])
{
  // MEC: 2008-08-04 : I believe the main problem w/ this is not being framework friendly.
  // so I'm (over) using the "main" of CMSSW

  std::string const kProgramName = argv[0];
  int rc = 0;

  // Copied from example stand-alone program in Message Logger July 18, 2007
  try {

    // A.  Instantiate a plug-in manager first.
    edm::AssertHandler ah;

    // B.  Load the message service plug-in.  Forget this and bad things happen!
    //     In particular, the job hangs as soon as the output buffer fills up.
    //     That's because, without the message service, there is no mechanism for
    //     emptying the buffers.
    boost::shared_ptr<edm::Presence> theMessageServicePresence;
    theMessageServicePresence = boost::shared_ptr<edm::Presence>(edm::PresenceFactory::get()->
								 makePresence("MessageServicePresence").release());

    // C.  Manufacture a configuration and establish it.
    std::string config =
      "import FWCore.ParameterSet.Config as cms\n"
      "process = cms.Process('TEST')\n"
      "process.maxEvents = cms.untracked.PSet(\n"
      "    input = cms.untracked.int32(5)\n"
      ")\n"
      "process.source = cms.Source('EmptySource')\n"
      "process.JobReportService = cms.Service('JobReportService')\n"
      "process.InitRootHandlers = cms.Service('InitRootHandlers')\n"
      // "process.MessageLogger = cms.Service('MessageLogger')\n"
      "process.m1 = cms.EDProducer('IntProducer',\n"
      "    ivalue = cms.int32(11)\n"
      ")\n"
      "process.out = cms.OutputModule('PoolOutputModule',\n"
      "    fileName = cms.untracked.string('testStandalone.root')\n"
      ")\n"
      "process.p = cms.Path(process.m1)\n"
      "process.e = cms.EndPath(process.out)\n";

    boost::shared_ptr<std::vector<edm::ParameterSet> > pServiceSets;
    boost::shared_ptr<edm::ParameterSet>          params_;
    edm::makeParameterSets(config, params_, pServiceSets);

    // D.  Create the services.
    edm::ServiceToken tempToken(edm::ServiceRegistry::createSet(*pServiceSets.get()));

    // E.  Make the services available.
    edm::ServiceRegistry::Operate operate(tempToken);

    // END Copy from example stand-alone program in Message Logger July 18, 2007

    cout  << "Initialize DDD (call AlgoInit)" << endl;

    AlgoInit();

    cout << "Initialize DDL parser (get the first instance)" << endl;
    DDLParser* myP = DDLParser::instance();

    if (argc < 2) {
      cout << "DEFAULT test using testConfiguration.xml" << endl;

      DDLTestDoc dp; //DDLConfiguration dp;

      dp.readConfig("testConfiguration.xml");
      dp.dumpFileList();

      cout << "About to start parsing..." << endl;

      myP->parse(dp);

      cout << "Completed Parser" << endl;
  
      cout << endl << endl << "Start checking!" << endl << endl;
      cout << "Call DDCheckMaterials and other DDChecks." << endl;
      DDCheckMaterials(cout);

      cout << "======== Navigate a little bit  ======" << endl;
      try {
	DDCompactView cpv;
	DDExpandedView ev(cpv);
	ev.firstChild();
	ev.nextSibling();
	cout << ev.geoHistory() << endl;
	ev.nextSibling();
	cout << ev.geoHistory() << endl;
	ev.firstChild();
	cout << ev.geoHistory() << endl;
	ev.nextSibling();
	cout << ev.geoHistory() << endl;
	ev.nextSibling();
	cout << ev.geoHistory() << endl;
	ev.firstChild();
	cout << ev.geoHistory() << endl;
	ev.nextSibling();
	cout << ev.geoHistory() << endl;
	ev.firstChild();
	cout << ev.geoHistory() << endl;
	ev.nextSibling();
	cout << ev.geoHistory() << endl;
	ev.firstChild();
	cout << ev.geoHistory() << endl;
	ev.nextSibling();
	cout << ev.geoHistory() << endl;
      }
      catch (DDException& e) {
	cout << e.what() << endl;
      }

      cout << "--------------- Parser testing started --------------" << endl;
      cout << endl << "Run the XML tests." << endl;
      testMaterials();
      testRotations();
      testSolids();
      testLogicalParts();
      testPosParts();
    } else if (argc < 3) {

      //just to have something!
      DDRootDef::instance().set(DDName("LP1", "testNoSections"));

      string fname = string(argv[1]);
      DDLTestDoc dp;
      while (fname != "q") {
	cout << "about to try to process the file " << fname << endl;
	dp.push_back(fname);
	myP->parse(dp);
	cout << "next file name:" ;
	cin >> fname;
	dp.clear();
      }
    }
  }
  catch (DDException& e)
    {
      std::cerr << "DDD-PROBLEM:" << std::endl 
		<< e << std::endl;
    }  
  //  Deal with any exceptions that may have been thrown.
  catch (cms::Exception& e) {
    std::cout << "cms::Exception caught in "
	      << kProgramName
	      << "\n"
	      << e.explainSelf();
    rc = 1;
  }
  catch (std::exception& e) {
    std::cout << "Standard library exception caught in "
	      << kProgramName
	      << "\n"
	      << e.what();
    rc = 1;
  }
  catch (...) {
    std::cout << "Unknown exception caught in "
	      << kProgramName;
    rc = 2;
  }

  return rc;

}
