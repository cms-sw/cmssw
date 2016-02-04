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

#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Parser/interface/DDLDocumentProvider.h"
#include "DetectorDescription/Parser/interface/DDLSAX2ConfigHandler.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Algorithm/src/AlgoInit.h"
#include "DetectorDescription/Core/src/DDCheck.h"
#include "DetectorDescription/Core/src/DDCheckMaterials.cc"
#include "DetectorDescription/Base/interface/DDException.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDExpandedNode.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDRoot.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Parser/src/StrX.h"
#include "DetectorDescription/Core/src/Material.h"

class DDLTestDoc : public DDLDocumentProvider
{
public:

  DDLTestDoc( void );
  virtual ~DDLTestDoc();

  /// Return a list of files as a vector of strings.
  virtual const std::vector < std::string >&  getFileList( void ) const;

  /// Return a list of urls as a vector of strings.
  virtual const std::vector < std::string >&  getURLList( void ) const;

  /// Print out the list of files.
  virtual void dumpFileList( void ) const;

  /// Return whether Validation should be on or off and where the DDL SchemaLocation is.
  virtual bool doValidation( void ) const;

  /// Return the designation for where to look for the schema.
  std::string getSchemaLocation( void ) const;

  /// ReadConfig
  virtual int readConfig( const std::string& filename );

  void push_back( std::string fileName, std::string url = std::string( "./" ));

  void setSchemaLocation( std::string path = std::string( "../../DDSchema" ));

  void setValidation( bool val );

  void clear( void );

  // test routines
  //   void testRotations();

  //    void testMaterials();

  //    void testSolids();

  //    void testLogicalParts();

  //    void testPosParts();

  //    void testAlgoPosParts();

  //    void testAlgorithm();

private:
  std::vector < std::string > fnames_;
  std::vector < std::string > urls_;
  std::string schemaLoc_;
  bool validate_;
};

//--------------------------------------------------------------------------
//  DDLTestDoc:  Default constructor and destructor.
//--------------------------------------------------------------------------
DDLTestDoc::~DDLTestDoc( void )
{ 
}

DDLTestDoc::DDLTestDoc( void )
  : validate_(true)
{ 
  schemaLoc_ = "http://www.cern.ch/cms/DDL ../../Schema/DDLSchema.xsd";
}

const std::vector<std::string>&
DDLTestDoc::getFileList( void ) const
{
  return fnames_;
}

const std::vector<std::string>&
DDLTestDoc::getURLList( void ) const
{
  return urls_;
}

void
DDLTestDoc::push_back( std::string fileName, std::string url ) 
{
  fnames_.push_back(fileName);
  urls_.push_back(url);
}

void
DDLTestDoc::setValidation( bool val )
{ validate_= val; }

bool
DDLTestDoc::doValidation( void ) const
{ return validate_; }

void
DDLTestDoc::setSchemaLocation( std::string path )
{ schemaLoc_ = path; }

std::string
DDLTestDoc::getSchemaLocation( void ) const
{ std::cout << schemaLoc_ << std::endl; return schemaLoc_; }

void
DDLTestDoc::dumpFileList( void ) const
{
  std::cout << "File List:" << std::endl;
  std::vector<std::string> vst = getFileList();
  std::cout << "  number of files=" << vst.size() << std::endl;
  for (std::vector<std::string>::const_iterator it = vst.begin(); it != vst.end(); ++it)
    std::cout << *it << std::endl;
}

void
DDLTestDoc::clear( void )
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
int
DDLTestDoc::readConfig( const std::string& filename )
{

  std::cout << "readConfig" << std::endl;
  //  configFileName_ = filename;

  // Set the parser to use the handler for the configuration file.
  // This makes sure the Parser is initialized and gets a handle to it.
  DDCompactView cpv;
  DDLParser parser(cpv);// = DDLParser::instance();
  DDLSAX2Handler* errHandler;
  DDLSAX2ConfigHandler * sch;

  sch = new DDLSAX2ConfigHandler(cpv);
  errHandler = new DDLSAX2Handler;

  parser.getXMLParser()->setContentHandler(sch);
  parser.getXMLParser()->setErrorHandler(errHandler);

  try {
    parser.getXMLParser()->parse(filename.c_str());
  }
  catch (const XERCES_CPP_NAMESPACE::XMLException& toCatch) {
    std::cout << "\nXMLException: parsing '" << filename << "'\n"
	      << "Exception message is: \n"
	      << std::string(StrX(toCatch.getMessage()).localForm()) << "\n" ;
    return 1;
  }
  catch (...)
  {
    std::cout << "\nUnexpected exception during parsing: '" << filename << "'\n";
    return 3;
  }

  fnames_ = sch->getFileNames();
  urls_ = sch->getURLs();
  std::cout << "there are " << fnames_.size() << " files." << std::endl;
  for (size_t i = 0; i < fnames_.size(); ++i)
    //std::cout << "url=" << sch->getURLs()[i] << " file=" << sch->getFileNames()[i] << std::endl;
    std::cout << "url=" << urls_[i] << " file=" << fnames_[i] << std::endl;
  return 0;
}


void
testRotations( void )
{
  std::cout << "--------------- Parser testing Rotations --------------" << std::endl;
  std::cout << "z30 should be a rotation of 30 degrees around the z axis:" << std::endl;
  std::cout << DDRotation(DDName( "z30", "testRotations")) << std::endl;
  std::cout << std::endl;
  std::cout << "z30x20 should be a rotation 30 degrees around z, then 20 degrees around x:" << std::endl;
  std::cout << DDRotation(DDName( "z30x20", "testRotations")) << std::endl;
  std::cout << std::endl;
  std::cout << "x90y45 should be a rotation 90 degrees around x, then 45 degrees around y:" << std::endl;
  std::cout << DDRotation(DDName( "x90y45", "testRotations")) << std::endl;
  std::cout << std::endl;
  std::cout << "x90y90 should be a rotation 90 degrees around x, then 90 degrees around y:" << std::endl;
  std::cout << DDRotation(DDName( "x90y90", "testRotations")) << std::endl;
  std::cout << std::endl;
  std::cout << "x90y135 should be a rotation 90 degrees around x, then 135 degrees around y:" << std::endl;
  std::cout << DDRotation(DDName( "x90y135", "testRotations")) << std::endl;
  std::cout << std::endl;
  std::cout << "x90y180 should be a rotation 90 degrees around x, then 180 degrees around y:" << std::endl;
  std::cout << DDRotation(DDName( "x90y180", "testRotations")) << std::endl;
  std::cout << std::endl;
  std::cout << "x90 should be a rotation of 90 degrees around the x axis:" << std::endl;
  std::cout << DDRotation(DDName( "x90", "testRotations")) << std::endl;
  std::cout << std::endl;
  std::cout << "cmsimIdentity makes the identity rotation matrix using the cmsim method (phi and theta of each axis):" << std::endl;
  std::cout << DDRotation(DDName("cmsimIdentity", "testRotations")) << std::endl;
  std::cout << std::endl;
  std::cout << "newrotIdentity makes the identity rotation matrix by rotating 0 degrees around the z axis:" << std::endl;
  std::cout << DDRotation(DDName("newrotIdentity", "testRotations")) << std::endl;
  std::cout << std::endl;
  std::cout << "180R should be a REFLECTION rotation.  It is defined using the cmsim way:" << std::endl;
  std::cout << DDRotation(DDName("180R", "testRotations")) << std::endl;
  std::cout << std::endl;
}

void testMaterials() { 
  std::cout << "--------------- Parser testing Materials --------------" << std::endl;
  std::cout << "There should be 4 Elementary Materials: Nitrogen," << std::endl;
  std::cout << "Oxygen,Argon and Hydrogen.  There should be one composite" << std::endl;
  std::cout << " material:Air, made up of those 4 components." << std::endl;
  std::cout << DDMaterial(DDName("Nitrogen", "testMaterials")) << std::endl;
  std::cout << std::endl;
  std::cout << DDMaterial(DDName("Oxygen", "testMaterials")) << std::endl;
  std::cout << std::endl;
  std::cout << DDMaterial(DDName("Argon", "testMaterials")) << std::endl;
  std::cout << std::endl;
  std::cout << DDMaterial(DDName("Hydrogen", "testMaterials")) << std::endl;
  std::cout << std::endl;
  std::cout << DDMaterial(DDName("Air", "testMaterials")) << std::endl;
  std::cout << std::endl;
}

void
testSolids( void )
{ 
  std::cout << "--------------- Parser testing Solids --------------" << std::endl;
  std::cout << "trap1 is a trapezoid:" << std::endl;
  std::cout << DDSolid(DDName("trap1", "testSolids")) << std::endl;
  std::cout << std::endl;
  std::cout << "trap2 is a trapezoid with more symmetry:" << std::endl;
  std::cout << DDSolid(DDName("trap2", "testSolids")) << std::endl;
  std::cout << std::endl;
  std::cout << "ptrap1 is a pseudo Trapezoid with atMinusZ=false:" << std::endl;
  std::cout << DDSolid(DDName("ptrap1", "testSolids")) << std::endl;
  std::cout << std::endl;
  std::cout << "ptrap2 is a psuedo Trapezoid with atMinusZ=true:" << std::endl;
  std::cout << DDSolid(DDName("ptrap2", "testSolids")) << std::endl;
  std::cout << std::endl;
  std::cout << "box1 is a Box:" << std::endl;
  std::cout << DDSolid(DDName("box1", "testSolids")) << std::endl;
  std::cout << std::endl;
  std::cout << "cone1 is a conical section (full 360 degree):" << std::endl;
  std::cout << DDSolid(DDName("cone1", "testSolids")) << std::endl;
  std::cout << std::endl;
  std::cout << "cone2 is a conical section (full 360 degree) which is actually a tube:" << std::endl;
  std::cout << DDSolid(DDName("cone2", "testSolids")) << std::endl;
  std::cout << std::endl;
  std::cout << "cone2hole is a conical section (20 degree):" << std::endl;
  std::cout << DDSolid(DDName("cone2hole", "testSolids")) << std::endl;
  std::cout << std::endl;
  std::cout << "pczsect is a polycone defined using z-sections:" << std::endl;
  std::cout << DDSolid(DDName("pczsect", "testSolids")) << std::endl;
  std::cout << std::endl;
  std::cout << "pcrz is a polycone defined using r & z points:" << std::endl;
  std::cout << DDSolid(DDName("pcrz", "testSolids")) << std::endl;
  std::cout << std::endl;
  std::cout << "phzsect is a polyhedra defined using z-sections:" << std::endl;
  std::cout << DDSolid(DDName("phzsect", "testSolids")) << std::endl;
  std::cout << std::endl;
  std::cout << "phrz is a polyhedra defined using r & z points:" << std::endl;
  std::cout << DDSolid(DDName("phrz", "testSolids")) << std::endl;
  std::cout << std::endl;
  std::cout << "trd1 is a \"special\" trapezoid declaration with fewer" ;
  std::cout << " parameters (Trd1):" << std::endl;
  std::cout << DDSolid(DDName("trd1", "testSolids")) << std::endl;
  std::cout << std::endl;
  std::cout << "trd2 is a \"special\" trapezoid declaration with fewer" ;
  std::cout << " parameters (Trd1):" << std::endl;
  std::cout << DDSolid(DDName("trd2", "testSolids")) << std::endl;
  std::cout << std::endl;
  std::cout << "tube1 is a tube:" << std::endl;
  std::cout << DDSolid(DDName("tube1", "testSolids")) << std::endl;
  std::cout << std::endl;
  std::cout << "tube2 is a tubs(Tube Section):" << std::endl;
  std::cout << DDSolid(DDName("tube2", "testSolids")) << std::endl;
  std::cout << std::endl;
  std::cout << "trunctubs1 is a trunctubs(Cut or truncated Tube Section):" << std::endl;
  std::cout << DDSolid(DDName("trunctubs1", "testSolids")) << std::endl;
  std::cout << std::endl;
  std::cout << "momma is a shapeless solid, a way of \"grouping\" things:" << std::endl;
  std::cout << DDSolid(DDName("momma", "testSolids")) << std::endl;
  std::cout << std::endl;
  std::cout << "MotherOfAllBoxes is a box and is the root's solid:" << std::endl;
  std::cout << DDSolid(DDName("MotherOfAllBoxes", "testSolids")) << std::endl;
  std::cout << std::endl;
  std::cout << "trd2mirror is a ReflectionSolid of trd2:" << std::endl;
  std::cout << DDSolid(DDName("trd2mirror", "testSolids")) << std::endl;
  std::cout << std::endl;
  std::cout << "subsolid is a subtraction solid, cone2-cone2hole:" << std::endl;
  std::cout << DDSolid(DDName("subsolid", "testSolids")) << std::endl;
  std::cout << std::endl;
  std::cout << "unionsolid is a union of pcrz and cone1:" << std::endl;
  std::cout << DDSolid(DDName("unionsolid", "testSolids")) << std::endl;
  std::cout << std::endl;
  std::cout << "intsolid is an Intersection(Solid) of cone1 and cone2:" << std::endl;
  std::cout << DDSolid(DDName("intsolid", "testSolids")) << std::endl;
  std::cout << std::endl;
}

void
testLogicalParts( void )
{ 
  std::cout << "--------------- Parser testing LogicalParts --------------" << std::endl;
  std::cout << "LogicalPart trap1:" << std::endl;
  std::cout << DDLogicalPart(DDName("trap1", "testLogicalParts")) << std::endl;
  std::cout << std::endl;
  std::cout << "LogicalPart trap2:" << std::endl;
  std::cout << DDLogicalPart(DDName("trap2", "testLogicalParts")) << std::endl;
  std::cout << std::endl;
  std::cout << "LogicalPart ptrap1:" << std::endl;
  std::cout << DDLogicalPart(DDName("ptrap1", "testLogicalParts")) << std::endl;
  std::cout << std::endl;
  std::cout << "LogicalPart ptrap2:" << std::endl;
  std::cout << DDLogicalPart(DDName("ptrap2", "testLogicalParts")) << std::endl;
  std::cout << std::endl;
  std::cout << "LogicalPart box1:" << std::endl;
  std::cout << DDLogicalPart(DDName("box1", "testLogicalParts")) << std::endl;
  std::cout << std::endl;
  std::cout << "LogicalPart cone1:" << std::endl;
  std::cout << DDLogicalPart(DDName("cone1", "testLogicalParts")) << std::endl;
  std::cout << std::endl;
  std::cout << "LogicalPart cone2:" << std::endl;
  std::cout << DDLogicalPart(DDName("cone2", "testLogicalParts")) << std::endl;
  std::cout << std::endl;
  std::cout << "LogicalPart cone2hole:" << std::endl;
  std::cout << DDLogicalPart(DDName("cone2hole", "testLogicalParts")) << std::endl;
  std::cout << std::endl;
  std::cout << "LogicalPart pczsect:" << std::endl;
  std::cout << DDLogicalPart(DDName("pczsect", "testLogicalParts")) << std::endl;
  std::cout << std::endl;
  std::cout << "LogicalPart pcrz:" << std::endl;
  std::cout << DDLogicalPart(DDName("pcrz", "testLogicalParts")) << std::endl;
  std::cout << std::endl;
  std::cout << "LogicalPart phzsect:" << std::endl;
  std::cout << DDLogicalPart(DDName("phzsect", "testLogicalParts")) << std::endl;
  std::cout << std::endl;
  std::cout << "LogicalPart phrz:" << std::endl;
  std::cout << DDLogicalPart(DDName("phrz", "testLogicalParts")) << std::endl;
  std::cout << std::endl;
  std::cout << "LogicalPart trd1:" ;
  std::cout << DDLogicalPart(DDName("trd1", "testLogicalParts")) << std::endl;
  std::cout << std::endl;
  std::cout << "LogicalPart trd2:" << std::endl;
  std::cout << DDLogicalPart(DDName("trd2", "testLogicalParts")) << std::endl;
  std::cout << std::endl;
  std::cout << "LogicalPart tube1:" << std::endl;
  std::cout << DDLogicalPart(DDName("tube1", "testLogicalParts")) << std::endl;
  std::cout << std::endl;
  std::cout << "LogicalPart tube2:" << std::endl;
  std::cout << DDLogicalPart(DDName("tube2", "testLogicalParts")) << std::endl;
  std::cout << std::endl;
  std::cout << "LogicalPart trunctubs1:" << std::endl;
  std::cout << DDLogicalPart(DDName("trunctubs1", "testLogicalParts")) << std::endl;
  std::cout << std::endl;
  std::cout << "LogicalPart momma:" << std::endl;
  std::cout << DDLogicalPart(DDName("momma", "testLogicalParts")) << std::endl;
  std::cout << std::endl;
  std::cout << "LogicalPart MotherOfAllBoxes:" << std::endl;
  std::cout << DDLogicalPart(DDName("MotherOfAllBoxes", "testLogicalParts")) << std::endl;
  std::cout << std::endl;
  std::cout << "LogicalPart torus:" << std::endl;
  std::cout << DDLogicalPart(DDName("torus", "testLogicalParts")) << std::endl;
  std::cout << std::endl;
  std::cout << "LogicalPart trd2mirror:" << std::endl;
  std::cout << DDLogicalPart(DDName("trd2mirror", "testLogicalParts")) << std::endl;
  std::cout << std::endl;
  std::cout << "LogicalPart subsolid:" << std::endl;
  std::cout << DDLogicalPart(DDName("subsolid", "testLogicalParts")) << std::endl;
  std::cout << std::endl;
  std::cout << "LogicalPart unionsolid:" << std::endl;
  std::cout << DDLogicalPart(DDName("unionsolid", "testLogicalParts")) << std::endl;
  std::cout << std::endl;
  std::cout << "LogicalPart intsolid:" << std::endl;
  std::cout << DDLogicalPart(DDName("intsolid", "testLogicalParts")) << std::endl;
  std::cout << std::endl;
}

void
testPosParts( void )
{}

void
testAlgoPosParts( void )
{}

void
testAlgorithm( void )
{}

int main(int argc, char *argv[])
{
  // MEC: 2008-08-04 : I believe the main problem w/ this is not being framework fristd::endly.
  // so I'm (over) using the "main" of CMSSW

  std::string const kProgramName = argv[0];
  int rc = 0;
  if (argc < 2 || argc > 2 ) {
    std::cout << "This is a polite exit so that scram b runtests' first run of this program does not give an error" << std::endl;
    exit(0); // SUCCESS;
  }
  // Copied from example stand-alone program in Message Logger July 18, 2007
  try {

    //     // A.  Instantiate a plug-in manager first.
    //     edm::AssertHandler ah;

    //     // B.  Load the message service plug-in.  Forget this and bad things happen!
    //     //     In particular, the job hangs as soon as the output buffer fills up.
    //     //     That's because, without the message service, there is no mechanism for
    //     //     emptying the buffers.
    //     boost::shared_ptr<edm::Presence> theMessageServicePresence;
    //     theMessageServicePresence = boost::shared_ptr<edm::Presence>(edm::PresenceFactory::get()->
    // 								 makePresence("MessageServicePresence").release());

    //     // C.  Manufacture a configuration and establish it.
    //     std::string config =
    //       "import FWCore.ParameterSet.Config as cms\n"
    //       "process = cms.Process('TEST')\n"
    //       "process.maxEvents = cms.untracked.PSet(\n"
    //       "    input = cms.untracked.int32(5)\n"
    //       ")\n"
    //       "process.source = cms.Source('EmptySource')\n"
    //       "process.JobReportService = cms.Service('JobReportService')\n"
    //       "process.InitRootHandlers = cms.Service('InitRootHandlers')\n"
    //       // "process.MessageLogger = cms.Service('MessageLogger')\n"
    //       "process.m1 = cms.EDProducer('IntProducer',\n"
    //       "    ivalue = cms.int32(11)\n"
    //       ")\n"
    //       "process.out = cms.OutputModule('PoolOutputModule',\n"
    //       "    fileName = cms.untracked.string('testStandalone.root')\n"
    //       ")\n"
    //       "process.p = cms.Path(process.m1)\n"
    //       "process.e = cms.EndPath(process.out)\n";

    //     // D.  Create the services.
    //     edm::ServiceToken tempToken(edm::ServiceRegistry::createServicesFromConfig(config));

    //     // E.  Make the services available.
    //     edm::ServiceRegistry::Operate operate(tempToken);

    //     // END Copy from example stand-alone program in Message Logger July 18, 2007

    std::cout  << "Initialize DDD (call AlgoInit)" << std::endl;

    AlgoInit();

    std::cout << "Initialize a DDL parser " << std::endl;
    DDCompactView cpv;
    DDLParser myP(cpv);// = DDLParser::instance();
    //until scram b runtests (if ever) does not run it by default, we will not run without at least one argument.
    //    if (argc < 2) {
    //       std::cout << "DEFAULT test using testConfiguration.xml" << std::endl;
    if ( argc == 2 ) {
      DDLTestDoc dp; //DDLConfiguration dp;

      dp.readConfig(argv[1]);
      dp.dumpFileList();

      std::cout << "About to start parsing..." << std::endl;

      myP.parse(dp);

      std::cout << "Completed Parser" << std::endl;
  
      std::cout << std::endl << std::endl << "Start checking!" << std::endl << std::endl;
      std::cout << "Call DDCheckMaterials and other DDChecks." << std::endl;
      DDCheckMaterials(std::cout);

      std::cout << "======== Navigate a little bit  ======" << std::endl;
      try {
	//	DDCompactView cpv;
	// 2010::FIX Once I get rid of DDRootDef then this problem goes away!
	if (!cpv.root().isDefined().second) {
	  cpv.setRoot(DDRootDef::instance().root());
	}
	DDExpandedView ev(cpv);
	while (ev.next()) {
	  std::cout << ev.geoHistory() << std::endl;
	}
	// 	  if (ev.firstChild()) {
	// 	    ev.firstChild();
	// 	    ev.nextSibling();
	// 	    std::cout << ev.geoHistory() << std::endl;
	// 	    ev.nextSibling();
	// 	    std::cout << ev.geoHistory() << std::endl;
	// 	    ev.firstChild();
	// 	    std::cout << ev.geoHistory() << std::endl;
	// 	    ev.nextSibling();
	// 	    std::cout << ev.geoHistory() << std::endl;
	// 	    ev.nextSibling();
	// 	    std::cout << ev.geoHistory() << std::endl;
	// 	    ev.firstChild();
	// 	    std::cout << ev.geoHistory() << std::endl;
	// 	    ev.nextSibling();
	// 	    std::cout << ev.geoHistory() << std::endl;
	// 	    ev.firstChild();
	// 	    std::cout << ev.geoHistory() << std::endl;
	// 	    ev.nextSibling();
	// 	    std::cout << ev.geoHistory() << std::endl;
	// 	    ev.firstChild();
	// 	    std::cout << ev.geoHistory() << std::endl;
	// 	    ev.nextSibling();
	// 	    std::cout << ev.geoHistory() << std::endl;
	// 	  }
	// // 	} else {
	// // 	  cpv.setRoot(DDRootDef::instance().root());
	// // 	  std::cout << cpv.root() << std::endl;
	// 	} 
      }
      catch (DDException& e) {
	std::cout << e.what() << std::endl;
      }
      std::cout << "--------------- Parser testing started --------------" << std::endl;
      std::cout << std::endl << "Run the XML tests." << std::endl;
      testMaterials();
      testRotations();
      testSolids();
      testLogicalParts();
      testPosParts();
    } else if (argc < 3) {
      // scram b runtests for now this should not work.
      // just to have something!
      DDRootDef::instance().set(DDName("LP1", "testNoSections"));
      
      std::string fname = std::string(argv[1]);
      DDLTestDoc dp;
      while (fname != "q") {
	std::cout << "about to try to process the file " << fname << std::endl;
	dp.push_back(fname);
	myP.parse(dp);
	std::cout << "next file name:" ;
	std::cin >> fname;
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
  //   catch (...) {
  //     std::cout << "Unknown exception caught in "
  // 	      << kProgramName;
  //     rc = 2;
  //   }

  return rc;
}
