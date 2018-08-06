#include <exception>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDPosData.h"
#include "DetectorDescription/RegressionTest/src/DDCheck.h"
#include "DetectorDescription/Core/src/Material.h"
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Parser/interface/FIPConfiguration.h"
#include "FWCore/PluginManager/interface/PresenceFactory.h"
#include "FWCore/PluginManager/interface/ProblemTracker.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/Presence.h"

int main(int argc, char *argv[])
{
  using Graph = DDCompactView::Graph;
  using adjl_iterator = Graph::const_adj_iterator;

  // Copied from example stand-alone program in Message Logger July 18, 2007
  std::string const kProgramName = argv[0];
  int rc = 0;

  try {

    // A.  Instantiate a plug-in manager first.
    edm::AssertHandler ah;

    // B.  Load the message service plug-in.  Forget this and bad things happen!
    //     In particular, the job hangs as soon as the output buffer fills up.
    //     That's because, without the message service, there is no mechanism for
    //     emptying the buffers.
    std::shared_ptr<edm::Presence> theMessageServicePresence;
    theMessageServicePresence = std::shared_ptr<edm::Presence>(edm::PresenceFactory::get()->
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

    // D.  Create the services.
    edm::ServiceToken tempToken(edm::ServiceRegistry::createServicesFromConfig(config));

    // E.  Make the services available.
    edm::ServiceRegistry::Operate operate(tempToken);

    // END Copy from example stand-alone program in Message Logger July 18, 2007

    std::cout << "main::initialize DDL parser" << std::endl;
    DDCompactView cpv;
    DDLParser myP(cpv);// = DDLParser::instance();

    //   std::cout << "main:: about to start parsing field configuration..." << std::endl;
    //   FIPConfiguration dp2;
    //   dp2.readConfig("Geometry/CMSCommonData/data/FieldConfiguration.xml");
    //   myP->parse(dp2);

    std::cout << "main::about to start parsing main configuration... " << std::endl;
    FIPConfiguration dp(cpv);
    dp.readConfig("DetectorDescription/Parser/test/cmsIdealGeometryXML.xml");
    myP.parse(dp);
  
    std::cout << "main::completed Parser" << std::endl;

    std::cout << std::endl << std::endl << "main::Start checking!" << std::endl << std::endl;
    DDCheckMaterials(std::cout);

    //  cpv.setRoot(DDLogicalPart(DDName("cms:World")));

    std::cout << "edge size of produce graph:" << cpv.graph().edge_size() << std::endl;
    const auto& gt = cpv.graph();
    adjl_iterator git = gt.begin();
    adjl_iterator gend = gt.end();    

    Graph::index_type i=0;
    for (; git != gend; ++git) {
      const DDLogicalPart & ddLP = gt.nodeData(git);
      std::cout << ++i << " P " << ddLP.name() << std::endl;
      if (!git->empty()) { 
	auto cit  = git->begin();
	auto cend = git->end();
	for (; cit != cend; ++cit) {
	  const DDLogicalPart & ddcurLP = gt.nodeData(cit->first);
	  std::cout << ++i << " c--> " << gt.edgeData(cit->second)->copyno() << " " << ddcurLP.name() << std::endl;
	}
      }
    }
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
