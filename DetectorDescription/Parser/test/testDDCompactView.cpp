/***************************************************************************
                          main.cpp  -  description
                             -------------------
    begin                : Wed Oct 24 17:36:15 PDT 2001
    copyright            : (C) 2001 by Michael Case
    email                : case@ucdhep.ucdavis.edu
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   mec: 2008-08-05 : need to make this an analyzer instead?              *
 *                     for now use the "main" of cmssw                     *
 ***************************************************************************/

#include <iostream>
#include <stdlib.h>
#include "FWCore/PluginManager/interface/ProblemTracker.h"
#include "FWCore/PluginManager/interface/PresenceFactory.h"
#include "FWCore/PythonParameterSet/interface/MakeParameterSets.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Parser/interface/FIPConfiguration.h"
#include "DetectorDescription/Algorithm/src/AlgoInit.h"
#include "DetectorDescription/Core/src/DDCheck.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDExpandedNode.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"

int main(int argc, char *argv[])
{
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

    // D.  Create the services.
    edm::ServiceToken tempToken(edm::ServiceRegistry::createServicesFromConfig(config));

    // E.  Make the services available.
    edm::ServiceRegistry::Operate operate(tempToken);

    // END Copy from example stand-alone program in Message Logger July 18, 2007

    std::cout << "main:: initialize" << std::endl;

    AlgoInit();

    std::cout << "main::initialize DDL parser" << std::endl;
    DDCompactView cpv;

    DDLParser myP(cpv); // = DDLParser::instance();

    FIPConfiguration dp(cpv);

    dp.readConfig("DetectorDescription/Parser/test/cmsIdealGeometryXML.xml");

    std::cout << "main::about to start parsing" << std::endl;
 
    myP.parse(dp);

    std::cout << "main::completed Parser" << std::endl;
  
    std::cout << std::endl << std::endl << "main::Start checking!" << std::endl << std::endl;
    DDCheckMaterials(std::cout);

    DDExpandedView ev(cpv);
    std::cout << "== got the epv ==" << std::endl;

    while ( ev.next() ) {
      if ( ev.logicalPart().name().name() == "MBAT" ) {
	std::cout << ev.geoHistory() << std::endl;
      }
      if ( ev.logicalPart().name().name() == "MUON" ) {
	std::cout << ev.geoHistory() << std::endl;
      }
    }
    //    cpv.clear();
    std::cout << "cleared DDCompactView.  " << std::endl;
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
