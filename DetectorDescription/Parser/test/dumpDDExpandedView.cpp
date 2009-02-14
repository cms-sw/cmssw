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
 ***************************************************************************/


#include <iostream>
#include <stdlib.h>
#include <fstream>

#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Parser/interface/FIPConfiguration.h"
#include "DetectorDescription/RegressionTest/interface/DDErrorDetection.h"
#include <boost/shared_ptr.hpp>
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/PluginManager/interface/ProblemTracker.h"
#include "FWCore/Utilities/interface/Presence.h"
#include "FWCore/PluginManager/interface/PresenceFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/MakeParameterSets.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Parser/interface/FIPConfiguration.h"
#include "DetectorDescription/Core/interface/DDMap.h"
#include "DetectorDescription/Core/interface/DDRoot.h"
#include "DetectorDescription/Core/interface/DDVector.h"
#include "DetectorDescription/Core/interface/DDNumeric.h"
#include "DetectorDescription/Core/interface/DDString.h"
#include "DetectorDescription/Algorithm/src/AlgoInit.h"
#include "DetectorDescription/Core/src/DDCheck.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/src/DDCheckMaterials.cc"
#include "DetectorDescription/Base/interface/DDException.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDExpandedNode.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/adjgraph.h"

int main(int argc, char *argv[])
{
  typedef DDCompactView::graph_type::const_adj_iterator adjl_iterator;

  std::cout << "main:: initialize" << std::endl;

  //  AlgoInit();

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
      "process x = {"
      "service = MessageLogger {"
      "untracked vstring destinations = {'infos.mlog','warnings.mlog'}"
      "untracked PSet infos = {"
      "untracked string threshold = 'INFO'"
      "untracked PSet default = {untracked int32 limit = 1000000}"
      "untracked PSet FwkJob = {untracked int32 limit = 0}"
      "}"
      "untracked PSet warnings = {"
      "untracked string threshold = 'WARNING'"
      "untracked PSet default = {untracked int32 limit = 1000000}"
      "}"
      "untracked vstring fwkJobReports = {'FrameworkJobReport.xml'}"
      "untracked vstring categories = {'FwkJob'}"
      "untracked PSet FrameworkJobReport.xml = {"
      "untracked PSet default = {untracked int32 limit = 0}"
      "untracked PSet FwkJob = {untracked int32 limit = 10000000}"
      "}"
      "}"
      "service = JobReportService{}"
      "service = SiteLocalConfigService{}"
      "}";


    boost::shared_ptr<std::vector<edm::ParameterSet> > pServiceSets;
    boost::shared_ptr<edm::ParameterSet>          params_;
    edm::makeParameterSets(config, params_, pServiceSets);

    // D.  Create the services.
    edm::ServiceToken tempToken(edm::ServiceRegistry::createSet(*pServiceSets.get()));

    // E.  Make the services available.
    edm::ServiceRegistry::Operate operate(tempToken);

    // END Copy from example stand-alone program in Message Logger July 18, 2007

    std::cout << "main::initialize DDL parser" << std::endl;
    DDLParser* myP = DDLParser::instance();

    //   std::cout << "main:: about to start parsing field configuration..." << std::endl;
    //   FIPConfiguration dp2;
    //   dp2.readConfig("Geometry/CMSCommonData/data/FieldConfiguration.xml");
    //   myP->parse(dp2);

    std::cout << "main::about to start parsing main configuration... " << std::endl;
    FIPConfiguration dp;
    dp.readConfig("DetectorDescription/Parser/test/cmsIdealGeometryXML.xml");
    myP->parse(dp);
  
    std::cout << "main::completed Parser" << std::endl;

    std::cout << std::endl << std::endl << "main::Start checking!" << std::endl << std::endl;
    DDCheckMaterials(std::cout);

    DDCompactView cpv;
    //  cpv.setRoot(DDLogicalPart(DDName("cms:World")));

    std::cout << "edge size of produce graph:" << cpv.writeableGraph().edge_size() << std::endl;

    DDExpandedView ev(cpv);
    std::cout << "== got the epv ==" << std::endl;
    // for now just count!
    std::ofstream plist("plist.out");
    int numPhysParts(0);    
    while ( ev.next() ) {
      ++numPhysParts;
      plist << ev.geoHistory() << std::endl;
    }
    plist.close();
    std::cout << "Traversing the tree went to " << numPhysParts << " nodes, or \"PhysicalParts\" in online db terms." << std::endl;
    cpv.writeableGraph().clear();
    cpv.clear();
    std::cout << "cleared DDCompactView.  " << std::endl;

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
