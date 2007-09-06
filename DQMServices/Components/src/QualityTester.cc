/*
 * \file QualityTester.cc
 * 
 * $Date: 2007/09/06 13:21:32 $
 * $Revision: 1.5 $
 * \author M. Zanetti - CERN PH
 *
 */

#include "DQMServices/Components/interface/QualityTester.h"

// Framework
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <DQMServices/UI/interface/MonitorUIRoot.h>
#include "DQMServices/ClientConfig/interface/QTestHandle.h"


#include <iostream>
#include <stdio.h>
#include <string>
#include <sstream>
#include <math.h>

using namespace edm;
using namespace std;


QualityTester::QualityTester(const ParameterSet& ps){
  
  prescaleFactor = ps.getUntrackedParameter<int>("prescaleFactor", 1);
  getQualityTestsFromFile = ps.getUntrackedParameter<bool>("getQualityTestsFromFile", true);

  mui = new MonitorUIRoot();
  bei = mui->getBEInterface();

  qtHandler=new QTestHandle;

  // if you use this module, it's non-sense not to provide the QualityTests.xml
  if (getQualityTestsFromFile)
    qtHandler->configureTests(ps.getUntrackedParameter<string>("qtList", "QualityTests.xml"),bei);
  
}


QualityTester::~QualityTester() { 
 
  delete mui;
  delete qtHandler;

}


//void QualityTester::analyze(const Event& e, const EventSetup& c){

void QualityTester::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {

  if (getQualityTestsFromFile && lumiSeg.id().luminosityBlock()%prescaleFactor == 0) {

    // always needed..
    mui->doMonitoring();

    // done here because new ME can appear while processing data
    qtHandler->attachTests(bei);

    edm::LogVerbatim ("QualityTester") <<"Running the Quality Test";    

    bei->runQTests();

  }

}
