/*
 * \file QualityTester.cc
 * 
 * $Date: 2007/04/16 22:08:22 $
 * $Revision: 1.2.2.1 $
 * \author M. Zanetti - CERN PH
 *
 */

#include "QualityTester.h"

// Framework
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>


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
  
  parameters = ps;

  nevents = 0;

  mui = new MonitorUIRoot();

  qtHandler=new QTestHandle;

  // if you use this module, it's non-sense not to provide the QualityTests.xml
  if (parameters.getUntrackedParameter<bool>("getQualityTestsFromFile", true))
    qtHandler->configureTests(parameters.getUntrackedParameter<string>("qtList", "QualityTests.xml"),mui);
  
}


QualityTester::~QualityTester() { 
 
  delete mui;
  delete qtHandler;

}


void QualityTester::analyze(const Event& e, const EventSetup& c){

  nevents++;

  // run QT test every "QualityTestPrescaler" event
  if (parameters.getUntrackedParameter<bool>("getQualityTestsFromFile", true) &&
      nevents%parameters.getUntrackedParameter<int>("QualityTestPrescaler", 1000) == 0) {

    // always needed..
    mui->doMonitoring();

    // done here because new ME can appear while processing data
    qtHandler->attachTests(mui);

    mui->runQTests();

  }

}



