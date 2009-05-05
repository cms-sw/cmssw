// -*- C++ -*-
//
// Package:    RCTConfigTester
// Class:      RCTConfigTester
// 
/**\class RCTConfigTester RCTConfigTester.h L1TriggerConfig/RCTConfigTester/src/RCTConfigTester.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sridhara Dasu
//         Created:  Mon Jul 16 23:48:35 CEST 2007
// $Id: RCTConfigTester.cc,v 1.10 2008/07/31 14:13:46 lgray Exp $
//
//
// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DataRecord/interface/L1RCTParametersRcd.h"
#include "CondFormats/L1TObjects/interface/L1RCTParameters.h"
//#include "CondFormats/DataRecord/interface/L1RCTChannelMaskRcd.h"
//#include "CondFormats/L1TObjects/interface/L1RCTChannelMask.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"


using std::cout;
using std::endl;
//
// class declaration
//

class RCTConfigTester : public edm::EDAnalyzer {
public:
  explicit RCTConfigTester(const edm::ParameterSet&) {}
  virtual  ~RCTConfigTester() {}
      virtual void analyze(const edm::Event&, const edm::EventSetup&);  

};



void RCTConfigTester::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup)
{
  

  edm::ESHandle< L1RCTParameters > rctParam;
   evSetup.get< L1RCTParametersRcd >().get( rctParam) ;


   rctParam->print(std::cout);

   /*
   ESHandle< L1RCTChannelMask > rctMask;
   es.get< L1RCTChannelMaskRcd >().get( rctMask) ;

   cout << "L1RCTChannelMaskRcd :" << endl;
   //   rctMask->print(cout);
   cout << endl;
   */
}





