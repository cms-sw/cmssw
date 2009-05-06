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
// $Id: L1RCTChannelMaskTester.cc,v 1.1 2009/05/06 13:40:46 efron Exp $
//
//
// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DataRecord/interface/L1RCTChannelMaskRcd.h"
#include "CondFormats/L1TObjects/interface/L1RCTChannelMask.h"


#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"


#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"



using std::cout;
using std::endl;
//
// class declaration
//

class L1RCTChannelMaskTester : public edm::EDAnalyzer {
public:
  explicit L1RCTChannelMaskTester(const edm::ParameterSet&) {}
  virtual  ~L1RCTChannelMaskTester() {}
      virtual void analyze(const edm::Event&, const edm::EventSetup&);  

};



void L1RCTChannelMaskTester::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup)
{
  

  edm::ESHandle< L1RCTChannelMask > rctChanMask;
   evSetup.get< L1RCTChannelMaskRcd >().get( rctChanMask) ;


   rctChanMask->print(std::cout);


}


DEFINE_FWK_MODULE(L1RCTChannelMaskTester);



