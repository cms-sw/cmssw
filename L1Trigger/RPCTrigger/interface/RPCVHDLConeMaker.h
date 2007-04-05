#ifndef RPCVHDLConeMaker_h
#define RPCVHDLConeMaker_h
// -*- C++ -*-
//
// Package:    RPCVHDLConeMaker
// Class:      RPCVHDLConeMaker
// 
/**\class RPCVHDLConeMaker RPCVHDLConeMaker.cc src/RPCVHDLConeMaker/src/RPCVHDLConeMaker.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Artur Kalinowski
//         Created:  Tue Aug  1 13:54:56 CEST 2006
// $Id$
//
//


// system include files
#include <memory>
#include <fstream>

// Framework include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// RPC trigger include files
#include "L1Trigger/RPCTrigger/interface/RPCTriggerGeo.h"
//
// class decleration
//

class RPCVHDLConeMaker : public edm::EDAnalyzer {
   public:
      explicit RPCVHDLConeMaker(const edm::ParameterSet&);
      ~RPCVHDLConeMaker();


      virtual void analyze(const edm::Event&, const edm::EventSetup&);
   private:

      void initRPCLinks(const edm::EventSetup& iSetup);
      void writeLogCones(int towMin, int towMax, int secMin, int secMax,
			 const edm::EventSetup& iSetup);
      void writeHeader(int aTower, int aSector, std::ofstream & out);
      void writeConesDef(int iTower, int aSector, std::ofstream & out, const edm::EventSetup& iSetup);
      void writeQualityDef(std::ofstream & out);
      void writePatternsDef(std::ofstream & out);
      void writeSorterDef(std::ofstream & out);
      void writeXMLPatternsDef(std::ofstream & out);

      // ----------member data ---------------------------
      bool RPCLinksDone;
      RPCTriggerGeo theLinksystem;  
      RPCRingFromRolls::RPCLinks aLinks;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

#endif
