#ifndef Modules_EventContentAnalyzer_h
#define Modules_EventContentAnalyzer_h
// -*- C++ -*-
//
// Package:     Modules
// Class  :     EventContentAnalyzer
// 
/**\class EventContentAnalyzer EventContentAnalyzer.h FWCore/Modules/src/EventContentAnalyzer.h

 Description: prints out what data is contained within an Event at that point in the path

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Mon Sep 19 11:49:35 CEST 2005
// $Id: EventContentAnalyzer.h,v 1.2 2005/10/06 22:36:21 marafino Exp $
//

// system include files
#include <string>
#include <map>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"

// forward declarations

class EventContentAnalyzer : public edm::EDAnalyzer {
public:
   explicit EventContentAnalyzer(const edm::ParameterSet&);
   ~EventContentAnalyzer();
   
   virtual void analyze(const edm::Event&, const edm::EventSetup&);
   virtual void endJob();

private:

      // ----------member data ---------------------------
      std::string indentation_;
      int         evno_;
      std::map<std::string, int>  cumulates_;
};



#endif
