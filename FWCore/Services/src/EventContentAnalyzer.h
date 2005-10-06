#ifndef Services_EventContentAnalyzer_h
#define Services_EventContentAnalyzer_h
// -*- C++ -*-
//
// Package:     Services
// Class  :     EventContentAnalyzer
// 
/**\class EventContentAnalyzer EventContentAnalyzer.h FWCore/Services/src/EventContentAnalyzer.h

 Description: prints out what data is contained within an Event at that point in the path

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Mon Sep 19 11:49:35 CEST 2005
// $Id: EventContentAnalyzer.h,v 1.1 2005/09/19 12:16:05 chrjones Exp $
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
