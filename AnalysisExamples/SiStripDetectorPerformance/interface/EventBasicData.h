// Author : Samvel Khalatian (samvel at fnal dot gov)
// Created: 04/10/07
// Licence: GPL

#ifndef ANALYSISEXAMPLES_SISTRIPDETECTORPERFORMANCE_EVENTBASICDATA_H
#define ANALYSISEXAMPLES_SISTRIPDETECTORPERFORMANCE_EVENTBASICDATA_H

#include <memory>
#include <string>

#include "FWCore/Framework/interface/EDAnalyzer.h"

// Save Compile time by forwarding declarations
#include "FWCore/Framework/interface/Frameworkfwd.h"

class TFile;
class TTree;

class EventBasicData: public edm::EDAnalyzer {
  public:
    // Constructor
    EventBasicData( const edm::ParameterSet &roCONFIG);
    virtual ~EventBasicData() {}

  protected:
    // Leave possibility of inheritance
    virtual void beginJob( const edm::EventSetup &roEVENT_SETUP);
    virtual void analyze ( const edm::Event      &roEVENT,
                           const edm::EventSetup &roEVENT_SETUP);
    virtual void endJob  ();

  private:
    // Prevent objects copying
    EventBasicData( const EventBasicData &);
    EventBasicData &operator =( const EventBasicData &);

    std::string           oOFileName_;
    std::auto_ptr<TFile>  poOFile_;
    TTree                 *poGenTree_;

    // --[ Group General Variables ]-------------------------------------------
    struct GenVal {
      GenVal():
        nRun     ( -10),
        nLclEvent( -10),
        nLTime   ( -10)  {}

      int nRun;
      int nLclEvent;
      int nLTime;
    } oGenVal_;

}; // End EventBasicData class

#endif // ANALYSISEXAMPLES_SISTRIPDETECTORPERFORMANCE_EVENTBASICDATA_H
