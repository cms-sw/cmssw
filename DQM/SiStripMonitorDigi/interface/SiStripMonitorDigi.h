#ifndef SiStripMonitorDigi_SiStripMonitorDigi_h
#define SiStripMonitorDigi_SiStripMonitorDigi_h
// -*- C++ -*-
// Package:     SiStripMonitorDigi
// Class  :     SiStripMonitorDigi
/**\class SiStripMonitorDigi SiStripMonitorDigi.h DQM/SiStripMonitorDigi/interface/SiStripMonitorDigi.h
   Data Quality Monitoring source of the Silicon Strip Tracker. Produces histograms related to digis.
*/
// Original Author:  dkcira
//         Created:  Sat Feb  4 20:49:51 CET 2006
// $Id: SiStripMonitorDigi.h,v 1.4 2007/05/08 21:37:27 dkcira Exp $
#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"
class SiStripMonitorDigi : public edm::EDAnalyzer {
   public:
      explicit SiStripMonitorDigi(const edm::ParameterSet&);
      ~SiStripMonitorDigi();
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void beginJob(edm::EventSetup const&) ;
      virtual void endJob() ;
      virtual void beginRun(const edm::Run&, const edm::EventSetup&);
      virtual void endRun(const edm::Run&, const edm::EventSetup&);
      struct ModMEs{
            MonitorElement* NumberOfDigis;
            MonitorElement* ADCsHottestStrip;
            MonitorElement* ADCsCoolestStrip;
            MonitorElement* DigiADCs;
            MonitorElement* StripOccupancy;
      };
  private:
      void FillStripOccupancy(MonitorElement*,  std::vector<uint16_t> &);
      void ResetME(MonitorElement* h1);
   private:
       DaqMonitorBEInterface* dbe_;
       edm::ParameterSet conf_;
       std::map<uint32_t, ModMEs> DigiMEs; // uint32_t me_type: 1=#digis/module; 2=adcs of hottest strip/module; 3= adcs of coolest strips/module.
       bool show_mechanical_structure_view, show_readout_view, show_control_view, select_all_detectors, calculate_strip_occupancy, reset_each_run;
};
#endif
