#ifndef SiStripMonitorPedestals_SiStripMonitorPedestals_h
#define SiStripMonitorPedestals_SiStripMonitorPedestals_h
// -*- C++ -*-
//
// Package:     SiStripMonitorPedestals
// Class  :     SiStripMonitorPedestals
// 
/**\class SiStripMonitorPedestals SiStripMonitorPedestals.h 

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  gennai, dutta
//         Created:  Sat Feb  4 20:49:51 CET 2006
// $Id: SiStripMonitorPedestals.h,v 1.11 2007/06/01 17:19:26 dutta Exp $
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


// data formats
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"

// cabling
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"


#include "boost/cstdint.hpp"
#include <iomanip>
#include <string>

class ApvAnalysisFactory;
class MonitorElement;
class DaqMonitorBEInterface;
class SiStripDetCabling;

class SiStripMonitorPedestals : public edm::EDAnalyzer {
   public:
      explicit SiStripMonitorPedestals(const edm::ParameterSet&);
      ~SiStripMonitorPedestals();

      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void beginJob(edm::EventSetup const&) ;
      virtual void endJob() ;
      
   
 private:
       struct ModMEs{
	 MonitorElement* PedsPerStrip;
	 MonitorElement* PedsDistribution;
	 MonitorElement* PedsEvolution;

	 MonitorElement* CMSubNoisePerStrip;
	 MonitorElement* RawNoisePerStrip;
	 MonitorElement* CMSubNoiseProfile;
	 MonitorElement* RawNoiseProfile;
	 MonitorElement* NoisyStrips;
	 MonitorElement* NoisyStripDistribution;

	 MonitorElement* CMDistribution;
	 MonitorElement* CMSlopeDistribution;


	 //MonitorElements for CondDB data display
	 MonitorElement* PedsPerStripDB;
	 MonitorElement* CMSubNoisePerStripDB;
	 MonitorElement* NoisyStripsDB;
       };

       DaqMonitorBEInterface* dbe_;
       edm::ParameterSet conf_;
       std::map<uint32_t, ModMEs> PedMEs;
       edm::ESHandle<SiStripDetCabling> detcabling;
       edm::ParameterSet pedsPSet_;
       bool analyzed;
       bool firstEvent;

       //The following to be put inside the parametersets
       int16_t nEvUpdate_;
       int16_t signalCutPeds_;
       int16_t nEvTot_;
       int16_t nEvInit_;
       int nIteration_;
       ApvAnalysisFactory* apvFactory_;
       int  theEventInitNumber_; 
       int theEventIterNumber_;
       int NumCMstripsInGroup_;
       std::string outPutFileName;
};

#endif
