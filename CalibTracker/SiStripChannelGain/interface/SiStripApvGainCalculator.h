// -*- C++ -*-
//
// Package:    SiStripApvGainCalculator
// Class:      SiStripApvGainCalculator
// 
/**\class SiStripApvGainCalculator SiStripApvGainCalculator.cc CalibTracker/SiStripChannelGain/src/SiStripApvGainCalculator.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Dorian Kcira, Pierre Rodeghiero
//         Created:  Mon Nov 20 10:04:31 CET 2006
// $Id: SiStripApvGainCalculator.h,v 1.1 2006/12/07 18:18:18 dkcira Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CalibTracker/SiStripChannelGain/interface/TrackLocalAngle.h"


#include "TH1F.h"
#include "TObjArray.h"
//
// class decleration
//

class SiStripApvGainCalculator : public edm::EDAnalyzer {
   public:
      explicit SiStripApvGainCalculator(const edm::ParameterSet&);
      ~SiStripApvGainCalculator();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      std::pair<double,double> getPeakOfLandau( TH1F * inputHisto );
      double moduleWidth(const uint32_t detid, const edm::EventSetup* iSetup);
      double moduleThickness(const uint32_t detid, const edm::EventSetup* iSetup);


      // ----------member data ---------------------------
      edm::ParameterSet conf_;
      TrackLocalAngle *anglefinder_;
      TObjArray * HlistAPVPairs;
      TObjArray * HlistOtherHistos;
      std::string TrackProducer;
      std::string TrackLabel;
      uint32_t total_nr_of_events;
      double ExpectedChargeDeposition;
      std::map<uint32_t, double> thickness_map; // map of detector id to respective thickness
      std::vector<uint32_t> SelectedDetIds;
      std::vector<uint32_t> detModulesToBeExcluded;
      const edm::EventSetup * eventSetupCopy_;
      unsigned int MinNrEntries;
      double MaxChi2OverNDF;

};
