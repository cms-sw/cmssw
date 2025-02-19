// -*- C++ -*-
//
// Package:    CalibTracker/SiStripDCS/test/Synchronization
// Class:      SyncDCSO2O
// 
/**\class SyncDCSO2O SyncDCSO2O.cc

 Description: Produces a histogram with the digi occupancy vs time and the change of payload.

*/
//
// Original Author:  Marco DE MATTIA
//         Created:  Tue Feb  9 15:38:18 CET 2010
// $Id: SyncDCSO2O.h,v 1.2 2010/03/29 12:32:38 demattia Exp $
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

#include "DataFormats/Common/interface/DetSetNew.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"

#include <vector>

#include "TH1F.h"
#include "TGraph.h"

class SyncDCSO2O : public edm::EDAnalyzer {
public:
  explicit SyncDCSO2O(const edm::ParameterSet&);
  ~SyncDCSO2O();

private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  void getDigis(const edm::Event& iEvent);
  /// Build TGraphs with quantity vs time
  TGraph * buildGraph(TH1F * histo, Float_t * timeArray);

  // ----------member data ---------------------------
  edm::Handle< edm::DetSetVector<SiStripDigi> > digiDetsetVector_[4];
  typedef std::vector<edm::ParameterSet> Parameters;
  Parameters digiProducersList_;

  struct TimeInfo
  {
    TimeInfo(unsigned long long inputTime, unsigned int inputDigiOccupancy, unsigned int inputDigiOccupancyWithMasking, unsigned int inputHVoff) :
      time(inputTime), digiOccupancy(inputDigiOccupancy), digiOccupancyWithMasking(inputDigiOccupancyWithMasking), HVoff(inputHVoff)
    {}

    unsigned long long time;
    unsigned int digiOccupancy;
    unsigned int digiOccupancyWithMasking;
    unsigned int HVoff;
  };

  struct SortByTime
  {
    bool operator()(const TimeInfo & timeInfo1, const TimeInfo & timeInfo2)
    {
      return( timeInfo1.time < timeInfo2.time );
    }
  };

  std::vector<TimeInfo> timeInfo_;
};
