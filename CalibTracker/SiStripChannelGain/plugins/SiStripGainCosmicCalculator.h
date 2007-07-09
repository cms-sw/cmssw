#ifndef CalibTracker_SiStripChannelGain_SiStripGainCosmicCalculator_h
#define CalibTracker_SiStripChannelGain_SiStripGainCosmicCalculator_h
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
// $Id: SiStripGainCosmicCalculator.h,v 1.1 2007/06/13 14:03:35 gbruno Exp $
//
//


#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include <vector>
#include "TH1F.h"
#include "TObjArray.h"
#include "TFile.h"
#include "TString.h"
#include <fstream>
#include <sstream>

class SiStripGainCosmicCalculator : public ConditionDBWriter<SiStripApvGain> {

public:

  explicit SiStripGainCosmicCalculator(const edm::ParameterSet&);
  ~SiStripGainCosmicCalculator();

private:

  void algoAnalyze(const edm::Event &, const edm::EventSetup &);

  SiStripApvGain * getNewObject();

  void bookHistos(const edm::EventSetup& iSetup);

  std::pair<double,double> getPeakOfLandau( TH1F * inputHisto );

  double moduleWidth(const uint32_t detid, const edm::EventSetup* iSetup);

  double moduleThickness(const uint32_t detid, const edm::EventSetup* iSetup);


private:

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
  bool outputHistogramsInRootFile;
  TString outputFileName ;


  //  DaqMonitorBEInterface* bei_;
  

  bool printdebug_;

};
#endif
