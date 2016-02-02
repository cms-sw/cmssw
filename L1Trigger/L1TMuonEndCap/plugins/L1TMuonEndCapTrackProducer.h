#ifndef _L1ITMu_L1TMuonUpgradedTrackFinder_h_
#define _L1ITMu_L1TMuonUpgradedTrackFinder_h_
//asd
#include <memory>
#include <map>

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "L1Trigger/L1TMuon/interface/deprecate/SubsystemCollectorFactory.h"

#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include <TMath.h>
#include <TCanvas.h>
#include <TLorentzVector.h>

#include <TStyle.h>
#include <TLegend.h>
#include <TF1.h>
#include <TH2.h>
#include <TH1F.h>
#include <TFile.h>
#include "L1Trigger/L1TMuon/interface/deprecate/GeometryTranslator.h"
#include "L1Trigger/L1TMuon/interface/deprecate/MuonTriggerPrimitive.h"

#include "L1Trigger/L1TMuonEndCap/interface/MuonInternalTrack.h"
#include "L1Trigger/L1TMuonEndCap/interface/MuonInternalTrackFwd.h"


#include "L1Trigger/L1TMuonEndCap/interface/PhiMemoryImage.h"
#include "L1Trigger/L1TMuonEndCap/interface/EmulatorClasses.h"
#include "L1Trigger/CSCTrackFinder/interface/CSCTFPtLUT.h"
#include "L1Trigger/CSCTrackFinder/interface/CSCSectorReceiverLUT.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"

typedef edm::ParameterSet PSet;


//class L1TMuonEndCapTrackProducer : public edm::EDAnalyzer {

class L1TMuonEndCapTrackProducer : public edm::EDProducer {
public:
  L1TMuonEndCapTrackProducer(const PSet&);
  ~L1TMuonEndCapTrackProducer() {}
	
 //void analyze(const edm::Event&, const edm::EventSetup&); 
 void produce(edm::Event&, const edm::EventSetup&); 
  void beginJob();
  void endJob();
  
  ///////////////////////////////////////
  //// For Emulator with timing /////////
  /////  we need all of these ///////////
  ///////////////////////////////////////
 // MatchingOutput Mout;
 // ZonesOutput Zout;
 // ExtenderOutput Eout;
 // PatternOutput Pout;
 // SortingOutput Sout;
 // std::vector<ConvertedHit> ConvHits;
 // std::vector<std::vector<DeltaOutput>> Dout;
  ///////////////////////////////////////
  ///////////////////////////////////////
  ///////////////////////////////////////
  ///////////////////////////////////////
  
  
  const float ptscale[33] = { 
  	-1.,   0.0,   1.5,   2.0,   2.5,   3.0,   3.5,   4.0,
    4.5,   5.0,   6.0,   7.0,   8.0,  10.0,  12.0,  14.0,  
    16.0,  18.0,  20.0,  25.0,  30.0,  35.0,  40.0,  45.0, 
    50.0,  60.0,  70.0,  80.0,  90.0, 100.0, 120.0, 140.0, 1.E6 };
  

private:

  edm::EDGetTokenT<CSCCorrelatedLCTDigiCollection> inputTokenCSC;
  
};




#endif
