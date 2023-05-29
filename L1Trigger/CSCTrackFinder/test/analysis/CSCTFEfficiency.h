
// -*- C++ -*-
//
// Package:    CSCTFEfficiency
// Class:      CSCTFEfficiency
//
/**\class CSCTFEfficiency CSCTFEfficiency.cc jhugon/CSCTFEfficiency/src/CSCTFEfficiency.cc
   
Description: [one line class summary]

Implementation:
[Notes on implementation]
*/
//
// Original Author:  Justin Hugon,Ivan's graduate student,jhugon@phys.ufl.edu
//         Created:  Thu Jun 10 10:40:10 EDT 2010
// $Id: CSCTFEfficiency.h,v 1.2 2012/02/10 10:54:57 jhugon Exp $
//
//
#ifndef jhugon_CSCTFEfficiency_h
#define jhugon_CSCTFEfficiency_h

// system include files
#include <memory>
#include <vector>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <SimDataFormats/Track/interface/SimTrackContainer.h>
#include <SimDataFormats/Track/interface/SimTrack.h>
#include <SimDataFormats/Vertex/interface/SimVertexContainer.h>

#include <L1Trigger/CSCTrackFinder/interface/CSCTrackFinderDataTypes.h>

#include "DataFormats/L1CSCTrackFinder/interface/TrackStub.h"
#include <DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <DataFormats/MuonDetId/interface/CSCTriggerNumbering.h>
#include <DataFormats/L1CSCTrackFinder/interface/CSCTriggerContainer.h>
#include <DataFormats/L1CSCTrackFinder/interface/TrackStub.h>
#include <DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h>
#include <DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h>
#include <DataFormats/L1CSCTrackFinder/interface/L1CSCStatusDigiCollection.h>
#include <DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h>
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include <TMath.h>
#include <TCanvas.h>
#include <TLorentzVector.h>

#include <TStyle.h>
#include <TLegend.h>
#include <TF1.h>
#include <TH2.h>

#include <L1Trigger/CSCTrackFinder/test/src/Track.h>
#include <L1Trigger/CSCTrackFinder/test/src/RefTrack.h>
#include <L1Trigger/CSCTrackFinder/test/src/TFTrack.h>

#include <L1Trigger/CSCTrackFinder/test/src/TrackHistogramList.h>
#include <L1Trigger/CSCTrackFinder/test/src/EffHistogramList.h>
#include <L1Trigger/CSCTrackFinder/test/src/ResolutionHistogramList.h>
#include <L1Trigger/CSCTrackFinder/test/src/MultiplicityHistogramList.h>
#include <L1Trigger/CSCTrackFinder/test/src/StatisticsFile.h>
#include <L1Trigger/CSCTrackFinder/test/src/RHistogram.h>

namespace csctf_analysis {
  unsigned int minIndex(const std::vector<int>*);
  unsigned int minIndex(const std::vector<double>*);
}  // namespace csctf_analysis

//
// class declaration
//

class CSCTFEfficiency : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit CSCTFEfficiency(const edm::ParameterSet&);
  ~CSCTFEfficiency();

private:
  virtual void beginJob();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();

  // ----------member data ---------------------------

  csctf_analysis::TrackHistogramList* tfHistList;
  csctf_analysis::TrackHistogramList* refHistList;
  csctf_analysis::EffHistogramList* effhistlist;
  csctf_analysis::ResolutionHistogramList* resHistList;
  csctf_analysis::MultiplicityHistogramList* multHistList;
  csctf_analysis::StatisticsFile* statFile;
  csctf_analysis::RHistogram* rHistogram;

  unsigned int nEvents;

  edm::InputTag inputTag;
  double minPtSim;
  double maxPtSim;
  double minEtaSim;
  double maxEtaSim;
  double minPtTF;
  double minQualityTF;
  std::string ghostLoseParam;
  std::string statsFilename;
  double minMatchRParam;  // Added by Daniel 07/02

  const edm::ParameterSet* configuration;

  bool inputData;
  bool wantmodes;
  bool saveHistImages;
  bool singleMuSample;
  bool noRefTracks;
  int dataType;
  std::vector<unsigned> cutOnModes;
};
#endif
