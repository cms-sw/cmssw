#ifndef DPGAnalysisSkims_CSCSkim_H
#define DPGAnalysisSkims_CSCSkim_H

/** \class CSCSkim
 *
 * This simple program selects minimal CSC events for output.
 *
 * Michael Schmitt, Northwestern University, July 2008
 */

// system include files
#include <memory>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <iomanip>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include <DataFormats/CSCRecHit/interface/CSCRecHit2D.h>
#include <DataFormats/CSCRecHit/interface/CSCSegmentCollection.h>

#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCChamber.h"
#include "Geometry/CSCGeometry/interface/CSCLayer.h"
#include "Geometry/CSCGeometry/interface/CSCLayerGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h"

#include "TVector3.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TFile.h"
#include "TString.h"
#include "TTree.h"


class CSCSkim : public edm::EDFilter {
 public:
  // Constructor
  explicit CSCSkim(const edm::ParameterSet& pset);
   
  // Destructor
  ~CSCSkim();
     
  // Analysis routines
  virtual void beginJob(const edm::EventSetup&) ;
  virtual bool filter(edm::Event& event, const edm::EventSetup& eventSetup);
  virtual void endJob() ;
       
protected:
       
private: 

  // main skimming routine
  bool doCSCSkimming(edm::Handle<CSCRecHit2DCollection> cscRecHits, 
                     edm::Handle<CSCSegmentCollection> cscSegments);

  // extra skimming routine for alignment studies
  bool doOverlapSkimming(edm::Handle<CSCSegmentCollection> cscSegments);

  // skimming routine for messy events
  bool doMessyEventSkimming(edm::Handle<CSCRecHit2DCollection> cscRecHits, 
                            edm::Handle<CSCSegmentCollection> cscSegments);


  // some useful functions
  int    chamberSerial(int kE, int kS, int kR, int kCh);

  // counters
  int nEventsAnalyzed;
  int nEventsSelected;
  int nEventsChambersBothSides;
  int nEventsOverlappingChambers;
  int nEventsMessy;

  // run and event number
  int iRun;
  int iEvent;

  // The root file for the histograms.
  TFile *theHistogramFile;

  // file names
  std::string outputFileName;
  std::string histogramFileName;

  // parameters for the selection
  int typeOfSkim;
  int nLayersWithHitsMinimum;
  int minimumHitChambers;
  int minimumSegments;
  bool demandChambersBothSides;
  bool makeHistograms;
  bool makeHistogramsForMessyEvents;

  // histograms for skimming module
  TH1F *hxnRecHits;
  TH1F *hxnSegments;
  TH1F *hxnHitChambers;
  TH1F *hxnRecHitsSel;
  TH1F *mevnRecHits0;
  TH1F *mevnChambers0;
  TH1F *mevnSegments0;
  TH1F *mevnRecHits1;
  TH1F *mevnChambers1;
  TH1F *mevnSegments1;

};
#endif
