#ifndef RecoLocalMuon_CSCSegmentReader_H
#define RecoLocalMuon_CSCSegmentReader_H

/** \class CSCSegmentReader
 *  Basic analyzer class which accesses CSCSegment
 *  and plot efficiency of the builder
 *
 *  $Date: 2008/01/30 03:41:53 $
 *  $Revision: 1.10 $
 *  \author M. Sani
 */

#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/Frameworkfwd.h>
#include <DataFormats/Common/interface/Handle.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h> 

#include <SimDataFormats/Track/interface/SimTrackContainer.h>
#include <SimDataFormats/TrackingHit/interface/PSimHitContainer.h>
#include <DataFormats/CSCRecHit/interface/CSCSegmentCollection.h>

#include <Geometry/CSCGeometry/interface/CSCGeometry.h>

#include <vector>
#include <map>
#include <string>

#include "TFile.h"
#include "TH1F.h"

class CSCSegmentReader : public edm::EDAnalyzer {
public:
  /// Constructor
  CSCSegmentReader(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~CSCSegmentReader();

  // Operations

  /// Perform the real analysis
  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup);
  
  /// Phi and theta resolution of the built segments
  void resolution(const edm::Handle<edm::PSimHitContainer> sH, 
            const edm::Handle<CSCRecHit2DCollection> rH, const edm::Handle<CSCSegmentCollection> seg, 
            const CSCGeometry* geom);
  
  /// Simulation info
  void simInfo(const edm::Handle<edm::SimTrackContainer> simTracks);
  
  /// Segment building info
  void recInfo( const edm::Handle<edm::PSimHitContainer> sH, 
                const edm::Handle<CSCRecHit2DCollection> rH, const edm::Handle<CSCSegmentCollection> seg, 
                const CSCGeometry* geom);

  /// Find best rec segment for set of muon sim hits
  int bestMatch( CSCDetId id0,
                 const edm::Handle<edm::PSimHitContainer> simHits,
                 const edm::Handle<CSCSegmentCollection> cscSegments,
                 const CSCGeometry* geom );

  
protected:

private: 

    std::string filename;
    TH1F *heff0, *heff1, *heff2, *heff3, *hchi2, *hpt, *heta, *hx, *hy;
    TH1I *hrechit, *hsegment;
    TH1F *hphiDir[9], *hthetaDir[9];
    TH1F *hdxOri[9], *hdyOri[9];
    
    TFile* file;  
    std::map<std::string, int> segMap1;
    std::map<std::string, int> segMap2;
    std::map<std::string, int> segMap3;
    std::map<std::string, int> chaMap1;
    std::map<std::string, int> chaMap2;
    std::map<std::string, int> chaMap3;
    int minLayerWithRechitChamber;
    int minRechitSegment;
    int minLayerWithSimhitChamber;
    double maxPhi, maxTheta;
    int simhit;
    int maxNhits;
    int minNhits;
    int n6hitSegmentMC[9];
    int n6hitSegmentReco[9];
    int near_segment;
};

#endif
