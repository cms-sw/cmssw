#ifndef RecoLocalMuon_CSCSegmentReader_H
#define RecoLocalMuon_CSCSegmentReader_H

/** \class CSCSegmentReader
 *  Basic analyzer class which accesses CSCSegment
 *  and plot efficiency of the builder
 *
 *  $Date: 2006/06/01 08:44:02 $
 *  $Revision: 1.3 $
 *  \author M. Sani
 */

#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/Frameworkfwd.h>
#include <FWCore/Framework/interface/Handle.h>
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
            const edm::Handle<CSCSegmentCollection> seg, const CSCGeometry* geom);
  
  /// Simulation info
  void simInfo(const edm::Handle<edm::SimTrackContainer> simTracks);
  
  /// Segment building info
  void recInfo(const edm::Handle<edm::PSimHitContainer> sH, 
            const edm::Handle<CSCRecHit2DCollection> rH, const edm::Handle<CSCSegmentCollection> seg, 
            const CSCGeometry* geom);
  
protected:

private: 

    std::string filename;
    TH1F* heff, *hchi2, *hpt, *heta, *hx, *hy;
    TH1I* hrechit, *hsegment;
    TH1F* hphi[4], *htheta[4];
    
    TFile* file;  
    std::map<std::string, int> segMap;
    std::map<std::string, int> chaMap;
    int minRechitChamber, minRechitSegment;
    double maxPhi, maxTheta;
    int simhit;
    int near_segment;
};

#endif
