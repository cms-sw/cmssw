#ifndef RecoLocalMuon_CSCSegmentVisualise_H
#define RecoLocalMuon_CSCSegmentVisualise_H
/** class CSCSegmentVisualise
 *
 *  \author D. Fortin - UC Riverside
 *
 * Class to produce simple projects of rechits and segments to study
 * performance of segment reconstruction
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"

#include <Geometry/CSCGeometry/interface/CSCChamberSpecs.h>
#include <Geometry/CSCGeometry/interface/CSCChamber.h>
#include <Geometry/CSCGeometry/interface/CSCLayer.h>
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <DataFormats/CSCRecHit/interface/CSCRecHit2D.h>
#include <DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h>
#include <DataFormats/CSCRecHit/interface/CSCRangeMapAccessor.h>

#include <FWCore/Framework/interface/MakerMacros.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>

#include <vector>
#include <string>

#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"

class CSCSegmentVisualise : public edm::EDAnalyzer {
 public:

  /// Constructor
  explicit CSCSegmentVisualise(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~CSCSegmentVisualise();

  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup);


private: 

    std::string filename;
    TH2F *hxvsz[100];
    TH2F *hyvsz[100];
    TH2F *hxvszE[100];
    TH2F *hyvszE[100];
    TH2F *hxvszSeg[100];
    TH2F *hyvszSeg[100];
    TH2F *hxvszSegP[100];
    TH2F *hyvszSegP[100];
    
    TFile* file;  
    int idxHisto;
    int minRechitChamber;
    int maxRechitChamber;
    //    double maxPhi, maxTheta;
    int ievt;
};

#endif

