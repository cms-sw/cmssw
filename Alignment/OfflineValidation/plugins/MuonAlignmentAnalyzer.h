#ifndef Alignment_OfflineValidation_MuonAlignmentAnalyzer_H
#define Alignment_OfflineValidation_MuonAlignmentAnalyzer_H

/** \class MuonAlignmentANalyzer
 *  MuonAlignment offline Monitor Analyzer 
 *  Makes histograms of high level Muon objects/quantities
 *  and residuals (base EDAnalyzer for Muon Alignment Offline DQM)
 *  for Alignment Scenarios/DB comparison
 *
 *  $Date: 2010/03/29 13:18:44 $
 *  $Revision: 1.8 $
 *  \author J. Fernandez - Univ. Oviedo <Javier.Fernandez@cern.ch>
 */

// Base Class Headers
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include <vector>

namespace edm {
    class ParameterSet;
    class EventSetup;
}

class TH1F;
class TH2F;

typedef std::vector< std::vector<int> > intDVector;
typedef std::vector<TrackingRecHit *> RecHitVector;

class MuonAlignmentAnalyzer: public edm::EDAnalyzer {
public:
    /// Constructor
    MuonAlignmentAnalyzer(const edm::ParameterSet& pset);

    /// Destructor
    virtual ~MuonAlignmentAnalyzer();

    // Operations

    void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);

    virtual void beginJob() ;
    virtual void endJob() ;
protected:

private:

    RecHitVector doMatching(const reco::Track &, edm::Handle<DTRecSegment4DCollection> &, edm::Handle<CSCSegmentCollection> &, intDVector *, intDVector *, edm::ESHandle<GlobalTrackingGeometry> &); 

    edm::Service<TFileService> fs;

// InputTags
    edm::InputTag theGLBMuonTag; 
    edm::InputTag theSTAMuonTag; 

// Collections needed
    edm::InputTag theRecHits4DTagDT;
    edm::InputTag theRecHits4DTagCSC;

// To switch between real data and MC
    std::string theDataType;

    bool doSAplots,doGBplots,doResplots;

    // Histograms

    //# muons per event
    TH1F	*hGBNmuons;
    TH1F	*hSANmuons;
    TH1F	*hSimNmuons;
    TH1F	*hGBNmuons_Barrel;
    TH1F	*hSANmuons_Barrel;
    TH1F	*hSimNmuons_Barrel;
    TH1F	*hGBNmuons_Endcap;
    TH1F	*hSANmuons_Endcap;
    TH1F	*hSimNmuons_Endcap;

    // # hits per track
    TH1F  *hGBNhits;
    TH1F  *hGBNhits_Barrel;
    TH1F  *hGBNhits_Endcap;
    TH1F  *hSANhits;
    TH1F  *hSANhits_Barrel;
    TH1F  *hSANhits_Endcap;

    // Chi2 of Track
    TH1F *hGBChi2;
    TH1F *hSAChi2;
    TH1F *hGBChi2_Barrel;
    TH1F *hSAChi2_Barrel;
    TH1F *hGBChi2_Endcap;
    TH1F *hSAChi2_Endcap;

    // Invariant mass for dimuons
    TH1F *hGBInvM;
    TH1F *hSAInvM;
    TH1F *hSimInvM;
    // Invariant Mass distributions in Barrel (eta<1.04) region
    TH1F *hGBInvM_Barrel;
    TH1F *hSAInvM_Barrel;
    TH1F *hSimInvM_Barrel;
    // Invariant Mass distributions in Endcap (eta>=1.04) region
    TH1F *hGBInvM_Endcap;
    TH1F *hSAInvM_Endcap;
    TH1F *hSimInvM_Endcap;
    // Invariant Mass distributions in Barrel-Endcap overlap region 
    // 1 muon barrel & 1 muon endcap
    TH1F *hGBInvM_Overlap;
    TH1F *hSAInvM_Overlap;
    TH1F *hSimInvM_Overlap;

    // pT 
    TH1F *hSAPTRec;
    TH1F *hGBPTRec;
    TH1F *hSimPT; 
    TH1F *hSAPTRec_Barrel;
    TH1F *hGBPTRec_Barrel;
    TH1F *hSimPT_Barrel; 
    TH1F *hSAPTRec_Endcap;
    TH1F *hGBPTRec_Endcap;
    TH1F *hSimPT_Endcap; 
    TH2F *hGBPTvsEta;
    TH2F *hGBPTvsPhi;
    TH2F *hSAPTvsEta;
    TH2F *hSAPTvsPhi;
    TH2F *hSimPTvsEta;
    TH2F *hSimPTvsPhi;

    // For reco efficiency calculations
    TH2F *hSimPhivsEta;
    TH2F *hSAPhivsEta;
    TH2F *hGBPhivsEta;

    // pT resolution
    TH1F *hSAPTres;
    TH1F *hSAinvPTres;
    TH1F *hGBPTres;
    TH1F *hGBinvPTres;
    TH1F *hSAPTres_Barrel;
    TH1F *hSAPTres_Endcap;
    TH1F *hGBPTres_Barrel;
    TH1F *hGBPTres_Endcap;
    //pT rec - pT gen
    TH1F *hSAPTDiff;
    TH1F *hGBPTDiff;
    TH2F *hSAPTDiffvsEta;
    TH2F *hSAPTDiffvsPhi;
    TH2F *hGBPTDiffvsEta;
    TH2F *hGBPTDiffvsPhi;
    TH2F *hGBinvPTvsEta;
    TH2F *hGBinvPTvsPhi;
    TH2F *hSAinvPTvsEta;
    TH2F *hSAinvPTvsPhi;
    TH2F *hSAinvPTvsNhits;
    TH2F *hGBinvPTvsNhits;

    // Vector of chambers Residuals
    std::vector<TH1F *> unitsLocalX;
    std::vector<TH1F *> unitsLocalPhi;
    std::vector<TH1F *> unitsLocalTheta;
    std::vector<TH1F *> unitsLocalY;
    std::vector<TH1F *> unitsGlobalRPhi;
    std::vector<TH1F *> unitsGlobalPhi;
    std::vector<TH1F *> unitsGlobalTheta;
    std::vector<TH1F *> unitsGlobalRZ;

    // DT & CSC Residuals
    TH1F *hResidualLocalXDT; 
    TH1F *hResidualLocalPhiDT; 
    TH1F *hResidualLocalThetaDT; 
    TH1F *hResidualLocalYDT; 
    TH1F *hResidualLocalXCSC; 
    TH1F *hResidualLocalPhiCSC; 
    TH1F *hResidualLocalThetaCSC; 
    TH1F *hResidualLocalYCSC; 
    std::vector<TH1F*> hResidualLocalXDT_W;
    std::vector<TH1F*> hResidualLocalPhiDT_W;
    std::vector<TH1F*> hResidualLocalThetaDT_W;
    std::vector<TH1F*> hResidualLocalYDT_W;	
    std::vector<TH1F*> hResidualLocalXCSC_ME;
    std::vector<TH1F*> hResidualLocalPhiCSC_ME;
    std::vector<TH1F*> hResidualLocalThetaCSC_ME;
    std::vector<TH1F*> hResidualLocalYCSC_ME;
    std::vector<TH1F*> hResidualLocalXDT_MB;
    std::vector<TH1F*> hResidualLocalPhiDT_MB;
    std::vector<TH1F*> hResidualLocalThetaDT_MB;
    std::vector<TH1F*> hResidualLocalYDT_MB;
    TH1F *hResidualGlobalRPhiDT; 
    TH1F *hResidualGlobalPhiDT; 
    TH1F *hResidualGlobalThetaDT; 
    TH1F *hResidualGlobalZDT; 
    TH1F *hResidualGlobalRPhiCSC; 
    TH1F *hResidualGlobalPhiCSC; 
    TH1F *hResidualGlobalThetaCSC; 
    TH1F *hResidualGlobalRCSC; 
    std::vector<TH1F*> hResidualGlobalRPhiDT_W;
    std::vector<TH1F*> hResidualGlobalPhiDT_W;
    std::vector<TH1F*> hResidualGlobalThetaDT_W;
    std::vector<TH1F*> hResidualGlobalZDT_W;	
    std::vector<TH1F*> hResidualGlobalRPhiCSC_ME;
    std::vector<TH1F*> hResidualGlobalPhiCSC_ME;
    std::vector<TH1F*> hResidualGlobalThetaCSC_ME;
    std::vector<TH1F*> hResidualGlobalRCSC_ME;
    std::vector<TH1F*> hResidualGlobalRPhiDT_MB;
    std::vector<TH1F*> hResidualGlobalPhiDT_MB;
    std::vector<TH1F*> hResidualGlobalThetaDT_MB;
    std::vector<TH1F*> hResidualGlobalZDT_MB;

    // Mean and RMS of residuals for DQM
    TH2F *hprofLocalPositionCSC;
    TH2F *hprofLocalAngleCSC;
    TH2F *hprofLocalPositionRmsCSC;
    TH2F *hprofLocalAngleRmsCSC;
    TH2F *hprofGlobalPositionCSC;
    TH2F *hprofGlobalAngleCSC;
    TH2F *hprofGlobalPositionRmsCSC;
    TH2F *hprofGlobalAngleRmsCSC;
    TH2F *hprofLocalPositionDT;
    TH2F *hprofLocalAngleDT;
    TH2F *hprofLocalPositionRmsDT;
    TH2F *hprofLocalAngleRmsDT;
    TH2F *hprofGlobalPositionDT;
    TH2F *hprofGlobalAngleDT;
    TH2F *hprofGlobalPositionRmsDT;
    TH2F *hprofGlobalAngleRmsDT;
  
    TH1F *hprofLocalXDT;
    TH1F *hprofLocalPhiDT;
    TH1F *hprofLocalThetaDT;
    TH1F *hprofLocalYDT;
    TH1F *hprofLocalXCSC;
    TH1F *hprofLocalPhiCSC;
    TH1F *hprofLocalThetaCSC;
    TH1F *hprofLocalYCSC;
    TH1F *hprofGlobalRPhiDT;
    TH1F *hprofGlobalPhiDT;
    TH1F *hprofGlobalThetaDT;
    TH1F *hprofGlobalZDT;
    TH1F *hprofGlobalRPhiCSC;
    TH1F *hprofGlobalPhiCSC;
    TH1F *hprofGlobalThetaCSC;
    TH1F *hprofGlobalRCSC;

    std::vector<long> detectorCollection;  

//  ESHandle<MagneticField> theMGField;

    Propagator * thePropagator;

    // Counters
    int numberOfSimTracks;
    int numberOfGBRecTracks;
    int numberOfSARecTracks;
    int numberOfHits;

    // hist kinematic range
    double ptRangeMin,ptRangeMax,invMassRangeMin,invMassRangeMax;
    // hist residual range
    double resLocalXRangeStation1,resLocalXRangeStation2,resLocalXRangeStation3,resLocalXRangeStation4;
    double resLocalYRangeStation1,resLocalYRangeStation2,resLocalYRangeStation3,resLocalYRangeStation4;
    double resPhiRange,resThetaRange;
    unsigned int nbins,min1DTrackRecHitSize,min4DTrackSegmentSize;
};
#endif

