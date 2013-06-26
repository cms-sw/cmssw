// -*- C++ -*-
//
// Package:    MuonAlignmentAlgorithms
// Class:      CSCOverlapsAlignmentAlgorithm
// 
/**\class CSCOverlapsAlignmentAlgorithm CSCOverlapsAlignmentAlgorithm.cc Alignment/CSCOverlapsAlignmentAlgorithm/interface/CSCOverlapsAlignmentAlgorithm.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jim Pivarski,,,
//         Created:  Tue Oct  7 14:56:49 CDT 2008
// $Id: CSCOverlapsAlignmentAlgorithm.h,v 1.7 2013/01/07 19:58:00 wmtan Exp $
//
//

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmBase.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterSelector.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterStore.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"  
#include "Alignment/CommonAlignment/interface/AlignableNavigator.h"  
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"  
#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"
#include "Alignment/MuonAlignment/interface/AlignableMuon.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TrackRefitter/interface/TrackTransformer.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "Alignment/MuonAlignmentAlgorithms/interface/CSCPairConstraint.h"
#include "Alignment/MuonAlignmentAlgorithms/plugins/CSCChamberFitter.h"
#include "Alignment/MuonAlignmentAlgorithms/plugins/CSCAlignmentCorrections.h"

#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TStyle.h"

#include <sstream>
#include <map>

class CSCOverlapsAlignmentAlgorithm : public AlignmentAlgorithmBase {
public:
  CSCOverlapsAlignmentAlgorithm(const edm::ParameterSet& iConfig);
  ~CSCOverlapsAlignmentAlgorithm();
  
  void initialize(const edm::EventSetup& iSetup, AlignableTracker* alignableTracker, AlignableMuon* alignableMuon,
		  AlignableExtras* alignableExtras, AlignmentParameterStore* alignmentParameterStore);
  void run(const edm::EventSetup& iSetup, const EventInfo &eventInfo);

  void terminate(const edm::EventSetup& iSetup);

  // having to make read-only accessors for all of these would be ridiculous, so they're public
  TH1F *m_hitsPerChamber;

  TProfile *m_fiducial_ME11;
  TProfile *m_fiducial_ME12;
  TProfile *m_fiducial_MEx1;
  TProfile *m_fiducial_MEx2;

  TH1F *m_slope;
  TH1F *m_slope_MEp4;
  TH1F *m_slope_MEp3;
  TH1F *m_slope_MEp2;
  TH1F *m_slope_MEp1;
  TH1F *m_slope_MEm1;
  TH1F *m_slope_MEm2;
  TH1F *m_slope_MEm3;
  TH1F *m_slope_MEm4;

  TH1F *m_slopeResiduals;
  TH1F *m_slopeResiduals_weighted;
  TH1F *m_slopeResiduals_normalized;
  TH1F *m_offsetResiduals;
  TH1F *m_offsetResiduals_weighted;
  TH1F *m_offsetResiduals_normalized;

  TH1F *m_drdz;

  TH2F *m_occupancy;
  TH2F *m_XYpos_mep1;
  TH2F *m_XYpos_mep2;
  TH2F *m_XYpos_mep3;
  TH2F *m_XYpos_mep4;
  TH2F *m_XYpos_mem1;
  TH2F *m_XYpos_mem2;
  TH2F *m_XYpos_mem3;
  TH2F *m_XYpos_mem4;
  TH2F *m_RPhipos_mep1;
  TH2F *m_RPhipos_mep2;
  TH2F *m_RPhipos_mep3;
  TH2F *m_RPhipos_mep4;
  TH2F *m_RPhipos_mem1;
  TH2F *m_RPhipos_mem2;
  TH2F *m_RPhipos_mem3;
  TH2F *m_RPhipos_mem4;

  int m_mode;
  int m_minHitsPerChamber;
  double m_maxdrdz;
  bool m_fiducial;
  bool m_useHitWeights;
  bool m_slopeFromTrackRefit;
  int m_minStationsInTrackRefits;
  double m_truncateSlopeResid;
  double m_truncateOffsetResid;
  bool m_combineME11;
  bool m_useTrackWeights;
  bool m_errorFromRMS;
  int m_minTracksPerOverlap;
  bool m_makeHistograms;

private:
  std::string m_mode_string;
  std::string m_reportFileName;
  double m_minP;
  double m_maxRedChi2;
  std::string m_writeTemporaryFile;
  std::vector<std::string> m_readTemporaryFiles;
  bool m_doAlignment;

  AlignmentParameterStore* m_alignmentParameterStore;
  std::vector<Alignable*> m_alignables;
  AlignableNavigator *m_alignableNavigator;
  std::vector<CSCChamberFitter> m_fitters;
  std::vector<CSCPairResidualsConstraint*> m_residualsConstraints;
  std::map<std::pair<CSCDetId,CSCDetId>,CSCPairResidualsConstraint*> m_quickChamberLookup;

  TrackTransformer *m_trackTransformer;
  std::string m_propagatorName;
  const Propagator *m_propagatorPointer;

  TH1F *m_histP10;
  TH1F *m_histP100;
  TH1F *m_histP1000;
};
