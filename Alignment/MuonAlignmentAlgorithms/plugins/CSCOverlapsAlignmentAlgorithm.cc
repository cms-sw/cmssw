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
// $Id: CSCOverlapsAlignmentAlgorithm.cc,v 1.2 2008/12/12 11:37:02 pivarski Exp $
//
//

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmBase.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterSelector.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterStore.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"  
#include "Alignment/CommonAlignment/interface/AlignableNavigator.h"
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
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"

#include "TH1F.h"
#include "TProfile.h"
#include "TStyle.h"

#include <sstream>
#include <map>

class CSCOverlapsAlignmentAlgorithm : public AlignmentAlgorithmBase {
   public:
      CSCOverlapsAlignmentAlgorithm(const edm::ParameterSet& iConfig);
      ~CSCOverlapsAlignmentAlgorithm();
  
      void initialize(const edm::EventSetup& iSetup, AlignableTracker* alignableTracker, AlignableMuon* alignableMuon, AlignmentParameterStore* alignmentParameterStore);
      void startNewLoop();
      void run(const edm::EventSetup& iSetup, const ConstTrajTrackPairCollection& trajtracks);
      void terminate();
  
   private:
      double striperr2(const TrackingRecHit* hit);
      double radiusFit(const std::vector<const TrackingRecHit*> &evenhits, const std::vector<const TrackingRecHit*> &oddhits, double zcenter);
      void trackFit(const std::vector<const TrackingRecHit*> &hits, bool convention2pi, double zcenter, double &intercept, double &intercept_err2, double &slope, double &slope_err2);
      double redChi2(const std::vector<const TrackingRecHit*> &hits, bool convention2pi, double zcenter, double intercept, double slope);
      bool summarize(Alignable *ali, double &rotyDiff, double &rotyDiff_err2, double &twistDiff, double &twistDiff_err2, double &phiPosDiff, double &phiPosDiff_err2, double &rotzDiff, double &rotzDiff_err2, double &rotyDiffRMS, double &phiPosDiffRMS, double &rotzDiffRMS);
      bool summarize2(Alignable *ai, Alignable *aiprev, double &resid_i, double &resid_err2_i, double &resid_iprev, double &resid_err2_iprev, double &residRMS_i, double &residRMS_iprev);
      bool matrixSolution(int endcap, int station, int ring);

      AlignmentParameterStore* m_alignmentParameterStore;
      std::vector<Alignable*> m_alignables;
      AlignableNavigator* m_alignableNavigator;
      std::map<int,bool> m_quickMap;
      edm::ESHandle<CSCGeometry> m_cscGeometry;
      TrajectoryStateCombiner m_tsoscomb;

      std::map<int,TFileDirectory*> m_hist_rings;
      std::map<int,TH1F*> m_hist_all_vertpos;
      std::map<int,TH1F*> m_hist_all_relativephi;
      std::map<int,TH1F*> m_hist_all_slope;
      std::map<int,TProfile*> m_hist_intercept_vertpos;
      std::map<int,TProfile*> m_hist_intercept_relativephi;
      std::map<int,TProfile*> m_hist_intercept_slope;

      std::map<Alignable*,TH1F*> m_hist_redChi2;
      std::map<Alignable*,TH1F*> m_hist_RotYDiff;
      std::map<Alignable*,TH1F*> m_hist_TwistDiff;
      std::map<Alignable*,TH1F*> m_hist_PhiPosDiff;
      std::map<Alignable*,TH1F*> m_hist_RPhiPosDiff;
      std::map<Alignable*,TH1F*> m_hist_RotZDiff;
      std::map<Alignable*,TH1F*> m_hist_vertpos;
      std::map<Alignable*,TProfile*> m_hist_slopeVsY;
      std::map<Alignable*,TProfile*> m_hist_interceptVsY;
      std::map<Alignable*,TProfile*> m_hist_interceptVsY2;

      std::map<Alignable*,TH1F*> m_hist_indiv_relativephi;
      std::map<Alignable*,TProfile*> m_hist_indiv_intercept_relativephi;

      std::string m_mode;
      double m_maxHitErr;
      int m_minHitsPerChamber;
      double m_maxRotYDiff;
      double m_maxRPhiDiff;
      double m_maxRedChi2;
//   double m_fiducialY;
//   double m_fiducialMinPhi, m_fiducialMaxPhi;
      int m_minTracksPerAlignable;
// bool m_usePurePhiPositions;
      bool m_useHitWeightsInTrackFit;
      bool m_useFitWeightsInMean;
      bool m_makeHistograms;

      std::map<Alignable*,int> m_rotyDiff_N;
      std::map<Alignable*,double> m_rotyDiff_y;
      std::map<Alignable*,double> m_rotyDiff_yy;
      std::map<Alignable*,double> m_rotyDiff_xw;
      std::map<Alignable*,double> m_rotyDiff_yw;
      std::map<Alignable*,double> m_rotyDiff_xyw;
      std::map<Alignable*,double> m_rotyDiff_xxw;
      std::map<Alignable*,double> m_rotyDiff_w;

      std::map<Alignable*,int> m_phiPosDiff_N;
      std::map<Alignable*,double> m_phiPosDiff_y;
      std::map<Alignable*,double> m_phiPosDiff_yy;
      std::map<Alignable*,double> m_phiPosDiff_xw;
      std::map<Alignable*,double> m_phiPosDiff_yw;
      std::map<Alignable*,double> m_phiPosDiff_xyw;
      std::map<Alignable*,double> m_phiPosDiff_xxw;
      std::map<Alignable*,double> m_phiPosDiff_w;

      std::map<Alignable*,int> m_rphiPosDiff_N;
      std::map<Alignable*,double> m_rphiPosDiff_y;
      std::map<Alignable*,double> m_rphiPosDiff_yy;
      std::map<Alignable*,double> m_rphiPosDiff_xw;
      std::map<Alignable*,double> m_rphiPosDiff_yw;
      std::map<Alignable*,double> m_rphiPosDiff_xyw;
      std::map<Alignable*,double> m_rphiPosDiff_xxw;
      std::map<Alignable*,double> m_rphiPosDiff_w;

      std::map<Alignable*,double> m_radius;
};

CSCOverlapsAlignmentAlgorithm::CSCOverlapsAlignmentAlgorithm(const edm::ParameterSet& iConfig)
   : AlignmentAlgorithmBase(iConfig)
   , m_mode(iConfig.getParameter<std::string>("mode"))
   , m_maxHitErr(iConfig.getParameter<double>("maxHitErr"))
   , m_minHitsPerChamber(iConfig.getParameter<int>("minHitsPerChamber"))
   , m_maxRotYDiff(iConfig.getParameter<double>("maxRotYDiff"))
   , m_maxRPhiDiff(iConfig.getParameter<double>("maxRPhiDiff"))
   , m_maxRedChi2(iConfig.getParameter<double>("maxRedChi2"))
//    , m_fiducialY(iConfig.getParameter<double>("fiducialY"))
//    , m_fiducialMinPhi(iConfig.getParameter<double>("fiducialMinPhi"))
//    , m_fiducialMaxPhi(iConfig.getParameter<double>("fiducialMaxPhi"))
   , m_minTracksPerAlignable(iConfig.getParameter<int>("minTracksPerAlignable"))
//    , m_usePurePhiPositions(iConfig.getParameter<bool>("usePurePhiPositions"))
   , m_useHitWeightsInTrackFit(iConfig.getParameter<bool>("useHitWeightsInTrackFit"))
   , m_useFitWeightsInMean(iConfig.getParameter<bool>("useFitWeightsInMean"))
   , m_makeHistograms(iConfig.getParameter<bool>("makeHistograms"))
{
  if (m_mode == std::string("roty")) {}
  else if (m_mode == std::string("phipos")) {}
  else if (m_mode == std::string("rotz")) {}
  else {
    throw cms::Exception("CSCOverlapsAlignmentAlgorithm") << "Allowed modes are \"roty\", \"phipos\", \"rotz\"." << std::endl;
  }

   if (m_makeHistograms) {
     edm::Service<TFileService> tfileService;

     for (int endcap = 1;  endcap <= 2;  endcap++) {
       for (int station = 1;  station <= 4;  station++) {
	 int rings = 2;
	 if (station == 1) rings = 4;
	 if (station == 4) rings = 1;
	 for (int ring = 1;  ring <= rings;  ring++) {
	   std::stringstream name;
	   name << "ME" << (endcap == 1 ? "p" : "m") << station << "_" << ring;
	   m_hist_rings[endcap * 100 + station * 10 + ring] = new TFileDirectory(tfileService->mkdir(name.str()));
	 } // ring
       } // station
     } // endcap
   }
}

CSCOverlapsAlignmentAlgorithm::~CSCOverlapsAlignmentAlgorithm() {}

void CSCOverlapsAlignmentAlgorithm::initialize(const edm::EventSetup& iSetup, AlignableTracker* alignableTracker, AlignableMuon* alignableMuon, AlignmentParameterStore* alignmentParameterStore) {
   m_alignmentParameterStore = alignmentParameterStore;
   m_alignables = m_alignmentParameterStore->alignables();
   m_quickMap.clear();

   if (alignableMuon == NULL) {
     throw cms::Exception("CSCOverlapsAlignmentAlgorithm") << "doMuon must be set to True" << std::endl;
   }

   for (std::vector<Alignable*>::const_iterator ali = m_alignables.begin();  ali != m_alignables.end();  ++ali) {
      DetId id = (*ali)->geomDetId();
      if (id.det() != DetId::Muon  ||  id.subdetId() != MuonSubdetId::CSC) {
	throw cms::Exception("CSCOverlapsAlignmentAlgorithm") << "Only CSCs may be alignable" << std::endl;
      }

      m_quickMap[id.rawId()] = true;

      std::string selector_str;
      std::vector<bool> selector = (*ali)->alignmentParameters()->selector();
      for (std::vector<bool>::const_iterator sel = selector.begin();  sel != selector.end();  ++sel) {
	selector_str += (*sel ? std::string("1") : std::string("0"));
      }

      if (m_mode == std::string("roty")) {
	if (selector_str != std::string("000010")) {
	  throw cms::Exception("CSCOverlapsAlignmentAlgorithm") << "In \"roty\" mode, all selectors must be \"000010\", not \"" << selector_str << "\"" << std::endl;
	}
      }
      else if (m_mode == std::string("phipos")) {
	if (selector_str != std::string("110001")) {
	  throw cms::Exception("CSCOverlapsAlignmentAlgorithm") << "In \"phipos\" mode, all selectors must be \"110001\", not \"" << selector_str << "\"" << std::endl;
	}
      }
      else if (m_mode == std::string("rotz")) {
	if (selector_str != std::string("000001")) {
	  throw cms::Exception("CSCOverlapsAlignmentAlgorithm") << "In \"rotz\" mode, all selectors must be \"000001\", not \"" << selector_str << "\"" << std::endl;
	}
      }

      if (m_makeHistograms) {
	CSCDetId cscId(id.rawId());
	const TFileDirectory *ring = m_hist_rings[cscId.endcap() * 100 + cscId.station() * 10 + cscId.ring()];
	
	int nchambers = 36;
	if (cscId.station() > 1  &&  cscId.ring() == 1) nchambers = 18;
	int i = cscId.chamber() - 1;
	int inext = (i + 1) % nchambers;

	std::stringstream numberi, numberinext;
	if (i+1 < 10) numberi << "0";
	numberi << (i+1);
	if (inext+1 < 10) numberinext << "0";
	numberinext << (inext+1);

	std::stringstream name, title;
	name << "_ME" << (cscId.endcap() == 1 ? "p" : "m") << cscId.station() << "_" << cscId.ring() << "_" << numberi.str();
	title << " for ME" << (cscId.endcap() == 1 ? "+" : "-") << cscId.station() << "/" << cscId.ring() << " " << (i+1);

	m_hist_redChi2[*ali] = ring->make<TH1F>((std::string("redChi2") + name.str()).c_str(), (std::string("redChi2") + title.str()).c_str(), 100, 0., 10.);

	std::stringstream name2, title2;
	name2 << "_ME" << (cscId.endcap() == 1 ? "p" : "m") << cscId.station() << "_" << cscId.ring() << "_" << numberinext.str() << "_" << numberi.str();
	title2 << " for ME" << (cscId.endcap() == 1 ? "+" : "-") << cscId.station() << "/" << cscId.ring() << " " << (inext+1) << "-" << (i+1);

	m_hist_RotYDiff[*ali] = ring->make<TH1F>((std::string("RotYDiff") + name2.str()).c_str(), (std::string("RotYDiff (mrad)") + title2.str()).c_str(), 100, -60., 60.);
	m_hist_TwistDiff[*ali] = ring->make<TH1F>((std::string("TwistDiff") + name2.str()).c_str(), (std::string("TwistDiff (mrad/m)") + title2.str()).c_str(), 100, -300., 300.);
	m_hist_PhiPosDiff[*ali] = ring->make<TH1F>((std::string("PhiPosDiff") + name2.str()).c_str(), (std::string("PhiPosDiff (mrad)") + title2.str()).c_str(), 100, -15., 15.);
	m_hist_RPhiPosDiff[*ali] = ring->make<TH1F>((std::string("RPhiPosDiff") + name2.str()).c_str(), (std::string("RPhiPosDiff (mm)") + title2.str()).c_str(), 100, -15., 15.);
	m_hist_RotZDiff[*ali] = ring->make<TH1F>((std::string("RotZDiff") + name2.str()).c_str(), (std::string("RotZDiff (mrad)") + title2.str()).c_str(), 100, -30., 30.);

	double length = (*ali)->surface().length();
	m_hist_vertpos[*ali] = ring->make<TH1F>((std::string("vertpos") + name2.str()).c_str(), (std::string("radial distribution (cm)") + title2.str()).c_str(), 100, -length/2., length/2.);
	m_hist_slopeVsY[*ali] = ring->make<TProfile>((std::string("slopeVsY") + name2.str()).c_str(), (std::string("dphi/dz slope versus R (unitless vs. cm)") + title2.str()).c_str(), 10, -length/2., length/2.);
	m_hist_interceptVsY[*ali] = ring->make<TProfile>((std::string("interceptVsY") + name2.str()).c_str(), (std::string("phi-intercept versus R (mrad vs. cm)") + title2.str()).c_str(), 10, -length/2., length/2.);
	m_hist_interceptVsY2[*ali] = ring->make<TProfile>((std::string("interceptVsY2") + name2.str()).c_str(), (std::string("phi-intercept versus R (mrad vs. cm)") + title2.str()).c_str(), 10, -length/2., length/2.);

	m_hist_indiv_relativephi[*ali] = ring->make<TH1F>((std::string("indiv_relativephi") + name2.str()).c_str(), (std::string("#phi") + title2.str()).c_str(), 64, 0.165, 0.185);
	m_hist_indiv_intercept_relativephi[*ali] = ring->make<TProfile>((std::string("indiv_intercept_relativephi") + name2.str()).c_str(), (std::string("#phi residual versus #phi") + title2.str()).c_str(), 16, 0.165, 0.185);

	m_hist_redChi2[*ali]->StatOverflows(kTRUE);
	m_hist_RotYDiff[*ali]->StatOverflows(kTRUE);
	m_hist_TwistDiff[*ali]->StatOverflows(kTRUE);
	m_hist_PhiPosDiff[*ali]->StatOverflows(kTRUE);
	m_hist_RPhiPosDiff[*ali]->StatOverflows(kTRUE);
	m_hist_RotZDiff[*ali]->StatOverflows(kTRUE);
	m_hist_vertpos[*ali]->StatOverflows(kTRUE);
	m_hist_indiv_relativephi[*ali]->StatOverflows(kTRUE);
	m_hist_indiv_intercept_relativephi[*ali]->StatOverflows(kTRUE);

      } // end if makeHistograms
   }

   if (alignableTracker == NULL) m_alignableNavigator = new AlignableNavigator(alignableMuon);
   else m_alignableNavigator = new AlignableNavigator(alignableTracker, alignableMuon);

   for (std::map<int,TFileDirectory*>::const_iterator ringiter = m_hist_rings.begin();  ringiter != m_hist_rings.end();  ++ringiter) {
     int index = ringiter->first;
     int iendcap = index / 100;
     int istation = (index % 100) / 10;
     int iring = index % 10;

     bool aligning = false;
     CSCDetId id;
     for (std::map<int,bool>::const_iterator epair = m_quickMap.begin();  epair != m_quickMap.end();  ++epair) {
       id = CSCDetId(epair->first);
       if (id.endcap() == iendcap  &&  id.station() == istation  &&  id.ring() == iring) {
	 aligning = true;
	 break;
       }
     }

     if (aligning) {
       const TFileDirectory *ring = m_hist_rings[index];

       std::stringstream name2, title2;
       name2 << "_ME" << (iendcap == 1 ? "p" : "m") << istation << "_" << iring;
       title2 << " for ME" << (iendcap == 1 ? "+" : "-") << istation << "/" << iring;

       Alignable *ali = m_alignableNavigator->alignableFromDetId(id).alignable();
       double length = ali->surface().length();
       
       m_hist_all_vertpos[index] = ring->make<TH1F>((std::string("vertpos") + name2.str()).c_str(), (std::string("vertpos") + title2.str()).c_str(), 100, -length/2., length/2.);
       m_hist_all_relativephi[index] = ring->make<TH1F>((std::string("relativephi") + name2.str()).c_str(), (std::string("relativephi") + title2.str()).c_str(), 100, 0.165, 0.185);
       m_hist_all_slope[index] = ring->make<TH1F>((std::string("slope") + name2.str()).c_str(), (std::string("slope") + title2.str()).c_str(), 100, -0.0005, 0.0005);
       m_hist_intercept_vertpos[index] = ring->make<TProfile>((std::string("intercept_vertpos") + name2.str()).c_str(), (std::string("residual vs. radial position") + title2.str()).c_str(), 100, -length/2., length/2.);
       m_hist_intercept_relativephi[index] = ring->make<TProfile>((std::string("intercept_relativephi") + name2.str()).c_str(), (std::string("residual vs. #phi") + title2.str()).c_str(), 100, 0.165, 0.185);
       m_hist_intercept_slope[index] = ring->make<TProfile>((std::string("intercept_slope") + name2.str()).c_str(), (std::string("residual vs. track slope") + title2.str()).c_str(), 100, -0.0005, 0.0005);
     }
   }   
}

void CSCOverlapsAlignmentAlgorithm::startNewLoop() {
  m_rotyDiff_N.clear();
  m_rotyDiff_y.clear();
  m_rotyDiff_yy.clear();
  m_rotyDiff_xw.clear();
  m_rotyDiff_yw.clear();
  m_rotyDiff_xyw.clear();
  m_rotyDiff_xxw.clear();
  m_rotyDiff_w.clear();

  m_phiPosDiff_N.clear();
  m_phiPosDiff_y.clear();
  m_phiPosDiff_yy.clear();
  m_phiPosDiff_xw.clear();
  m_phiPosDiff_yw.clear();
  m_phiPosDiff_xyw.clear();
  m_phiPosDiff_xxw.clear();
  m_phiPosDiff_w.clear();

  m_rphiPosDiff_N.clear();
  m_rphiPosDiff_y.clear();
  m_rphiPosDiff_yy.clear();
  m_rphiPosDiff_xw.clear();
  m_rphiPosDiff_yw.clear();
  m_rphiPosDiff_xyw.clear();
  m_rphiPosDiff_xxw.clear();
  m_rphiPosDiff_w.clear();

  m_radius.clear();

  // differences are always next-up-neighbor minus this alignable
  for (std::vector<Alignable*>::const_iterator ali = m_alignables.begin();  ali != m_alignables.end();  ++ali) {
    m_rotyDiff_N[*ali] = 0;
    m_rotyDiff_y[*ali] = 0.;
    m_rotyDiff_yy[*ali] = 0.;
    m_rotyDiff_xw[*ali] = 0.;
    m_rotyDiff_yw[*ali] = 0.;
    m_rotyDiff_xyw[*ali] = 0.;
    m_rotyDiff_xxw[*ali] = 0.;
    m_rotyDiff_w[*ali] = 0.;

    m_phiPosDiff_N[*ali] = 0;
    m_phiPosDiff_y[*ali] = 0.;
    m_phiPosDiff_yy[*ali] = 0.;
    m_phiPosDiff_xw[*ali] = 0.;
    m_phiPosDiff_yw[*ali] = 0.;
    m_phiPosDiff_xyw[*ali] = 0.;
    m_phiPosDiff_xxw[*ali] = 0.;
    m_phiPosDiff_w[*ali] = 0.;

    m_rphiPosDiff_N[*ali] = 0;
    m_rphiPosDiff_y[*ali] = 0.;
    m_rphiPosDiff_yy[*ali] = 0.;
    m_rphiPosDiff_xw[*ali] = 0.;
    m_rphiPosDiff_yw[*ali] = 0.;
    m_rphiPosDiff_xyw[*ali] = 0.;
    m_rphiPosDiff_xxw[*ali] = 0.;
    m_rphiPosDiff_w[*ali] = 0.;

    m_radius[*ali] = -1.;
  }
}

void CSCOverlapsAlignmentAlgorithm::run(const edm::EventSetup& iSetup, const ConstTrajTrackPairCollection& trajtracks) {
  iSetup.get<MuonGeometryRecord>().get(m_cscGeometry);
  
  for (ConstTrajTrackPairCollection::const_iterator trajtrack = trajtracks.begin();  trajtrack != trajtracks.end();  ++trajtrack) {
    // const Trajectory* traj = (*trajtrack).first;
    const reco::Track* track = (*trajtrack).second;

    std::vector<std::vector<const TrackingRecHit*> > hits_by_station;
    std::vector<const TrackingRecHit*> current_station;
    int current_evenhits = 0;
    int current_oddhits = 0;
    int last_station = 0;

    for (trackingRecHit_iterator hit = track->recHitsBegin();  hit != track->recHitsEnd();  ++hit) {
      DetId id = (*hit)->geographicalId();
      if (id.det() == DetId::Muon  &&  id.subdetId() == MuonSubdetId::CSC) {
	CSCDetId cscId(id.rawId());
	CSCDetId chamberId(cscId.endcap(), cscId.station(), cscId.ring(), cscId.chamber());
	int station = (cscId.endcap() == 1 ? 1 : -1) * cscId.station();

	if (last_station == 0) last_station = station;
	if (last_station != station) {
	  if (m_minHitsPerChamber < 0  ||  (current_evenhits >= m_minHitsPerChamber  &&  current_oddhits >= m_minHitsPerChamber)) {
	    hits_by_station.push_back(current_station);
	  }
	  current_station.clear();
	  current_evenhits = 0;
	  current_oddhits = 0;
	}

	if ((*hit)->isValid()  &&  (m_maxHitErr < 0.  ||  (striperr2(&(**hit)) < m_maxHitErr*m_maxHitErr))  &&  (m_quickMap.find(chamberId.rawId()) != m_quickMap.end())) {
	  current_station.push_back(&(**hit));
	  if (cscId.chamber() % 2 == 0) current_evenhits++;
	  else current_oddhits++;

	  last_station = station;
	}

      } // end if CSC
    } // end loop over hits (collating station-by-station)

    if (m_minHitsPerChamber < 0  ||  (current_evenhits >= m_minHitsPerChamber  &&  current_oddhits >= m_minHitsPerChamber)) {
      hits_by_station.push_back(current_station);
    } // end get that last station

    for (std::vector<std::vector<const TrackingRecHit*> >::const_iterator station = hits_by_station.begin();  station != hits_by_station.end();  ++station) {
      std::map<CSCDetId,bool> distinct_chambers;
      CSCDetId evenChamber, oddChamber;

      int iendcap = -1000;
      int istation = -1000;
      int iring = -1000;
      std::vector<const TrackingRecHit*> evenhits, oddhits;
      for (std::vector<const TrackingRecHit*>::const_iterator hit = station->begin();  hit != station->end();  ++hit) {
	CSCDetId id((*hit)->geographicalId().rawId());
	iendcap = id.endcap();
	istation = id.station();
	iring = id.ring();
	CSCDetId chamberId(iendcap, istation, iring, id.chamber());

	distinct_chambers[chamberId] = true;
	if (id.chamber() % 2 == 0) {
	  evenChamber = chamberId;
	  evenhits.push_back(*hit);
	}
	else {
	  oddChamber = chamberId;
	  oddhits.push_back(*hit);
	}
      } // end loop over hits to find the even and odd chambers
      
      if (distinct_chambers.size() != 2) break;  // how could that happen?  Be careful anyway...

      int nchambers = 36;
      if (evenChamber.station() > 1  &&  evenChamber.ring() == 1) nchambers = 18;

      GlobalPoint evenCenter = m_cscGeometry->idToDet(evenChamber)->toGlobal(LocalPoint(0., 0., 0.));
      GlobalPoint oddCenter = m_cscGeometry->idToDet(oddChamber)->toGlobal(LocalPoint(0., 0., 0.));

      double evenphi = evenCenter.phi();
      double oddphi = oddCenter.phi();
      double evenz = evenCenter.z();
      double oddz = oddCenter.z();
      double zcenter = (evenz + oddz)/2.;
      double evenR0 = sqrt(pow(evenCenter.x(), 2.) + pow(evenCenter.y(), 2.));
      double oddR0 = sqrt(pow(oddCenter.x(), 2.) + pow(oddCenter.y(), 2.));
      
      // determine which phi convention to use so that we never cross the boundary
      bool convention2pi = false;
      while (evenphi > M_PI) evenphi -= 2.*M_PI;
      while (evenphi < -M_PI) evenphi += 2.*M_PI;
      while (oddphi > M_PI) oddphi -= 2.*M_PI;
      while (oddphi < -M_PI) oddphi += 2.*M_PI;

      if (fabs(evenphi - oddphi) > M_PI) convention2pi = true;  // these two chambers are on the loop-around boundary; switch conventions
      if (convention2pi) {
	if (evenphi < 0.) evenphi += 2.*M_PI;
	if (oddphi < 0.) oddphi += 2.*M_PI;
      }

      if (fabs(evenphi - oddphi) > M_PI) break;  // how could that happen?

      double even_intercept, even_intercept_err2, even_slope, even_slope_err2;
      trackFit(evenhits, convention2pi, zcenter, even_intercept, even_intercept_err2, even_slope, even_slope_err2);
      double even_redChi2 = redChi2(evenhits, convention2pi, zcenter, even_intercept, even_slope);

      double odd_intercept, odd_intercept_err2, odd_slope, odd_slope_err2;
      trackFit(oddhits, convention2pi, zcenter, odd_intercept, odd_intercept_err2, odd_slope, odd_slope_err2);
      double odd_redChi2 = redChi2(oddhits, convention2pi, zcenter, odd_intercept, odd_slope);

      double radius = radiusFit(evenhits, oddhits, zcenter);
      std::vector<const TrackingRecHit*> emptyhitlist;      
      double radius_justeven = radiusFit(evenhits, emptyhitlist, evenz);
      double radius_justodd = radiusFit(emptyhitlist, oddhits, oddz);

      double vertpos = radius - (evenR0 + oddR0)/2.;

      double even_phipos = even_intercept;
      double even_phipos_err2 = even_intercept_err2;
      double even_roty = even_slope * radius_justeven;  // actually the atan() of that
      double even_roty_err2 = even_slope_err2 * radius_justeven * radius_justeven;  // but atan() would complicate the uncertainty

      double odd_phipos = odd_intercept;
      double odd_phipos_err2 = odd_intercept_err2;
      double odd_roty = odd_slope * radius_justodd;  // actually the atan() of that
      double odd_roty_err2 = odd_slope_err2 * radius_justodd * radius_justodd;  // but atan() would complicate the uncertainty

      Alignable *chamber_i, *chamber_inext;
      bool even_is_lesser = (evenChamber.chamber() < oddChamber.chamber());
      if (evenChamber.chamber() == nchambers  &&  oddChamber.chamber() == 1) even_is_lesser = true;   // handle wrap-around cases
      else if (oddChamber.chamber() == nchambers  &&  evenChamber.chamber() == 1) even_is_lesser = false;

      double rotyDiff, phiPosDiff, rotyDiff_err2, phiPosDiff_err2, redChi2_i, redChi2_inext, relativephi_i, slope_i;

      if (even_is_lesser) {
	chamber_i = m_alignableNavigator->alignableFromDetId(evenChamber).alignable();
	chamber_inext = m_alignableNavigator->alignableFromDetId(oddChamber).alignable();

	m_radius[chamber_i] = evenR0;
	m_radius[chamber_inext] = oddR0;
	redChi2_i = even_redChi2;
	redChi2_inext = odd_redChi2;
	relativephi_i = even_phipos - evenphi;
	slope_i = even_slope;

	rotyDiff = odd_roty - even_roty;  // the (N+1) - (N) difference is odd minus even
	phiPosDiff = odd_phipos - even_phipos;
      }
      else {
	chamber_i = m_alignableNavigator->alignableFromDetId(oddChamber).alignable();
	chamber_inext = m_alignableNavigator->alignableFromDetId(evenChamber).alignable();

	m_radius[chamber_i] = oddR0;
	m_radius[chamber_inext] = evenR0;
	redChi2_i = odd_redChi2;
	redChi2_inext = even_redChi2;
	relativephi_i = odd_phipos - oddphi;
	slope_i = odd_slope;
	
	rotyDiff = even_roty - odd_roty;  // the (N+1) - (N) difference is even minus odd
	phiPosDiff = even_phipos - odd_phipos;
      }

      if (m_useFitWeightsInMean) {
	rotyDiff_err2 = odd_roty_err2 + even_roty_err2;
	phiPosDiff_err2 = odd_phipos_err2 + even_phipos_err2;
      }
      else {
	rotyDiff_err2 = 1.;
	phiPosDiff_err2 = 1.;
      }

      double rphiPosDiff = phiPosDiff * radius;
      double rphiPosDiff_err2 = phiPosDiff_err2 * radius * radius;

      // double length = chamber_i->surface().length();

      if ((m_maxRotYDiff < 0.  ||  fabs(rotyDiff) < m_maxRotYDiff)  &&
	  (m_maxRPhiDiff < 0.  ||  fabs(phiPosDiff) * m_radius[chamber_i] < m_maxRPhiDiff)  &&
	  (m_maxRedChi2 < 0.  ||  (even_redChi2 > 0.  &&  even_redChi2 < m_maxRedChi2  &&  odd_redChi2 > 0.  &&  odd_redChi2 < m_maxRedChi2))  &&
// 	  (m_fiducialY < 0.  ||  (fabs(vertpos) < m_fiducialY * length/2.))  &&
// 	  (m_fiducialMinPhi < 0.  ||  m_fiducialMaxPhi < 0.  ||  (m_fiducialMinPhi < relativephi_i  &&  relativephi_i < m_fiducialMaxPhi))  &&
	  true) {

	if (m_makeHistograms) {
	  m_hist_redChi2[chamber_i]->Fill(redChi2_i);
	  m_hist_redChi2[chamber_inext]->Fill(redChi2_inext);
	
	  m_hist_RotYDiff[chamber_i]->Fill(rotyDiff * 1000.);
	  m_hist_TwistDiff[chamber_i]->Fill((rotyDiff * 1000.) / (vertpos / 100.));
	  m_hist_PhiPosDiff[chamber_i]->Fill(phiPosDiff * 1000.);
	  m_hist_RPhiPosDiff[chamber_i]->Fill(rphiPosDiff * 10.);
	  m_hist_RotZDiff[chamber_i]->Fill((rphiPosDiff / vertpos) * 1000.);
	  m_hist_vertpos[chamber_i]->Fill(vertpos);
	
	  m_hist_slopeVsY[chamber_i]->Fill(vertpos, rotyDiff);
	  m_hist_interceptVsY[chamber_i]->Fill(vertpos, rphiPosDiff);
	  m_hist_interceptVsY2[chamber_i]->Fill(vertpos, phiPosDiff * 1000.);

	  int index = iendcap * 100 + istation * 10 + iring;
	  m_hist_all_vertpos[index]->Fill(vertpos);
	  m_hist_all_relativephi[index]->Fill(relativephi_i);
	  m_hist_all_slope[index]->Fill(slope_i);
	  m_hist_intercept_vertpos[index]->Fill(vertpos, phiPosDiff);
	  m_hist_intercept_relativephi[index]->Fill(relativephi_i, phiPosDiff);
	  m_hist_intercept_slope[index]->Fill(slope_i, phiPosDiff);

	  m_hist_indiv_relativephi[chamber_i]->Fill(relativephi_i);
	  m_hist_indiv_intercept_relativephi[chamber_i]->Fill(relativephi_i, phiPosDiff);
	}

	// "x" is vertpos, "y" is the quantity under study
	m_rotyDiff_N[chamber_i]++;
	m_rotyDiff_y[chamber_i] += rotyDiff;
	m_rotyDiff_yy[chamber_i] += rotyDiff * rotyDiff;
	m_rotyDiff_xw[chamber_i] += vertpos / rotyDiff_err2;
	m_rotyDiff_yw[chamber_i] += rotyDiff / rotyDiff_err2;
	m_rotyDiff_xyw[chamber_i] += vertpos * rotyDiff / rotyDiff_err2;
	m_rotyDiff_xxw[chamber_i] += vertpos * vertpos / rotyDiff_err2;
	m_rotyDiff_w[chamber_i] += 1. / rotyDiff_err2;
	
	m_phiPosDiff_N[chamber_i]++;
	m_phiPosDiff_y[chamber_i] += phiPosDiff;
	m_phiPosDiff_yy[chamber_i] += phiPosDiff * phiPosDiff;
	m_phiPosDiff_xw[chamber_i] += vertpos / phiPosDiff_err2;
	m_phiPosDiff_yw[chamber_i] += phiPosDiff / phiPosDiff_err2;
	m_phiPosDiff_xyw[chamber_i] += vertpos * phiPosDiff / phiPosDiff_err2;
	m_phiPosDiff_xxw[chamber_i] += vertpos * vertpos / phiPosDiff_err2;
	m_phiPosDiff_w[chamber_i] += 1. / phiPosDiff_err2;

	m_rphiPosDiff_N[chamber_i]++;
	m_rphiPosDiff_y[chamber_i] += rphiPosDiff / vertpos;
	m_rphiPosDiff_yy[chamber_i] += rphiPosDiff * rphiPosDiff / vertpos / vertpos;
	m_rphiPosDiff_xw[chamber_i] += vertpos / rphiPosDiff_err2;
	m_rphiPosDiff_yw[chamber_i] += rphiPosDiff / rphiPosDiff_err2;
	m_rphiPosDiff_xyw[chamber_i] += vertpos * rphiPosDiff / rphiPosDiff_err2;
	m_rphiPosDiff_xxw[chamber_i] += vertpos * vertpos / rphiPosDiff_err2;
	m_rphiPosDiff_w[chamber_i] += 1. / rphiPosDiff_err2;
      }

    } // end loop over stations on this track
  } // end loop over tracks
}

void CSCOverlapsAlignmentAlgorithm::terminate() {
  std::map<int,bool> exists;
  for (std::vector<Alignable*>::const_iterator ali = m_alignables.begin();  ali != m_alignables.end();  ++ali) {
    CSCDetId id((*ali)->geomDetId().rawId());
    exists[id.endcap() * 100 + id.station() * 10 + id.ring()] = true;
  }

  for (std::map<int,bool>::const_iterator epair = exists.begin();  epair != exists.end();  ++epair) {
    int endcap = (epair->first) / 100;
    int station = ((epair->first) % 100) / 10;
    int ring = (epair->first) % 10;

    matrixSolution(endcap, station, ring);  // returns false if fails, but sets error message
  }
}

double CSCOverlapsAlignmentAlgorithm::striperr2(const TrackingRecHit* hit) {
  DetId id = hit->geographicalId();

  int strip = m_cscGeometry->layer(id)->geometry()->nearestStrip(hit->localPosition());
  double angle = m_cscGeometry->layer(id)->geometry()->stripAngle(strip) - M_PI/2.;
  double sinAngle = sin(angle);
  double cosAngle = cos(angle);

  double xx = hit->localPositionError().xx();
  double xy = hit->localPositionError().xy();
  double yy = hit->localPositionError().yy();

  return xx*cosAngle*cosAngle + 2.*xy*sinAngle*cosAngle + yy*sinAngle*sinAngle;
}

double CSCOverlapsAlignmentAlgorithm::radiusFit(const std::vector<const TrackingRecHit*> &evenhits, const std::vector<const TrackingRecHit*> &oddhits, double zcenter) {
  double sum = 0.;
  double sum_z = 0.;
  double sum_r = 0.;
  double sum_zr = 0.;
  double sum_zz = 0.;

  for (std::vector<const TrackingRecHit*>::const_iterator hit = evenhits.begin();  hit != evenhits.end();  ++hit) {
    DetId id = (*hit)->geographicalId();
    GlobalPoint globalPoint = m_cscGeometry->idToDet(id)->toGlobal((*hit)->localPosition());

    double z = globalPoint.z() - zcenter;
    double radius = sqrt(pow(globalPoint.x(), 2.) + pow(globalPoint.y(), 2.));

    sum += 1.;
    sum_z += z;
    sum_r += radius;
    sum_zr += z*radius;
    sum_zz += z*z;
  }

  for (std::vector<const TrackingRecHit*>::const_iterator hit = oddhits.begin();  hit != oddhits.end();  ++hit) {
    DetId id = (*hit)->geographicalId();
    GlobalPoint globalPoint = m_cscGeometry->idToDet(id)->toGlobal((*hit)->localPosition());

    double z = globalPoint.z() - zcenter;
    double radius = sqrt(pow(globalPoint.x(), 2.) + pow(globalPoint.y(), 2.));

    sum += 1.;
    sum_z += z;
    sum_r += radius;
    sum_zr += z*radius;
    sum_zz += z*z;
  }

  return ((sum_zz * sum_r) - (sum_z * sum_zr)) / ((sum * sum_zz) - (sum_z * sum_z));
}

void CSCOverlapsAlignmentAlgorithm::trackFit(const std::vector<const TrackingRecHit*> &hits, bool convention2pi, double zcenter, double &intercept, double &intercept_err2, double &slope, double &slope_err2) {
  double sum_w = 0.;
  double sum_zw = 0.;
  double sum_zzw = 0.;
  double sum_zphiw = 0.;
  double sum_phiw = 0.;

  for (std::vector<const TrackingRecHit*>::const_iterator hit = hits.begin();  hit != hits.end();  ++hit) {
    CSCDetId id((*hit)->geographicalId().rawId());  // doesn't need to be a CSCDetId

    LocalPoint localPoint = (*hit)->localPosition();
    GlobalPoint globalPoint = m_cscGeometry->idToDet(id)->toGlobal(localPoint);

    double phi = globalPoint.phi();
    if (convention2pi) {
      while (phi < 0.) phi += 2.*M_PI;
      while (phi >= 2.*M_PI) phi -= 2.*M_PI;
    }
    else {
      while (phi < -M_PI) phi += 2.*M_PI;
      while (phi >= M_PI) phi -= 2.*M_PI;
    }

    double z = globalPoint.z() - zcenter;
    double radius = sqrt(pow(globalPoint.x(), 2.) + pow(globalPoint.y(), 2.));

    double phierr2 = striperr2(*hit) / radius / radius;
    if (!m_useHitWeightsInTrackFit) phierr2 = 1.;

    sum_w += 1./phierr2;
    sum_zw += z/phierr2;
    sum_zzw += z*z/phierr2;
    sum_zphiw += z*phi/phierr2;
    sum_phiw += phi/phierr2;
  }

  double delta = (sum_w * sum_zzw) - (sum_zw * sum_zw);
  intercept = ((sum_zzw * sum_phiw) - (sum_zw * sum_zphiw)) / delta;
  intercept_err2 = sum_zzw / delta;
  slope = ((sum_w * sum_zphiw) - (sum_zw * sum_phiw)) / delta;
  slope_err2 = sum_w / delta;

  if (!m_useHitWeightsInTrackFit) {
    intercept_err2 = 1.;
    slope_err2 = 1.;
  }
}

double CSCOverlapsAlignmentAlgorithm::redChi2(const std::vector<const TrackingRecHit*> &hits, bool convention2pi, double zcenter, double intercept, double slope) {
  double chi2 = 0.;
  double dof = 0.;

  for (std::vector<const TrackingRecHit*>::const_iterator hit = hits.begin();  hit != hits.end();  ++hit) {
    DetId id = (*hit)->geographicalId();
    GlobalPoint globalPoint = m_cscGeometry->idToDet(id)->toGlobal((*hit)->localPosition());

    double phi = globalPoint.phi();
    if (convention2pi) {
      while (phi < 0.) phi += 2.*M_PI;
      while (phi >= 2.*M_PI) phi -= 2.*M_PI;
    }
    else {
      while (phi < -M_PI) phi += 2.*M_PI;
      while (phi >= M_PI) phi -= 2.*M_PI;
    }

    double z = globalPoint.z() - zcenter;
    double radius = sqrt(pow(globalPoint.x(), 2.) + pow(globalPoint.y(), 2.));

    double phierr2 = striperr2(*hit) / radius / radius;
    if (!m_useHitWeightsInTrackFit) phierr2 = 1.;

    double trackphi = intercept + z * slope;
    
    chi2 += (trackphi - phi) * (trackphi - phi) / phierr2;
    dof += 1.;
  }
  dof -= 2.;

  if (dof > 0.) return chi2 / dof;
  else return -1.;
}

bool CSCOverlapsAlignmentAlgorithm::summarize(Alignable *ali, double &rotyDiff, double &rotyDiff_err2, double &twistDiff, double &twistDiff_err2, double &phiPosDiff, double &phiPosDiff_err2, double &rotzDiff, double &rotzDiff_err2, double &rotyDiffRMS, double &phiPosDiffRMS, double &rotzDiffRMS) {
  if (m_rotyDiff_N[ali] < m_minTracksPerAlignable  ||  m_phiPosDiff_N[ali] < m_minTracksPerAlignable  ||  m_rphiPosDiff_N[ali] < m_minTracksPerAlignable) return false;

  double rotyDiff_delta = (m_rotyDiff_w[ali] * m_rotyDiff_xxw[ali]) - (m_rotyDiff_xw[ali] * m_rotyDiff_xw[ali]);

  rotyDiff = ((m_rotyDiff_xxw[ali] * m_rotyDiff_yw[ali]) - (m_rotyDiff_xw[ali] * m_rotyDiff_xyw[ali])) / rotyDiff_delta;
  rotyDiff_err2 = m_rotyDiff_xxw[ali] / rotyDiff_delta;

  double rotyDiff_simplemean = m_rotyDiff_y[ali] / m_rotyDiff_N[ali];
  rotyDiffRMS = sqrt(m_rotyDiff_yy[ali] / m_rotyDiff_N[ali] - pow(rotyDiff_simplemean, 2.));

  twistDiff = ((m_rotyDiff_w[ali] * m_rotyDiff_xyw[ali]) - (m_rotyDiff_xw[ali] * m_rotyDiff_yw[ali])) / rotyDiff_delta;
  twistDiff_err2 = m_rotyDiff_w[ali] / rotyDiff_delta;

  double phiPosDiff_delta = (m_phiPosDiff_w[ali] * m_phiPosDiff_xxw[ali]) - (m_phiPosDiff_xw[ali] * m_phiPosDiff_xw[ali]);

  phiPosDiff = ((m_phiPosDiff_xxw[ali] * m_phiPosDiff_yw[ali]) - (m_phiPosDiff_xw[ali] * m_phiPosDiff_xyw[ali])) / phiPosDiff_delta;
  phiPosDiff_err2 = m_phiPosDiff_xxw[ali] / phiPosDiff_delta;

  double phiPosDiff_simplemean = m_phiPosDiff_y[ali] / m_phiPosDiff_N[ali];
  phiPosDiffRMS = sqrt(m_phiPosDiff_yy[ali] / m_phiPosDiff_N[ali] - pow(phiPosDiff_simplemean, 2.));

  double rphiPosDiff_delta = (m_rphiPosDiff_w[ali] * m_rphiPosDiff_xxw[ali]) - (m_rphiPosDiff_xw[ali] * m_rphiPosDiff_xw[ali]);

  rotzDiff = ((m_rphiPosDiff_w[ali] * m_rphiPosDiff_xyw[ali]) - (m_rphiPosDiff_xw[ali] * m_rphiPosDiff_yw[ali])) / rphiPosDiff_delta;
  rotzDiff_err2 = m_rphiPosDiff_w[ali] / rphiPosDiff_delta;

  double rotzDiff_simplemean = m_rphiPosDiff_y[ali] / m_rphiPosDiff_N[ali];
  rotzDiffRMS = sqrt(m_rphiPosDiff_yy[ali] / m_rphiPosDiff_N[ali] - pow(rotzDiff_simplemean, 2.));

  return true;
}

bool CSCOverlapsAlignmentAlgorithm::summarize2(Alignable *ai, Alignable *aiprev, double &resid_i, double &resid_err2_i, double &resid_iprev, double &resid_err2_iprev, double &residRMS_i, double &residRMS_iprev) {
  double rotyDiff_i, rotyDiff_err2_i, twistDiff_i, twistDiff_err2_i, phiPosDiff_i, phiPosDiff_err2_i, rotzDiff_i, rotzDiff_err2_i, rotyDiffRMS_i, phiPosDiffRMS_i, rotzDiffRMS_i;
  double rotyDiff_iprev, rotyDiff_err2_iprev, twistDiff_iprev, twistDiff_err2_iprev, phiPosDiff_iprev, phiPosDiff_err2_iprev, rotzDiff_iprev, rotzDiff_err2_iprev, rotyDiffRMS_iprev, phiPosDiffRMS_iprev, rotzDiffRMS_iprev;

  if (!summarize(ai, rotyDiff_i, rotyDiff_err2_i, twistDiff_i, twistDiff_err2_i, phiPosDiff_i, phiPosDiff_err2_i, rotzDiff_i, rotzDiff_err2_i, rotyDiffRMS_i, phiPosDiffRMS_i, rotzDiffRMS_i)) return false;
  if (!summarize(aiprev, rotyDiff_iprev, rotyDiff_err2_iprev, twistDiff_iprev, twistDiff_err2_iprev, phiPosDiff_iprev, phiPosDiff_err2_iprev, rotzDiff_iprev, rotzDiff_err2_iprev, rotyDiffRMS_iprev, phiPosDiffRMS_iprev, rotzDiffRMS_iprev)) return false; 
    
  if (m_mode == std::string("roty")) {
    resid_i = rotyDiff_i;
    resid_err2_i = rotyDiff_err2_i;
    resid_iprev = rotyDiff_iprev;
    resid_err2_iprev = rotyDiff_err2_iprev;
    residRMS_i = rotyDiffRMS_i;
    residRMS_iprev = rotyDiffRMS_iprev;
  }
  else if (m_mode == std::string("phipos")) {
    resid_i = phiPosDiff_i;
    resid_err2_i = phiPosDiff_err2_i;
    resid_iprev = phiPosDiff_iprev;
    resid_err2_iprev = phiPosDiff_err2_iprev;
    residRMS_i = phiPosDiffRMS_i;
    residRMS_iprev = phiPosDiffRMS_iprev;
  }
  else if (m_mode == std::string("rotz")) {
    resid_i = rotzDiff_i;
    resid_err2_i = rotzDiff_err2_i;
    resid_iprev = rotzDiff_iprev;
    resid_err2_iprev = rotzDiff_err2_iprev;
    residRMS_i = rotzDiffRMS_i;
    residRMS_iprev = rotzDiffRMS_iprev;
  }
  
  return true;
}

bool CSCOverlapsAlignmentAlgorithm::matrixSolution(int endcap, int station, int ring) {
  int nchambers = 36;
  if (station > 1  &&  ring == 1) nchambers = 18;

  std::vector<Alignable*> a;
  for (int i = 0;  i < nchambers;  i++) a.push_back(NULL);

  AlgebraicVector v(nchambers);
  AlgebraicMatrix m(nchambers, nchambers);

  double radiusmm = -1.;
  for (std::vector<Alignable*>::const_iterator ali = m_alignables.begin();  ali != m_alignables.end();  ++ali) {
    CSCDetId id((*ali)->geomDetId().rawId());
    if (id.endcap() == endcap  &&  id.station() == station  &&  id.ring() == ring) {
      a[id.chamber() - 1] = *ali;
      radiusmm = m_radius[*ali] * 10.;
    }
  }
  for (int i = 0;  i < nchambers;  i++) {
    if (a[i] == NULL) throw cms::Exception("CSCOverlapsAlignmentAlgorithm") << "ME" << (endcap == 1 ? "+" : "-") << station << "/" << ring << " is incomplete (chamber " << i+1 << " is not alignable; please correct the configuration)" << std::endl;
  }

  for (int i = 0;  i < nchambers;  i++) {
    int iprev = (i + nchambers - 1) % nchambers;
    int inext = (i + 1) % nchambers;

    double resid_i, resid_err2_i, resid_iprev, resid_err2_iprev, residRMS_i, residRMS_iprev;
    if (!summarize2(a[i], a[iprev], resid_i, resid_err2_i, resid_iprev, resid_err2_iprev, residRMS_i, residRMS_iprev)) {
      edm::LogError("CSCOverlapsAlignmentAlgorithm") << "ME" << (endcap == 1 ? "+" : "-") << station << "/" << ring << " is incomplete (chamber " << i+1 << " has " << m_rotyDiff_N[a[i]] << " and chamber " << iprev+1 << " has " << m_rotyDiff_N[a[iprev]] << " tracks, while the minimum is " << m_minTracksPerAlignable << "); skipping this ring." << std::endl;
      return false;
    }

    v[i] = resid_iprev - resid_i;

    for (int j = 0;  j < nchambers;  j++) {
      if      (iprev == j) m[i][j] = -1.;
      else if (i == j)     m[i][j] =  2.;
      else if (inext == j) m[i][j] = -1.;
      else                 m[i][j] =  0.;

      // This term forces the average of all corrections to be zero;
      // equivalent to fixing one chamber, but more symmetric
      m[i][j] += 1./nchambers/nchambers;
    }

  } // end loop over chambers in ring

  int ierr;
  m.invert(ierr);
  if (ierr != 0) {
    edm::LogError("CSCOverlapsAlignmentAlgorithm") << "Matrix inversion failed for ME" << (endcap == 1 ? "+" : "-") << station << "/" << ring << " (skipping).  Matrix is " << m << std::endl;
    return false;
  }

  // calculate all corrections for this ring
  AlgebraicVector p = m * v;
    
  // correct for goofy sign conventions
  bool backward = ((endcap == 1  &&  station > 2.5)  ||  (endcap != 1  &&  station < 2.5));
  if (m_mode == std::string("phipos")  &&  !backward) p = -p;
  if (m_mode == std::string("rotz")  &&  backward) p = -p;

  // apply corrections
  for (int i = 0;  i < nchambers;  i++) {
    if (m_mode == std::string("roty")) {
      AlgebraicVector params(1);
      params[0] = p[i];

      AlgebraicSymMatrix cov(1);
      cov[0][0] = 1e-6;

      AlignmentParameters *parnew = a[i]->alignmentParameters()->cloneFromSelected(params, cov);
      a[i]->setAlignmentParameters(parnew);
      m_alignmentParameterStore->applyParameters(a[i]);
      a[i]->alignmentParameters()->setValid(true);
    }

    else if (m_mode == std::string("phipos")) {
      double phi_correction = p[i];

      AlgebraicVector params(3);
      params[0] = m_radius[a[i]] * sin(phi_correction);
      params[1] = m_radius[a[i]] * (cos(phi_correction) - 1.);
      params[2] = -phi_correction;

      AlgebraicSymMatrix cov(3);
      cov[0][0] = 1e-6;
      cov[1][1] = 1e-6;
      cov[2][2] = 1e-6;
	 
      AlignmentParameters *parnew = a[i]->alignmentParameters()->cloneFromSelected(params, cov);
      a[i]->setAlignmentParameters(parnew);
      m_alignmentParameterStore->applyParameters(a[i]);
      a[i]->alignmentParameters()->setValid(true);
    }

    else if (m_mode == std::string("rotz")) {
      AlgebraicVector params(1);
      params[0] = p[i];

      AlgebraicSymMatrix cov(1);
      cov[0][0] = 1e-6;

      AlignmentParameters *parnew = a[i]->alignmentParameters()->cloneFromSelected(params, cov);
      a[i]->setAlignmentParameters(parnew);
      m_alignmentParameterStore->applyParameters(a[i]);
      a[i]->alignmentParameters()->setValid(true);
    }

  } // end loop over chambers to apply corrections

  // print out useful information
  std::stringstream output;

  output << "********** " << m_mode << " results for ME" << (endcap == 1 ? "+" : "-") << station << "/" << ring << " ***********************************************************" << std::endl;
  if (m_mode == std::string("roty"))
    output << "(residuals, parameters, and closure are phi_y rotations in radians, multiply by 1000 for mrad)" << std::endl;
  else if (m_mode == std::string("phipos"))
    output << "(residuals, parameters, and closure are phi positions in radians, multiply by 1000 for mrad or " << radiusmm << " for rphi displacement in mm)" << std::endl;
  else if (m_mode == std::string("rotz"))
    output << "(residuals, parameters, and closure are phi_z rotations in radians, multiply by 1000 for mrad)" << std::endl;
  output << std::endl;

  double total_closure = 0.;
  double total_closure_err2 = 0.;
  double total_closure_errFromRMS = 0.;
  double old_chi2 = 0.;
  double new_chi2 = 0.;
  double mean_term = 0.;

  for (int j = 0;  j < nchambers;  j++) {
    int i = (j + nchambers - 1) % nchambers;
    int iprev = (i + nchambers - 1) % nchambers;
    int inext = (i + 1) % nchambers;

    double resid_i, resid_err2_i, resid_iprev, resid_err2_iprev, residRMS_i, residRMS_iprev;
    summarize2(a[i], a[iprev], resid_i, resid_err2_i, resid_iprev, resid_err2_iprev, residRMS_i, residRMS_iprev);  // already verified to return true (above)

    double chi2i = pow(resid_i - p[inext] + p[i], 2.);

    int Ntracks = -1;
    if (m_mode == std::string("roty")) Ntracks = m_rotyDiff_N[a[i]];
    else if (m_mode == std::string("phipos")) Ntracks = m_phiPosDiff_N[a[i]];
    else if (m_mode == std::string("rotz")) Ntracks = m_rphiPosDiff_N[a[i]];
    double RMSoverSqrtN = residRMS_i / sqrt(Ntracks);
    total_closure_errFromRMS += RMSoverSqrtN * RMSoverSqrtN;

    std::stringstream plusorminus;
    if (m_useFitWeightsInMean) plusorminus << " +- " << sqrt(resid_err2_i);
    output << (inext+1) << "-" << (i+1) << " residuals: " << resid_i << plusorminus.str() << " RMS/sqrt(" << Ntracks << "): " << RMSoverSqrtN << " old chi2i: " << (resid_i*resid_i) << " new chi2i: " << chi2i << std::endl;
    output << "delta parameter " << (j+1) << ": " << p[j] << std::endl;

    total_closure += resid_i;
    total_closure_err2 += resid_err2_i;
    old_chi2 += pow(resid_i, 2.);
    new_chi2 += chi2i;
    mean_term += -p[i];
  }
  output << std::endl;

  mean_term = mean_term / nchambers / nchambers;
  new_chi2 += pow(mean_term, 2.);

  output << "Mean of corrections: " << mean_term << std::endl;
  std::stringstream plusorminus;
  if (m_useFitWeightsInMean) plusorminus << " +- " << sqrt(total_closure_err2);
  output << "Total closure: " << total_closure << plusorminus.str() << " ErrFromRMS: " << sqrt(total_closure_errFromRMS) << " closure per chamber: " << total_closure / nchambers << std::endl;
  output << "Old chi2: " << old_chi2 << " New chi2: " << new_chi2 << std::endl;
  output << std::endl;

  // FIXME: do this through the MessageLogger
  std::cout << output.str();

  return true;
}

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmPluginFactory.h"
DEFINE_EDM_PLUGIN(AlignmentAlgorithmPluginFactory, CSCOverlapsAlignmentAlgorithm, "CSCOverlapsAlignmentAlgorithm");
