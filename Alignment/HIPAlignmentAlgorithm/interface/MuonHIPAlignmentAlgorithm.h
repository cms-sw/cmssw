// -*- C++ -*-
//
// Package:    MuonHIPAlignmentAlgorithm
// Class:      MuonHIPAlignmentAlgorithm
// 
/**\class MuonHIPAlignmentAlgorithm MuonHIPAlignmentAlgorithm.cc Alignment/MuonHIPAlignmentAlgorithm/interface/MuonHIPAlignmentAlgorithm.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jim Pivarski,,,
//         Created:  Wed Nov  5 21:47:33 CET 2008
// $Id$
//
//

#ifndef Alignment_MuonHIPAlignmentAlgorithm_MuonHIPAlignmentAlgorithm_h
#define Alignment_MuonHIPAlignmentAlgorithm_MuonHIPAlignmentAlgorithm_h

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmBase.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignableNavigator.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"

#include "TFile.h"
#include "TList.h"
#include "TH1F.h"
#include "TProfile.h"
#include "TF1.h"

#include <map>

class MuonHIPAlignmentAlgorithm : public AlignmentAlgorithmBase {
   public:
      MuonHIPAlignmentAlgorithm(const edm::ParameterSet& iConfig);
      ~MuonHIPAlignmentAlgorithm();
  
      void initialize(const edm::EventSetup& iSetup, AlignableTracker* alignableTracker, AlignableMuon* alignableMuon, AlignmentParameterStore* alignmentParameterStore);
      void startNewLoop();
      void run(const edm::EventSetup& iSetup, const ConstTrajTrackPairCollection& trajtracks);
      void terminate();
  
   private:
      AlignmentParameterStore* m_alignmentParameterStore;
      std::vector<Alignable*> m_alignables;
      AlignableNavigator* m_alignableNavigator;
      TrajectoryStateCombiner m_tsoscomb;
      std::map<Alignable*,bool> m_nearlyGlobalCoords;

      TH1F* m_hist_qoverpt;
      TProfile* m_hist_qoverpt_vs_eta;
      TProfile* m_hist_plusoverpt_vs_eta;
      TProfile* m_hist_minusoverpt_vs_eta;
      TH1F* m_hist_redChi2;
      std::map<Alignable*,TH1F*> m_hist_xresidual;
      std::map<Alignable*,TH1F*> m_hist_yresidual;
      std::map<Alignable*,TH1F*> m_hist_xresidual10GeV;
      std::map<Alignable*,TH1F*> m_hist_yresidual10GeV;
      std::map<Alignable*,TH1F*> m_hist_xresidual20GeV;
      std::map<Alignable*,TH1F*> m_hist_yresidual20GeV;
      std::map<std::pair<Alignable*,int>,TH1F*> m_hist_parameter;
      std::map<std::pair<Alignable*,int>,TH1F*> m_hist_parameter10GeV;
      std::map<std::pair<Alignable*,int>,TH1F*> m_hist_parameter20GeV;
      std::map<std::pair<Alignable*,int>,TProfile*> m_hist_prof;
      std::map<std::pair<Alignable*,int>,TH1F*> m_hist_oparameter;
      std::map<std::pair<Alignable*,int>,TH1F*> m_hist_oparameter10GeV;
      std::map<std::pair<Alignable*,int>,TH1F*> m_hist_oparameter20GeV;
      std::map<std::pair<Alignable*,int>,TProfile*> m_hist_oprof;
      std::vector<TH1*> m_all_histograms;
      
      int m_minTrackerHits;
      double m_maxRedChi2;
      int m_minStations;
      int m_minHitsPerDT;
      int m_minHitsPerDT4;
      int m_minHitsPerCSC;
      double m_maxResidualDT13;
      double m_maxResidualDT2;
      double m_maxResidualCSC;
      std::vector<int> m_ignoreCSCRings;
      int m_minTracksPerAlignable;
      bool m_useHitWeightsInDTAlignment;
      bool m_useHitWeightsInCSCAlignment;
      bool m_useOneDTSuperLayerPerEntry;
      bool m_useOneCSCChamberPerEntry;
      double m_fitRangeDTrphi;
      double m_fitRangeDTz;
      double m_fitRangeCSCrphi;
      double m_fitRangeCSCz;

      bool m_align;
      std::vector<std::string> m_collector;
      std::string m_collectorDirectory;
};

#endif // Alignment_MuonHIPAlignmentAlgorithm_MuonHIPAlignmentAlgorithm_h
