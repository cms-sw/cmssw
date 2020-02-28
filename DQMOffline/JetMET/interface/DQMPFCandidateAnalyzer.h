#ifndef DQMPFCandidateAnalyzer_H
#define DQMPFCandidateAnalyzer_H

/** \class JetMETAnalyzer
 *
 *  DQM jetMET analysis monitoring
 *
 *  \author F. Chlebana - Fermilab
 *          K. Hatakeyama - Rockefeller University
 *
 *          Jan. '14: modified by
 *
 *          M. Artur Weber
 *          R. Schoefbeck
 *          V. Sordini
 */

#include <memory>
#include <fstream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "DQMOffline/JetMET/interface/JetMETDQMDCSFilter.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DataFormats/Scalers/interface/DcsStatus.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include <map>
#include <string>

//namespace jetAnalysis {
//class TrackPropagatorToCalo;
//class StripSignalOverNoiseCalculator;
//}

class DQMPFCandidateAnalyzer : public DQMEDAnalyzer {
public:
  /// Constructor
  DQMPFCandidateAnalyzer(const edm::ParameterSet&);

  /// Destructor
  ~DQMPFCandidateAnalyzer() override;

  /// Inizialize parameters for histo binning
  //  void beginJob(void);
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  /// Get the analysis
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  /// Initialize run-based parameters
  void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;

private:
  // ----------member data ---------------------------
  static bool jetSortingRule(reco::Jet x, reco::Jet y) { return x.pt() > y.pt(); }

  //try to put one collection as start

  edm::InputTag vertexTag_;
  edm::EDGetTokenT<std::vector<reco::Vertex>> vertexToken_;

  edm::InputTag mInputCollection_;
  edm::InputTag theTriggerResultsLabel_;
  edm::InputTag hbheNoiseFilterResultTag_;
  edm::EDGetTokenT<bool> hbheNoiseFilterResultToken_;
  std::string HBHENoiseStringMiniAOD;

  HLTConfigProvider FilterhltConfig_;
  edm::InputTag METFilterMiniAODLabel_;
  edm::EDGetTokenT<edm::TriggerResults> METFilterMiniAODToken_;
  edm::InputTag METFilterMiniAODLabel2_;  //needed for RECO and reRECO differntiation
  edm::EDGetTokenT<edm::TriggerResults> METFilterMiniAODToken2_;

  bool hbhenoifilterdecision;  //we only care about HBHENoiseFilter here
  int miniaodfilterindex;      //-1 if not found/RECO, else put to a number >=0
  //needed to decide which filterresults are supposed to be called
  int miniaodfilterdec;  //if RECO set to 0, if reRECO set to 1, else to -1

  edm::EDGetTokenT<std::vector<reco::PFCandidate>> pflowToken_;
  edm::EDGetTokenT<std::vector<pat::PackedCandidate>> pflowPackedToken_;

  std::string candidateType_;

  bool isMiniAO_;

  // DCS filter
  JetMETDQMDCSFilter* DCSFilter_;

  edm::ParameterSet cleaningParameters_;
  std::vector<edm::ParameterSet> diagnosticsParameters_;

  double ptMinCand_;  //pt min of candidates
  // Smallest raw HCAL energy linked to the track
  double hcalMin_;

  MonitorElement* m_HOverTrackP_trackPtVsEta;
  MonitorElement* m_HOverTrackPVsTrackP_Barrel;
  MonitorElement* m_HOverTrackPVsTrackP_EndCap;

  MonitorElement* m_HOverTrackPVsTrackPt_Barrel;
  MonitorElement* m_HOverTrackPVsTrackPt_EndCap;

  MonitorElement* m_HOverTrackPVsEta_hPt_1_10;
  MonitorElement* m_HOverTrackPVsEta_hPt_10_20;
  MonitorElement* m_HOverTrackPVsEta_hPt_20_50;
  MonitorElement* m_HOverTrackPVsEta_hPt_50;

  MonitorElement* m_HOverTrackP_Barrel_hPt_1_10;
  MonitorElement* m_HOverTrackP_Barrel_hPt_10_20;
  MonitorElement* m_HOverTrackP_Barrel_hPt_20_50;
  MonitorElement* m_HOverTrackP_Barrel_hPt_50;

  MonitorElement* m_HOverTrackP_EndCap_hPt_1_10;
  MonitorElement* m_HOverTrackP_EndCap_hPt_10_20;
  MonitorElement* m_HOverTrackP_EndCap_hPt_20_50;
  MonitorElement* m_HOverTrackP_EndCap_hPt_50;

  MonitorElement* mProfileIsoPFChHad_HadPtCentral;
  MonitorElement* mProfileIsoPFChHad_HadPtEndcap;
  MonitorElement* mProfileIsoPFChHad_EMPtCentral;
  MonitorElement* mProfileIsoPFChHad_EMPtEndcap;
  MonitorElement* mProfileIsoPFChHad_TrackPt;

  MonitorElement* mProfileIsoPFChHad_HcalOccupancyCentral;
  MonitorElement* mProfileIsoPFChHad_HcalOccupancyEndcap;
  MonitorElement* mProfileIsoPFChHad_EcalOccupancyCentral;
  MonitorElement* mProfileIsoPFChHad_EcalOccupancyEndcap;
  MonitorElement* mProfileIsoPFChHad_TrackOccupancy;

  //PFcandidate maps
  std::vector<MonitorElement*> occupancyPFCand_, ptPFCand_, multiplicityPFCand_;
  std::vector<std::string> occupancyPFCand_name_, ptPFCand_name_, multiplicityPFCand_name_;
  std::vector<MonitorElement*> occupancyPFCand_puppiNolepWeight_, ptPFCand_puppiNolepWeight_;
  std::vector<std::string> occupancyPFCand_name_puppiNolepWeight_, ptPFCand_name_puppiNolepWeight_;
  std::vector<double> etaMinPFCand_, etaMaxPFCand_;
  std::vector<int> typePFCand_, countsPFCand_;

  //PFcandidate maps
  std::vector<MonitorElement*> occupancyPFCandRECO_, ptPFCandRECO_, multiplicityPFCandRECO_;
  std::vector<std::string> occupancyPFCand_nameRECO_, ptPFCand_nameRECO_, multiplicityPFCand_nameRECO_;
  std::vector<double> etaMinPFCandRECO_, etaMaxPFCandRECO_;
  std::vector<int> typePFCandRECO_, countsPFCandRECO_;

  int numPV_;
  int verbose_;

  int LSBegin_;
  int LSEnd_;

  bool bypassAllPVChecks_;
  bool bypassAllDCSChecks_;

  std::map<std::string, MonitorElement*> map_of_MEs;

  bool isMiniAOD_;
};
#endif
