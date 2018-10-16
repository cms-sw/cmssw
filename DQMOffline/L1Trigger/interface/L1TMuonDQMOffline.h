#ifndef DQMOFFLINE_L1TRIGGER_L1TMUONDQMOFFLINE_H
#define DQMOFFLINE_L1TRIGGER_L1TMUONDQMOFFLINE_H

/**
 * \file L1TMuonDQMOffline.h
 *
 * \author J. Pela, C. Battilana
*
* Stage2 Muons implementation: Anna Stakia 
*
 */

// system include files
#include <memory>

// user include files
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "MuonAnalysis/MuonAssociators/interface/PropagateToMuon.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TrackTransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "TRegexp.h"
#include "TString.h"
#include <utility>
#include <vector>

class MuonGmtPair;

//
// DQM class declaration
//

class L1TMuonDQMOffline : public DQMEDAnalyzer {
    public:
        enum Control {kCtrlTagPt, kCtrlTagEta, kCtrlTagPhi, kCtrlProbePt, kCtrlProbeEta, kCtrlProbePhi, kCtrlTagProbeDr, kCtrlTagHltDr, kCtrlMuonGmtDeltaR, kCtrlNTightVsAll, kCtrlNProbesVsTight};
        enum EffType {kEffPt, kEffPhi, kEffEta, kEffVtx};
        enum ResType {kResPt, kRes1OverPt, kResQOverPt, kResPhi, kResEta, kResCh};
        enum EtaRegion {kEtaRegionAll, kEtaRegionBmtf, kEtaRegionOmtf, kEtaRegionEmtf, kEtaRegionOut};
        enum QualLevel {kQualAll, kQualOpen, kQualDouble, kQualSingle};

        L1TMuonDQMOffline(const edm::ParameterSet& ps);
        ~L1TMuonDQMOffline() override;

    protected:
        void beginLuminosityBlock(edm::LuminosityBlock const& lumiBlock, edm::EventSetup const& c) override;
        virtual void dqmEndLuminosityBlock  (edm::LuminosityBlock const& lumiBlock, edm::EventSetup const& c);
        void dqmBeginRun(const edm::Run& run, const edm::EventSetup& iSetup) override;
        virtual void bookControlHistos(DQMStore::IBooker &);
        virtual void bookEfficiencyHistos(DQMStore::IBooker &ibooker);
        virtual void bookResolutionHistos(DQMStore::IBooker &ibooker);
        void bookHistograms(DQMStore::IBooker &ibooker, const edm::Run& run, const edm::EventSetup& iSetup) override;
        void analyze (const edm::Event& e, const edm::EventSetup& c) override;

    private:
        // Helper Functions
        const unsigned int getNVertices(edm::Handle<reco::VertexCollection> & vertex);
        const reco::Vertex getPrimaryVertex(edm::Handle<reco::VertexCollection> & vertex,edm::Handle<reco::BeamSpot> & beamSpot);
        double matchHlt(edm::Handle<trigger::TriggerEvent>  & triggerEvent, const reco::Muon * mu);

        // Cut and Matching
        void getMuonGmtPairs(edm::Handle<l1t::MuonBxCollection> & gmtCands);
        void getTightMuons(edm::Handle<reco::MuonCollection> & muons, const reco::Vertex & vertex);
        void getProbeMuons(edm::Handle<edm::TriggerResults> & trigResults,edm::Handle<trigger::TriggerEvent> & trigEvent);

        HLTConfigProvider m_hltConfig;

        PropagateToMuon m_propagator;

        std::vector<float> getHistBinsEff(EffType eff);
        std::tuple<int, double, double> getHistBinsRes(ResType res);

        // Keys for histogram maps
        typedef std::tuple<ResType, EtaRegion, QualLevel> m_histoKeyResType; // resolution histograms
        typedef std::tuple<EffType, int, EtaRegion, QualLevel> m_histoKeyEffNumVarType; // efficiency numerator histograms for all variables except eta
        typedef std::pair<int, QualLevel> m_histoKeyEffNumEtaType; // efficiency numerator histograms for eta variable
        typedef std::tuple<EffType, int, EtaRegion> m_histoKeyEffDenVarType; // efficiency denominator histograms for all variables except eta

        // Histograms and histogram containers
        std::map<std::tuple<EffType, int, EtaRegion, QualLevel>, MonitorElement*> m_EfficiencyNumVarHistos;
        std::map<std::pair<int, QualLevel>, MonitorElement*> m_EfficiencyNumEtaHistos;
        std::map<std::tuple<EffType, int, EtaRegion>, MonitorElement*> m_EfficiencyDenVarHistos;
        std::map<EtaRegion, MonitorElement*> m_EfficiencyDenPtHistos;
        std::map<int, MonitorElement*> m_EfficiencyDenEtaHistos;
        std::map<std::tuple<ResType, EtaRegion, QualLevel>, MonitorElement*> m_ResolutionHistos;
        std::map<Control, MonitorElement*> m_ControlHistos;

        // helper variables
        std::vector<const reco::Muon*> m_TightMuons;
        std::vector<const reco::Muon*> m_ProbeMuons;
        std::vector<MuonGmtPair>  m_MuonGmtPairs;

        std::vector<reco::MuonCollection> m_RecoMuons;
        std::vector<l1t::MuonBxCollection> m_L1tMuons;
        std::vector<reco::Muon> m_RecoRecoMuons;
        BXVector<l1t::Muon> m_L1tL1tMuons;

        std::vector<std::pair<int, QualLevel>> m_cuts;

        // vectors of enum values to loop over
        const std::vector<EffType> m_effTypes;
        const std::vector<ResType> m_resTypes;
        const std::vector<EtaRegion> m_etaRegions;
        const std::vector<QualLevel> m_qualLevelsRes;

        // maps with histogram name bits
        std::map<EffType, std::string> m_effStrings;
        std::map<EffType, std::string> m_effLabelStrings;
        std::map<ResType, std::string> m_resStrings;
        std::map<ResType, std::string> m_resLabelStrings;
        std::map<EtaRegion, std::string> m_etaStrings;
        std::map<QualLevel, std::string> m_qualStrings;

        // config params
        bool  m_verbose;
        std::string m_HistFolder;
        double m_TagPtCut;
        double m_recoToL1PtCutFactor;
        std::vector<edm::ParameterSet> m_cutsVPSet;
        edm::EDGetTokenT<reco::MuonCollection> m_MuonInputTag;
        edm::EDGetTokenT<l1t::MuonBxCollection> m_GmtInputTag;
        edm::EDGetTokenT<reco::VertexCollection> m_VtxInputTag;
        edm::EDGetTokenT<reco::BeamSpot> m_BsInputTag;
        edm::EDGetTokenT<trigger::TriggerEvent> m_trigInputTag;
        std::string m_trigProcess;
        edm::EDGetTokenT<edm::TriggerResults> m_trigProcess_token;

        std::vector<std::string> m_trigNames;
        std::vector<double> m_effVsPtBins;
        std::vector<double> m_effVsPhiBins;
        std::vector<double> m_effVsEtaBins;
        std::vector<double> m_effVsVtxBins;

        std::vector<int> m_trigIndices;

        bool m_useAtVtxCoord;
        float m_maxGmtMuonDR;
        float m_minTagProbeDR;
        float m_maxHltMuonDR;
};

//
// helper class to manage GMT-Muon pairing
//
class MuonGmtPair {
    public :
        MuonGmtPair(const reco::Muon *muon, const l1t::Muon *regMu, const PropagateToMuon& propagator, bool useAtVtxCoord);
        MuonGmtPair(const MuonGmtPair& muonGmtPair);
        ~MuonGmtPair() { };

        double dR();
        double pt()  const { return m_muon->pt(); };
        double eta() const { return m_muon->eta(); };
        double phi() const { return m_muon->phi(); };
        int charge() const { return m_muon->charge(); };
        double gmtPt() const { return m_regMu ? m_regMu->pt() : -1.; };
        double gmtEta() const { return m_regMu ? m_gmtEta : -5.; };
        double gmtPhi() const { return m_regMu ? m_gmtPhi : -5.; };
        int gmtCharge() const {return m_regMu ? m_regMu->charge() : -5; };
        int gmtQual() const { return m_regMu ? m_regMu->hwQual() : -1; };

        L1TMuonDQMOffline::EtaRegion etaRegion() const;
        double getDeltaVar(const L1TMuonDQMOffline::ResType) const;
        double getVar(const L1TMuonDQMOffline::EffType) const;

    private :
        const reco::Muon *m_muon;
        const l1t::Muon *m_regMu;

        // L1T muon eta and phi coordinates to be used
        // Can be the coordinates from the 2nd muon station or from the vertex
        double m_gmtEta;
        double m_gmtPhi;

        double m_eta;
        double m_phi;
};

#endif
