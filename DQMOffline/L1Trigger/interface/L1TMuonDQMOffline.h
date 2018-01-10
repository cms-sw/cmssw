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
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TrackTransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Math/interface/deltaR.h"
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
        enum control {CTRL_TAGPT, CTRL_TAGETA, CTRL_TAGPHI, CTRL_PROBEPT, CTRL_PROBEETA, CTRL_PROBEPHI, CTRL_TAGPROBEDR, CTRL_MUONGMTDELTAR, CTRL_NTIGHTVSALL, CTRL_NPROBESVSTIGHT};
        enum effType {EFF_PT, EFF_PHI, EFF_ETA};
        enum resType {RES_PT, RES_1OVERPT, RES_PHI, RES_ETA, RES_CH};
        enum etaRegion {ETAREGION_ALL, ETAREGION_BMTF, ETAREGION_OMTF, ETAREGION_EMTF, ETAREGION_OUT};
        enum qualLevel {QUAL_ALL, QUAL_OPEN, QUAL_DOUBLE, QUAL_SINGLE};
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
        const reco::Vertex getPrimaryVertex(edm::Handle<reco::VertexCollection> & vertex,edm::Handle<reco::BeamSpot> & beamSpot);
        bool matchHlt(edm::Handle<trigger::TriggerEvent>  & triggerEvent, const reco::Muon * mu);

        // Cut and Matching
        void getMuonGmtPairs(edm::Handle<l1t::MuonBxCollection> & gmtCands);
        void getTightMuons(edm::Handle<reco::MuonCollection> & muons, const reco::Vertex & vertex);
        void getProbeMuons(edm::Handle<edm::TriggerResults> & trigResults,edm::Handle<trigger::TriggerEvent> & trigEvent);

        HLTConfigProvider m_hltConfig;

        edm::ESHandle<MagneticField> m_BField;
        edm::ESHandle<Propagator> m_propagatorAlong;
        edm::ESHandle<Propagator> m_propagatorOpposite;

        std::vector<float> getHistBinsEff(effType eff);
        std::tuple<int, double, double> getHistBinsRes(resType res);

        // Keys for histogram maps
        typedef std::tuple<resType, etaRegion, qualLevel> m_histoKeyResType; // resolution histograms
        typedef std::tuple<effType, int, etaRegion, qualLevel> m_histoKeyEffNumVarType; // efficiency numerator histograms for all variables except eta
        typedef std::pair<int, qualLevel> m_histoKeyEffNumEtaType; // efficiency numerator histograms for eta variable
        typedef std::tuple<effType, int, etaRegion> m_histoKeyEffDenVarType; // efficiency denominator histograms for all variables except eta

        // Histograms and histogram containers
        std::map<std::tuple<effType, int, etaRegion, qualLevel>, MonitorElement*> m_EfficiencyNumVarHistos;
        std::map<std::pair<int, qualLevel>, MonitorElement*> m_EfficiencyNumEtaHistos;
        std::map<std::tuple<effType, int, etaRegion>, MonitorElement*> m_EfficiencyDenVarHistos;
        std::map<etaRegion, MonitorElement*> m_EfficiencyDenPtHistos;
        std::map<int, MonitorElement*> m_EfficiencyDenEtaHistos;
        std::map<std::tuple<resType, etaRegion, qualLevel>, MonitorElement*> m_ResolutionHistos;
        std::map<control, MonitorElement*> m_ControlHistos;

        // helper variables
        std::vector<const reco::Muon*> m_TightMuons;
        std::vector<const reco::Muon*> m_ProbeMuons;
        std::vector<MuonGmtPair>  m_MuonGmtPairs;

        std::vector<reco::MuonCollection> m_RecoMuons;
        std::vector<l1t::MuonBxCollection> m_L1tMuons;
        std::vector<reco::Muon> m_RecoRecoMuons;
        BXVector<l1t::Muon> m_L1tL1tMuons;

        std::vector<std::pair<int, qualLevel>> m_cuts;

        // vectors of enum values to loop over
        const std::vector<effType> m_effTypes;
        const std::vector<resType> m_resTypes;
        const std::vector<etaRegion> m_etaRegions;
        const std::vector<qualLevel> m_qualLevelsRes;

        // maps with histogram name bits
        std::map<effType, std::string> m_effStrings;
        std::map<effType, std::string> m_effLabelStrings;
        std::map<resType, std::string> m_resStrings;
        std::map<resType, std::string> m_resLabelStrings;
        std::map<etaRegion, std::string> m_etaStrings;
        std::map<qualLevel, std::string> m_qualStrings;

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

        std::vector<int> m_trigIndices;

        float m_maxGmtMuonDR;
        float m_minTagProbeDR;
        float m_maxHltMuonDR;
};

//
// helper class to manage GMT-Muon pairing
//
class MuonGmtPair {
    public :
        MuonGmtPair(const reco::Muon *muon, const l1t::Muon *regMu) :
        m_muon(muon), m_regMu(regMu), m_eta(999.), m_phi_bar(999.), m_phi_end(999.) { };
        MuonGmtPair(const MuonGmtPair& muonGmtPair);

        ~MuonGmtPair() { };

        double dR();
        double pt()  const { return m_muon->pt(); };
        double eta() const { return m_muon->eta(); };
        double phi() const { return m_muon->phi(); };
        int charge() const { return m_muon->charge(); };
        double gmtPt() const { return m_regMu ? m_regMu->pt() : -1.; };
        double gmtEta() const { return m_regMu ? m_regMu->eta() : -5.; };
        double gmtPhi() const { return m_regMu ? m_regMu->phi() : -5.; };
        int gmtCharge() const {return m_regMu ? m_regMu->charge() : -5; };
        int gmtQual() const { return m_regMu ? m_regMu->hwQual() : -1; };

        L1TMuonDQMOffline::etaRegion etaRegion() const;
        double getDeltaVar(const L1TMuonDQMOffline::resType) const;
        double getVar(const L1TMuonDQMOffline::effType) const;

        void propagate(edm::ESHandle<MagneticField> bField,
        edm::ESHandle<Propagator> propagatorAlong,
        edm::ESHandle<Propagator> propagatorOpposite);

    private :
    // propagation private members
        TrajectoryStateOnSurface cylExtrapTrkSam(reco::TrackRef track, double rho);
        TrajectoryStateOnSurface surfExtrapTrkSam(reco::TrackRef track, double z);
        FreeTrajectoryState freeTrajStateMuon(reco::TrackRef track);

    private :
        const reco::Muon *m_muon;
        const l1t::Muon *m_regMu;

        edm::ESHandle<MagneticField> m_BField;
        edm::ESHandle<Propagator> m_propagatorAlong;
        edm::ESHandle<Propagator> m_propagatorOpposite;

        double m_eta;
        double m_phi_bar;
        double m_phi_end;
};

#endif
