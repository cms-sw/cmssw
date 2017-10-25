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
#include <unistd.h>

// user include files
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"
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
#include <iostream>
#include <fstream>
#include <utility>
#include <vector>

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
        int charge() const {return m_muon->charge(); };
        double gmtPt() const { return m_regMu ? m_regMu->pt() : -1.; };
        double gmtEta() const { return m_regMu ? m_regMu->eta() : -5.; };
        double gmtPhi() const { return m_regMu ? m_regMu->phi() : -5.; };
        int gmtCharge() const {return m_regMu ? m_regMu->charge() : -5; };
        int gmtQual() const { return m_regMu ? m_regMu->hwQual() : -1; };
        
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

//
// DQM class declaration
//

class L1TMuonDQMOffline : public DQMEDAnalyzer {
    public:
        L1TMuonDQMOffline(const edm::ParameterSet& ps);
        ~L1TMuonDQMOffline() override;

    protected:
   // Luminosity Block
        void beginLuminosityBlock(edm::LuminosityBlock const& lumiBlock, edm::EventSetup const& c) override;
        virtual void dqmEndLuminosityBlock  (edm::LuminosityBlock const& lumiBlock, edm::EventSetup const& c);
        void dqmBeginRun(const edm::Run& run, const edm::EventSetup& iSetup) override;
        virtual void bookControlHistos(DQMStore::IBooker &);
        virtual void bookEfficiencyHistos(DQMStore::IBooker &ibooker, int ptCut);
        virtual void bookResolutionHistos(DQMStore::IBooker &ibooker);
        void bookHistograms(DQMStore::IBooker &ibooker, const edm::Run& run, const edm::EventSetup& iSetup) override;
       //virtual void analyze (const edm::Event& e, const edm::EventSetup& c);

    private:
        void analyze (const edm::Event& e, const edm::EventSetup& c) override;

        // Helper Functions
        const reco::Vertex getPrimaryVertex(edm::Handle<reco::VertexCollection> & vertex,edm::Handle<reco::BeamSpot> & beamSpot);
        bool matchHlt(edm::Handle<trigger::TriggerEvent>  & triggerEvent, const reco::Muon * mu);

        // Cut and Matching
        void getMuonGmtPairs(edm::Handle<l1t::MuonBxCollection> & gmtCands);
        void getTightMuons(edm::Handle<reco::MuonCollection> & muons, const reco::Vertex & vertex);
        void getProbeMuons(edm::Handle<edm::TriggerResults> & trigResults,edm::Handle<trigger::TriggerEvent> & trigEvent);

    private:
        HLTConfigProvider m_hltConfig;

        edm::ESHandle<MagneticField> m_BField;
        edm::ESHandle<Propagator> m_propagatorAlong;
        edm::ESHandle<Propagator> m_propagatorOpposite;

        enum Resol
        {
            RESOL_Pt,
            RESOL_1overPt,
            RESOL_Eta,
            RESOL_Phi,
            RESOL_Charge,

            RESOL_Pt_OPEN,
            RESOL_1overPt_OPEN,
            RESOL_Eta_OPEN,
            RESOL_Phi_OPEN,
            RESOL_Charge_OPEN,

            RESOL_Pt_DOUBLE,
            RESOL_1overPt_DOUBLE,
            RESOL_Eta_DOUBLE,
            RESOL_Phi_DOUBLE,
            RESOL_Charge_DOUBLE,

            RESOL_Pt_SINGLE,
            RESOL_1overPt_SINGLE,
            RESOL_Eta_SINGLE,
            RESOL_Phi_SINGLE,
            RESOL_Charge_SINGLE,
        };

        enum Control
        {
            CONTROL_MuonGmtDeltaR,
            CONTROL_NTightVsAll,
            CONTROL_NProbesVsTight,
        };

        // histos
        std::map<int, std::map<std::string, MonitorElement*> > m_EfficiencyHistos;
        std::map<Resol, MonitorElement*> m_ResolutionHistos;
        std::map<Control, MonitorElement*> m_ControlHistos;

        // helper variables
        std::vector<const reco::Muon*>  m_TightMuons;
        std::vector<const reco::Muon*>  m_ProbeMuons;
        std::vector<MuonGmtPair>  m_MuonGmtPairs;

        std::vector<reco::MuonCollection>  m_RecoMuons;
        std::vector<l1t::MuonBxCollection>  m_L1tMuons;
        std::vector<reco::Muon>  m_RecoRecoMuons;
        BXVector<l1t::Muon>  m_L1tL1tMuons;

        // config params
        bool  m_verbose;
        std::string m_HistFolder;
        std::vector<int> m_GmtPtCuts;
        double m_TagPtCut;
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

        float m_MaxMuonEta;
        float m_MaxGmtMuonDR;
        float m_MaxHltMuonDR;

   // CB ignored at present
  // float m_MinMuonDR;
};

#endif
