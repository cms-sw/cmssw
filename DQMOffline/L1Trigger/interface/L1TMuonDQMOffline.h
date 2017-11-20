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

#include <boost/preprocessor.hpp>

#define X_DEFINE_ENUM_WITH_STRING_CONVERSIONS_TOSTRING_CASE(r, data, elem)    \
    case elem : return BOOST_PP_STRINGIZE(elem);

#define DEFINE_ENUM_WITH_STRING_CONVERSIONS(name, enumerators)                \
    enum name {                                                               \
        BOOST_PP_SEQ_ENUM(enumerators)                                        \
    };                                                                        \
                                                                              \
    inline const char* ToString(name v)                                       \
    {                                                                         \
        switch (v)                                                            \
        {                                                                     \
            BOOST_PP_SEQ_FOR_EACH(                                            \
                X_DEFINE_ENUM_WITH_STRING_CONVERSIONS_TOSTRING_CASE,          \
                name,                                                         \
                enumerators                                                   \
            )                                                                 \
            default: return "[Unknown " BOOST_PP_STRINGIZE(name) "]";         \
        }                                                                     \
    }

DEFINE_ENUM_WITH_STRING_CONVERSIONS(Eff, (EFF_Pt)(EFF_Phi)(EFF_Eta))
DEFINE_ENUM_WITH_STRING_CONVERSIONS(Res, (RES_Pt)(RES_1overPt)(RES_Phi)(RES_Eta)(RES_Charge))
DEFINE_ENUM_WITH_STRING_CONVERSIONS(EtaRegion, (ETAREGION_any)(ETAREGION_BMTF)(ETAREGION_OMTF)(ETAREGION_EMTF)(ETAREGION_out))
DEFINE_ENUM_WITH_STRING_CONVERSIONS(Qual, (QUAL_any)(QUAL_Open)(QUAL_Double)(QUAL_Single)(QUAL_else)(QUAL_no))
DEFINE_ENUM_WITH_STRING_CONVERSIONS(Type, (TYPE_Num)(TYPE_Den)(TYPE_Div))

        enum Control
        {
            CONTROL_MuonGmtDeltaR,
            CONTROL_NTightVsAll,
            CONTROL_NProbesVsTight,
        };

float maxEta = 2.4;
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
//        static const double pt = m_muon->pt();
        double eta() const { return m_muon->eta(); };
        double phi() const { return m_muon->phi(); };
        int charge() const {return m_muon->charge(); };
        double gmtPt() const { return m_regMu ? m_regMu->pt() : -1.; };
        double gmtEta() const { return m_regMu ? m_regMu->eta() : -5.; };
        double gmtPhi() const { return m_regMu ? m_regMu->phi() : -5.; };
        int gmtCharge() const {return m_regMu ? m_regMu->charge() : -5; };
        int gmtQual() const { return m_regMu ? m_regMu->hwQual() : -1; };
//        std::tuple<m_muon->charge(), m_regMu->hwQual()> pairInfo;       //lathos
//        std::tuple<MuonGmtPair.charge(), MuonGmtPair.gmtQual()> pairInfo;       //lathos
 //       double anna = pt();
  //      double vava = eta();
//        mama = (5,5);
//        double anna = phi();  
//double pt = pt();
//        std::tuple<double, double, double, double, double, double, double, EtaRegion, int, Qual> pairInfo = std::make_tuple(pt(), pt() - gmtPt(), 1/pt() - 1/gmtPt(), phi(), phi() - gmtPhi(), eta(), eta() - gmtEta(), ETAREGION_BMTF, charge() - gmtCharge(), QUAL_Open);

        std::tuple<double, double, double> pairInfoEff = {pt(), phi(), eta()};
        std::tuple<double, double, double, double, int> pairInfoRes = {pt() - gmtPt(), 1/pt() - 1/gmtPt(), phi() - gmtPhi(), eta() - gmtEta(), charge() - gmtCharge()};

        EtaRegion etaRegion() const {
            if (fabs(eta()) <= 0.83) {
                return ETAREGION_BMTF;
                if (fabs(eta()) < 1.24) {
                    return ETAREGION_OMTF;
                    if (fabs(eta()) < maxEta) return ETAREGION_EMTF;
                }
            }
            return ETAREGION_out;
        }
        Qual quality() const {
            if (gmtQual() >= 12) {
                return QUAL_Single;
                if (gmtQual() >= 8) {
                    return QUAL_Double;
                    if (gmtQual() >= 4) return QUAL_Open;
                }
            }
        return QUAL_else;
        }
        std::tuple<EtaRegion, Qual> etaQual(int etaQualCase) const {
            if (etaQualCase == 1)   return {ETAREGION_any, QUAL_any};
            if (etaQualCase == 2)   return {ETAREGION_any, quality()};
            if (etaQualCase == 3)   return {etaRegion(), QUAL_any};
            if (etaQualCase == 4)   return {etaRegion(), quality()};
            throw std::invalid_argument("etaQualCase");
        }
//        std::tuple<double, double, double, double, double, double, double, EtaRegion, int, Qual> pairInfo ;
//        std::get<0>(pairInfo) = pt();

        void propagate(edm::ESHandle<MagneticField> bField,
         edm::ESHandle<Propagator> propagatorAlong,
         edm::ESHandle<Propagator> propagatorOpposite);

        //std::tuple<double, double, double, double, double, double, double, EtaRegion, int, Qual> pairInfo;
      //  std::tie(std::tuple<float, float> pairInfo) 
            //= std::make_tuple(0,0);
//        std::tuple<int, int> pairInfo;
//        std::tuple<int m_muon->charge(), int m_regMu->hwQual()> pairInfo;
            //= std::make_tuple(0,0);

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
/*
        std::vector<double> m_effVsPtBins;
        std::vector<double> m_effVsPhiBins;
        std::vector<double> m_effVsEtaBins;
*/
        int nEffVsPtBins, nEffVsPhiBins, nEffVsEtaBins;
        float* ptBinsArray, phiBinsArray, etaBinsArray;
/*
        std::tuple<int, float*> getHistBinsEff(Eff eff){
            if (eff == EFF_Pt)        return {nEffVsPtBins,ptBinsArray};
            if (eff == EFF_Phi)       return {nEffVsPhiBins,phiBinsArray};
            if (eff == EFF_Eta)       return {nEffVsEtaBins,etaBinsArray};
            throw std::invalid_argument("eff");
        }*/
        std::tuple<int, double, double> getHistBinsRes(Res res){
            if (res == RES_Pt)        return {100,-50.,-50.};
            if (res == RES_1overPt)   return {50,-0.05,0.05};
            if (res == RES_Phi)       return {96,-0.2,-0.2};
            if (res == RES_Eta)       return {100,-0.1,0.1};
            if (res == RES_Charge)    return {5,-2,-3};
            throw std::invalid_argument("res");
        }

//        std::tuple<Res, double, double, EtaRegion, Qual, int, double, double> histoInfoEff, histoInfoRes;
        std::tuple<Eff, double, int, double, EtaRegion, Qual> histoInfoEffNum, histoInfoEffDen, histoInfoEff,  histoInfoEffTagPt,  histoInfoEffTagPhi,  histoInfoEffTagEta,  histoInfoEffProbePt,  histoInfoEffProbePhi,  histoInfoEffProbeEta;
        std::tuple<Res, double, double, EtaRegion, Qual> histoInfoRes;
//        std::tuple<Res, double, double, annoula, Qual, int, double, double> testing;
  //      std::tuple<double, double, double> getHistBins(Plot);

// histos
//        std::map<int, std::map<std::string, MonitorElement*> > m_EfficiencyHistos;
        std::map<std::tuple<Eff, double, int, double, EtaRegion, Qual, Type>, MonitorElement*> m_EfficiencyHistos;
        std::map<std::tuple<Res, double, double, EtaRegion, Qual>, MonitorElement*> m_ResolutionHistos;

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


        float m_MaxGmtMuonDR;
        float m_MaxHltMuonDR;

   // CB ignored at present
  // float m_MinMuonDR;
};

#endif
