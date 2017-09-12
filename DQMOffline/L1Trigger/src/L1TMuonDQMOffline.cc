/**
 * \file L1TMuonDQMOffline.cc
 *
 * \author J. Pela, C. Battilana
 *
 * Stage2 Muons implementation: Anna Stakia
 *
 */

#include "DQMOffline/L1Trigger/interface/L1TMuonDQMOffline.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "TMath.h"

using namespace reco;
using namespace trigger;
using namespace edm;
using namespace std;
using namespace l1t;

//__________RECO-GMT Muon Pair Helper Class____________________________

MuonGmtPair::MuonGmtPair(const MuonGmtPair& muonGmtPair) {
    m_muon    = muonGmtPair.m_muon;
    m_regMu     = muonGmtPair.m_regMu;

    m_eta     = muonGmtPair.m_eta;
    m_phi_bar = muonGmtPair.m_phi_bar;
    m_phi_end = muonGmtPair.m_phi_end;
}

double MuonGmtPair::dR() {
    float dEta = m_regMu ? (m_regMu->eta() - eta()) : 999.;
    float dPhi = m_regMu ? (m_regMu->phi() - phi()) : 999.;
    return sqrt(dEta*dEta + dPhi*dPhi);
}

void MuonGmtPair::propagate(ESHandle<MagneticField> bField,
                ESHandle<Propagator> propagatorAlong,
                ESHandle<Propagator> propagatorOpposite) {
    m_BField = bField;
    m_propagatorAlong = propagatorAlong;
    m_propagatorOpposite = propagatorOpposite;
    TrackRef standaloneMuon = m_muon->outerTrack();
    TrajectoryStateOnSurface trajectory;
    trajectory = cylExtrapTrkSam(standaloneMuon, 500);  // track at MB2 radius - extrapolation
    if (trajectory.isValid()) {
        m_eta     = trajectory.globalPosition().eta();
        m_phi_bar = trajectory.globalPosition().phi();
    }
    trajectory = surfExtrapTrkSam(standaloneMuon, 790);   // track at ME2+ plane - extrapolation
    if (trajectory.isValid()) {
        m_eta     = trajectory.globalPosition().eta();
        m_phi_end = trajectory.globalPosition().phi();
    }
    trajectory = surfExtrapTrkSam(standaloneMuon, -790); // track at ME2- disk - extrapolation
    if (trajectory.isValid()) {
        m_eta     = trajectory.globalPosition().eta();
        m_phi_end = trajectory.globalPosition().phi();
    }
}

TrajectoryStateOnSurface MuonGmtPair::cylExtrapTrkSam(TrackRef track, double rho)
{
    Cylinder::PositionType pos(0, 0, 0);
    Cylinder::RotationType rot;
    Cylinder::CylinderPointer myCylinder = Cylinder::build(pos, rot, rho);

    FreeTrajectoryState recoStart = freeTrajStateMuon(track);
    TrajectoryStateOnSurface recoProp;
    recoProp = m_propagatorAlong->propagate(recoStart, *myCylinder);
    if (!recoProp.isValid()) {
      recoProp = m_propagatorOpposite->propagate(recoStart, *myCylinder);
    }
    return recoProp;
}

TrajectoryStateOnSurface MuonGmtPair::surfExtrapTrkSam(TrackRef track, double z)
{
    Plane::PositionType pos(0, 0, z);
    Plane::RotationType rot;
    Plane::PlanePointer myPlane = Plane::build(pos, rot);

    FreeTrajectoryState recoStart = freeTrajStateMuon(track);
    TrajectoryStateOnSurface recoProp;
    recoProp = m_propagatorAlong->propagate(recoStart, *myPlane);
    if (!recoProp.isValid()) {
      recoProp = m_propagatorOpposite->propagate(recoStart, *myPlane);
    }
    return recoProp;
}

FreeTrajectoryState MuonGmtPair::freeTrajStateMuon(TrackRef track)
{
    GlobalPoint  innerPoint(track->innerPosition().x(), track->innerPosition().y(),  track->innerPosition().z());
    GlobalVector innerVec  (track->innerMomentum().x(),  track->innerMomentum().y(),  track->innerMomentum().z());
    FreeTrajectoryState recoStart(innerPoint, innerVec, track->charge(), &*m_BField);
    return recoStart;
}

//__________DQM_base_class_______________________________________________
L1TMuonDQMOffline::L1TMuonDQMOffline(const ParameterSet & ps) :
    m_verbose(ps.getUntrackedParameter<bool>("verbose")),
    m_HistFolder(ps.getUntrackedParameter<string>("histFolder")),
    m_GmtPtCuts(ps.getUntrackedParameter< vector<int> >("gmtPtCuts")),
    m_MuonInputTag(consumes<reco::MuonCollection>(ps.getUntrackedParameter<InputTag>("muonInputTag"))),
    m_GmtInputTag(consumes<l1t::MuonBxCollection>(ps.getUntrackedParameter<InputTag>("gmtInputTag"))),
    m_VtxInputTag(consumes<VertexCollection>(ps.getUntrackedParameter<InputTag>("vtxInputTag"))),
    m_BsInputTag(consumes<BeamSpot>(ps.getUntrackedParameter<InputTag>("bsInputTag"))),
    m_trigInputTag(consumes<trigger::TriggerEvent>(ps.getUntrackedParameter<InputTag>("trigInputTag"))),
    m_trigProcess(ps.getUntrackedParameter<string>("trigProcess")),
    m_trigProcess_token(consumes<edm::TriggerResults>(ps.getUntrackedParameter<InputTag>("trigProcess_token"))),
    m_trigNames(ps.getUntrackedParameter<vector<string> >("triggerNames")),
    m_effVsPtBins(ps.getUntrackedParameter<std::vector<double>>("efficiencyVsPtBins")),
    m_effVsPhiBins(ps.getUntrackedParameter<std::vector<double>>("efficiencyVsPhiBins")),
    m_effVsEtaBins(ps.getUntrackedParameter<std::vector<double>>("efficiencyVsEtaBins"))
{
    if (m_verbose) cout << "[L1TMuonDQMOffline:] ____________ Storage initialization ____________ " << endl;
    // CB do we need them from cfi?
    m_MaxMuonEta   = 2.4;
    m_MaxGmtMuonDR = 0.7;
    m_MaxHltMuonDR = 0.1;
    // CB ignored at present
    //m_MinMuonDR    = 1.2;
}

//_____________________________________________________________________
L1TMuonDQMOffline::~L1TMuonDQMOffline(){ }
//----------------------------------------------------------------------
void L1TMuonDQMOffline::dqmBeginRun(const edm::Run& run, const edm::EventSetup& iSetup){
    if (m_verbose) cout << "[L1TMuonDQMOffline:] Called beginRun." << endl;
    bool changed = true;
    m_hltConfig.init(run,iSetup,m_trigProcess,changed);
}

//_____________________________________________________________________
void L1TMuonDQMOffline::bookHistograms(DQMStore::IBooker &ibooker, const edm::Run& run, const edm::EventSetup& iSetup){
    //book histos
    bookControlHistos(ibooker);
    vector<int>::const_iterator gmtPtCutsIt  = m_GmtPtCuts.begin();
    vector<int>::const_iterator gmtPtCutsEnd = m_GmtPtCuts.end();

    for (; gmtPtCutsIt!=gmtPtCutsEnd; ++ gmtPtCutsIt) {
        bookEfficiencyHistos(ibooker, (*gmtPtCutsIt));
    }

    vector<string>::const_iterator trigNamesIt  = m_trigNames.begin();
    vector<string>::const_iterator trigNamesEnd = m_trigNames.end();

    for (; trigNamesIt!=trigNamesEnd; ++trigNamesIt) {
        TString tNameTmp = TString(*trigNamesIt); // use TString as it handles regex
        TRegexp tNamePattern = TRegexp(tNameTmp,true);
        int tIndex = -1;

        for (unsigned ipath = 0; ipath < m_hltConfig.size(); ++ipath) {
            TString tmpName = TString(m_hltConfig.triggerName(ipath));
            if (tmpName.Contains(tNamePattern)) {
                tIndex = int(ipath);
                m_trigIndices.push_back(tIndex);
            }
        }
        if (tIndex < 0 && m_verbose) cout << "[L1TMuonDQMOffline:] Warning: Could not find trigger " << (*trigNamesIt) << endl;   
    }
}

//_____________________________________________________________________
void L1TMuonDQMOffline::beginLuminosityBlock(LuminosityBlock const& lumiBlock, EventSetup const& c) {
    if(m_verbose) cout << "[L1TMuonDQMOffline:] Called beginLuminosityBlock at LS=" << lumiBlock.id().luminosityBlock() << endl;
}

//_____________________________________________________________________
void L1TMuonDQMOffline::dqmEndLuminosityBlock(LuminosityBlock const& lumiBlock, EventSetup const& c) {
    if(m_verbose) cout << "[L1TMuonDQMOffline:] Called endLuminosityBlock at LS=" << lumiBlock.id().luminosityBlock() << endl;
}

//_____________________________________________________________________
void L1TMuonDQMOffline::analyze(const Event & iEvent, const EventSetup & eventSetup){

    Handle<reco::MuonCollection> muons;
    iEvent.getByToken(m_MuonInputTag, muons);

    Handle<BeamSpot> beamSpot;
    iEvent.getByToken(m_BsInputTag, beamSpot);

    Handle<VertexCollection> vertex;
    iEvent.getByToken(m_VtxInputTag, vertex);

    Handle<l1t::MuonBxCollection> gmtCands;
    iEvent.getByToken(m_GmtInputTag,gmtCands);

    Handle<edm::TriggerResults> trigResults;
    iEvent.getByToken(m_trigProcess_token,trigResults);

    edm::Handle<trigger::TriggerEvent> trigEvent;
    iEvent.getByToken(m_trigInputTag,trigEvent);

    eventSetup.get<IdealMagneticFieldRecord>().get(m_BField);
    eventSetup.get<TrackingComponentsRecord>().get("PropagatorWithMaterial",m_propagatorAlong);
    eventSetup.get<TrackingComponentsRecord>().get("PropagatorWithMaterialOpposite",m_propagatorOpposite);

    const Vertex primaryVertex = getPrimaryVertex(vertex,beamSpot);

    getTightMuons(muons,primaryVertex);
    getProbeMuons(trigResults,trigEvent); // CB add flag to run on orthogonal datasets (no T&P)
    getMuonGmtPairs(gmtCands);

//    MuonCollection::const_iterator muonIt  = muons->begin();
//    MuonCollection::const_iterator muonEnd = muons->end();

    vector<l1t::Muon> gmtContainer;// = gmtCands->getRecord(0).getGMTCands();                       

    for (auto mu = gmtCands->begin(0); mu != gmtCands->end(0); ++mu) {
        gmtContainer.push_back(*mu);
    }

//    vector<l1t::Muon>::const_iterator gmtIt = gmtContainer.begin();
//    vector<l1t::Muon>::const_iterator gmtEnd = gmtContainer.end();

    if (m_verbose) cout << "[L1TMuonDQMOffline:] Computing efficiencies" << endl;

    vector<MuonGmtPair>::const_iterator muonGmtPairsIt  = m_MuonGmtPairs.begin();
    vector<MuonGmtPair>::const_iterator muonGmtPairsEnd = m_MuonGmtPairs.end();

    for(; muonGmtPairsIt!=muonGmtPairsEnd; ++muonGmtPairsIt) {
        float eta = muonGmtPairsIt->eta();
        float phi = muonGmtPairsIt->phi();
        float pt  = muonGmtPairsIt->pt();

        float gmtPt  = muonGmtPairsIt->gmtPt();
        int qual = muonGmtPairsIt->gmtQual();

        vector<int>::const_iterator gmtPtCutsIt  = m_GmtPtCuts.begin();
        vector<int>::const_iterator gmtPtCutsEnd = m_GmtPtCuts.end();

        for (; gmtPtCutsIt!=gmtPtCutsEnd; ++ gmtPtCutsIt) {
            int gmtPtCut = (*gmtPtCutsIt);
            bool gmtAboveCut = (gmtPt > gmtPtCut);

            stringstream ptCutToTag; ptCutToTag << gmtPtCut;
            string ptTag = ptCutToTag.str();

            if (fabs(eta) < m_MaxMuonEta) {
                m_EfficiencyHistos[gmtPtCut]["EffvsPt_" + ptTag + "_Den"]->Fill(pt);
                m_EfficiencyHistos[gmtPtCut]["EffvsPt_OPEN_" + ptTag + "_Den"]->Fill(pt);
                m_EfficiencyHistos[gmtPtCut]["EffvsPt_DOUBLE_" + ptTag + "_Den"]->Fill(pt);
                m_EfficiencyHistos[gmtPtCut]["EffvsPt_SINGLE_" + ptTag + "_Den"]->Fill(pt);

                if (gmtAboveCut) {
                    m_EfficiencyHistos[gmtPtCut]["EffvsPt_" + ptTag + "_Num"]->Fill(pt);

                    if (qual >= 4) m_EfficiencyHistos[gmtPtCut]["EffvsPt_OPEN_" + ptTag + "_Num"]->Fill(pt);
                    if (qual >= 8) m_EfficiencyHistos[gmtPtCut]["EffvsPt_DOUBLE_" + ptTag + "_Num"]->Fill(pt);
                    if (qual >= 12) m_EfficiencyHistos[gmtPtCut]["EffvsPt_SINGLE_" + ptTag + "_Num"]->Fill(pt);
                }

                // efficiency in eta/phi at plateau
                if (pt > 1.25*gmtPtCut) {       // efficiency in eta/phi at plateau

                    m_EfficiencyHistos[gmtPtCut]["EffvsPhi_" + ptTag + "_Den"]->Fill(phi);
                    m_EfficiencyHistos[gmtPtCut]["EffvsEta_" + ptTag + "_Den"]->Fill(eta);

                    m_EfficiencyHistos[gmtPtCut]["EffvsPhi_OPEN_" + ptTag + "_Den"]->Fill(phi);
                    m_EfficiencyHistos[gmtPtCut]["EffvsEta_OPEN_" + ptTag + "_Den"]->Fill(eta);

                    m_EfficiencyHistos[gmtPtCut]["EffvsPhi_DOUBLE_" + ptTag + "_Den"]->Fill(phi);
                    m_EfficiencyHistos[gmtPtCut]["EffvsEta_DOUBLE_" + ptTag + "_Den"]->Fill(eta);

                    m_EfficiencyHistos[gmtPtCut]["EffvsPhi_SINGLE_" + ptTag + "_Den"]->Fill(phi);
                    m_EfficiencyHistos[gmtPtCut]["EffvsEta_SINGLE_" + ptTag + "_Den"]->Fill(eta);

                    if (gmtAboveCut) {
                        m_EfficiencyHistos[gmtPtCut]["EffvsPhi_" + ptTag + "_Num"]->Fill(phi);
                        m_EfficiencyHistos[gmtPtCut]["EffvsEta_" + ptTag + "_Num"]->Fill(eta);

                        if (qual >= 4) {
                            m_EfficiencyHistos[gmtPtCut]["EffvsPhi_OPEN_" + ptTag + "_Num"]->Fill(phi);
                            m_EfficiencyHistos[gmtPtCut]["EffvsEta_OPEN_" + ptTag + "_Num"]->Fill(eta);
                        }
                        if (qual >= 8) {
                            m_EfficiencyHistos[gmtPtCut]["EffvsPhi_DOUBLE_" + ptTag + "_Num"]->Fill(phi);
                            m_EfficiencyHistos[gmtPtCut]["EffvsEta_DOUBLE_" + ptTag + "_Num"]->Fill(eta);
                        }
                        if (qual >= 12) {
                            m_EfficiencyHistos[gmtPtCut]["EffvsPhi_SINGLE_" + ptTag + "_Num"]->Fill(phi);
                            m_EfficiencyHistos[gmtPtCut]["EffvsEta_SINGLE_" + ptTag + "_Num"]->Fill(eta);
                        }
                    }
                }
            }
        }
    }
    if (m_verbose) cout << "[L1TMuonDQMOffline:] Computation finished" << endl;
}

//_____________________________________________________________________
void L1TMuonDQMOffline::bookControlHistos(DQMStore::IBooker& ibooker) {
    if(m_verbose) cout << "[L1TMuonDQMOffline:] Booking Control Plot Histos" << endl;

    ibooker.setCurrentFolder(m_HistFolder+"/control_variables");

    string name = "MuonGmtDeltaR";
    m_ControlHistos[name] = ibooker.book1D(name.c_str(),name.c_str(),25.,0.,2.5);

    name = "NTightVsAll";
    m_ControlHistos[name] = ibooker.book2D(name.c_str(),name.c_str(),5,-0.5,4.5,5,-0.5,4.5);

    name = "NProbesVsTight";
    m_ControlHistos[name] = ibooker.book2D(name.c_str(),name.c_str(),5,-0.5,4.5,5,-0.5,4.5);

////////////////////////////////////////////////////////////////////////////////////////////////////
    string name1 = "TagMuonEta_Histo";
    m_EfficiencyHistos[0][name1] = ibooker.book1D(name1.c_str(),name1.c_str(),50,-2.5,2.5);
    string name2 = "TagMuonPhi_Histo";
    m_EfficiencyHistos[0][name2] = ibooker.book1D(name2.c_str(),name2.c_str(),24,-TMath::Pi(),TMath::Pi());
    string name3 = "TagMuonPt_Histo";
    m_EfficiencyHistos[0][name3] = ibooker.book1D(name3.c_str(),name3.c_str(),50,0.,100.);
//*****
    name1 = "ProbeMuonEta_Histo";
    m_EfficiencyHistos[0][name1] = ibooker.book1D(name1.c_str(),name1.c_str(),50,-2.5,2.5);
    name2 = "ProbeMuonPhi_Histo";
    m_EfficiencyHistos[0][name2] = ibooker.book1D(name2.c_str(),name2.c_str(),24,-TMath::Pi(),TMath::Pi());
    name3 = "ProbeMuonPt_Histo";
    m_EfficiencyHistos[0][name3] = ibooker.book1D(name3.c_str(),name3.c_str(),50,0.,100.);
}

//_____________________________________________________________________
void L1TMuonDQMOffline::bookEfficiencyHistos(DQMStore::IBooker &ibooker, int ptCut) {
    if(m_verbose) cout << "[L1TMuonDQMOffline:] Booking Efficiency Plot Histos for pt cut = " << ptCut << endl;

    stringstream ptCutToTag; ptCutToTag << ptCut;
    string ptTag = ptCutToTag.str();

    ibooker.setCurrentFolder(m_HistFolder+"/numerators_and_denominators");

    std::vector<float> effVsPtBins(m_effVsPtBins.begin(), m_effVsPtBins.end());
    int nEffVsPtBins = effVsPtBins.size() - 1;
    float* ptBinsArray = &(effVsPtBins[0]);

    string name1 = "EffvsPt_" + ptTag + "_Den";
    m_EfficiencyHistos[ptCut][name1] = ibooker.book1D(name1.c_str(), name1.c_str(), nEffVsPtBins, ptBinsArray);
    string name2 = "EffvsPt_" + ptTag + "_Num";
    m_EfficiencyHistos[ptCut][name2] = ibooker.book1D(name2.c_str(), name2.c_str(), nEffVsPtBins, ptBinsArray);

    name1 = "EffvsPt_OPEN_" + ptTag + "_Den";
    m_EfficiencyHistos[ptCut][name1] = ibooker.book1D(name1.c_str(), name1.c_str(), nEffVsPtBins, ptBinsArray);
    name2 = "EffvsPt_OPEN_" + ptTag + "_Num";
    m_EfficiencyHistos[ptCut][name2] = ibooker.book1D(name2.c_str(), name2.c_str(), nEffVsPtBins, ptBinsArray);

    name1 = "EffvsPt_DOUBLE_" + ptTag + "_Den";
    m_EfficiencyHistos[ptCut][name1] = ibooker.book1D(name1.c_str(), name1.c_str(), nEffVsPtBins, ptBinsArray);
    name2 = "EffvsPt_DOUBLE_" + ptTag + "_Num";
    m_EfficiencyHistos[ptCut][name2] = ibooker.book1D(name2.c_str(), name2.c_str(), nEffVsPtBins, ptBinsArray);

    name1 = "EffvsPt_SINGLE_" + ptTag + "_Den";
    m_EfficiencyHistos[ptCut][name1] = ibooker.book1D(name1.c_str(), name1.c_str(), nEffVsPtBins, ptBinsArray);
    name2 = "EffvsPt_SINGLE_" + ptTag + "_Num";
    m_EfficiencyHistos[ptCut][name2] = ibooker.book1D(name2.c_str(), name2.c_str(), nEffVsPtBins, ptBinsArray);

////////////////////////////////////////////////

    std::vector<float> effVsPhiBins(m_effVsPhiBins.begin(), m_effVsPhiBins.end());
    int nEffVsPhiBins = effVsPhiBins.size() - 1;
    float* phiBinsArray = &(effVsPhiBins[0]);

    std::vector<float> effVsEtaBins(m_effVsEtaBins.begin(), m_effVsEtaBins.end());
    int nEffVsEtaBins = effVsEtaBins.size() - 1;
    float* etaBinsArray = &(effVsEtaBins[0]);

    name1 = "EffvsPhi_" + ptTag + "_Den";
    m_EfficiencyHistos[ptCut][name1] = ibooker.book1D(name1.c_str(), name1.c_str(), nEffVsPhiBins, phiBinsArray);
    name2 = "EffvsPhi_" + ptTag + "_Num";
    m_EfficiencyHistos[ptCut][name2] = ibooker.book1D(name2.c_str(), name2.c_str(), nEffVsPhiBins, phiBinsArray);

    name1 = "EffvsEta_" + ptTag + "_Den";
    m_EfficiencyHistos[ptCut][name1] = ibooker.book1D(name1.c_str(), name1.c_str(), nEffVsEtaBins, etaBinsArray);
    name2 = "EffvsEta_" + ptTag + "_Num";
    m_EfficiencyHistos[ptCut][name2] = ibooker.book1D(name2.c_str(), name2.c_str(), nEffVsEtaBins, etaBinsArray);

//////////////////////////////////////////////

    name1 = "EffvsPhi_OPEN_" + ptTag + "_Den";
    m_EfficiencyHistos[ptCut][name1] = ibooker.book1D(name1.c_str(), name1.c_str(), nEffVsPhiBins, phiBinsArray);
    name2 = "EffvsPhi_OPEN_" + ptTag + "_Num";
    m_EfficiencyHistos[ptCut][name2] = ibooker.book1D(name2.c_str(), name2.c_str(), nEffVsPhiBins, phiBinsArray);

    name1 = "EffvsEta_OPEN_" + ptTag + "_Den";
    m_EfficiencyHistos[ptCut][name1] = ibooker.book1D(name1.c_str(), name1.c_str(), nEffVsEtaBins, etaBinsArray);
    name2 = "EffvsEta_OPEN_" + ptTag + "_Num";
    m_EfficiencyHistos[ptCut][name2] = ibooker.book1D(name2.c_str(), name2.c_str(), nEffVsEtaBins, etaBinsArray);

//////////////////////////////////////////////

    name1 = "EffvsPhi_DOUBLE_" + ptTag + "_Den";
    m_EfficiencyHistos[ptCut][name1] = ibooker.book1D(name1.c_str(), name1.c_str(), nEffVsPhiBins, phiBinsArray);
    name2 = "EffvsPhi_DOUBLE_" + ptTag + "_Num";
    m_EfficiencyHistos[ptCut][name2] = ibooker.book1D(name2.c_str(), name2.c_str(), nEffVsPhiBins, phiBinsArray);

    name1 = "EffvsEta_DOUBLE_" + ptTag + "_Den";
    m_EfficiencyHistos[ptCut][name1] = ibooker.book1D(name1.c_str(), name1.c_str(), nEffVsEtaBins, etaBinsArray);
    name2 = "EffvsEta_DOUBLE_" + ptTag + "_Num";
    m_EfficiencyHistos[ptCut][name2] = ibooker.book1D(name2.c_str(), name2.c_str(), nEffVsEtaBins, etaBinsArray);

//////////////////////////////////////////////

    name1 = "EffvsPhi_SINGLE_" + ptTag + "_Den";
    m_EfficiencyHistos[ptCut][name1] = ibooker.book1D(name1.c_str(), name1.c_str(), nEffVsPhiBins, phiBinsArray);
    name2 = "EffvsPhi_SINGLE_" + ptTag + "_Num";
    m_EfficiencyHistos[ptCut][name2] = ibooker.book1D(name2.c_str(), name2.c_str(), nEffVsPhiBins, phiBinsArray);

    name1 = "EffvsEta_SINGLE_" + ptTag + "_Den";
    m_EfficiencyHistos[ptCut][name1] = ibooker.book1D(name1.c_str(), name1.c_str(), nEffVsEtaBins, etaBinsArray);
    name2 = "EffvsEta_SINGLE_" + ptTag + "_Num";
    m_EfficiencyHistos[ptCut][name2] = ibooker.book1D(name2.c_str(), name2.c_str(), nEffVsEtaBins, etaBinsArray);
}

//_____________________________________________________________________
const reco::Vertex L1TMuonDQMOffline::getPrimaryVertex( Handle<VertexCollection> & vertex,
                                 Handle<BeamSpot> & beamSpot ) {
    Vertex::Point posVtx;
    Vertex::Error errVtx;

    bool hasPrimaryVertex = false;

    if (vertex.isValid()) {
        vector<Vertex>::const_iterator vertexIt  = vertex->begin();
        vector<Vertex>::const_iterator vertexEnd = vertex->end();

        for (;vertexIt!=vertexEnd;++vertexIt) {
            if (vertexIt->isValid() && !vertexIt->isFake()) {
                posVtx = vertexIt->position();
                errVtx = vertexIt->error();
                hasPrimaryVertex = true;
                break;
            }
        }
    }

    if ( !hasPrimaryVertex ) {
        posVtx = beamSpot->position();
        errVtx(0,0) = beamSpot->BeamWidthX();
        errVtx(1,1) = beamSpot->BeamWidthY();
        errVtx(2,2) = beamSpot->sigmaZ();
    }
    const Vertex primaryVertex(posVtx,errVtx);
    return primaryVertex;
}

//_____________________________________________________________________
void L1TMuonDQMOffline::getTightMuons(edm::Handle<reco::MuonCollection> & muons,  const Vertex & vertex) {

    if (m_verbose) cout << "[L1TMuonDQMOffline:] Getting tight muons" << endl;
    m_TightMuons.clear();
    MuonCollection::const_iterator muonIt  = muons->begin();
    MuonCollection::const_iterator muonEnd = muons->end();

    for(; muonIt!=muonEnd; ++muonIt) {
        if (muon::isTightMuon((*muonIt), vertex)) {
            m_TightMuons.push_back(&(*muonIt));
        }
    }
    m_ControlHistos["NTightVsAll"]->Fill(muons->size(),m_TightMuons.size());
}

//_____________________________________________________________________
void L1TMuonDQMOffline::getProbeMuons(Handle<edm::TriggerResults> & trigResults,
                           edm::Handle<trigger::TriggerEvent> & trigEvent) {

    if (m_verbose) cout << "[L1TMuonDQMOffline:] getting probe muons" << endl;
    m_ProbeMuons.clear();

    vector<const reco::Muon*>::const_iterator probeCandIt   = m_TightMuons.begin();
//    vector<const reco::Muon*>::const_iterator probeMuIt  = m_ProbeMuons.begin();
    vector<const reco::Muon*>::const_iterator tightMuonsEnd = m_TightMuons.end();

    for (; probeCandIt!=tightMuonsEnd; ++probeCandIt) {
        bool tagHasTrig = false;
        vector<const reco::Muon*>::const_iterator tagCandIt  = m_TightMuons.begin();
        float deltar = 0.;

        for (; tagCandIt!=tightMuonsEnd; ++tagCandIt) {
            float eta = (*tagCandIt)->eta();
            float phi = (*tagCandIt)->phi();
            float pt  = (*tagCandIt)->pt();
            float dEta = eta - (*probeCandIt)->eta();
            float dPhi = phi - (*probeCandIt)->phi();

            deltar = sqrt(dEta*dEta + dPhi*dPhi);

            if ( (*tagCandIt) == (*probeCandIt) || (deltar<0.7) ) continue; // CB has a little bias for closed-by muons     
                tagHasTrig |= matchHlt(trigEvent,(*tagCandIt));
            if (tagHasTrig) {
                m_EfficiencyHistos[0]["TagMuonEta_Histo"]->Fill(eta);
                m_EfficiencyHistos[0]["TagMuonPhi_Histo"]->Fill(phi);
                m_EfficiencyHistos[0]["TagMuonPt_Histo"]->Fill(pt);
             }
        }
        if (tagHasTrig) m_ProbeMuons.push_back((*probeCandIt));
    }
    m_ControlHistos["NProbesVsTight"]->Fill(m_TightMuons.size(),m_ProbeMuons.size());
}

//_____________________________________________________________________
void L1TMuonDQMOffline::getMuonGmtPairs(edm::Handle<l1t::MuonBxCollection> & gmtCands) {

    m_MuonGmtPairs.clear();
    if (m_verbose) cout << "[L1TMuonDQMOffline:] Getting muon GMT pairs" << endl;

    vector<const reco::Muon*>::const_iterator probeMuIt  = m_ProbeMuons.begin();
    vector<const reco::Muon*>::const_iterator probeMuEnd = m_ProbeMuons.end();
    vector<l1t::Muon> gmtContainer;

    for (auto mu = gmtCands->begin(0); mu != gmtCands->end(0); ++mu) {
        gmtContainer.push_back(*mu);
    }

    vector<l1t::Muon>::const_iterator gmtIt;
    vector<l1t::Muon>::const_iterator gmtEnd = gmtContainer.end();

    for (; probeMuIt!=probeMuEnd; ++probeMuIt) {
        float eta = (*probeMuIt)->eta();
        float phi = (*probeMuIt)->phi();
        float pt  = (*probeMuIt)->pt();

        m_EfficiencyHistos[0]["ProbeMuonEta_Histo"]->Fill(eta);
        m_EfficiencyHistos[0]["ProbeMuonPhi_Histo"]->Fill(phi);
        m_EfficiencyHistos[0]["ProbeMuonPt_Histo"]->Fill(pt);

        MuonGmtPair pairBestCand((*probeMuIt),0);
//      pairBestCand.propagate(m_BField,m_propagatorAlong,m_propagatorOpposite);
        gmtIt = gmtContainer.begin();

        for(; gmtIt!=gmtEnd; ++gmtIt) {
            MuonGmtPair pairTmpCand((*probeMuIt),&(*gmtIt));
//          pairTmpCand.propagate(m_BField,m_propagatorAlong,m_propagatorOpposite);     

            if ( (pairTmpCand.dR() < m_MaxGmtMuonDR) && (pairTmpCand.dR() < pairBestCand.dR() ) ) {
                pairBestCand = pairTmpCand;
            }

        }
        m_MuonGmtPairs.push_back(pairBestCand);
        m_ControlHistos["MuonGmtDeltaR"]->Fill(pairBestCand.dR());
    }
}

//_____________________________________________________________________
bool L1TMuonDQMOffline::matchHlt(edm::Handle<TriggerEvent>  & triggerEvent, const reco::Muon * mu) {

    double matchDeltaR = 9999;

    TriggerObjectCollection trigObjs = triggerEvent->getObjects();

    vector<int>::const_iterator trigIndexIt  = m_trigIndices.begin();
    vector<int>::const_iterator trigIndexEnd = m_trigIndices.end();

    for(; trigIndexIt!=trigIndexEnd; ++trigIndexIt) {
        const vector<string> moduleLabels(m_hltConfig.moduleLabels(*trigIndexIt));
        const unsigned moduleIndex = m_hltConfig.size((*trigIndexIt))-2;
        const unsigned hltFilterIndex = triggerEvent->filterIndex(InputTag(moduleLabels[moduleIndex],"",m_trigProcess));

        if (hltFilterIndex < triggerEvent->sizeFilters()) {
            const Keys triggerKeys(triggerEvent->filterKeys(hltFilterIndex));
            const Vids triggerVids(triggerEvent->filterIds(hltFilterIndex));
            const unsigned nTriggers = triggerVids.size();
            for (size_t iTrig = 0; iTrig < nTriggers; ++iTrig) {
                const TriggerObject trigObject = trigObjs[triggerKeys[iTrig]];
                double dRtmp = deltaR((*mu),trigObject);
                if (dRtmp < matchDeltaR) matchDeltaR = dRtmp;
            }
        }
    }
    return (matchDeltaR < m_MaxHltMuonDR);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TMuonDQMOffline);
