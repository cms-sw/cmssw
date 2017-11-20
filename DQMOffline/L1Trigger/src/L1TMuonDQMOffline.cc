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
/*
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
DEFINE_ENUM_WITH_STRING_CONVERSIONS(Qual, (QUAL_any)(QUAL_Open)(QUAL_Double)(QUAL_Single)(QUAL_else))
*/
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
    m_TagPtCut(ps.getUntrackedParameter<double>("tagPtCut")),
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
//    m_MaxMuonEta   = 2.4;
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
    bookResolutionHistos(ibooker);

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

    if (m_verbose) cout << "[L1TMuonDQMOffline:] Computing efficiencies" << endl;

    vector<MuonGmtPair>::const_iterator muonGmtPairsIt  = m_MuonGmtPairs.begin();
    vector<MuonGmtPair>::const_iterator muonGmtPairsEnd = m_MuonGmtPairs.end();

    for(; muonGmtPairsIt!=muonGmtPairsEnd; ++muonGmtPairsIt) {
        vector<int>::const_iterator gmtPtCutsIt  = m_GmtPtCuts.begin();
        vector<int>::const_iterator gmtPtCutsEnd = m_GmtPtCuts.end();

              //  const int resLoopNo = 0;
        double gmtPt = muonGmtPairsIt->gmtPt();
        if( (muonGmtPairsIt->etaRegion() != ETAREGION_out) && (gmtPt > 0) ){
            for(int resLoop = RES_Pt; resLoop <= RES_Charge; resLoop++){
                Res res = static_cast<Res>(resLoop);
                int* resLoopNo; (*resLoopNo) = 0;
                for(int etaQualCaseLoop = 1; etaQualCaseLoop <= 4; etaQualCaseLoop++){
                    EtaRegion eta =  std::get<0>(muonGmtPairsIt->etaQual(etaQualCaseLoop));
                    Qual qual =  std::get<1>(muonGmtPairsIt->etaQual(etaQualCaseLoop));
                    histoInfoRes = {res, gmtPt, 0., eta, qual};
                    m_ResolutionHistos[histoInfoRes]->Fill(std::get<(*resLoopNo)>(muonGmtPairsIt->pairInfoRes));
                }
            }
        }
/*
        for (; gmtPtCutsIt!=gmtPtCutsEnd; ++ gmtPtCutsIt) {
            int gmtPtCut = (*gmtPtCutsIt);
            bool gmtAboveCut = (gmtPt > gmtPtCut);
            std::string ptTag = std::to_string(gmtPtCut);

            if( (muonGmtPairsIt->etaRegion()) != ETAREGION_out){
                for(int effLoop = EFF_Pt; effLoop <= EFF_Eta; effLoop++){
                    Eff eff = static_cast<Eff>(effLoop);
                    if(eff != EFF_Pt){
                       if(muonGmtPairsIt->pt() < 1.25*gmtPtCut) break;    // efficiency in eta/phi at plateau
                    }
                    for(int etaQualCaseLoop = 1; etaQualCaseLoop <= 4; etaQualCaseLoop++){
                        EtaRegion eta =  std::get<0>(muonGmtPairsIt->etaQual(etaQualCaseLoop));
                        Qual qual =  std::get<1>(muonGmtPairsIt->etaQual(etaQualCaseLoop));
                        histoInfoEffDen = {eff, gmtPt, gmtPtCut, 0., eta, QUAL_no};
                        if( (etaQualCaseLoop == 1) || (etaQualCaseLoop == 3) )
                            m_EfficiencyHistos[histoInfoEffDen]->Fill(std::get<0>(muonGmtPairsIt->pairInfoEff));
                        if(gmtAboveCut){
                            histoInfoEffNum = {eff, gmtPt, gmtPtCut, 0., eta, qual};
                            m_EfficiencyHistos[histoInfoEffNum]->Fill(std::get<effLoop>(muonGmtPairsIt->pairInfoEff));
                        }
                    }
                }
            }
        }
*/
    }

    if (m_verbose) cout << "[L1TMuonDQMOffline:] Computation finished" << endl;
}

//_____________________________________________________________________
void L1TMuonDQMOffline::bookControlHistos(DQMStore::IBooker& ibooker) {
    if(m_verbose) cout << "[L1TMuonDQMOffline:] Booking Control Plot Histos" << endl;

    ibooker.setCurrentFolder(m_HistFolder+"/control_variables");

    string name = "MuonGmtDeltaR";
    m_ControlHistos[CONTROL_MuonGmtDeltaR] = ibooker.book1D(name.c_str(),name.c_str(),25.,0.,2.5);

    name = "NTightVsAll";
    m_ControlHistos[CONTROL_NTightVsAll] = ibooker.book2D(name.c_str(),name.c_str(),16,-0.5,15.5,16,-0.5,15.5);

    name = "NProbesVsTight";
    m_ControlHistos[CONTROL_NProbesVsTight] = ibooker.book2D(name.c_str(),name.c_str(),8,-0.5,7.5,8,-0.5,7.5);

////////////////////////////////////////////////////////////////////////////////////////////////////
/*
    string name3 = "TagMuonPt_Histo";
    m_EfficiencyHistos[histoInfoEffTagPt] = ibooker.book1D(name3.c_str(),name3.c_str(),50,0.,100.);
    string name2 = "TagMuonPhi_Histo";
    m_EfficiencyHistos[histoInfoEffTagPhi] = ibooker.book1D(name2.c_str(),name2.c_str(),24,-TMath::Pi(),TMath::Pi());
    string name1 = "TagMuonEta_Histo";
    m_EfficiencyHistos[histoInfoEffTagEta] = ibooker.book1D(name1.c_str(),name1.c_str(),50,-2.5,2.5);
    
    //*****
    name3 = "ProbeMuonPt_Histo";
    m_EfficiencyHistos[histoInfoEffProbePt] = ibooker.book1D(name3.c_str(),name3.c_str(),50,0.,100.);
    name2 = "ProbeMuonPhi_Histo";
    m_EfficiencyHistos[histoInfoEffTagPhi] = ibooker.book1D(name2.c_str(),name2.c_str(),24,-TMath::Pi(),TMath::Pi());
    name1 = "ProbeMuonEta_Histo";
    m_EfficiencyHistos[histoInfoEffTagEta] = ibooker.book1D(name1.c_str(),name1.c_str(),50,-2.5,2.5);
*/
}

//_____________________________________________________________________
void L1TMuonDQMOffline::bookEfficiencyHistos(DQMStore::IBooker &ibooker, int ptCut) {
    if(m_verbose) cout << "[L1TMuonDQMOffline:] Booking Efficiency Plot Histos for pt cut = " << ptCut << endl;

    stringstream ptCutToTag; ptCutToTag << ptCut;
    string ptTag = ptCutToTag.str();

    ibooker.setCurrentFolder(m_HistFolder+"/numerators_and_denominators");
/*
    std::vector<float> effVsPtBins(m_effVsPtBins.begin(), m_effVsPtBins.end());
    nEffVsPtBins = effVsPtBins.size() - 1;
    ptBinsArray = &(effVsPtBins[0]);

    std::vector<float> effVsPhiBins(m_effVsPhiBins.begin(), m_effVsPhiBins.end());
    nEffVsPhiBins = effVsPhiBins.size() - 1;
    phiBinsArray = &(effVsPhiBins[0]);

    std::vector<float> effVsEtaBins(m_effVsEtaBins.begin(), m_effVsEtaBins.end());
    nEffVsEtaBins = effVsEtaBins.size() - 1;
    etaBinsArray = &(effVsEtaBins[0]);

    std::tuple<int, float*> getHistBinsEff(Eff eff1){
        if (eff1 == EFF_Pt)        return {nEffVsPtBins,ptBinsArray};
        if (eff1 == EFF_Phi)       return {nEffVsPhiBins,phiBinsArray};
        if (eff1 == EFF_Eta)       return {nEffVsEtaBins,etaBinsArray};
        throw std::invalid_argument("eff1");
    }

    for(int effLoop = EFF_Pt; effLoop <= EFF_Eta; effLoop++){
        Eff eff = static_cast<Eff>(effLoop);
        int nbins = std::get<0>(getHistBinsEff(eff));
        float* binsArray = std::get<1>(getHistBinsEff(eff));
        for(int etaRegionLoop = ETAREGION_any; etaRegionLoop != ETAREGION_out; etaRegionLoop++){
            EtaRegion etaReg = static_cast<EtaRegion>(etaRegionLoop);
            histoInfoEffDen = {eff, 0., ptCut, 0., etaReg, QUAL_no};
            std::string name = std::string(ToString(eff)) + "__" + std::to_string(0) + "__" + std::to_string(ptCut) + "__" + std::to_string(0) + "__" + ToString(etaReg) + "__" + ToString(QUAL_no) + "__Den";
            m_EfficiencyHistos[histoInfoEffDen] = ibooker.book1D(name.c_str(),name.c_str(), nbins, binsArray);
            for(int qualLoop = QUAL_any; qualLoop != QUAL_else; qualLoop++){
                Qual qual = static_cast<Qual>(qualLoop);
                histoInfoEffNum = {eff, 0., ptCut, 0., etaReg, qual};
                std::string name = std::string(ToString(eff)) + "__" + std::to_string(0) + "__" + std::to_string(ptCut) + "__" + std::to_string(0) + "__" + ToString(etaReg) + "__" + ToString(qual) + "__Num";
                m_EfficiencyHistos[histoInfoEffNum] = ibooker.book1D(name.c_str(),name.c_str(), nbins, binsArray);
            }
        }
    }
*/
}

void L1TMuonDQMOffline::bookResolutionHistos(DQMStore::IBooker &ibooker) {
    if(m_verbose) cout << "[L1TMuonOffline:] Booking Resolution Plot Histos" << endl;
    ibooker.setCurrentFolder(m_HistFolder+"/resolution");

    for(int resLoop = RES_Pt; resLoop <= RES_Charge; resLoop++){
        Res res = static_cast<Res>(resLoop);
        int nbins = std::get<0>(getHistBinsRes(res));
        double xmin = std::get<1>(getHistBinsRes(res));
        double xmax = std::get<2>(getHistBinsRes(res));
        for(int etaRegionLoop = ETAREGION_any; etaRegionLoop != ETAREGION_out; etaRegionLoop++){
            EtaRegion etaReg = static_cast<EtaRegion>(etaRegionLoop);
            for(int qualLoop = QUAL_any; qualLoop != QUAL_else; qualLoop++){
                Qual qual = static_cast<Qual>(qualLoop);
                histoInfoRes = {res, 0., 0., etaReg, qual};
                std::string name = std::string(ToString(res)) + "__" + std::to_string(0) + "__" + std::to_string(0) + "__" + ToString(etaReg) + "__" + ToString(qual);
                m_ResolutionHistos[histoInfoRes] = ibooker.book1D(name.c_str(),name.c_str(), nbins, xmin, xmax);
            }
        }
    }

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
    m_ControlHistos[CONTROL_NTightVsAll]->Fill(muons->size(),m_TightMuons.size());
}

//_____________________________________________________________________
void L1TMuonDQMOffline::getProbeMuons(Handle<edm::TriggerResults> & trigResults,
                           edm::Handle<trigger::TriggerEvent> & trigEvent) {

    if (m_verbose) cout << "[L1TMuonDQMOffline:] getting probe muons" << endl;
    m_ProbeMuons.clear();
    std::vector<const reco::Muon*>  tagMuonsInHist;

    tagMuonsInHist.clear();

    vector<const reco::Muon*>::const_iterator probeCandIt   = m_TightMuons.begin();
//    vector<const reco::Muon*>::const_iterator probeMuIt  = m_ProbeMuons.begin();
    vector<const reco::Muon*>::const_iterator tightMuonsEnd = m_TightMuons.end();

    for (; probeCandIt!=tightMuonsEnd; ++probeCandIt) {
        bool isProbe = false;
        vector<const reco::Muon*>::const_iterator tagCandIt  = m_TightMuons.begin();
        float deltar = 0.;

        for (; tagCandIt!=tightMuonsEnd; ++tagCandIt) {
            bool tagMuonAlreadyInHist = false;
            bool tagHasTrig = false;
            float eta = (*tagCandIt)->eta();
            float phi = (*tagCandIt)->phi();
            float pt  = (*tagCandIt)->pt();
            float dEta = eta - (*probeCandIt)->eta();
            float dPhi = phi - (*probeCandIt)->phi();
            deltar = sqrt(dEta*dEta + dPhi*dPhi);

            if ( (*tagCandIt) == (*probeCandIt) || (deltar<0.7) ) continue; // CB has a little bias for closed-by muons     
            tagHasTrig = matchHlt(trigEvent,(*tagCandIt)) && (pt > m_TagPtCut);
            isProbe |= tagHasTrig;
            if (tagHasTrig) {
                if (std::distance(m_TightMuons.begin(), m_TightMuons.end()) > 2 ) {
                    for (vector<const reco::Muon*>::const_iterator tagMuonsInHistIt = tagMuonsInHist.begin(); tagMuonsInHistIt!=tagMuonsInHist.end(); ++tagMuonsInHistIt) {
                        if ( (*tagCandIt) == (*tagMuonsInHistIt) )  {
                            tagMuonAlreadyInHist = true;
                           break;
                        }
                    }
                    if (tagMuonAlreadyInHist == false) tagMuonsInHist.push_back((*tagCandIt));
                }
                if (tagMuonAlreadyInHist == false) {
 //                   for(int effLoop = EFF_Pt; effLoop <= EFF_Eta; effLoop++){
     //                   Eff eff = static_cast<Eff>(effLoop);
/*
                        histoInfoEffTagPt = {EFF_Pt, 0., 0, 0., 0, QUAL_no};
                        m_EfficiencyHistos[histoInfoEffTagPt]->Fill(pt);
                        histoInfoEffTagPhi = {EFF_Phi, 0., 0, 0., 0, QUAL_no};
                        m_EfficiencyHistos[histoInfoEffTagPhi]->Fill(phi);
                        histoInfoEffTagEta = {EFF_Eta, 0., 0, 0., 0, QUAL_no};
                        m_EfficiencyHistos[histoInfoEffTagEta]->Fill(eta);
   //                 }
*/
                }
            }
        }
        if (isProbe) m_ProbeMuons.push_back((*probeCandIt));
    }
    m_ControlHistos[CONTROL_NProbesVsTight]->Fill(m_TightMuons.size(),m_ProbeMuons.size());
}

//_____________________________________________________________________
void L1TMuonDQMOffline::getMuonGmtPairs(edm::Handle<l1t::MuonBxCollection> & gmtCands) {

    m_MuonGmtPairs.clear();
    if (m_verbose) cout << "[L1TMuonDQMOffline:] Getting muon GMT pairs" << endl;

    vector<const reco::Muon*>::const_iterator probeMuIt  = m_ProbeMuons.begin();
    vector<const reco::Muon*>::const_iterator probeMuEnd = m_ProbeMuons.end();

    l1t::MuonBxCollection::const_iterator gmtIt;
    l1t::MuonBxCollection::const_iterator gmtEnd = gmtCands->end();

    for (; probeMuIt!=probeMuEnd; ++probeMuIt) {
        float eta = (*probeMuIt)->eta();
        float phi = (*probeMuIt)->phi();
        float pt  = (*probeMuIt)->pt();
/*
        histoInfoEffProbePt = {EFF_Pt, 0., 0, 0., 0, QUAL_no};
        m_EfficiencyHistos[histoInfoEffProbePt]->Fill(pt);
        histoInfoEffProbePhi = {EFF_Phi, 0., 0, 0., 0, QUAL_no};
        m_EfficiencyHistos[histoInfoEffProbePhi]->Fill(phi);
        histoInfoEffProbeEta = {EFF_Eta, 0., 0, 0., 0, QUAL_no};
        m_EfficiencyHistos[histoInfoEffProbeEta]->Fill(eta);
*/
        MuonGmtPair pairBestCand((*probeMuIt),nullptr);
//      pairBestCand.propagate(m_BField,m_propagatorAlong,m_propagatorOpposite);
        gmtIt = gmtCands->begin();

        for(; gmtIt!=gmtEnd; ++gmtIt) {
            MuonGmtPair pairTmpCand((*probeMuIt),&(*gmtIt));
//          pairTmpCand.propagate(m_BField,m_propagatorAlong,m_propagatorOpposite);     

            if ( (pairTmpCand.dR() < m_MaxGmtMuonDR) && (pairTmpCand.dR() < pairBestCand.dR() ) ) {
                pairBestCand = pairTmpCand;
            }

        }
        m_MuonGmtPairs.push_back(pairBestCand);
        m_ControlHistos[CONTROL_MuonGmtDeltaR]->Fill(pairBestCand.dR());
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
