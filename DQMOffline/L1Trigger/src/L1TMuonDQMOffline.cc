/**
 * \file L1TMuonDQMOffline.cc
 *
 * \author J. Pela, C. Battilana
 *
 * Stage2 Muons implementation: Anna Stakia
 *
 */

#include "DQMOffline/L1Trigger/interface/L1TMuonDQMOffline.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"

#include <array>

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

L1TMuonDQMOffline::etaRegion MuonGmtPair::etaRegion() const {
    if (std::abs(eta()) < 0.83) return L1TMuonDQMOffline::ETAREGION_BMTF;
    if (std::abs(eta()) < 1.24) return L1TMuonDQMOffline::ETAREGION_OMTF;
    if (std::abs(eta()) < 2.4)  return L1TMuonDQMOffline::ETAREGION_EMTF;
    return L1TMuonDQMOffline::ETAREGION_OUT;
}

double MuonGmtPair::getDeltaVar(const L1TMuonDQMOffline::resType type) const {
    if (type == L1TMuonDQMOffline::RES_PT)      return gmtPt() - pt();
    if (type == L1TMuonDQMOffline::RES_1OVERPT) return 1/gmtPt() - 1/pt();
    if (type == L1TMuonDQMOffline::RES_QOVERPT) return gmtCharge()/gmtPt() - charge()/pt();
    if (type == L1TMuonDQMOffline::RES_PHI)     return gmtPhi() - phi();
    if (type == L1TMuonDQMOffline::RES_ETA)     return gmtEta() - eta();
    if (type == L1TMuonDQMOffline::RES_CH)      return gmtCharge() - charge();
    return -999.;
}

double MuonGmtPair::getVar(const L1TMuonDQMOffline::effType type) const {
    if (type == L1TMuonDQMOffline::EFF_PT)  return pt();
    if (type == L1TMuonDQMOffline::EFF_PHI) return phi();
    if (type == L1TMuonDQMOffline::EFF_ETA) return eta();
    return -999.;
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
    m_effTypes({EFF_PT, EFF_PHI, EFF_ETA, EFF_VTX}),
    m_resTypes({RES_PT, RES_1OVERPT, RES_QOVERPT, RES_PHI, RES_ETA}),
    m_etaRegions({ETAREGION_ALL, ETAREGION_BMTF, ETAREGION_OMTF, ETAREGION_EMTF}),
    m_qualLevelsRes({QUAL_ALL}),
    m_effStrings({ {EFF_PT, "pt"}, {EFF_PHI, "phi"}, {EFF_ETA, "eta"}, {EFF_VTX, "vtx"} }),
    m_effLabelStrings({ {EFF_PT, "p_{T} (GeV)"}, {EFF_PHI, "#phi"}, {EFF_ETA, "#eta"}, {EFF_VTX, "# vertices"} }),
    m_resStrings({ {RES_PT, "pt"}, {RES_1OVERPT, "1overpt"}, {RES_QOVERPT, "qoverpt"}, {RES_PHI, "phi"}, {RES_ETA, "eta"}, {RES_CH, "charge"} }),
    m_resLabelStrings({ {RES_PT, "p_{T}^{L1} - p_{T}^{reco}"}, {RES_1OVERPT, "1/p_{T}^{L1} - 1/p_{T}^{reco}"}, {RES_QOVERPT, "q^{L1}/p_{T}^{L1} - q^{reco}/p_{T}^{reco}"}, {RES_PHI, "#phi_{L1} - #phi_{reco}"}, {RES_ETA, "#eta_{L1} - #eta_{reco}"}, {RES_CH, "charge^{L1} - charge^{reco}"} }),
    m_etaStrings({ {ETAREGION_ALL, "etaMin0_etaMax2p4"}, {ETAREGION_BMTF, "etaMin0_etaMax0p83"}, {ETAREGION_OMTF, "etaMin0p83_etaMax1p24"}, {ETAREGION_EMTF, "etaMin1p24_etaMax2p4"} }),
    m_qualStrings({ {QUAL_ALL, "qualAll"}, {QUAL_OPEN, "qualOpen"}, {QUAL_DOUBLE, "qualDouble"}, {QUAL_SINGLE, "qualSingle"} }),
    m_verbose(ps.getUntrackedParameter<bool>("verbose")),
    m_HistFolder(ps.getUntrackedParameter<string>("histFolder")),
    m_TagPtCut(ps.getUntrackedParameter<double>("tagPtCut")),
    m_recoToL1PtCutFactor(ps.getUntrackedParameter<double>("recoToL1PtCutFactor")),
    m_cutsVPSet(ps.getUntrackedParameter<std::vector<edm::ParameterSet>>("cuts")),
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
    m_effVsEtaBins(ps.getUntrackedParameter<std::vector<double>>("efficiencyVsEtaBins")),
    m_effVsVtxBins(ps.getUntrackedParameter<std::vector<double>>("efficiencyVsVtxBins")),
    m_maxGmtMuonDR(0.3),
    m_minTagProbeDR(0.5),
    m_maxHltMuonDR(0.3)
{
    if (m_verbose) cout << "[L1TMuonDQMOffline:] ____________ Storage initialization ____________ " << endl;

    for (const auto cutsPSet : m_cutsVPSet) {
        const auto qCut = cutsPSet.getUntrackedParameter<int>("qualCut");
        qualLevel qLevel = QUAL_ALL;
        if (qCut > 11) {
            qLevel = QUAL_SINGLE;
        } else if (qCut > 7) {
            qLevel = QUAL_DOUBLE;
        } else if (qCut > 3) {
            qLevel = QUAL_OPEN;
        }
        m_cuts.emplace_back(std::make_pair(cutsPSet.getUntrackedParameter<int>("ptCut"), qLevel));
    }
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
    bookEfficiencyHistos(ibooker);
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

    const auto nVtx = getNVertices(vertex);
    const Vertex primaryVertex = getPrimaryVertex(vertex,beamSpot);

    getTightMuons(muons,primaryVertex);
    getProbeMuons(trigResults,trigEvent); // CB add flag to run on orthogonal datasets (no T&P)

    getMuonGmtPairs(gmtCands);

    if (m_verbose) cout << "[L1TMuonDQMOffline:] Computing efficiencies" << endl;

    vector<MuonGmtPair>::const_iterator muonGmtPairsIt  = m_MuonGmtPairs.begin();
    vector<MuonGmtPair>::const_iterator muonGmtPairsEnd = m_MuonGmtPairs.end();

    // To fill once for global eta and once for TF eta region of the L1T muon.
    // The second entry is a placeholder and will be replaced by the TF eta region of the L1T muon.
    std::array<etaRegion, 2> regsToFill { {ETAREGION_ALL, ETAREGION_ALL} };

    for(; muonGmtPairsIt!=muonGmtPairsEnd; ++muonGmtPairsIt) {
        // Fill the resolution histograms
        if( (muonGmtPairsIt->etaRegion() != ETAREGION_OUT) && (muonGmtPairsIt->gmtPt() > 0) ){
            regsToFill[1] = muonGmtPairsIt->etaRegion();
            m_histoKeyResType histoKeyRes = {RES_PT, ETAREGION_ALL, QUAL_ALL};
            for (const auto var : m_resTypes) {
                const auto varToFill = muonGmtPairsIt->getDeltaVar(var);
                std::get<0>(histoKeyRes) = var;
                // Fill for the global eta and for TF eta region that the probe muon is in
                for (const auto regToFill : regsToFill) {
                    std::get<1>(histoKeyRes) = regToFill;
                    for (const auto qualLevel : m_qualLevelsRes) {
                        // This assumes that the qualLevel enum has increasing qualities
                        // HW quality levels can be 0, 4, 8, or 12
                        int qualCut = qualLevel * 4;
                        if (muonGmtPairsIt->gmtQual() >= qualCut) {
                            std::get<2>(histoKeyRes) = qualLevel;
                            m_ResolutionHistos[histoKeyRes]->Fill(varToFill);
                        }
                    }
                }
            }
        }

        // Fill the efficiency numerator and denominator histograms
        if (muonGmtPairsIt->etaRegion() != ETAREGION_OUT) {
            unsigned int cutsCounter = 0;
            for (const auto cut : m_cuts) {
                const auto gmtPtCut = cut.first;
                const auto qualLevel = cut.second;
                const bool gmtAboveCut = (muonGmtPairsIt->gmtPt() > gmtPtCut);

                // default keys
                m_histoKeyEffDenVarType histoKeyEffDenVar = {EFF_PT, gmtPtCut, ETAREGION_ALL};
                m_histoKeyEffNumVarType histoKeyEffNumVar = {EFF_PT, gmtPtCut, ETAREGION_ALL, qualLevel};

                regsToFill[1] = muonGmtPairsIt->etaRegion();
                for(const auto var : m_effTypes) {
                    if(var != EFF_PT){
                       if (muonGmtPairsIt->pt() < m_recoToL1PtCutFactor * gmtPtCut) break; // efficiency at plateau
                    }
                    double varToFill;
                    if (var == EFF_VTX) {
                        varToFill = static_cast<double>(nVtx);
                    } else {
                        varToFill = muonGmtPairsIt->getVar(var);
                    }
                    // Fill denominators
                    if (var == EFF_ETA) {
                        m_EfficiencyDenEtaHistos[gmtPtCut]->Fill(varToFill);
                    } else {
                        std::get<0>(histoKeyEffDenVar) = var;
                        // Fill for the global eta and for TF eta region that the probe muon is in
                        for (const auto regToFill : regsToFill) {
                            if (var == EFF_PT) {
                                if (cutsCounter == 0) {
                                    m_EfficiencyDenPtHistos[regToFill]->Fill(varToFill);
                                }
                            } else {
                                std::get<2>(histoKeyEffDenVar) = regToFill;
                                m_EfficiencyDenVarHistos[histoKeyEffDenVar]->Fill(varToFill);
                            }
                        }
                    }
                    // Fill numerators
                    std::get<0>(histoKeyEffNumVar) = var;
                    // This assumes that the qualLevel enum has increasing qualities
                    if (gmtAboveCut && muonGmtPairsIt->gmtQual() >= qualLevel * 4) {
                        if (var == EFF_ETA) {
                            m_histoKeyEffNumEtaType histoKeyEffNumEta = {gmtPtCut, qualLevel};
                            m_EfficiencyNumEtaHistos[histoKeyEffNumEta]->Fill(varToFill);
                        } else {
                            std::get<3>(histoKeyEffNumVar) = qualLevel;
                            // Fill for the global eta and for TF eta region that the probe muon is in
                            for (const auto regToFill : regsToFill) {
                                std::get<2>(histoKeyEffNumVar) = regToFill;
                                m_EfficiencyNumVarHistos[histoKeyEffNumVar]->Fill(varToFill);
                            }
                        }
                    }
                }
                ++cutsCounter;
            }
        }
    }

    if (m_verbose) cout << "[L1TMuonDQMOffline:] Computation finished" << endl;
}

//_____________________________________________________________________
void L1TMuonDQMOffline::bookControlHistos(DQMStore::IBooker& ibooker) {
    if(m_verbose) cout << "[L1TMuonDQMOffline:] Booking Control Plot Histos" << endl;

    ibooker.setCurrentFolder(m_HistFolder+"/control_variables");

    m_ControlHistos[CTRL_MUONGMTDELTAR] = ibooker.book1D("MuonGmtDeltaR", "MuonGmtDeltaR; #DeltaR", 25., 0., 2.5);
    m_ControlHistos[CTRL_NTIGHTVSALL] = ibooker.book2D("NTightVsAll", "NTightVsAll; # muons; # tight muons", 20, -0.5, 19.5, 16, -0.5, 15.5);
    m_ControlHistos[CTRL_NPROBESVSTIGHT] = ibooker.book2D("NProbesVsTight", "NProbesVsTight; # tight muons; # probe muons", 8, -0.5, 7.5, 8, -0.5, 7.5);

    m_ControlHistos[CTRL_TAGPT] = ibooker.book1D("TagMuonPt", "TagMuonPt; p_{T}", 50, 0., 100.);
    m_ControlHistos[CTRL_TAGPHI] = ibooker.book1D("TagMuonPhi", "TagMuonPhi; #phi", 66, -3.3, 3.3);
    m_ControlHistos[CTRL_TAGETA] = ibooker.book1D("TagMuonEta", "TagMuonEta; #eta", 50, -2.5, 2.5);

    m_ControlHistos[CTRL_PROBEPT] = ibooker.book1D("ProbeMuonPt", "ProbeMuonPt; p_{T}", 50, 0., 100.);
    m_ControlHistos[CTRL_PROBEPHI] = ibooker.book1D("ProbeMuonPhi", "ProbeMuonPhi; #phi", 66, -3.3, 3.3);
    m_ControlHistos[CTRL_PROBEETA] = ibooker.book1D("ProbeMuonEta", "ProbeMuonEta; #eta", 50, -2.5, 2.5);

    m_ControlHistos[CTRL_TAGPROBEDR] = ibooker.book1D("TagMuonProbeMuonDeltaR", "TagMuonProbeMuonDeltaR; #DeltaR", 50, 0.,5.0);
}

//_____________________________________________________________________
void L1TMuonDQMOffline::bookEfficiencyHistos(DQMStore::IBooker &ibooker) {
    ibooker.setCurrentFolder(m_HistFolder+"/numerators_and_denominators");

    for(const auto var : m_effTypes) {
        auto histBins = getHistBinsEff(var);
        // histograms for eta variable get a special treatment
        if (var == EFF_ETA) {
            for (const auto cut : m_cuts) {
                const auto gmtPtCut = cut.first;
                const auto qualLevel = cut.second;
                std::string name = "effDen_"+m_effStrings[var]+"_"+std::to_string(gmtPtCut);
                m_EfficiencyDenEtaHistos[gmtPtCut] = ibooker.book1D(name, name+";"+m_effLabelStrings[var], histBins.size()-1, &histBins[0]);
                name = "effNum_"+m_effStrings[var]+"_"+std::to_string(gmtPtCut)+"_"+m_qualStrings[qualLevel];
                m_histoKeyEffNumEtaType histoKeyEffNumEta = {gmtPtCut, qualLevel};
                m_EfficiencyNumEtaHistos[histoKeyEffNumEta] = ibooker.book1D(name, name+";"+m_effLabelStrings[var], histBins.size()-1, &histBins[0]);
            }
        } else {
            for (const auto etaReg : m_etaRegions) {
                // denominator histograms for pt variable get a special treatment
                if (var == EFF_PT) {
                    std::string name = "effDen_"+m_effStrings[var]+"_"+m_etaStrings[etaReg];
                    m_EfficiencyDenPtHistos[etaReg] = ibooker.book1D(name, name+";"+m_effLabelStrings[var], histBins.size()-1, &histBins[0]);
                } else {
                    for (const auto cut : m_cuts) {
                        const int gmtPtCut = cut.first;
                        std::string name = "effDen_"+m_effStrings[var]+"_"+std::to_string(gmtPtCut)+"_"+m_etaStrings[etaReg];
                        m_histoKeyEffDenVarType histoKeyEffDenVar = {var, gmtPtCut, etaReg};
                        m_EfficiencyDenVarHistos[histoKeyEffDenVar] = ibooker.book1D(name, name+";"+m_effLabelStrings[var], histBins.size()-1, &histBins[0]);
                    }
                }
                for (const auto cut : m_cuts) {
                    const auto gmtPtCut = cut.first;
                    const auto qualLevel = cut.second;
                    std::string name = "effNum_"+m_effStrings[var]+"_"+std::to_string(gmtPtCut)+"_"+m_etaStrings[etaReg]+"_"+m_qualStrings[qualLevel];
                    m_histoKeyEffNumVarType histoKeyEffNum = {var, gmtPtCut, etaReg, qualLevel};
                    m_EfficiencyNumVarHistos[histoKeyEffNum] = ibooker.book1D(name, name+";"+m_effLabelStrings[var], histBins.size()-1, &histBins[0]);
                }
            }
        }
    }
}

void L1TMuonDQMOffline::bookResolutionHistos(DQMStore::IBooker &ibooker) {
    if(m_verbose) cout << "[L1TMuonOffline:] Booking Resolution Plot Histos" << endl;
    ibooker.setCurrentFolder(m_HistFolder+"/resolution");

    for (const auto var : m_resTypes) {
        auto nbins = std::get<0>(getHistBinsRes(var));
        auto xmin = std::get<1>(getHistBinsRes(var));
        auto xmax = std::get<2>(getHistBinsRes(var));
        for (const auto etaReg : m_etaRegions) {
            for (const auto qualLevel : m_qualLevelsRes) {
                m_histoKeyResType histoKeyRes = {var, etaReg, qualLevel};
                std::string name = "resolution_"+m_resStrings[var]+"_"+m_etaStrings[etaReg]+"_"+m_qualStrings[qualLevel];
                m_ResolutionHistos[histoKeyRes] = ibooker.book1D(name, name+";"+m_resLabelStrings[var], nbins, xmin, xmax);
            }
        }
    }
}

//_____________________________________________________________________
const unsigned int L1TMuonDQMOffline::getNVertices(Handle<VertexCollection> & vertex) {
    unsigned int nVtx = 0;

    if (vertex.isValid()) {
        for (const auto vertexIt : *vertex) {
            if (vertexIt.isValid() && !vertexIt.isFake()) {
                ++nVtx;
            }
        }
    }
    return nVtx;
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
    m_ControlHistos[CTRL_NTIGHTVSALL]->Fill(muons->size(), m_TightMuons.size());
}

//_____________________________________________________________________
void L1TMuonDQMOffline::getProbeMuons(Handle<edm::TriggerResults> & trigResults,
                           edm::Handle<trigger::TriggerEvent> & trigEvent) {

    if (m_verbose) cout << "[L1TMuonDQMOffline:] getting probe muons" << endl;
    m_ProbeMuons.clear();
    std::vector<const reco::Muon*> tagMuonsInHist;

    tagMuonsInHist.clear();

    vector<const reco::Muon*>::const_iterator probeCandIt   = m_TightMuons.begin();
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

            if ( (*tagCandIt) == (*probeCandIt) || deltar<m_minTagProbeDR ) continue; // CB has a little bias for closed-by muons     
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
                    m_ControlHistos[CTRL_TAGETA]->Fill(eta);
                    m_ControlHistos[CTRL_TAGPHI]->Fill(phi);
                    m_ControlHistos[CTRL_TAGPT]->Fill(pt);
                    m_ControlHistos[CTRL_TAGPROBEDR]->Fill(deltar);
                }
            }
        }
        if (isProbe) m_ProbeMuons.push_back((*probeCandIt));
    }
    m_ControlHistos[CTRL_NPROBESVSTIGHT]->Fill(m_TightMuons.size(), m_ProbeMuons.size());
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
        m_ControlHistos[CTRL_PROBEETA]->Fill((*probeMuIt)->eta());
        m_ControlHistos[CTRL_PROBEPHI]->Fill((*probeMuIt)->phi());
        m_ControlHistos[CTRL_PROBEPT]->Fill((*probeMuIt)->pt());

        MuonGmtPair pairBestCand((*probeMuIt), nullptr);
//      pairBestCand.propagate(m_BField,m_propagatorAlong,m_propagatorOpposite);
        gmtIt = gmtCands->begin();

        for(; gmtIt!=gmtEnd; ++gmtIt) {
            MuonGmtPair pairTmpCand((*probeMuIt),&(*gmtIt));
//          pairTmpCand.propagate(m_BField,m_propagatorAlong,m_propagatorOpposite);     

            if ( (pairTmpCand.dR() < m_maxGmtMuonDR) && (pairTmpCand.dR() < pairBestCand.dR() ) ) {
                pairBestCand = pairTmpCand;
            }

        }
        m_MuonGmtPairs.push_back(pairBestCand);
        m_ControlHistos[CTRL_MUONGMTDELTAR]->Fill(pairBestCand.dR());
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
    return (matchDeltaR < m_maxHltMuonDR);
}

std::vector<float> L1TMuonDQMOffline::getHistBinsEff(effType eff) {
    if (eff == EFF_PT) {
        std::vector<float> effVsPtBins(m_effVsPtBins.begin(), m_effVsPtBins.end());
        return effVsPtBins;
    }
    if (eff == EFF_PHI) {
        std::vector<float> effVsPhiBins(m_effVsPhiBins.begin(), m_effVsPhiBins.end());
        return effVsPhiBins;
    }
    if (eff == EFF_ETA) {
        std::vector<float> effVsEtaBins(m_effVsEtaBins.begin(), m_effVsEtaBins.end());
        return effVsEtaBins;
    }
    if (eff == EFF_VTX) {
        std::vector<float> effVsVtxBins(m_effVsVtxBins.begin(), m_effVsVtxBins.end());
        return effVsVtxBins;
    }
    return {0., 1.};
}

std::tuple<int, double, double> L1TMuonDQMOffline::getHistBinsRes(resType res){
    if (res == RES_PT)      return {100, -50., 50.};
    if (res == RES_1OVERPT) return {50, -0.05, 0.05};
    if (res == RES_QOVERPT) return {50, -0.05, 0.05};
    if (res == RES_PHI)     return {96, -0.2, 0.2};
    if (res == RES_ETA)     return {100, -0.1, 0.1};
    if (res == RES_CH)      return {5, -2, 3};
    return {1, 0, 1};
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TMuonDQMOffline);
