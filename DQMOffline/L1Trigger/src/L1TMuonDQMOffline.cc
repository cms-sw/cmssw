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

#include <array>

using namespace reco;
using namespace trigger;
using namespace edm;
using namespace std;
using namespace l1t;

//__________RECO-GMT Muon Pair Helper Class____________________________

MuonGmtPair::MuonGmtPair(const reco::Muon *muon, const l1t::Muon *regMu, const PropagateToMuon& propagator, bool useAtVtxCoord) :
        m_muon(muon), m_regMu(regMu)
{
    if (m_regMu) {
        if (useAtVtxCoord) {
            m_gmtEta = m_regMu->etaAtVtx();
            m_gmtPhi = m_regMu->phiAtVtx();
        } else {
            m_gmtEta = m_regMu->eta();
            m_gmtPhi = m_regMu->phi();
        }
    } else {
        m_gmtEta = -5.;
        m_gmtPhi = -5.;
    }
    if (m_muon) {
        TrajectoryStateOnSurface trajectory = propagator.extrapolate(*m_muon);
        if (trajectory.isValid()) {
            m_eta = trajectory.globalPosition().eta();
            m_phi = trajectory.globalPosition().phi();
        }
    } else {
        m_eta = 999.;
        m_phi = 999.;
    }
};

MuonGmtPair::MuonGmtPair(const MuonGmtPair& muonGmtPair) {
    m_muon    = muonGmtPair.m_muon;
    m_regMu   = muonGmtPair.m_regMu;

    m_gmtEta  = muonGmtPair.m_gmtEta;
    m_gmtPhi  = muonGmtPair.m_gmtPhi;

    m_eta     = muonGmtPair.m_eta;
    m_phi     = muonGmtPair.m_phi;
}

double MuonGmtPair::dR() {
    float dEta = m_regMu ? (m_gmtEta - m_eta) : 999.;
    float dPhi = m_regMu ? reco::deltaPhi(m_gmtPhi, m_phi) : 999.;
    return sqrt(dEta*dEta + dPhi*dPhi);
}

L1TMuonDQMOffline::EtaRegion MuonGmtPair::etaRegion() const {
    if (std::abs(m_eta) < 0.83) return L1TMuonDQMOffline::kEtaRegionBmtf;
    if (std::abs(m_eta) < 1.24) return L1TMuonDQMOffline::kEtaRegionOmtf;
    if (std::abs(m_eta) < 2.4)  return L1TMuonDQMOffline::kEtaRegionEmtf;
    return L1TMuonDQMOffline::kEtaRegionOut;
}

double MuonGmtPair::getDeltaVar(const L1TMuonDQMOffline::ResType type) const {
    if (type == L1TMuonDQMOffline::kResPt)      return (gmtPt() - pt()) / pt();
    if (type == L1TMuonDQMOffline::kRes1OverPt) return (pt() - gmtPt()) / gmtPt(); // (1/gmtPt - 1/pt) / (1/pt)
    if (type == L1TMuonDQMOffline::kResQOverPt) return (gmtCharge()*charge()*pt() - gmtPt()) / gmtPt(); // (gmtCharge/gmtPt - charge/pt) / (charge/pt) with gmtCharge/charge = gmtCharge*charge
    if (type == L1TMuonDQMOffline::kResPhi)     return reco::deltaPhi(gmtPhi(), m_phi);
    if (type == L1TMuonDQMOffline::kResEta)     return gmtEta() - m_eta;
    if (type == L1TMuonDQMOffline::kResCh)      return gmtCharge() - charge();
    return -999.;
}

double MuonGmtPair::getVar(const L1TMuonDQMOffline::EffType type) const {
    if (type == L1TMuonDQMOffline::kEffPt)  return pt();
    if (type == L1TMuonDQMOffline::kEffPhi) return m_phi;
    if (type == L1TMuonDQMOffline::kEffEta) return m_eta;
    return -999.;
}

//__________DQM_base_class_______________________________________________
L1TMuonDQMOffline::L1TMuonDQMOffline(const ParameterSet & ps) :
    m_propagator(ps.getParameter<edm::ParameterSet>("muProp")),
    m_effTypes({kEffPt, kEffPhi, kEffEta, kEffVtx}),
    m_resTypes({kResPt, kResQOverPt, kResPhi, kResEta}),
    m_etaRegions({kEtaRegionAll, kEtaRegionBmtf, kEtaRegionOmtf, kEtaRegionEmtf}),
    m_qualLevelsRes({kQualAll}),
    m_effStrings({ {kEffPt, "pt"}, {kEffPhi, "phi"}, {kEffEta, "eta"}, {kEffVtx, "vtx"} }),
    m_effLabelStrings({ {kEffPt, "p_{T} (GeV)"}, {kEffPhi, "#phi"}, {kEffEta, "#eta"}, {kEffVtx, "# vertices"} }),
    m_resStrings({ {kResPt, "pt"}, {kRes1OverPt, "1overpt"}, {kResQOverPt, "qoverpt"}, {kResPhi, "phi"}, {kResEta, "eta"}, {kResCh, "charge"} }),
    m_resLabelStrings({ {kResPt, "(p_{T}^{L1} - p_{T}^{reco})/p_{T}^{reco}"}, {kRes1OverPt, "(p_{T}^{reco} - p_{T}^{L1})/p_{T}^{L1}"}, {kResQOverPt, "(q^{L1}*q^{reco}*p_{T}^{reco} - p_{T}^{L1})/p_{T}^{L1}"}, {kResPhi, "#phi_{L1} - #phi_{reco}"}, {kResEta, "#eta_{L1} - #eta_{reco}"}, {kResCh, "charge^{L1} - charge^{reco}"} }),
    m_etaStrings({ {kEtaRegionAll, "etaMin0_etaMax2p4"}, {kEtaRegionBmtf, "etaMin0_etaMax0p83"}, {kEtaRegionOmtf, "etaMin0p83_etaMax1p24"}, {kEtaRegionEmtf, "etaMin1p24_etaMax2p4"} }),
    m_qualStrings({ {kQualAll, "qualAll"}, {kQualOpen, "qualOpen"}, {kQualDouble, "qualDouble"}, {kQualSingle, "qualSingle"} }),
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
    m_useAtVtxCoord(ps.getUntrackedParameter<bool>("useL1AtVtxCoord")),
    m_maxGmtMuonDR(0.3),
    m_minTagProbeDR(0.5),
    m_maxHltMuonDR(0.1)
{
    if (m_verbose) cout << "[L1TMuonDQMOffline:] ____________ Storage initialization ____________ " << endl;

    for (const auto cutsPSet : m_cutsVPSet) {
        const auto qCut = cutsPSet.getUntrackedParameter<int>("qualCut");
        QualLevel qLevel = kQualAll;
        if (qCut > 11) {
            qLevel = kQualSingle;
        } else if (qCut > 7) {
            qLevel = kQualDouble;
        } else if (qCut > 3) {
            qLevel = kQualOpen;
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
    m_propagator.init(iSetup);
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
    std::array<EtaRegion, 2> regsToFill { {kEtaRegionAll, kEtaRegionAll} };

    for(; muonGmtPairsIt!=muonGmtPairsEnd; ++muonGmtPairsIt) {
        // Fill the resolution histograms
        if( (muonGmtPairsIt->etaRegion() != kEtaRegionOut) && (muonGmtPairsIt->gmtPt() > 0) ){
            regsToFill[1] = muonGmtPairsIt->etaRegion();
            m_histoKeyResType histoKeyRes = {kResPt, kEtaRegionAll, kQualAll};
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
        if (muonGmtPairsIt->etaRegion() != kEtaRegionOut) {
            unsigned int cutsCounter = 0;
            for (const auto cut : m_cuts) {
                const auto gmtPtCut = cut.first;
                const auto qualLevel = cut.second;
                const bool gmtAboveCut = (muonGmtPairsIt->gmtPt() > gmtPtCut);

                // default keys
                m_histoKeyEffDenVarType histoKeyEffDenVar = {kEffPt, gmtPtCut, kEtaRegionAll};
                m_histoKeyEffNumVarType histoKeyEffNumVar = {kEffPt, gmtPtCut, kEtaRegionAll, qualLevel};

                regsToFill[1] = muonGmtPairsIt->etaRegion();
                for(const auto var : m_effTypes) {
                    if(var != kEffPt){
                       if (muonGmtPairsIt->pt() < m_recoToL1PtCutFactor * gmtPtCut) break; // efficiency at plateau
                    }
                    double varToFill;
                    if (var == kEffVtx) {
                        varToFill = static_cast<double>(nVtx);
                    } else {
                        varToFill = muonGmtPairsIt->getVar(var);
                    }
                    // Fill denominators
                    if (var == kEffEta) {
                        m_EfficiencyDenEtaHistos[gmtPtCut]->Fill(varToFill);
                    } else {
                        std::get<0>(histoKeyEffDenVar) = var;
                        // Fill for the global eta and for TF eta region that the probe muon is in
                        for (const auto regToFill : regsToFill) {
                            if (var == kEffPt) {
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
                        if (var == kEffEta) {
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

    m_ControlHistos[kCtrlMuonGmtDeltaR] = ibooker.book1D("MuonGmtDeltaR", "MuonGmtDeltaR; #DeltaR", 50, 0., 0.5);
    m_ControlHistos[kCtrlNTightVsAll] = ibooker.book2D("NTightVsAll", "NTightVsAll; # muons; # tight muons", 20, -0.5, 19.5, 16, -0.5, 15.5);
    m_ControlHistos[kCtrlNProbesVsTight] = ibooker.book2D("NProbesVsTight", "NProbesVsTight; # tight muons; # probe muons", 8, -0.5, 7.5, 8, -0.5, 7.5);

    m_ControlHistos[kCtrlTagPt] = ibooker.book1D("TagMuonPt", "TagMuonPt; p_{T}", 50, 0., 100.);
    m_ControlHistos[kCtrlTagPhi] = ibooker.book1D("TagMuonPhi", "TagMuonPhi; #phi", 66, -3.3, 3.3);
    m_ControlHistos[kCtrlTagEta] = ibooker.book1D("TagMuonEta", "TagMuonEta; #eta", 50, -2.5, 2.5);

    m_ControlHistos[kCtrlProbePt] = ibooker.book1D("ProbeMuonPt", "ProbeMuonPt; p_{T}", 50, 0., 100.);
    m_ControlHistos[kCtrlProbePhi] = ibooker.book1D("ProbeMuonPhi", "ProbeMuonPhi; #phi", 66, -3.3, 3.3);
    m_ControlHistos[kCtrlProbeEta] = ibooker.book1D("ProbeMuonEta", "ProbeMuonEta; #eta", 50, -2.5, 2.5);

    m_ControlHistos[kCtrlTagProbeDr] = ibooker.book1D("TagMuonProbeMuonDeltaR", "TagMuonProbeMuonDeltaR; #DeltaR", 50, 0.,5.0);
    m_ControlHistos[kCtrlTagHltDr] = ibooker.book1D("TagMuonHltDeltaR", "TagMuonHltDeltaR;#DeltaR", 55, 0., 0.11);
}

//_____________________________________________________________________
void L1TMuonDQMOffline::bookEfficiencyHistos(DQMStore::IBooker &ibooker) {
    ibooker.setCurrentFolder(m_HistFolder+"/numerators_and_denominators");

    for(const auto var : m_effTypes) {
        auto histBins = getHistBinsEff(var);
        // histograms for eta variable get a special treatment
        if (var == kEffEta) {
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
                if (var == kEffPt) {
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
    m_ControlHistos[kCtrlNTightVsAll]->Fill(muons->size(), m_TightMuons.size());
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
            auto matchHltDeltaR = matchHlt(trigEvent,(*tagCandIt));
            tagHasTrig = (matchHltDeltaR < m_maxHltMuonDR) && (pt > m_TagPtCut);
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
                    m_ControlHistos[kCtrlTagEta]->Fill(eta);
                    m_ControlHistos[kCtrlTagPhi]->Fill(phi);
                    m_ControlHistos[kCtrlTagPt]->Fill(pt);
                    m_ControlHistos[kCtrlTagProbeDr]->Fill(deltar);
                    m_ControlHistos[kCtrlTagHltDr]->Fill(matchHltDeltaR);
                }
            }
        }
        if (isProbe) m_ProbeMuons.push_back((*probeCandIt));
    }
    m_ControlHistos[kCtrlNProbesVsTight]->Fill(m_TightMuons.size(), m_ProbeMuons.size());
}

//_____________________________________________________________________
void L1TMuonDQMOffline::getMuonGmtPairs(edm::Handle<l1t::MuonBxCollection> & gmtCands) {

    m_MuonGmtPairs.clear();
    if (m_verbose) cout << "[L1TMuonDQMOffline:] Getting muon GMT pairs" << endl;

    vector<const reco::Muon*>::const_iterator probeMuIt  = m_ProbeMuons.begin();
    vector<const reco::Muon*>::const_iterator probeMuEnd = m_ProbeMuons.end();

    l1t::MuonBxCollection::const_iterator gmtIt;
    l1t::MuonBxCollection::const_iterator gmtEnd = gmtCands->end(0);

    for (; probeMuIt!=probeMuEnd; ++probeMuIt) {
        MuonGmtPair pairBestCand((*probeMuIt), nullptr, m_propagator, m_useAtVtxCoord);

        // Fill the control histograms with the probe muon kinematic variables used
        m_ControlHistos[kCtrlProbeEta]->Fill(pairBestCand.getVar(L1TMuonDQMOffline::kEffEta));
        m_ControlHistos[kCtrlProbePhi]->Fill(pairBestCand.getVar(L1TMuonDQMOffline::kEffPhi));
        m_ControlHistos[kCtrlProbePt]->Fill(pairBestCand.getVar(L1TMuonDQMOffline::kEffPt));

        gmtIt = gmtCands->begin(0); // use only on L1T muons from BX 0

        for(; gmtIt!=gmtEnd; ++gmtIt) {
            MuonGmtPair pairTmpCand((*probeMuIt), &(*gmtIt), m_propagator, m_useAtVtxCoord);

            if ( (pairTmpCand.dR() < m_maxGmtMuonDR) && (pairTmpCand.dR() < pairBestCand.dR() ) ) {
                pairBestCand = pairTmpCand;
            }

        }
        m_MuonGmtPairs.push_back(pairBestCand);
        m_ControlHistos[kCtrlMuonGmtDeltaR]->Fill(pairBestCand.dR());
    }
}

//_____________________________________________________________________
double L1TMuonDQMOffline::matchHlt(edm::Handle<TriggerEvent>  & triggerEvent, const reco::Muon * mu) {

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
    return matchDeltaR;
}

std::vector<float> L1TMuonDQMOffline::getHistBinsEff(EffType eff) {
    if (eff == kEffPt) {
        std::vector<float> effVsPtBins(m_effVsPtBins.begin(), m_effVsPtBins.end());
        return effVsPtBins;
    }
    if (eff == kEffPhi) {
        std::vector<float> effVsPhiBins(m_effVsPhiBins.begin(), m_effVsPhiBins.end());
        return effVsPhiBins;
    }
    if (eff == kEffEta) {
        std::vector<float> effVsEtaBins(m_effVsEtaBins.begin(), m_effVsEtaBins.end());
        return effVsEtaBins;
    }
    if (eff == kEffVtx) {
        std::vector<float> effVsVtxBins(m_effVsVtxBins.begin(), m_effVsVtxBins.end());
        return effVsVtxBins;
    }
    return {0., 1.};
}

std::tuple<int, double, double> L1TMuonDQMOffline::getHistBinsRes(ResType res){
    if (res == kResPt)      return {50, -2., 2.};
    if (res == kRes1OverPt) return {50, -2., 2.};
    if (res == kResQOverPt) return {50, -2., 2.};
    if (res == kResPhi)     return {96, -0.2, 0.2};
    if (res == kResEta)     return {100, -0.1, 0.1};
    if (res == kResCh)      return {5, -2, 3};
    return {1, 0, 1};
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TMuonDQMOffline);
