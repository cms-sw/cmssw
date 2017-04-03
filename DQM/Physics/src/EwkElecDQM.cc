// -*- C++ -*-
//
// Package:    DQM/DQM
// Class:      DQM
// 
/**\class DQM DQM.cc DQM/DQM/plugins/DQM.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Genius Walia
//         Created:  Thu, 26 Nov 2015 06:26:56 GMT
//
//

// system include files
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <fstream>
// user include files
#include "EwkElecDQM.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/Common/interface/View.h"

#include<TFile.h>
#include<TTree.h>
#include<TH1.h>
#include<TH2.h>

//
using namespace edm;
using namespace std;
using namespace reco;

// class declaration
// bracket 173

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
EwkElecDQM::EwkElecDQM(const edm::ParameterSet & iConfig)
{
    //now do what ever initialization is needed
    //usesResource("TFileService");
    metTag_ = iConfig.getUntrackedParameter < edm::InputTag > ("METTag", edm::InputTag("pfmet"));
    jetTag_ = iConfig.getUntrackedParameter < edm::InputTag > ("JetTag", edm::InputTag("ak4PFJets"));
    elecTag_ =
        consumes < edm::View < reco::GsfElectron >> (iConfig.getUntrackedParameter < edm::InputTag >
                                                     ("ElecTag", edm::InputTag("gsfElectrons")));
    metToken_ =
        consumes < edm::View < reco::MET > >(iConfig.getUntrackedParameter < edm::InputTag >
                                             ("METTag", edm::InputTag("pfmet")));
    jetToken_ =
        consumes < edm::View < reco::Jet > >(iConfig.getUntrackedParameter < edm::InputTag >
                                             ("JetTag", edm::InputTag("ak4PFJets")));
    vertexTag_ =
        consumes < edm::View < reco::Vertex > >(iConfig.getUntrackedParameter < edm::InputTag >
                                                ("VertexTag", edm::InputTag("offlinePrimaryVertices")));
    beamSpotTag_ =
        consumes < reco::BeamSpot > (iConfig.getUntrackedParameter < edm::InputTag >
                                     ("beamSpotTag", edm::InputTag("offlineBeamSpot")));

    //Main Cuts
    ptCut_ = iConfig.getUntrackedParameter < double >("PtCut", 25.);
    etaCut_ = iConfig.getUntrackedParameter < double >("EtaCut", 2.4);
    sieieCutBarrel_ = iConfig.getUntrackedParameter < double >("SieieBarrel", 0.0101);
    sieieCutEndcap_ = iConfig.getUntrackedParameter < double >("SieieEndcap", 0.0279);
    detainCutBarrel_ = iConfig.getUntrackedParameter < double >("DetainBarrel", 0.00926);
    detainCutEndcap_ = iConfig.getUntrackedParameter < double >("DetainEndcap", 0.00724);
    ecalIsoCutBarrel_ = iConfig.getUntrackedParameter < double >("EcalIsoCutBarrel", 5.7);
    ecalIsoCutEndcap_ = iConfig.getUntrackedParameter < double >("EcalIsoCutEndcap", 5.0);
    hcalIsoCutBarrel_ = iConfig.getUntrackedParameter < double >("HcalIsoCutBarrel", 8.1);
    hcalIsoCutEndcap_ = iConfig.getUntrackedParameter < double >("HcalIsoCutEndcap", 3.4);
    trkIsoCutBarrel_ = iConfig.getUntrackedParameter < double >("TrkIsoCutBarrel", 7.2);
    trkIsoCutEndcap_ = iConfig.getUntrackedParameter < double >("TrkIsoCutEndcap", 5.1);
    mtMin_ = iConfig.getUntrackedParameter < double >("MtMin", -999999);
    mtMax_ = iConfig.getUntrackedParameter < double >("MtMax", 999999.);
    metMin_ = iConfig.getUntrackedParameter < double >("MetMin", -999999.);
    metMax_ = iConfig.getUntrackedParameter < double >("MetMax", 999999.);
    eJetMin_ = iConfig.getUntrackedParameter < double >("EJetMin", 999999.);
    nJetMax_ = iConfig.getUntrackedParameter < int >("NJetMax", 999999);
    PUMax_ = iConfig.getUntrackedParameter < unsigned int >("PUMax", 60);
    PUBinCount_ = iConfig.getUntrackedParameter < unsigned int >("PUBinCount", 12);

}

void EwkElecDQM::dqmBeginRun(const Run & iRun, const EventSetup & iSet)
{
    nall = 0;
    nsel = 0;

    nrec = 0;
    neid = 0;
    niso = 0;

}

void EwkElecDQM::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const &, edm::EventSetup const &)
{

    ibooker.setCurrentFolder("Physics/EwkElecDQM");

    //histogram declaration
    char chtitle[256] = "";
    pt_before_ = ibooker.book1D("PT_BEFORECUTS", "Electron transverse momentum [GeV]", 100, 0., 100.);
    pt_after_ = ibooker.book1D("PT_LASTCUT", "Electron transverse momentum [GeV]", 100, 0., 100.);

    eta_before_ = ibooker.book1D("ETA_BEFORECUTS", "Electron pseudo-rapidity", 50, -2.5, 2.5);
    eta_after_ = ibooker.book1D("ETA_LASTCUT", "Electron pseudo-rapidity", 50, -2.5, 2.5);

    sieiebarrel_before_ = ibooker.book1D("SIEIEBARREL_BEFORECUTS",
                                         "Electron #sigma_{i#etai#eta} (barrel)", 70, 0., 0.07);
    sieiebarrel_after_ = ibooker.book1D("SIEIEBARREL_LASTCUT", "Electron #sigma_{i#etai#eta} (barrel)", 70, 0., 0.07);

    sieieendcap_before_ = ibooker.book1D("SIEIEENDCAP_BEFORECUTS",
                                         "Electron #sigma_{i#etai#eta} (endcap)", 70, 0., 0.07);
    sieieendcap_after_ = ibooker.book1D("SIEIEENDCAP_LASTCUT", "Electron #sigma_{i#etai#eta} (endcap)", 70, 0., 0.07);

    detainbarrel_before_ = ibooker.book1D("DETAINBARREL_BEFORECUTS",
                                          "Electron #Delta#eta_{in} (barrel)", 40, -0.02, 0.02);
    detainbarrel_after_ = ibooker.book1D("DETAINBARREL_LASTCUT", "Electron #Delta#eta_{in} (barrel)", 40, -0.02, 0.02);

    detainendcap_before_ = ibooker.book1D("DETAINENDCAP_BEFORECUTS",
                                          "Electron #Delta#eta_{in} (endcap)", 40, -0.02, 0.02);
    detainendcap_after_ = ibooker.book1D("DETAINENDCAP_LASTCUT", "Electron #Delta#eta_{in} (endcap)", 40, -0.02, 0.02);

    ecalisobarrel_before_ = ibooker.book1D("ECALISOBARREL_BEFORECUTS",
                                           "Absolute electron ECAL isolation variable (barrel) [GeV]", 50, 0., 50.);
    ecalisobarrel_after_ = ibooker.book1D("ECALISOBARREL_LASTCUT",
                                          "Absolute electron ECAL isolation variable (barrel) [GeV]", 50, 0., 50.);

    ecalisoendcap_before_ = ibooker.book1D("ECALISOENDCAP_BEFORECUTS",
                                           "Absolute electron ECAL isolation variable (endcap) [GeV]", 50, 0., 50.);
    ecalisobarrel_after_ = ibooker.book1D("ECALISOBARREL_LASTCUT",
                                          "Absolute electron ECAL isolation variable (barrel) [GeV]", 50, 0., 50.);

    ecalisoendcap_before_ = ibooker.book1D("ECALISOENDCAP_BEFORECUTS",
                                           "Absolute electron ECAL isolation variable (endcap) [GeV]", 50, 0., 50.);
    ecalisoendcap_after_ = ibooker.book1D("ECALISOENDCAP_LASTCUT",
                                          "Absolute electron ECAL isolation variable (endcap) [GeV]", 50, 0., 50.);

    hcalisobarrel_before_ = ibooker.book1D("HCALISOBARREL_BEFORECUTS",
                                           "Absolute electron HCAL isolation variable (barrel) [GeV]", 50, 0., 50.);
    hcalisobarrel_after_ = ibooker.book1D("HCALISOBARREL_LASTCUT",
                                          "Absolute electron HCAL isolation variable (barrel) [GeV]", 50, 0., 50.);

    hcalisoendcap_before_ = ibooker.book1D("HCALISOENDCAP_BEFORECUTS",
                                           "Absolute electron HCAL isolation variable (endcap) [GeV]", 50, 0., 50.);
    hcalisoendcap_after_ = ibooker.book1D("HCALISOENDCAP_LASTCUT",
                                          "Absolute electron HCAL isolation variable (endcap) [GeV]", 50, 0., 50.);

    trkisobarrel_before_ = ibooker.book1D("TRKISOBARREL_BEFORECUTS",
                                          "Absolute electron track isolation variable (barrel) [GeV]", 50, 0., 50.);
    trkisobarrel_after_ = ibooker.book1D("TRKISOBARREL_LASTCUT",
                                         "Absolute electron track isolation variable (barrel) [GeV]", 50, 0., 50.);

    trkisoendcap_before_ = ibooker.book1D("TRKISOENDCAP_BEFORECUTS",
                                          "Absolute electron track isolation variable (endcap) [GeV]", 50, 0., 50.);
    trkisoendcap_after_ = ibooker.book1D("TRKISOENDCAP_LASTCUT",
                                         "Absolute electron track isolation variable (endcap) [GeV]", 50, 0., 50.);

    Phistar_ = ibooker.book1D("PHISTAR", "Phi star", 100, 0., 10);
    Phistar_after_ = ibooker.book1D("PHISTAR_AFTER", "Phi star", 100, 0., 10);

    CosineThetaStar_ = ibooker.book1D("CosineThetastar_beforecuts", "Cos ThetaStar_beforecuts", 100, -1, 1);

    CosineThetaStar_afterZ_ = ibooker.book1D("CosineThetastar_aftercuts", "Cos ThetaStar_aftercuts", 100, -1, 1);

    const int ZMassBins = 4;
    double ZMassGrid[4] = { 60, 80, 100, 120 };

    char name[100], title[100];
    for (int m = 0; m < ZMassBins - 1; m++) {
        sprintf(name, "CosineThetastar_ZMassBin_%i", m);
        sprintf(title, "CosineThetaStar for %f<ZMASS<%f", ZMassGrid[m], ZMassGrid[m + 1]);
        CosineThetaStar_2D[m] = ibooker.book1D(name, title, 100, -1, 1);
    }

    char name1[100], title1[100];
    for (int m = 0; m < ZMassBins - 1; m++) {
        sprintf(name1, "CosineThetastar_AfterZcuts_ZMassBin_%i", m);
        sprintf(title1, "CosineThetaStar for %f<ZMASS<%f", ZMassGrid[m], ZMassGrid[m + 1]);
        CosineThetaStar_afterZ_2D[m] = ibooker.book1D(name1, title1, 100, -1., 1.);
    }

    const int MuRapBins = 4;
    double MuRapGrid[4] = { 0, 0.8, 1.6, 2.4 };

    char name2[100], title2[100];
    for (int m = 0; m < MuRapBins - 1; m++) {
        sprintf(name2, "CosineThetastar_YBin_%i", m);
        sprintf(title2, "CosineThetaStar for %f<Y<%f", MuRapGrid[m], MuRapGrid[m + 1]);
        CosineThetaStar_Y_2D[m] = ibooker.book1D(name2, title2, 100, -1, 1);
    }

    char name3[100], title3[100];
    for (int m = 0; m < MuRapBins - 1; m++) {
        sprintf(name3, "CosineThetastar_AFTERZCUTS_YBin_%i", m);
        sprintf(title3, "CosineThetaStar for %f<Y<%f", MuRapGrid[m], MuRapGrid[m + 1]);
        CosineThetaStar_Y_afterZ_2D[m] = ibooker.book1D(name3, title3, 100, -1, 1);
    }

    invmass_before_ = ibooker.book1D("INVMASS_BEFORECUTS", "Di-electron invariant mass [GeV]", 100, 0., 200.);
    invpt_before_ = ibooker.book1D("INVPT_BEFORECUTS", "Di-electron invariant pt [GeV]", 100, 0., 200.);

    deltaPhi_ = ibooker.book1D("DELTA_PHI_J1_J2", "deltaphi b/w j1 and j2", 50, 0., 3.);

    deltaPhi_afterZ_ = ibooker.book1D("DELTA_PHI_J1_J2_aftercuts", "deltaphi b/w j1 and j2 after cuts", 50, 0., 3.);

    invmass_after_ = ibooker.book1D("INVMASS_AFTERCUTS", "Di-electron invariant mass [GeV]", 100, 0., 200.);

    invpt_after_ = ibooker.book1D("INVPT_AFTERCUTS", "Di-electron invariant pt [GeV]", 100, 0., 200.);
    InVaMassJJ_ = ibooker.book1D("INVARIANT MASS_JJ", "Invariant mass", 100, 0, 700);

    InVaMassJJ_after_ = ibooker.book1D("INVARIANT MASS_JJ_AFTERCUTS", "Invariant mass", 100, 0, 700);

    invmassPU_before_ = ibooker.book2D("INVMASS_PU_BEFORECUTS",
                                       "Di-electron invariant mass [GeV] vs PU; mass [GeV]; PU count", 100, 0., 200.,
                                       PUBinCount_, -0.5, PUMax_ + 0.5);
    invmassPU_afterZ_ =
        ibooker.book2D("INVMASS_PU_AFTERZCUTS", "Di-electron invariant mass [GeV] vs PU; mass [GeV]; PU count", 100, 0.,
                       200., PUBinCount_, -0.5, PUMax_ + 0.5);

    npvs_before_ = ibooker.book1D("NPVs_BEFORECUTS",
                                  "Number of Valid Primary Vertices; nGoodPVs", PUMax_ + 1, -0.5, PUMax_ + 0.5);

    npvs_afterZ_ = ibooker.book1D("NPVs_AFTERZCUTS",
                                  "Number of Valid Primary Vertices; nGoodPVs", PUMax_ + 1, -0.5, PUMax_ + 0.5);

    nelectrons_before_ = ibooker.book1D("NELECTRONS_BEFORECUTS", "Number of electrons in event", 10, -0.5, 9.5);
    nelectrons_after_ = ibooker.book1D("NELECTRONS_AFTERCUTS", "Number of electrons in event", 10, -0.5, 9.5);

    snprintf(chtitle, 255, "Transverse mass (%s) [GeV]", metTag_.label().data());
    mt_before_ = ibooker.book1D("MT_BEFORECUTS", chtitle, 150, 0., 300.);
    mt_after_ = ibooker.book1D("MT_LASTCUT", chtitle, 150, 0., 300.);

    snprintf(chtitle, 255, "Missing transverse energy (%s) [GeV]", metTag_.label().data());
    met_before_ = ibooker.book1D("MET_BEFORECUTS", chtitle, 100, 0., 200.);
    met_after_ = ibooker.book1D("MET_LASTCUT", chtitle, 100, 0., 200.);
    snprintf(chtitle, 255, "Number of jets (%s) above %.2f GeV", jetTag_.label().data(), eJetMin_);
    njets_before_ = ibooker.book1D("NJETS_BEFORECUTS", chtitle, 10, -0.5, 9.5);
    njets_after_ = ibooker.book1D("NJETS_LASTCUT", chtitle, 10, -0.5, 9.5);

    snprintf(chtitle, 255, "Jet with highest E_{T} (%s)", jetTag_.label().data());
    jet_et_before_ = ibooker.book1D("JETET1_BEFORECUTS", chtitle, 20, 0., 200.0);
    jet_et_after_ = ibooker.book1D("JETET1_AFTERCUTS", chtitle, 20, 0., 200.0);

    snprintf(chtitle, 255, "Jet with second highest E_{T} (%s)", jetTag_.label().data());
    jet2_et_before_ = ibooker.book1D("JETET2_BEFORECUTS", chtitle, 20, 0., 200.0);
    jet2_et_after_ = ibooker.book1D("JETET2_AFTERCUTS", chtitle, 20, 0., 200.0);

    snprintf(chtitle, 255, "Jet with third highest E_{T} (%s)", jetTag_.label().data());

    jet3_et_before_ = ibooker.book1D("JETET3_BEFORECUTS", chtitle, 20, 0., 200.0);
    jet3_et_after_ = ibooker.book1D("JETET3_AFTERCUTS", chtitle, 20, 0., 200.0);

    snprintf(chtitle, 255, "Eta of Jet with highest E_{T} (%s)", jetTag_.label().data());
    jet_eta_before_ = ibooker.book1D("JETETA1_BEFORECUTS", chtitle, 20, -5, 5);
    jet_eta_after_ = ibooker.book1D("JETETA1_AFTERCUTS", chtitle, 20, -5, 5);
}

void EwkElecDQM::endRun(const Run & r, const EventSetup &)
{
    double all = nall;
    double esel = nsel / all;
    LogVerbatim("") << "\n>>>>>> SELECTION SUMMARY BEGIN >>>>>>>>>>>>>>>";
    LogVerbatim("") << "Total number of events analyzed: " << nall << " [events]";
    LogVerbatim("") << "Total number of events selected: " << nsel << " [events]";
    LogVerbatim("") << "Overall efficiency:             "
        << "(" << setprecision(4) << esel * 100. << " +/- "
        << setprecision(2) << sqrt(esel * (1 - esel) / all) * 100. << ")%";

    double erec = nrec / all;
    double eeid = neid / all;
    double eiso = niso / all;

    // general reconstruction step??
    double num = nrec;
    double eff = erec;
    double err = sqrt(eff * (1 - eff) / all);
    LogVerbatim("") << "Passing Pt/Eta/Quality cuts:    " << num << " [events], ("
        << setprecision(4) << eff * 100. << " +/- " << setprecision(2)
        << err * 100. << ")%";

    //electron identification
    num = neid;
    eff = eeid;
    err = sqrt(eff * (1 - eff) / all);
    double effstep = 0.;
    double errstep = 0.;
    if (nrec > 0)
        effstep = eeid / erec;
    if (nrec > 0)
        errstep = sqrt(effstep * (1 - effstep) / nrec);
    LogVerbatim("") << "Passing eID cuts:         " << num << " [events], ("
        << setprecision(4) << eff * 100. << " +/- " << setprecision(2)
        << err * 100. << ")%, to previous step: (" << setprecision(4)
        << effstep * 100. << " +/- " << setprecision(2)
        << errstep * 100. << ")%";

    //electron isolation
    num = niso;
    eff = eiso;
    err = sqrt(eff * (1 - eff) / all);
    effstep = 0.;
    errstep = 0.;
    if (neid > 0)
        effstep = eiso / eeid;
    if (neid > 0)
        errstep = sqrt(effstep * (1 - effstep) / neid);
    LogVerbatim("") << "Passing isolation cuts:         " << num << " [events], ("
        << setprecision(4) << eff * 100. << " +/- " << setprecision(2)
        << err * 100. << ")%, to previous step: (" << setprecision(4)
        << effstep * 100. << " +/- " << setprecision(2)
        << errstep * 100. << ")%";

    LogVerbatim("") << ">>>>>> SELECTION SUMMARY END   >>>>>>>>>>>>>>>\n";
}

inline void HERE(const char *msg)
{
    std::cout << msg << "\n";
}

// ------------ method called for each event  ------------

void EwkElecDQM::analyze(const edm::Event & iEvent, const edm::EventSetup & iSetup)
{

    //Reset  global event selection flags
    bool rec_sel = false;
    bool eid_sel = false;
    bool iso_sel = false;
    bool all_sel = false;

    // Electron collection
    Handle < View < GsfElectron > >electronCollection;
    if (!iEvent.getByToken(elecTag_, electronCollection)) {
        LogWarning("") << ">>> Electron collection does not exist !!!";
        return;
    }
    unsigned int electronCollectionSize = electronCollection->size();

    // Beam spot
    Handle < reco::BeamSpot > beamSpotHandle;
    if (!iEvent.getByToken(beamSpotTag_, beamSpotHandle)) {
        LogWarning ("") << ">>> No beam spot found !!!";
        return;
    }
    // MET
    double met_px = 0.;
    double met_py = 0.;
    Handle < View < MET > >metCollection;
    if (!iEvent.getByToken(metToken_, metCollection)) {
        LogWarning("") << ">>> MET collection does not exist !!!";
        return;
    }

    const MET & met = metCollection->at(0);
    met_px = met.px();
    met_py = met.py();
    double met_et = sqrt(met_px * met_px + met_py * met_py);
    LogTrace("") << ">>> MET, MET_px, MET_py: " << met_et << ", " << met_px << ", " << met_py << " [GeV]";
    met_before_->Fill(met_et);

    // Vertices in the event
    int npvCount = 0;
    Handle < View < reco::Vertex > >vertexCollection;
    if (!iEvent.getByToken(vertexTag_, vertexCollection)) {
        LogError("") << ">>> Vertex collection does not exist !!!";
        return;
    }

    for (unsigned int i = 0; i < vertexCollection->size(); i++) {
        const Vertex & vertex = vertexCollection->at(i);
        if (vertex.isValid())
            npvCount++;
    }
    npvs_before_->Fill(npvCount);

    // Jet collection
    Handle < View < Jet > >jetCollection;
    if (!iEvent.getByToken(jetToken_, jetCollection)) {
        LogError("") << ">>> JET collection does not exist !!!";
        return;
    }
    float electron_et = -8.0;
    float electron_eta = -8.0;
    float electron_phi = -8.0;
    float electron2_et = -9.0;
    float electron2_eta = -9.0;
    float electron2_phi = -9.0;

    // need to get some electron info so jets can be cleaned of them
    for (unsigned int i = 0; i < electronCollectionSize; i++) {
        const GsfElectron & elec = electronCollection->at(i);

        if (i < 1) {
            electron_et = elec.pt();
            electron_eta = elec.eta();
            electron_phi = elec.phi();
        }
        if (i == 2) {
            electron2_et = elec.pt();
            electron2_eta = elec.eta();
            electron2_phi = elec.phi();
        }
    }

    float delta = -1.0;
    double invariant_mass = -1;
    float jet_px = -1.0;
    float jet_py = -1.0;
    float jet_pz = -1.0;
    float jet_p = -1.0;
    float jet_phi = -1.0;
    float jet2_px = -1.0;
    float jet2_py = -1.0;
    float jet2_pz = -1.0;
    float jet2_p = -1.0;
    float jet2_phi = -1.0;
    float jet3_px = -1.0;
    float jet3_py = -1.0;
    float jet3_pz = -1.0;
    float jet3_p = -1.0;
    float jet3_phi = -1.0;
    float jet_et = -8.0;
    float jet_eta = -8.0;
    int jet_count = 0;
    float jet2_et = -9.0;
    float jet3_et = -10.0;
    unsigned int jetCollectionSize = jetCollection->size();
    int njets = 0;
    for (unsigned int i = 0; i < jetCollectionSize; i++) {
        const Jet & jet = jetCollection->at(i);
        float jet_current_et = jet.et();
        if (electron_et > 0.0 && fabs(jet.eta() - electron_eta) < 0.2 && calcDeltaPhi(jet.phi(), electron_phi) < 0.2)
            continue;
        if (electron2_et > 0.0 && fabs(jet.eta() - electron2_eta) < 0.2 && calcDeltaPhi(jet.phi(), electron2_phi) < 0.2)
            continue;
        if (jet.et() > eJetMin_) {
            njets++;
            jet_count++;
        }
        if (jet_current_et > jet_et) {
            jet2_et = jet_et;
            jet2_phi = jet_phi;
            jet2_px = jet_px;
            jet2_py = jet_py;
            jet2_pz = jet_pz;
            jet2_p = jet_p;
            jet_et = jet.et();
            jet_phi = jet.phi();
            jet_px = jet.px();
            jet_py = jet.py();
            jet_pz = jet.pz();
            jet_p = jet.p();
            jet_eta = jet.eta();
        } else if (jet_current_et > jet2_et) {
            jet3_et = jet2_et;
            jet2_et = jet.et();
            jet3_px = jet2_px;
            jet3_py = jet2_py;
            jet3_pz = jet2_pz;
            jet3_phi = jet2_phi;
            jet3_p = jet2_p;
            jet2_px = jet.px();
            jet2_py = jet.py();
            jet2_pz = jet.pz();
            jet2_phi = jet.phi();
            jet2_p = jet.p();
        } else if (jet_current_et > jet3_et) {
            jet3_et = jet.et();
            jet3_phi = jet.phi();
            jet3_px = jet.px();
            jet3_py = jet.py();
            jet3_pz = jet.pz();
            jet3_p = jet.p();
        }

        if (jet2_px != -1 && jet2_py != -1 && jet2_pz != -1 && jet2_p != -1 && jet3_px != -1 && jet3_py != -1
            && jet3_pz != -1 && jet3_p != -1) {
            const math::XYZTLorentzVector jets(jet3_px + jet2_px, jet3_py + jet2_py, jet3_pz + jet2_pz,
                                               jet3_p + jet2_p);
            invariant_mass = jets.M();
            InVaMassJJ_->Fill(invariant_mass);

        }

        if (jet_phi != -1 && jet2_phi != -1 && jet3_phi != -1) {
            if (fabs(jet_phi - jet2_phi) <= M_PI) {
                delta = (fabs(jet_phi - jet2_phi));
            } else {
                delta = (2 * M_PI - fabs(jet_phi - jet2_phi));
            }
            deltaPhi_->Fill(delta);

        }

    }

    if (jet_et > 10) {
        jet_et_before_->Fill(jet_et);
        jet2_et_before_->Fill(jet2_et);
        jet3_et_before_->Fill(jet3_et);
        jet_eta_before_->Fill(jet_eta);
    }

    LogTrace("") << ">>> Total number of jets: " << jetCollectionSize;
    LogTrace("") << ">>> Number of jets above " << eJetMin_ << " [GeV]: " << njets;
    njets_before_->Fill(njets);
    // Start counting
    nall++;
    bool njets_hist_done = false;
    bool met_hist_done = false;
    const int NFLAGS = 10;
    bool electron_sel[NFLAGS];
    double electron[2][9];
    double goodElectron[2][9];
    nGoodElectrons = 0;
    for (unsigned int i = 0; i < 2; i++) {
        for (unsigned int j = 0; j < 9; j++) {
            electron[i][j] = 0.;
            goodElectron[i][j] = 0.;
        }
    }

    for (unsigned int i = 0; i < electronCollectionSize; i++) {
        for (int j = 0; j < NFLAGS; ++j) {
            electron_sel[j] = false;
        }
        const GsfElectron & elec = electronCollection->at(i);

        if (i < 2) {
            electron[i][0] = 1.;
            electron[i][1] = elec.massSqr();
            electron[i][2] = elec.energy();
            electron[i][3] = elec.px();
            electron[i][4] = elec.py();
            electron[i][5] = elec.pz();
            electron[i][6] = elec.phi();
            electron[i][7] = elec.eta();
            electron[i][8] = elec.charge();
        }

        // Pt,eta cuts
        double pt = elec.pt();
        double px = elec.px();
        double py = elec.py();
        double eta = elec.eta();
        if (pt > ptCut_)
            electron_sel[0] = true;
        if (fabs(eta) < etaCut_)
            electron_sel[1] = true;

        bool isBarrel = false;
        bool isEndcap = false;
        if (eta < 1.4442 && eta > -1.4442) {
            isBarrel = true;
        } else if ((eta > 1.56 && eta < 2.4) || (eta < -1.56 && eta > -2.4)) {
            isEndcap = true;
        }
        pt_before_->Fill(pt);
        eta_before_->Fill(eta);
        // Electron ID cuts
        double sieie = (double)elec.sigmaIetaIeta();
        double detain = (double)elec.deltaEtaSuperClusterTrackAtVtx();  // think this is detain
        if (sieie < sieieCutBarrel_ && isBarrel)
            electron_sel[2] = true;
        if (sieie < sieieCutEndcap_ && isEndcap)
            electron_sel[2] = true;
        if (detain < detainCutBarrel_ && isBarrel)
            electron_sel[3] = true;
        if (detain < detainCutEndcap_ && isEndcap)
            electron_sel[3] = true;
        int out2 = 0;
        if (out2 == 1) {
            if (isBarrel) {
                 LogTrace("") << "\t... detain value " << detain << " (barrel), pass? " << electron_sel[3];
            } else if (isEndcap) {
                LogTrace("") << "\t... sieie value " << sieie << " (endcap), pass? " << electron_sel[2];
                LogTrace("") << "\t... detain value " << detain << " (endcap), pass? " << electron_sel[2];
            }
        }
        if (isBarrel) {
            sieiebarrel_before_->Fill(sieie);
            detainbarrel_before_->Fill(detain);
        } else if (isEndcap) {
            sieieendcap_before_->Fill(sieie);
            detainendcap_before_->Fill(detain);
        }
        // Isolation cuts
        double ecalisovar = elec.dr03EcalRecHitSumEt(); // picked one set!
        double hcalisovar = elec.dr03HcalTowerSumEt();  // try others if
        double trkisovar = elec.dr04TkSumPt();

        if (ecalisovar < ecalIsoCutBarrel_ && isBarrel)
            electron_sel[4] = true;
        if (ecalisovar < ecalIsoCutEndcap_ && isEndcap)
            electron_sel[4] = true;
        if (hcalisovar < hcalIsoCutBarrel_ && isBarrel)
            electron_sel[5] = true;
        if (hcalisovar < hcalIsoCutEndcap_ && isEndcap)
            electron_sel[5] = true;
        if (trkisovar < trkIsoCutBarrel_ && isBarrel)
            electron_sel[6] = true;
        if (trkisovar < trkIsoCutEndcap_ && isEndcap)
            electron_sel[6] = true;
        if (isBarrel) {
            LogTrace("") << "\t... ecal isolation value " << ecalisovar << " (barrel), pass? " << electron_sel[4];
            LogTrace("") << "\t... hcal isolation value " << hcalisovar << " (barrel), pass? " << electron_sel[5];
            LogTrace("") << "\t... trk isolation value " << trkisovar << " (barrel), pass? " << electron_sel[6];
        } else if (isEndcap) {
            LogTrace("") << "\t... ecal isolation value " << ecalisovar << " (endcap), pass? " << electron_sel[4];
            LogTrace("") << "\t... hcal isolation value " << hcalisovar << " (endcap), pass? " << electron_sel[5];
            LogTrace("") << "\t... trk isolation value " << trkisovar << " (endcap), pass? " << electron_sel[6];
        }
        if (isBarrel) {
            ecalisobarrel_before_->Fill(ecalisovar);
            hcalisobarrel_before_->Fill(hcalisovar);
            trkisobarrel_before_->Fill(trkisovar);
        } else if (isEndcap) {
            ecalisoendcap_before_->Fill(ecalisovar);
            hcalisoendcap_before_->Fill(hcalisovar);
            trkisoendcap_before_->Fill(trkisovar);
        }
        // MET/MT cuts
        double w_et = met_et + pt;
        double w_px = met_px + px;
        double w_py = met_py + py;

        double massT = w_et * w_et - w_px * w_px - w_py * w_py;
        massT = (massT > 0) ? sqrt(massT) : 0;

        LogTrace("") << "\t... W mass, W_et, W_px, W_py: " << massT << ", " << w_et
            << ", " << w_px << ", " << w_py << " [GeV]";
        if (massT > mtMin_ && massT < mtMax_)
            electron_sel[7] = true;
        mt_before_->Fill(massT);
        if (met_et > metMin_ && met_et < metMax_)
            electron_sel[8] = true;
        if (njets <= nJetMax_)
            electron_sel[9] = true;

        // Collect necessary flags "per electron"
        int flags_passed = 0;
        bool rec_sel_this = true;
        bool eid_sel_this = true;
        bool iso_sel_this = true;
        bool all_sel_this = true;
        for (int j = 0; j < NFLAGS; ++j) {
            if (electron_sel[j])
                flags_passed += 1;
            if (j < 2 && !electron_sel[j])
                rec_sel_this = false;
            if (j < 4 && !electron_sel[j])
                eid_sel_this = false;
            if (j < 7 && !electron_sel[j])
                iso_sel_this = false;
            if (!electron_sel[j])
                all_sel_this = false;
        }

        if (all_sel_this) {
            if (nGoodElectrons < 2) {
                goodElectron[nGoodElectrons][0] = 1.;
                goodElectron[nGoodElectrons][1] = elec.massSqr();
                goodElectron[nGoodElectrons][2] = elec.energy();
                goodElectron[nGoodElectrons][3] = elec.px();
                goodElectron[nGoodElectrons][4] = elec.py();
                goodElectron[nGoodElectrons][5] = elec.pz();
                goodElectron[nGoodElectrons][6] = elec.phi();
                goodElectron[nGoodElectrons][7] = elec.eta();
                goodElectron[nGoodElectrons][8] = elec.charge();

            }
            nGoodElectrons++;
        }

        if (rec_sel_this)
            rec_sel = true;
        if (eid_sel_this)
            iso_sel = true;
        if (iso_sel_this)
            iso_sel = true;
        if (all_sel_this)
            all_sel = true;

        // Do N-1 histograms now (and only once for global event quantities)
        if (flags_passed >= (NFLAGS - 1)) {
            if (!electron_sel[0] || flags_passed == NFLAGS) {
                pt_after_->Fill(pt);
            }
            if (!electron_sel[1] || flags_passed == NFLAGS) {
                eta_after_->Fill(eta);
            }
            if (!electron_sel[2] || flags_passed == NFLAGS) {
                if (isBarrel) {
                    sieiebarrel_after_->Fill(sieie);
                } else if (isEndcap) {
                    sieieendcap_after_->Fill(sieie);
                }
            }
            if (!electron_sel[3] || flags_passed == NFLAGS) {
                if (isBarrel) {
                    detainbarrel_after_->Fill(detain);
                } else if (isEndcap) {
                    detainendcap_after_->Fill(detain);
                }
            }
            if (!electron_sel[4] || flags_passed == NFLAGS) {
                if (isBarrel) {
                    ecalisobarrel_after_->Fill(ecalisovar);
                } else if (isEndcap) {
                    ecalisoendcap_after_->Fill(ecalisovar);
                }
            }
            if (!electron_sel[5] || flags_passed == NFLAGS) {
                if (isBarrel) {
                    hcalisobarrel_after_->Fill(hcalisovar);
                } else if (isEndcap) {
                    hcalisoendcap_after_->Fill(hcalisovar);
                }
            }
            if (!electron_sel[6] || flags_passed == NFLAGS) {
                if (isBarrel) {
                    trkisobarrel_after_->Fill(trkisovar);
                } else if (isEndcap) {
                    trkisoendcap_after_->Fill(trkisovar);
                }
            }
            if (!electron_sel[7] || flags_passed == NFLAGS) {
                mt_after_->Fill(massT);
            }
            if (!electron_sel[8] || flags_passed == NFLAGS) {
                if (!met_hist_done) {
                    met_after_->Fill(met_et);
                }
            }
            met_hist_done = true;

            if (!electron_sel[9] || flags_passed == NFLAGS) {
                if (!njets_hist_done) {
                    njets_after_->Fill(njets);
                    if (jet_et > 10)    // don't want low energy "jets"
                    {
                        jet_et_after_->Fill(jet_et);

                        jet2_et_after_->Fill(jet2_et);
                        jet3_et_after_->Fill(jet3_et);

                        deltaPhi_afterZ_->Fill(delta);
                        InVaMassJJ_after_->Fill(invariant_mass);
                        jet_eta_after_->Fill(jet_eta);
                    }
                }
            }
            njets_hist_done = true;
        }
    }

    /* float electron_eta = -8.0;
       float electron_phi = -8.0;
       //float electron2_et = -9.0;
       float electron2_eta = -9.0;
       float electron2_phi = -9.0;
     */
    double deltam = 0.;
    double acopl = 0.;
    double Rapdiff = 0.;
    double Thetastar = 0.;
    double halfacopl = 0.;
    double angle = 0;
    double Cos_thetastar = 0;
    double tan_acopl = 0.;
    double result_phistar = 0.;

    double rapidity = 0;
    double invMass = 0;
    double invPt = 0;
    nelectrons_before_->Fill(electronCollectionSize);
    if (electronCollectionSize > 1)
    {
        invMass = sqrt(electron[0][1] + electron[1][1] +
                       2 * (electron[0][2] * electron[1][2] -
                            (electron[0][3] * electron[1][3] +
                             electron[0][4] * electron[1][4] + electron[0][5] * electron[1][5])));
        //invPt = sqrt(electron[0][3] * electron[1][3] +
        //                    electron[0][4] * electron[1][4]); 

        const math::XYZTLorentzVector Reco(electron[0][3] + electron[1][3], electron[0][4] + electron[1][4],
                                           electron[0][5] + electron[1][5], electron[0][2] + electron[1][2]);

        invPt = Reco.pt();
        rapidity = Reco.Rapidity();
        invmass_before_->Fill(invMass);
        invpt_before_->Fill(invPt);
        invmassPU_before_->Fill(invMass, npvCount);
        if (fabs(electron[0][6] - electron[1][6]) <= M_PI) {
            deltam = (fabs(electron[0][6] - electron[1][6]));
        } else {
            deltam = (2 * M_PI - fabs(electron[0][6] - electron[1][6]));
        }

        acopl = (M_PI - deltam);
        if (electron[0][8] < 0) {
            Rapdiff = ((electron[0][7] - electron[1][7]) / 2);
        } else {
            Rapdiff = ((electron[1][7] - electron[0][7]) / 2);
        }

        Cos_thetastar = tanh(Rapdiff);
        Thetastar = acos(Cos_thetastar);
        angle = sin(Thetastar);
        halfacopl = acopl / 2;
        tan_acopl = tan(halfacopl);
        result_phistar = (tan_acopl * angle);
        Phistar_->Fill(result_phistar);
        const int ZMassBins = 4;
        double ZMassGrid[4] = { 60, 80, 100, 120 };
        int bin = FindMassBin(ZMassGrid, invMass, ZMassBins);   //calling function
        CosineThetaStar_->Fill(Cos_thetastar);
        CosineThetaStar_2D[bin]->Fill(Cos_thetastar);
        const int MuRapBins = 4;
        double MuRapGrid[4] = { 0, 0.8, 1.6, 2.4 };
        int Rapbin = FindRapBin(MuRapGrid, fabs(rapidity), MuRapBins);  //calling functio
        CosineThetaStar_Y_2D[Rapbin]->Fill(Cos_thetastar);
    }

    nelectrons_after_->Fill(nGoodElectrons);
    if (nGoodElectrons > 1) {
        invMass = sqrt(goodElectron[0][1] + goodElectron[1][1] +
                       2 * (goodElectron[0][2] * goodElectron[1][2] -
                            (goodElectron[0][3] * goodElectron[1][3] +
                             goodElectron[0][4] * goodElectron[1][4] + goodElectron[0][5] * goodElectron[1][5])));

        // invPt = sqrt(goodElectron[0][3] *goodElectron[1][3] +
        //                       goodElectron[0][4] * goodElectron[1][4]);

        const math::XYZTLorentzVector Reco(goodElectron[0][3] + goodElectron[1][3],
                                           goodElectron[0][4] + goodElectron[1][4],
                                           goodElectron[0][5] + goodElectron[1][5],
                                           goodElectron[0][2] + goodElectron[1][2]);

        rapidity = Reco.Rapidity();
        invPt = Reco.pt();
        invmass_after_->Fill(invMass);
        invpt_after_->Fill(invPt);
        invmassPU_afterZ_->Fill(invMass, npvCount);
        npvs_afterZ_->Fill(npvCount);

        if (fabs(goodElectron[0][6] - goodElectron[1][6]) <= M_PI) {
            deltam = (fabs(goodElectron[0][6] - goodElectron[1][6]));
        } else {
            deltam = (2 * M_PI - fabs(goodElectron[0][6] - goodElectron[1][6]));
        }

        acopl = (M_PI - deltam);
        if (goodElectron[0][8] < 0) {
            Rapdiff = ((goodElectron[0][7] - goodElectron[1][7]) / 2);
        } else {
            Rapdiff = ((goodElectron[1][7] - goodElectron[0][7]) / 2);
        }

        Cos_thetastar = tanh(Rapdiff);
        Thetastar = acos(Cos_thetastar);
        angle = sin(Thetastar);
        halfacopl = acopl / 2;
        tan_acopl = tan(halfacopl);
        result_phistar = (tan_acopl * angle);
        Phistar_after_->Fill(result_phistar);

        const int ZMassBins = 4;
        double ZMassGrid[4] = { 60, 80, 100, 120 };

        int bin = FindMassBin(ZMassGrid, invMass, ZMassBins);   //calling function

        CosineThetaStar_afterZ_2D[bin]->Fill(Cos_thetastar);

        const int MuRapBins = 4;
        double MuRapGrid[4] = { 0, 0.8, 1.6, 2.4 };

        int Rapbin = FindRapBin(MuRapGrid, fabs(rapidity), MuRapBins);  //calling functio

        CosineThetaStar_Y_afterZ_2D[Rapbin]->Fill(Cos_thetastar);

    }
    if (rec_sel)
        nrec++;
    if (eid_sel)
        neid++;
    if (iso_sel)
        niso++;

    if (all_sel) {
        nsel++;
        LogTrace("") << ">>>> Event ACCEPTED";
    } else {
        LogTrace("") << ">>>> Event REJECTED";
    }
    return;
}

int EwkElecDQM::FindMassBin(double MassGrid[], double Mass, const int size)
{
    for (int m = 0; m < size - 1; m++) {
        if (Mass >= MassGrid[m] && Mass < MassGrid[m + 1])
            return m;
    }
    return -1;
}

int EwkElecDQM::FindRapBin(double RapGrid[], double Rap, const int size)
{
    for (int m = 0; m < size - 1; m++) {
        if (Rap >= RapGrid[m] && Rap < RapGrid[m + 1])
            return m;
    }
    return -1;
}

// ------------ method called once each job just before starting event loop  ------------

double EwkElecDQM::calcDeltaPhi(double phi1, double phi2)
{

    double deltaPhi = phi1 - phi2;

    if (deltaPhi < 0)
        deltaPhi = -deltaPhi;

    if (deltaPhi > 3.1415926) {
        deltaPhi = 2 * 3.1415926 - deltaPhi;
    }

    return deltaPhi;
}

//define this as a plug-in
DEFINE_FWK_MODULE(EwkElecDQM);
