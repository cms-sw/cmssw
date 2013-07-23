/*
 *  See header file for a description of this class.
 *
 *  $Date: 2012/10/25 11:54:49 $
 *  $Revision: 1.22 $
 *  \author Valentina Gori, University of Firenze
 */

#include "DQM/Physics/src/EwkDQM.h"

#include <vector>
#include <string>
#include <cmath>

#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Common/interface/TriggerNames.h"

// Physics Objects
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "TLorentzVector.h"

using namespace std;
using namespace edm;
using namespace reco;



// EwkDQM::EwkDQM(const ParameterSet& parameters) {
EwkDQM::EwkDQM(const ParameterSet& parameters) {
  eJetMin_     = parameters.getUntrackedParameter<double>("EJetMin", 999999.);

  // riguardare questa sintassi
  // Get parameters from configuration file
  theElecTriggerPathToPass_ =
    parameters.getParameter<std::vector<string> >("elecTriggerPathToPass");
  theMuonTriggerPathToPass_ =
    parameters.getParameter<std::vector<string> >("muonTriggerPathToPass");
  //   eleTrigPathNames_ =
  //     parameters.getUntrackedParameter<std::vector<std::string> >("eleTrigPathNames");
  //   muTrigPathNames_ =
  //     parameters.getUntrackedParameter< std::vector<std::string> >("muTrigPathNames");
  theTriggerResultsCollection_ =
    parameters.getParameter<InputTag>("triggerResultsCollection");
  theMuonCollectionLabel_ = parameters.getParameter<InputTag>("muonCollection");
  theElectronCollectionLabel_ =
    parameters.getParameter<InputTag>("electronCollection");
  //  theCaloJetCollectionLabel_   = parameters.getParameter<InputTag>("caloJetCollection");
  thePFJetCollectionLabel_ =
    parameters.getParameter<InputTag>("PFJetCollection");
  theCaloMETCollectionLabel_ = parameters.getParameter<InputTag>("caloMETCollection");

  // just to initialize
  isValidHltConfig_ = false;

  h_vertex_number = 0;
  h_vertex_chi2 = 0;
  h_vertex_numTrks = 0;
  h_vertex_sumTrks = 0;
  h_vertex_d0 = 0;

  h_jet_count = 0;
  h_jet_et = 0;
  h_jet_pt = 0;
  h_jet_eta = 0;
  h_jet_phi = 0;
  h_jet2_et = 0;
  // h_jet2_pt = 0;
  h_jet2_eta = 0;
  h_jet2_phi = 0;

  h_e1_et = 0;
  h_e2_et = 0;
  h_e1_eta = 0;
  h_e2_eta = 0;
  h_e1_phi = 0;
  h_e2_phi = 0;

  h_m1_pt = 0;
  h_m2_pt = 0;
  h_m1_eta = 0;
  h_m2_eta = 0;
  h_m1_phi = 0;
  h_m2_phi = 0;

  // h_t1_et = 0;
  // h_t1_eta = 0;
  // h_t1_phi = 0;

  h_met = 0;
  h_met_phi = 0;

  h_e_invWMass = 0;
  h_m_invWMass = 0;
  h_mumu_invMass = 0;
  h_ee_invMass = 0;

  theDbe = Service<DQMStore>().operator->();
}

EwkDQM::~EwkDQM() {
}


void EwkDQM::beginJob() {
  char chtitle[256] = "";
  const size_t title_s = sizeof(chtitle);

  logTraceName = "EwkAnalyzer";

  LogTrace(logTraceName) << "Parameters initialization";
  theDbe->setCurrentFolder("Physics/EwkDQM");  // Use folder with name of PAG

  const float pi = 4*atan(1);

  // Keep the number of plots and number of bins to a minimum!
  h_vertex_number = theDbe->book1D("vertex_number",
                                   "Number of event vertices in collection",
                                   10, -0.5, 9.5);
  h_vertex_chi2  = theDbe->book1D("vertex_chi2",
                                  "Event Vertex #chi^{2}/n.d.o.f.",
                                  20, 0.0, 2.0);
  h_vertex_numTrks = theDbe->book1D("vertex_numTrks",
                                    "Event Vertex, number of tracks",
                                    20, -0.5, 59.5);
  h_vertex_sumTrks = theDbe->book1D("vertex_sumTrks",
                                    "Event Vertex, sum of track pt",
                                    20, 0.0, 100.0);
  h_vertex_d0 = theDbe->book1D("vertex_d0",
                               "Event Vertex d0", 20, 0.0, 0.05);

  snprintf(chtitle, title_s, "Number of %s (E_{T} > 15 GeV);Number of Jets",
           thePFJetCollectionLabel_.label().data());
  h_jet_count = theDbe->book1D("jet_count", chtitle, 8, -0.5, 7.5);

  snprintf(chtitle, title_s, "Leading jet E_{T} (from %s);E_{T}(1^{st} jet) (GeV)",
           thePFJetCollectionLabel_.label().data());
  h_jet_et = theDbe->book1D("jet_et", chtitle,    20, 0., 200.0);

  snprintf(chtitle, title_s, "Leading jet p_{T} (from %s);p_{T}(1^{st} jet) (GeV/c)",
           thePFJetCollectionLabel_.label().data());
  h_jet_pt = theDbe->book1D("jet_pt", chtitle,  20, 0., 200.0);

  snprintf(chtitle, title_s,  "Leading jet #eta (from %s); #eta (1^{st} jet)",
           thePFJetCollectionLabel_.label().data());
  h_jet_eta = theDbe->book1D("jet_eta", chtitle, 20, -10., 10.0);
  snprintf(chtitle, title_s, "Leading jet #phi (from %s); #phi(1^{st} jet)",
           thePFJetCollectionLabel_.label().data());
  h_jet_phi = theDbe->book1D("jet_phi", chtitle, 22, -1.1*pi, 1.1*pi);

  snprintf(chtitle, title_s, "2^{nd} leading jet E_{T} (from %s);E_{T}(2^{nd} jet) (GeV)",
           thePFJetCollectionLabel_.label().data());
  h_jet2_et = theDbe->book1D("jet2_et", chtitle,  20, 0., 200.0);
  // snprintf(chtitle, title_s, "2^{nd} leading jet p_{T} (from %s);p_{T}(2^{nd} jet) (GeV/c)",
  // thePFJetCollectionLabel_.label().data());
  // h_jet2_pt = theDbe->book1D("jet2_pt", chtitle,  20, 0., 200.0);

  snprintf(chtitle, title_s,  "2^{nd} leading jet #eta (from %s); #eta (2^{nd} jet)",
           thePFJetCollectionLabel_.label().data());
  h_jet2_eta = theDbe->book1D("jet2_eta", chtitle,  20, -10., 10.0);

  snprintf(chtitle, title_s, "2^{nd} leading jet #phi (from %s); #phi(2^{nd} jet)",
           thePFJetCollectionLabel_.label().data());
  h_jet2_phi = theDbe->book1D("jet2_phi", chtitle,  22, -1.1*pi, 1.1*pi);

  h_e1_et  = theDbe->book1D("e1_et",  "E_{T} of Leading Electron;E_{T} (GeV)",
                            20, 0.0, 100.0);
  h_e2_et  = theDbe->book1D("e2_et",  "E_{T} of Second Electron;E_{T} (GeV)",
                            20, 0.0, 100.0);
  h_e1_eta = theDbe->book1D("e1_eta", "#eta of Leading Electron;#eta",
                            20, -4.0, 4.0);
  h_e2_eta = theDbe->book1D("e2_eta", "#eta of Second Electron;#eta",
                            20, -4.0, 4.0);
  h_e1_phi = theDbe->book1D("e1_phi", "#phi of Leading Electron;#phi",
                            22, -1.1*pi, 1.1*pi);
  h_e2_phi = theDbe->book1D("e2_phi", "#phi of Second Electron;#phi",
                            22, -1.1*pi, 1.1*pi);
  h_m1_pt  = theDbe->book1D("m1_pt",  "p_{T} of Leading Muon;p_{T}(1^{st} #mu) (GeV)",
                            20, 0.0, 100.0);
  h_m2_pt  = theDbe->book1D("m2_pt",  "p_{T} of Second Muon;p_{T}(2^{nd} #mu) (GeV)",
                            20, 0.0, 100.0);
  h_m1_eta = theDbe->book1D("m1_eta", "#eta of Leading Muon;#eta(1^{st} #mu)",
                            20, -4.0, 4.0);
  h_m2_eta = theDbe->book1D("m2_eta", "#eta of Second Muon;#eta(2^{nd} #mu)",
                            20, -4.0, 4.0);
  h_m1_phi = theDbe->book1D("m1_phi", "#phi of Leading Muon;#phi(1^{st} #mu)",
                            20, (-1. - 1./10.)*pi, (1. + 1./10.)*pi);
  h_m2_phi = theDbe->book1D("m2_phi", "#phi of Second Muon;#phi(2^{nd} #mu)",
                            20, (-1. - 1./10.)*pi, (1. + 1./10.)*pi);
  //  h_t1_et = theDbe->book1D("t1_et", "E_{T} of Leading Tau;E_{T} (GeV)",
  //  20, 0.0 , 100.0);
  //  h_t1_eta = theDbe->book1D("t1_eta", "#eta of Leading Tau;#eta",
  //  20, -4.0, 4.0);
  //  h_t1_phi = theDbe->book1D("t1_phi", "#phi of Leading Tau;#phi",
  //  20, -4.0, 4.0);
  snprintf(chtitle, title_s, "Missing E_{T} (%s); GeV",
           theCaloMETCollectionLabel_.label().data());
  h_met          = theDbe->book1D("met", chtitle, 20, 0.0, 100);
  h_met_phi      = theDbe->book1D("met_phi", "Missing E_{T} #phi;#phi(MET)",
                                  22, (-1. - 1./10.)*pi, (1. + 1./10.)*pi);

  h_e_invWMass   = theDbe->book1D("we_invWMass", "W-> e #nu Transverse Mass;M_{T} (GeV)",
                                  20, 0.0, 140.0);
  h_m_invWMass   = theDbe->book1D("wm_invWMass", "W-> #mu #nu Transverse Mass;M_{T} (GeV)",
                                  20, 0.0, 140.0);
  h_mumu_invMass = theDbe->book1D("z_mm_invMass", "#mu#mu Invariant Mass;InvMass (GeV)",
                                  20, 40.0, 140.0);
  h_ee_invMass   = theDbe->book1D("z_ee_invMass", "ee Invariant Mass;InvMass (Gev)",
                                  20, 40.0, 140.0);
}

///
///
///
void EwkDQM::beginRun(const edm::Run& theRun, const edm::EventSetup& theSetup) {
  // passed as parameter to HLTConfigProvider::init(), not yet used
  bool isConfigChanged = false;

  // isValidHltConfig_ used to short-circuit analyze() in case of problems
  const std::string hltProcessName(theTriggerResultsCollection_.process());
  isValidHltConfig_ = hltConfigProvider_.init(theRun, theSetup,
                                              hltProcessName, isConfigChanged);
}


void EwkDQM::analyze(const Event& iEvent, const EventSetup& iSetup) {
  // short-circuit if hlt problems
  if (!isValidHltConfig_)
    return;

  LogTrace(logTraceName) << "Analysis of event # ";
  // Did it pass certain HLT path?
  Handle<TriggerResults> HLTresults;
  iEvent.getByLabel(theTriggerResultsCollection_, HLTresults);
  if (!HLTresults.isValid())
    return;

  const edm::TriggerNames & trigNames = iEvent.triggerNames(*HLTresults);

  // a temporary, until we have a list of triggers of interest
  std::vector<std::string> eleTrigPathNames;
  std::vector<std::string> muTrigPathNames;

  // eleTrigPathNames.push_back(theElecTriggerPathToPass_);
  // muTrigPathNames.push_back(theMuonTriggerPathToPass_);
  // end of temporary

  bool passed_electron_HLT = false;
  bool passed_muon_HLT     = false;
  for (unsigned int i = 0; i < HLTresults->size(); i++) {
    const std::string trigName = trigNames.triggerName(i);
    // check if triggerName matches electronPath
    for (unsigned int index = 0;
         index < theElecTriggerPathToPass_.size() && !passed_electron_HLT;
         index++) {
      // 0 if found, pos if not
      size_t trigPath = trigName.find(theElecTriggerPathToPass_[index]);
      if (trigPath == 0) {
        //      cout << "MuonTrigger passed (=trigName): " << trigName <<endl;
        passed_electron_HLT = HLTresults->accept(i);
      }
    }
    // check if triggerName matches muonPath
    for (unsigned int index = 0;
         index < theMuonTriggerPathToPass_.size() && !passed_muon_HLT;
         index++) {
      // 0 if found, pos if not
      size_t trigPath = trigName.find(theMuonTriggerPathToPass_[index]);
      if (trigPath == 0) {
        //      cout << "MuonTrigger passed (=trigName): " << trigName <<endl;
        passed_muon_HLT = HLTresults->accept(i);
      }
    }
  }

  // we are interested in events with a valid electron or muon
  if (!(passed_electron_HLT || passed_muon_HLT))
    return;

  ////////////////////////////////////////////////////////////////////////////////
  // Vertex information
  Handle<VertexCollection> vertexHandle;
  iEvent.getByLabel("offlinePrimaryVertices", vertexHandle);
  if (!vertexHandle.isValid())
    return;
  VertexCollection vertexCollection = *(vertexHandle.product());
  VertexCollection::const_iterator v = vertexCollection.begin();
  int vertex_number     = vertexCollection.size();
  double vertex_chi2    = v->normalizedChi2();  // v->chi2();
  double vertex_d0      = sqrt(v->x()*v->x()+v->y()*v->y());
  double vertex_numTrks = v->tracksSize();
  double vertex_sumTrks = 0.0;
  // std::cout << "vertex_d0=" << vertex_d0 << "\n";
  // double vertex_ndof    = v->ndof();cout << "ndof="<<vertex_ndof<<endl;
  for (Vertex::trackRef_iterator vertex_curTrack = v->tracks_begin();
       vertex_curTrack != v->tracks_end(); vertex_curTrack++)
    vertex_sumTrks += (*vertex_curTrack)->pt();

  ////////////////////////////////////////////////////////////////////////////////
  // Missing ET
  Handle< View<MET> > caloMETCollection;
  iEvent.getByLabel(theCaloMETCollectionLabel_, caloMETCollection);
  if (!caloMETCollection.isValid())
    return;
  float missing_et = caloMETCollection->begin()->et();
  float met_phi    = caloMETCollection->begin()->phi();


  ////////////////////////////////////////////////////////////////////////////////
  // grab "gaussian sum fitting" electrons
  Handle<GsfElectronCollection> electronCollection;
  iEvent.getByLabel(theElectronCollectionLabel_, electronCollection);
  if (!electronCollection.isValid())
    return;

  // Find the highest and 2nd highest electron
  float electron_et   = -8.0;
  float electron_eta  = -8.0;
  float electron_phi  = -8.0;
  float electron2_et  = -9.0;
  float electron2_eta = -9.0;
  float electron2_phi = -9.0;
  float ee_invMass    = -9.0;
  TLorentzVector e1, e2;

  // If it passed electron HLT and the collection was found, find electrons near Z mass
  if (passed_electron_HLT) {
    for (reco::GsfElectronCollection::const_iterator recoElectron = electronCollection->begin();
         recoElectron != electronCollection->end(); recoElectron++) {
      // Require electron to pass some basic cuts
      if (recoElectron->et() < 20 || fabs(recoElectron->eta()) > 2.5)
        continue;

      // Tighter electron cuts
      if (recoElectron->deltaPhiSuperClusterTrackAtVtx() > 0.58 ||
          recoElectron->deltaEtaSuperClusterTrackAtVtx() > 0.01 ||
          recoElectron->sigmaIetaIeta() > 0.027)
        continue;

      if (recoElectron->et() > electron_et) {
        electron2_et  = electron_et;  // 2nd highest gets values from current highest
        electron2_eta = electron_eta;
        electron2_phi = electron_phi;
        electron_et   = recoElectron->et();  // 1st highest gets values from new highest
        electron_eta  = recoElectron->eta();
        electron_phi  = recoElectron->phi();
        e1 = TLorentzVector(recoElectron->momentum().x(),
                            recoElectron->momentum().y(),
                            recoElectron->momentum().z(),
                            recoElectron->p());
      } else if (recoElectron->et() > electron2_et) {
        electron2_et  = recoElectron->et();
        electron2_eta = recoElectron->eta();
        electron2_phi = recoElectron->phi();
        e2 = TLorentzVector(recoElectron->momentum().x(),
                            recoElectron->momentum().y(),
                            recoElectron->momentum().z(),
                            recoElectron->p());
      }
    }  // end of loop over electrons
    if (electron2_et > 0.0) {
      TLorentzVector pair = e1+e2;
      ee_invMass = pair.M();
    }
  }  // end of "are electrons valid"
  ////////////////////////////////////////////////////////////////////////////////



  ////////////////////////////////////////////////////////////////////////////////
  // Take the STA muon container
  Handle<MuonCollection> muonCollection;
  iEvent.getByLabel(theMuonCollectionLabel_, muonCollection);
  if (!muonCollection.isValid())
    return;

  // Find the highest pt muons
  float mm_invMass = -9.0;
  float muon_pt    = -9.0;
  float muon_eta   = -9.0;
  float muon_phi   = -9.0;
  float muon2_pt   = -9.0;
  float muon2_eta  = -9.0;
  float muon2_phi  = -9.0;
  TLorentzVector m1, m2;

  if (passed_muon_HLT) {
    for (reco::MuonCollection::const_iterator recoMuon = muonCollection->begin();
         recoMuon != muonCollection->end(); recoMuon++) {
      // Require muon to pass some basic cuts
      if (recoMuon->pt() < 20 || !recoMuon->isGlobalMuon())
        continue;
      // Some tighter muon cuts
      if (recoMuon->globalTrack()->normalizedChi2() > 10)
        continue;

      if (recoMuon->pt() > muon_pt) {
        muon2_pt  = muon_pt;  // 2nd highest gets values from current highest
        muon2_eta = muon_eta;
        muon2_phi = muon_phi;
        muon_pt   = recoMuon->pt();  // 1st highest gets values from new highest
        muon_eta  = recoMuon->eta();
        muon_phi  = recoMuon->phi();
        m1 = TLorentzVector(recoMuon->momentum().x(),
                            recoMuon->momentum().y(),
                            recoMuon->momentum().z(),
                            recoMuon->p());
      } else if (recoMuon->pt() > muon2_pt) {
        muon2_pt  = recoMuon->pt();
        muon2_eta = recoMuon->eta();
        muon2_phi = recoMuon->phi();
        m2 = TLorentzVector(recoMuon->momentum().x(),
                            recoMuon->momentum().y(),
                            recoMuon->momentum().z(),
                            recoMuon->p());
      }
    }
  }
  if (muon2_pt > 0.0) {
    TLorentzVector pair = m1+m2;
    mm_invMass = pair.M();
  }
  ////////////////////////////////////////////////////////////////////////////////


  ////////////////////////////////////////////////////////////////////////////////
  // Find the highest et jet

  //  Handle<CaloJetCollection> caloJetCollection;
  Handle<View<Jet> > PFJetCollection;
  //  iEvent.getByLabel (theCaloJetCollectionLabel,caloJetCollection);
  iEvent.getByLabel(thePFJetCollectionLabel_, PFJetCollection);
  //  if ( !caloJetCollection.isValid() ) return;
  if (!PFJetCollection.isValid())
    return;

  unsigned int muonCollectionSize = muonCollection->size();
  // unsigned int jetCollectionSize = jetCollection->size();
  unsigned int PFJetCollectionSize = PFJetCollection->size();
  int jet_count = 0;
  // int LEADJET=-1;  double max_pt=0;


  float jet_et    = -80.0;
  float jet_pt    = -80.0;  // prova
  float jet_eta   = -80.0;  // now USED
  float jet_phi   = -80.0;  // now USED
  float jet2_et   = -90.0;
  float jet2_eta  = -90.0;  // now USED
  float jet2_phi  = -90.0;  // now USED
  //  for (CaloJetCollection::const_iterator i_calojet = caloJetCollection->begin();
  //       i_calojet != caloJetCollection->end(); i_calojet++) {
  //  for (PFJetCollection::const_iterator i_pfjet = PFJetCollection->begin();
  //       i_pfjet != PFJetCollection->end(); i_pfjet++) {
  //  float jet_current_et = i_calojet->et();
  //  float jet_current_et = i_pfjet->et();            // e` identico a jet.et()
  //    jet_count++;

  // cleaning: va messo prima del riempimento dell'istogramma // This is in order to use PFJets
  for (unsigned int i = 0; i < PFJetCollectionSize; i++) {
    const Jet& jet = PFJetCollection->at(i);
    // la classe "jet" viene definita qui!!!
    double minDistance = 99999;
    for (unsigned int j = 0; j < muonCollectionSize; j++) {
      const Muon& mu = muonCollection->at(j);
      double distance = sqrt((mu.eta() - jet.eta()) * (mu.eta() - jet.eta()) +
                             (mu.phi() - jet.phi()) * (mu.phi() - jet.phi()));
      if (minDistance > distance)
        minDistance = distance;
    }
    if (minDistance < 0.3)
      continue;  // 0.3 is the isolation cone around the muon
    // se la distanza muone-cono del jet e` minore di 0.3, passo avanti e non conteggio il mio jet

    // If it overlaps with ELECTRON, it is not a jet
    if (electron_et > 0.0 &&
        fabs(jet.eta() - electron_eta) < 0.2 &&
        calcDeltaPhi(jet.phi(), electron_phi) < 0.2)
      continue;
    if (electron2_et > 0.0 &&
        fabs(jet.eta() - electron2_eta) < 0.2 &&
        calcDeltaPhi(jet.phi(), electron2_phi) < 0.2)
      continue;

    // provo a cambiare la parte degli elettroni in modo simmetrico alla parte per i muoni

    // ...
    // ...


    // if it has too low Et, throw away
    if (jet.et() < eJetMin_)
      continue;
    jet_count++;

    // ovvero: incrementa jet_count se:
    //   - non c'e un muone entro 0.3 di distanza dal cono del jet;
    //   - se il jet non si sovrappone ad un elettrone;
    //   - se l'energia trasversa e` maggiore della soglia impostata (15?)

    // if(jet.et()>max_pt) { LEADJET=i; max_pt=jet.et();}
    // se l'energia del jet e` maggiore di max_pt, diventa "i"
    // l'indice del jet piu` energetico e max_pt la sua energia

    // riguardare questo!!!
    // fino ad ora, jet_et era inizializzato a -8.0
    if (jet.et() > jet_et) {
      jet2_et  = jet_et;  // 2nd highest jet gets et from current highest
      // perche` prende l'energia del primo jet??
      jet2_eta = jet_eta;  // now USED
      jet2_phi = jet_phi;  // now USED
      // jet_et = i_calojet->et(); // current highest jet gets
      // et from the new highest
      jet_et   = jet.et();  // current highest jet gets et from the new highest
      // ah, ok! lo riaggiorna solo dopo!
      jet_pt   = jet.pt();   // e` il pT del leading jet
      jet_eta  = jet.eta();  // now USED
      jet_phi  = jet.phi() * (Geom::pi() / 180.);  // now USED
    } else if (jet.et() > jet2_et) {
      //      jet2_et  = i_calojet->et();
      jet2_et  = jet.et();
      //      jet2_eta = i_calojet->eta();  // UNUSED
      //      jet2_phi = i_calojet->phi();  // UNUSED
      jet2_eta = jet.eta();  // now USED
      jet2_phi = jet.phi();  // now USED
    }
    // questo elseif funziona
  }
  ////////////////////////////////////////////////////////////////////////////////



  ////////////////////////////////////////////////////////////////////////////////
  //                 Fill Histograms                                            //
  ////////////////////////////////////////////////////////////////////////////////

  bool fill_e1  = false;
  bool fill_e2  = false;
  bool fill_m1  = false;
  bool fill_m2  = false;
  bool fill_met = false;

  // Was Z->ee found?
  if (ee_invMass > 0.0) {
    h_ee_invMass->Fill(ee_invMass);
    fill_e1 = true;
    fill_e2 = true;
  }

  // Was Z->mu mu found?
  if (mm_invMass > 0.0) {
    h_mumu_invMass->Fill(mm_invMass);
    fill_m1 = true;
    fill_m2 = true;
    h_jet2_et ->Fill(jet2_et);
  }

  // Was W->e nu found?
  if (electron_et > 0.0 && missing_et > 20.0) {
    float dphiW  = fabs(met_phi-electron_phi);
    float W_mt_e = sqrt(2 * missing_et * electron_et * (1 - cos(dphiW)));
    h_e_invWMass->Fill(W_mt_e);
    fill_e1  = true;
    fill_met = true;
  }

  // Was W->mu nu found?
  if (muon_pt > 0.0 && missing_et > 20.0) {
    float dphiW = fabs(met_phi - muon_phi);
    float W_mt_m = sqrt(2 * missing_et * muon_pt * (1 - cos(dphiW)));
    h_m_invWMass->Fill(W_mt_m);
    fill_m1  = true;
    fill_met = true;
  }

  if (jet_et > -10.0) {
    h_jet_et->Fill(jet_et);
    h_jet_count->Fill(jet_count);
  }

  if (jet_pt > 0.) {
    h_jet_pt->Fill(jet_pt);
  }

  if (jet_eta > -50.) {
    h_jet_eta->Fill(jet_eta);
  }

  if (jet_phi > -10.) {
    h_jet_phi->Fill(jet_phi);
  }

  if (jet2_et > -10.0) {
    h_jet2_et->Fill(jet2_et);
  }

  // if (jet2_pt>0.) {
  //  h_jet2_pt   ->Fill(jet2_pt);
  // }

  if (jet2_eta > -50.) {
    h_jet2_eta->Fill(jet2_eta);
  }

  if (jet2_phi > -10.) {
    h_jet2_phi->Fill(jet2_phi);
  }



  if (fill_e1 || fill_m1) {
    h_vertex_number->Fill(vertex_number);
    h_vertex_chi2->Fill(vertex_chi2);
    h_vertex_d0->Fill(vertex_d0);
    h_vertex_numTrks->Fill(vertex_numTrks);
    h_vertex_sumTrks->Fill(vertex_sumTrks);
  }

  if (fill_e1) {
    h_e1_et->Fill(electron_et);
    h_e1_eta->Fill(electron_eta);
    h_e1_phi->Fill(electron_phi);
  }
  if (fill_e2) {
    h_e2_et->Fill(electron2_et);
    h_e2_eta->Fill(electron2_eta);
    h_e2_phi->Fill(electron2_phi);
  }
  if (fill_m1) {
    h_m1_pt->Fill(muon_pt);
    h_m1_eta->Fill(muon_eta);
    h_m1_phi->Fill(muon_phi);
  }
  if (fill_m2) {
    h_m2_pt->Fill(muon2_pt);
    h_m2_eta->Fill(muon2_eta);
    h_m2_phi->Fill(muon2_phi);
  }
  if (fill_met) {
    h_met->Fill(missing_et);
    h_met_phi->Fill(met_phi);
  }
  ////////////////////////////////////////////////////////////////////////////////
}


void EwkDQM::endJob(void) {}


// This always returns only a positive deltaPhi
double EwkDQM::calcDeltaPhi(double phi1, double phi2) {
  double deltaPhi = phi1 - phi2;

  if (deltaPhi < 0)
    deltaPhi = -deltaPhi;

  if (deltaPhi > 3.1415926)
    deltaPhi = 2 * 3.1415926 - deltaPhi;

  return deltaPhi;
}
