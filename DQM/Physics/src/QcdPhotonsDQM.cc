/*
 *  See header file for a description of this class.
 *
 *  $Date: 2010/06/16 15:53:52 $
 *  $Revision: 1.25 $
 *  \author Michael B. Anderson, University of Wisconsin Madison
 */

#include "DQM/Physics/src/QcdPhotonsDQM.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Physics Objects
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

// Vertex
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

// For removing ECAL Spikes
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalCondObjectContainer.h"

//geometry
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalPreshowerGeometry.h"

// Math stuff
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include <vector>

#include <string>
#include <cmath>
using namespace std;
using namespace edm;
using namespace reco;



QcdPhotonsDQM::QcdPhotonsDQM(const ParameterSet& parameters) {
  // Get parameters from configuration file
  theTriggerPathToPass        = parameters.getParameter<string>("triggerPathToPass");
  thePlotTheseTriggersToo     = parameters.getParameter<vector<string> >("plotTheseTriggersToo");
  theHltMenu                  = parameters.getParameter<string>("hltMenu");
  theTriggerResultsCollection = parameters.getParameter<string>("triggerResultsCollection");
  thePhotonCollectionLabel    = parameters.getParameter<InputTag>("photonCollection");
  theCaloJetCollectionLabel   = parameters.getParameter<InputTag>("caloJetCollection");
  theVertexCollectionLabel    = parameters.getParameter<InputTag>("vertexCollection");
  theMinCaloJetPt             = parameters.getParameter<double>("minCaloJetPt");
  theMinPhotonEt              = parameters.getParameter<double>("minPhotonEt");
  theRequirePhotonFound       = parameters.getParameter<bool>("requirePhotonFound");
  thePlotPhotonMaxEt          = parameters.getParameter<double>("plotPhotonMaxEt");
  thePlotPhotonMaxEta         = parameters.getParameter<double>("plotPhotonMaxEta");
  thePlotJetMaxEta            = parameters.getParameter<double>("plotJetMaxEta");
  // just to initialize
  isValidHltConfig_ = false;

}

QcdPhotonsDQM::~QcdPhotonsDQM() { 
}


void QcdPhotonsDQM::beginJob() {
 
  logTraceName = "QcdPhotonAnalyzer";

  LogTrace(logTraceName)<<"Parameters initialization";
  theDbe = Service<DQMStore>().operator->();
 
  theDbe->setCurrentFolder("Physics/QcdPhotons");  // Use folder with name of PAG

  std::stringstream aStringStream;
  std::string aString;
  aStringStream << theMinCaloJetPt;
  aString = aStringStream.str();

  // Monitor of triggers passed
  int numOfTriggersToMonitor = thePlotTheseTriggersToo.size();
  h_triggers_passed = theDbe->book1D("h_triggers_passed", "Events passing these trigger paths", numOfTriggersToMonitor, 0, numOfTriggersToMonitor);
  for (int i=0; i<numOfTriggersToMonitor; i++) {
    h_triggers_passed->setBinLabel(i+1,thePlotTheseTriggersToo[i]);
  }

  // Keep the number of plots and number of bins to a minimum!
  h_photon_et           = theDbe->book1D("h_photon_et",       "#gamma with highest E_{T};E_{T}(#gamma) (GeV)", 20, 0., thePlotPhotonMaxEt);
  h_photon_eta          = theDbe->book1D("h_photon_eta",      "#gamma with highest E_{T};#eta(#gamma)", 40, -thePlotPhotonMaxEta, thePlotPhotonMaxEta);
  h_photon_count_bar    = theDbe->book1D("h_photon_count_bar","Number of #gamma's passing selection (Barrel);Number of #gamma's", 8, -0.5, 7.5);
  h_photon_count_end    = theDbe->book1D("h_photon_count_end","Number of #gamma's passing selection (Endcap);Number of #gamma's", 8, -0.5, 7.5);

  h_jet_pt             = theDbe->book1D("h_jet_pt",   "Jet with highest p_{T} (from "+theCaloJetCollectionLabel.label()+");p_{T}(1^{st} jet) (GeV)",    20, 0., thePlotPhotonMaxEt);
  h_jet_eta             = theDbe->book1D("h_jet_eta", "Jet with highest p_{T} (from "+theCaloJetCollectionLabel.label()+");#eta(1^{st} jet)", 20, -thePlotJetMaxEta, thePlotJetMaxEta);
  h_deltaPhi_photon_jet = theDbe->book1D("h_deltaPhi_photon_jet", "#Delta#phi between Highest E_{T} #gamma and jet;#Delta#phi(#gamma,1^{st} jet)", 20, 0, 3.1415926);
  h_deltaPhi_jet_jet2   = theDbe->book1D("h_deltaPhi_jet_jet2", "#Delta#phi between Highest E_{T} jet and 2^{nd} jet;#Delta#phi(1^{st} jet,2^{nd} jet)", 20, 0, 3.1415926);
  h_deltaEt_photon_jet  = theDbe->book1D("h_deltaEt_photon_jet",  "(E_{T}(#gamma)-p_{T}(jet))/E_{T}(#gamma) when #Delta#phi(#gamma,1^{st} jet) > 2.8;#DeltaE_{T}(#gamma,1^{st} jet)/E_{T}(#gamma)", 20, -1.0, 1.0);
  h_jet_count           = theDbe->book1D("h_jet_count", "Number of "+theCaloJetCollectionLabel.label()+" (p_{T} > "+aString+" GeV);Number of Jets", 8, -0.5, 7.5);
  h_jet2_pt             = theDbe->book1D("h_jet2_pt",   "Jet with 2^{nd} highest p_{T} (from "+theCaloJetCollectionLabel.label()+");p_{T}(2^{nd} jet) (GeV)",    20, 0., thePlotPhotonMaxEt);
  h_jet2_eta            = theDbe->book1D("h_jet2_eta", "Jet with 2^{nd} highest p_{T} (from "+theCaloJetCollectionLabel.label()+");#eta(2^{nd} jet)", 20, -thePlotJetMaxEta, thePlotJetMaxEta);
  h_jet2_ptOverPhotonEt = theDbe->book1D("h_jet2_ptOverPhotonEt", "p_{T}(2^{nd} highest jet) / E_{T}(#gamma);p_{T}(2^{nd} Jet)/E_{T}(#gamma)", 20, 0.0, 4.0);
  h_deltaPhi_photon_jet2 = theDbe->book1D("h_deltaPhi_photon_jet2","#Delta#phi between Highest E_{T} #gamma and 2^{nd} highest jet;#Delta#phi(#gamma,2^{nd} jet)", 20, 0, 3.1415926);
  h_deltaR_jet_jet2      = theDbe->book1D("h_deltaR_jet_jet2", "#DeltaR between Highest Jet and 2^{nd} Highest;#DeltaR(1^{st} jet,2^{nd} jet)", 30, 0, 6.0);
  h_deltaR_photon_jet2   = theDbe->book1D("h_deltaR_photon_jet2", "#DeltaR between Highest E_{T} #gamma and 2^{nd} jet;#DeltaR(#gamma, 2^{nd} jet)", 30, 0, 6.0);

  // Photon Et for different jet configurations
  Float_t bins_et[] = {15,20,30,50,80};
  int num_bins_et = 4;
  h_photon_et_jetcs = theDbe->book1D("h_photon_et_jetcs", "#gamma with highest E_{T} (#eta(jet)<1.45, #eta(#gamma)#eta(jet)>0);E_{T}(#gamma) (GeV)", num_bins_et, bins_et);
  h_photon_et_jetco = theDbe->book1D("h_photon_et_jetco", "#gamma with highest E_{T} (#eta(jet)<1.45, #eta(#gamma)#eta(jet)<0);E_{T}(#gamma) (GeV)", num_bins_et, bins_et);
  h_photon_et_jetfs = theDbe->book1D("h_photon_et_jetfs", "#gamma with highest E_{T} (1.55<#eta(jet)<2.5, #eta(#gamma)#eta(jet)>0);E_{T}(#gamma) (GeV)", num_bins_et, bins_et);
  h_photon_et_jetfo = theDbe->book1D("h_photon_et_jetfo", "#gamma with highest E_{T} (1.55<#eta(jet)<2.5, #eta(#gamma)#eta(jet)<0);E_{T}(#gamma) (GeV)", num_bins_et, bins_et);
  h_photon_et_jetcs->getTH1F()->Sumw2();
  h_photon_et_jetco->getTH1F()->Sumw2();
  h_photon_et_jetfs->getTH1F()->Sumw2();
  h_photon_et_jetfo->getTH1F()->Sumw2();
  // Ratio of the above Photon Et distributions
  h_photon_et_ratio_co_cs = theDbe->book1D("h_photon_et_ratio_00_co_cs", "D(|#eta(jet)|<1.45, #eta(jet)*#eta(#gamma)<0) / D(|#eta(jet)|<1.45, #eta(jet)*#eta(#gamma)>0);E_{T}(#gamma) (GeV); ratio", num_bins_et, bins_et);
  h_photon_et_ratio_fo_fs = theDbe->book1D("h_photon_et_ratio_01_fo_fs", "D(1.55<|#eta(jet)|<2.6, #eta(jet)*#eta(#gamma)<0) / D(1.55<|#eta(jet)|<2.6, #eta(jet)*#eta(#gamma)>0);E_{T}(#gamma) (GeV); ratio", num_bins_et, bins_et);
  h_photon_et_ratio_cs_fs = theDbe->book1D("h_photon_et_ratio_02_cs_fs", "D(|#eta(jet)|<1.45, #eta(jet)*#eta(#gamma)>0) / D(1.55<|#eta(jet)|<2.6, #eta(jet)*#eta(#gamma)>0);E_{T}(#gamma) (GeV); ratio", num_bins_et, bins_et);
  h_photon_et_ratio_co_fs = theDbe->book1D("h_photon_et_ratio_03_co_fs", "D(|#eta(jet)|<1.45, #eta(jet)*#eta(#gamma)<0) / D(1.55<|#eta(jet)|<2.6, #eta(jet)*#eta(#gamma)>0);E_{T}(#gamma) (GeV); ratio", num_bins_et, bins_et);
  h_photon_et_ratio_cs_fo = theDbe->book1D("h_photon_et_ratio_04_cs_fo", "D(|#eta(jet)|<1.45, #eta(jet)*#eta(#gamma)>0) / D(1.55<|#eta(jet)|<2.6, #eta(jet)*#eta(#gamma)<0);E_{T}(#gamma) (GeV); ratio", num_bins_et, bins_et);
  h_photon_et_ratio_co_fo = theDbe->book1D("h_photon_et_ratio_05_co_fo", "D(|#eta(jet)|<1.45, #eta(jet)*#eta(#gamma)<0) / D(1.55<|#eta(jet)|<2.6, #eta(jet)*#eta(#gamma)<0);E_{T}(#gamma) (GeV); ratio", num_bins_et, bins_et);
  h_photon_et_ratio_co_cs->getTH1F()->Sumw2();
  h_photon_et_ratio_fo_fs->getTH1F()->Sumw2();
  h_photon_et_ratio_cs_fs->getTH1F()->Sumw2();
  h_photon_et_ratio_co_fs->getTH1F()->Sumw2();
  h_photon_et_ratio_cs_fo->getTH1F()->Sumw2();
  h_photon_et_ratio_co_fo->getTH1F()->Sumw2();
}


///
///
///
void QcdPhotonsDQM::beginRun( const edm::Run &r, const edm::EventSetup &iSetup ) {

  // passed as parameter to HLTConfigProvider::init(), not yet used
  bool isConfigChanged = false;

  // isValidHltConfig_ used to short-circuit analyze() in case of problems
  isValidHltConfig_ = hltConfigProvider_.init( r, iSetup, theHltMenu, isConfigChanged );

  num_events_in_run = 0;
}


void QcdPhotonsDQM::analyze(const Event& iEvent, const EventSetup& iSetup) {
  num_events_in_run++;

  // short-circuit if hlt problems
  if( ! isValidHltConfig_ ) return;
  
  LogTrace(logTraceName)<<"Analysis of event # ";

  ////////////////////////////////////////////////////////////////////
  // Did event pass HLT paths?
  Handle<TriggerResults> HLTresults;
  iEvent.getByLabel(InputTag(theTriggerResultsCollection, "", theHltMenu), HLTresults); 
  if (!HLTresults.isValid()) return;

  unsigned int triggerIndex; // index of trigger path
  bool passed_HLT;

  // See if event passed trigger paths
  //  increment that bin in the trigger plot
  for (unsigned int i=0; i<thePlotTheseTriggersToo.size(); i++) {
    passed_HLT = false;
    triggerIndex = hltConfigProvider_.triggerIndex(thePlotTheseTriggersToo[i]);
    if (triggerIndex < HLTresults->size()) passed_HLT = HLTresults->accept(triggerIndex);
    if (passed_HLT) h_triggers_passed->Fill(i);
  }

  // Quit if the event did not pass the HLT path we care about
  passed_HLT = false;
  triggerIndex = hltConfigProvider_.triggerIndex(theTriggerPathToPass); // index of trigger path
  if (triggerIndex < HLTresults->size()) passed_HLT = HLTresults->accept(triggerIndex);
  if (!passed_HLT) return;
  ////////////////////////////////////////////////////////////////////


  ////////////////////////////////////////////////////////////////////
  // Does event have valid vertex?
  // Get the primary event vertex
  Handle<VertexCollection> vertexHandle;
  iEvent.getByLabel(theVertexCollectionLabel, vertexHandle);
  VertexCollection vertexCollection = *(vertexHandle.product());
  double vtx_ndof = -1.0;
  double vtx_z    = 0.0;
  bool   vtx_isFake = true;
  if (vertexCollection.size()>0) {
    vtx_ndof = vertexCollection.begin()->ndof();
    vtx_z    = vertexCollection.begin()->z();
    vtx_isFake = false;
  }
  if (vtx_isFake || fabs(vtx_z)>15 || vtx_ndof<4) return;
  ////////////////////////////////////////////////////////////////////


  ////////////////////////////////////////////////////////////////////
  // Did the event pass certain L1 Technical Trigger bits?
  // It's probably beam halo
  //  TODO: ADD code
  ////////////////////////////////////////////////////////////////////

  // grab photons
  Handle<PhotonCollection> photonCollection;
  iEvent.getByLabel(thePhotonCollectionLabel, photonCollection);

  // If photon collection is empty, exit
  if (!photonCollection.isValid()) return;

  // For finding spikes
  Handle<EcalRecHitCollection> EBReducedRecHits;
  iEvent.getByLabel("reducedEcalRecHitsEB", EBReducedRecHits);
  Handle<EcalRecHitCollection> EEReducedRecHits;
  iEvent.getByLabel("reducedEcalRecHitsEE", EEReducedRecHits); 
  EcalClusterLazyTools lazyTool(iEvent, iSetup, InputTag("reducedEcalRecHitsEB"), InputTag("reducedEcalRecHitsEE") );
  // get the channel status from the DB
  ESHandle<EcalChannelStatus> chStatus;
  iSetup.get<EcalChannelStatusRcd>().get(chStatus);

  // Find the highest et "decent" photon
  float photon_et  = -9.0;
  float photon_eta = -9.0;
  float photon_phi = -9.0;
  bool  photon_passPhotonID = false;
  bool  found_lead_pho = false;
  int   photon_count_bar = 0;
  int   photon_count_end = 0;
  // Assumption: reco photons are ordered by Et
  for (PhotonCollection::const_iterator recoPhoton = photonCollection->begin(); recoPhoton!=photonCollection->end(); recoPhoton++){

    // stop looping over photons once we get to too low Et
    if ( recoPhoton->et() < theMinPhotonEt ) break;

    //  Ignore ECAL Spikes
    const reco::CaloClusterPtr  seed = recoPhoton->superCluster()->seed();
    DetId id = lazyTool.getMaximum(*seed).first; // Cluster shape variables
    float time  = -999., outOfTimeChi2 = -999., chi2 = -999.;
    int   flags=-1, severity = -1; 
    const EcalRecHitCollection & rechits = ( recoPhoton->isEB() ? *EBReducedRecHits : *EEReducedRecHits); 
    EcalRecHitCollection::const_iterator it = rechits.find( id );
    if( it != rechits.end() ) {
      time = it->time(); 
      outOfTimeChi2 = it->outOfTimeChi2();
      chi2 = it->chi2();
      flags = it->recoFlag();
      severity = EcalSeverityLevelAlgo::severityLevel( id, rechits, *chStatus );
    }
    bool isNotSpike = ((recoPhoton->isEB() && (severity!=3 && severity!=4 ) && (flags != 2) ) || recoPhoton->isEE());
    if (!isNotSpike) continue;  // move on to next photon
    // END of determining ECAL Spikes

    bool pho_current_passPhotonID = false;
    bool pho_current_isEB = recoPhoton->isEB();
    bool pho_current_isEE = recoPhoton->isEE();

    if ( pho_current_isEB && (recoPhoton->sigmaIetaIeta() < 0.01 || recoPhoton->hadronicOverEm() < 0.05) ) {
      // Photon object in barrel passes photon ID
      pho_current_passPhotonID = true;
      photon_count_bar++;
    } else if ( pho_current_isEE && (recoPhoton->hadronicOverEm() < 0.05) ) {
      // Photon object in endcap passes photon ID
      pho_current_passPhotonID = true;
      photon_count_end++;
    }

    if (!found_lead_pho) {
      found_lead_pho = true;
      photon_passPhotonID = pho_current_passPhotonID;
      photon_et  = recoPhoton->et();
      photon_eta = recoPhoton->eta();
      photon_phi = recoPhoton->phi();
    }
  }
  
  // If user requires a photon to be found, but none is, return.
  //   theRequirePhotonFound should pretty much always be set to 'True'
  //    except when running on qcd monte carlo just to see the jets.
  if ( theRequirePhotonFound && (!photon_passPhotonID || photon_et<theMinPhotonEt) ) return;


  ////////////////////////////////////////////////////////////////////
  // Find the highest et jet
  Handle<CaloJetCollection> caloJetCollection;
  iEvent.getByLabel (theCaloJetCollectionLabel,caloJetCollection);
  if (!caloJetCollection.isValid()) return;

  float jet_pt    = -8.0;
  float jet_eta   = -8.0;
  float jet_phi   = -8.0;
  int   jet_count = 0;
  float jet2_pt   = -9.0;
  float jet2_eta  = -9.0;
  float jet2_phi  = -9.0;
  // Assumption: jets are ordered by Et
  for (CaloJetCollection::const_iterator i_calojet = caloJetCollection->begin(); i_calojet != caloJetCollection->end(); i_calojet++) {

    float jet_current_pt = i_calojet->pt();

    // don't care about jets that overlap with the lead photon
    if ( deltaR(i_calojet->eta(), i_calojet->phi(), photon_eta, photon_phi) < 0.5 ) continue;
    // stop looping over jets once we get to too low Et
    if (jet_current_pt < theMinCaloJetPt) break;

    jet_count++;
    if (jet_current_pt > jet_pt) {
      jet2_pt  = jet_pt;  // 2nd highest jet get's et from current highest
      jet2_eta = jet_eta;
      jet2_phi = jet_phi;
      jet_pt   = jet_current_pt; // current highest jet gets et from the new highest
      jet_eta  = i_calojet->eta();
      jet_phi  = i_calojet->phi();
    } else if (jet_current_pt > jet2_pt) {
      jet2_pt  = jet_current_pt;
      jet2_eta = i_calojet->eta();
      jet2_phi = i_calojet->phi();
    }
  }
  ////////////////////////////////////////////////////////////////////


  ////////////////////////////////////////////////////////////////////
  // Fill histograms if a jet found
  // NOTE: if a photon was required to be found, but wasn't
  //        we wouldn't have made it to this point in the code
  if ( jet_pt > 0.0 ) {

    // Photon Plots
    h_photon_et       ->Fill( photon_et  );
    h_photon_eta      ->Fill( photon_eta );
    h_photon_count_bar->Fill( photon_count_bar );
    h_photon_count_end->Fill( photon_count_end );

    // Photon Et hists for different orientations to the jet
    if ( fabs(photon_eta)<1.45 && photon_passPhotonID ) {  // Lead photon is in barrel
      if (fabs(jet_eta)<1.45){                          //   jet is in barrel
	if (photon_eta*jet_eta>0) {
	  h_photon_et_jetcs->Fill(photon_et);
	} else {
	  h_photon_et_jetco->Fill(photon_et);
	}
      } else if (jet_eta>1.55 && jet_eta<2.5) {         // jet is in endcap
	if (photon_eta*jet_eta>0) {
	  h_photon_et_jetfs->Fill(photon_et);
	} else {
	  h_photon_et_jetfo->Fill(photon_et);
	}
      }
    } // END of Lead Photon is in Barrel

    // Jet Plots
    h_jet_pt       ->Fill( jet_pt     );
    h_jet_eta      ->Fill( jet_eta    );
    h_jet_count    ->Fill( jet_count  );
    h_deltaPhi_photon_jet   ->Fill( abs(deltaPhi(photon_phi, jet_phi)) );
    if ( abs(deltaPhi(photon_phi,jet_phi))>2.8 ) h_deltaEt_photon_jet->Fill( (photon_et-jet_pt)/photon_et );

    // 2nd Highest Jet Plots
    if ( jet2_pt  > 0.0 ) {
      h_jet2_pt             ->Fill( jet2_pt  );
      h_jet2_eta            ->Fill( jet2_eta );
      h_jet2_ptOverPhotonEt ->Fill( jet2_pt/photon_et );
      h_deltaPhi_photon_jet2->Fill( abs(deltaPhi(photon_phi, jet2_phi)) );
      h_deltaPhi_jet_jet2   ->Fill( abs(deltaPhi(   jet_phi, jet2_phi)) );
      h_deltaR_jet_jet2     ->Fill( deltaR(   jet_eta,    jet_phi, jet2_eta, jet2_phi) );
      h_deltaR_photon_jet2  ->Fill( deltaR(photon_eta, photon_phi, jet2_eta, jet2_phi) );
    }
  } 
  // End of Filling histograms
  ////////////////////////////////////////////////////////////////////
}


void QcdPhotonsDQM::endJob(void) {}

void QcdPhotonsDQM::endRun(const edm::Run& run, const edm::EventSetup& es) {
  if (num_events_in_run>0) { 
    h_triggers_passed->getTH1F()->Scale(1.0/num_events_in_run);
  }
  h_photon_et_ratio_co_cs->getTH1F()->Divide( h_photon_et_jetco->getTH1F(), h_photon_et_jetcs->getTH1F() );
  h_photon_et_ratio_fo_fs->getTH1F()->Divide( h_photon_et_jetfo->getTH1F(), h_photon_et_jetfs->getTH1F() );
  h_photon_et_ratio_cs_fs->getTH1F()->Divide( h_photon_et_jetcs->getTH1F(), h_photon_et_jetfs->getTH1F() );
  h_photon_et_ratio_co_fs->getTH1F()->Divide( h_photon_et_jetco->getTH1F(), h_photon_et_jetfs->getTH1F() );
  h_photon_et_ratio_cs_fo->getTH1F()->Divide( h_photon_et_jetcs->getTH1F(), h_photon_et_jetfo->getTH1F() );
  h_photon_et_ratio_co_fo->getTH1F()->Divide( h_photon_et_jetco->getTH1F(), h_photon_et_jetfo->getTH1F() );
}

