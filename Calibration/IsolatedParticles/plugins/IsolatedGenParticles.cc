// -*- C++ -*-
//
// Package:    IsolatedGenParticles
// Class:      IsolatedGenParticles
// 
/**\class IsolatedGenParticles IsolatedGenParticles.cc Calibration/IsolatedParticles/plugins/IsolatedGenParticles.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Seema Sharma
//         Created:  Tue Oct 27 09:46:41 CDT 2009
//
//

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include <iostream>
#include <iomanip>
#include <list>
#include <vector>
#include <cmath>

#include "Calibration/IsolatedParticles/interface/CaloPropagateTrack.h"
#include "Calibration/IsolatedParticles/interface/ChargeIsolation.h"

#include "Calibration/IsolatedParticles/plugins/IsolatedGenParticles.h"

const int IsolatedGenParticles::PBins;
const int IsolatedGenParticles::EtaBins;

IsolatedGenParticles::IsolatedGenParticles(const edm::ParameterSet& iConfig) {

  genSrc_    = iConfig.getUntrackedParameter("GenSrc",std::string("generator"));

  tok_hepmc_        = consumes<edm::HepMCProduct>(edm::InputTag(genSrc_));
  tok_genParticles_ = consumes<reco::GenParticleCollection>(edm::InputTag(genSrc_));

  useHepMC   = iConfig.getUntrackedParameter<bool>("UseHepMC", false );
  pSeed      = iConfig.getUntrackedParameter<double>("ChargedHadronSeedP", 1.0);
  ptMin      = iConfig.getUntrackedParameter<double>("PTMin", 1.0);
  etaMax     = iConfig.getUntrackedParameter<double>("MaxChargedHadronEta", 2.5);
  a_coneR    = iConfig.getUntrackedParameter<double>("ConeRadius",34.98);
  a_charIsoR = a_coneR + 28.9;
  a_neutIsoR = a_charIsoR*0.726;
  a_mipR     = iConfig.getUntrackedParameter<double>("ConeRadiusMIP",14.0);
  a_Isolation= iConfig.getUntrackedParameter<bool>("UseConeIsolation",false);
  pCutIsolate= iConfig.getUntrackedParameter<double>("PMaxIsolation",20.0);
  verbosity  = iConfig.getUntrackedParameter<int>("Verbosity", 0);

  debugL1Info_           = iConfig.getUntrackedParameter<bool>( "DebugL1Info", false );

  edm::InputTag L1extraTauJetSource_   = iConfig.getParameter<edm::InputTag>("L1extraTauJetSource");
  edm::InputTag L1extraCenJetSource_   = iConfig.getParameter<edm::InputTag>("L1extraCenJetSource");
  edm::InputTag L1extraFwdJetSource_   = iConfig.getParameter<edm::InputTag>("L1extraFwdJetSource");
  edm::InputTag L1extraMuonSource_     = iConfig.getParameter<edm::InputTag>("L1extraMuonSource");
  edm::InputTag L1extraIsoEmSource_    = iConfig.getParameter<edm::InputTag>("L1extraIsoEmSource");
  edm::InputTag L1extraNonIsoEmSource_ = iConfig.getParameter<edm::InputTag>("L1extraNonIsoEmSource");
  edm::InputTag L1GTReadoutRcdSource_ = iConfig.getParameter<edm::InputTag>("L1GTReadoutRcdSource");
  edm::InputTag L1GTObjectMapRcdSource_= iConfig.getParameter<edm::InputTag>("L1GTObjectMapRcdSource");
  tok_L1GTrorsrc_   =  consumes<L1GlobalTriggerReadoutRecord>(L1GTReadoutRcdSource_);
  tok_L1GTobjmap_   =   consumes<L1GlobalTriggerObjectMapRecord>(L1GTObjectMapRcdSource_);
  tok_L1extMusrc_   =  consumes<l1extra::L1MuonParticleCollection>(L1extraMuonSource_);
  tok_L1Em_         =  consumes<l1extra::L1EmParticleCollection>(L1extraIsoEmSource_);
  tok_L1extNonIsoEm_= consumes<l1extra::L1EmParticleCollection>(L1extraNonIsoEmSource_);
  tok_L1extTauJet_  = consumes<l1extra::L1JetParticleCollection>(L1extraTauJetSource_);
  tok_L1extCenJet_  = consumes<l1extra::L1JetParticleCollection>(L1extraCenJetSource_);
  tok_L1extFwdJet_  = consumes<l1extra::L1JetParticleCollection>(L1extraFwdJetSource_);

  if (!strcmp("Dummy", genSrc_.c_str())) {
    if (useHepMC) genSrc_ = "generator";
    else          genSrc_ = "genParticles";
  }
  std::cout << "Generator Source " << genSrc_ << " Use HepMC " << useHepMC
	    << " pSeed " << pSeed << " ptMin " << ptMin << " etaMax " << etaMax
	    << "\n a_coneR " << a_coneR << " a_charIsoR " << a_charIsoR
	    << " a_neutIsoR " << a_neutIsoR << " a_mipR " << a_mipR 
	    << " debug " << verbosity << " debugL1Info " <<   debugL1Info_ << "\n"
	    << " Isolation Flag " << a_Isolation << " with cut "
	    << pCutIsolate << " GeV"
	    << std::endl;
}

IsolatedGenParticles::~IsolatedGenParticles() {

}

void IsolatedGenParticles::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){

  clearTreeVectors();

  nEventProc++;

  edm::ESHandle<MagneticField> bFieldH;
  iSetup.get<IdealMagneticFieldRecord>().get(bFieldH);
  bField = bFieldH.product();

  // get particle data table
  edm::ESHandle<ParticleDataTable> pdt;
  iSetup.getData(pdt);

  // get handle to HEPMCProduct
  edm::Handle<edm::HepMCProduct> hepmc;
  edm::Handle<reco::GenParticleCollection> genParticles;
  if (useHepMC) iEvent.getByToken(tok_hepmc_, hepmc);
  else          iEvent.getByToken(tok_genParticles_, genParticles);

  edm::ESHandle<CaloGeometry> pG;
  iSetup.get<CaloGeometryRecord>().get(pG);
  const CaloGeometry* geo = pG.product();

  edm::ESHandle<CaloTopology> theCaloTopology;
  iSetup.get<CaloTopologyRecord>().get(theCaloTopology);
  const CaloTopology *caloTopology = theCaloTopology.product();
  
  edm::ESHandle<HcalTopology> htopo;
  iSetup.get<IdealGeometryRecord>().get(htopo);
  const HcalTopology* theHBHETopology = htopo.product();

  //===================== save L1 Trigger information =======================
  // get L1TriggerReadout records
  edm::Handle<L1GlobalTriggerReadoutRecord>   gtRecord;
  iEvent.getByToken(tok_L1GTrorsrc_,   gtRecord);
  
  edm::Handle<L1GlobalTriggerObjectMapRecord> gtOMRec;
  iEvent.getByToken(tok_L1GTobjmap_, gtOMRec);
  
  // sanity check on L1 Trigger Records
  if (!gtRecord.isValid()) {
    std::cout << "\nL1GlobalTriggerReadoutRecord with \n \nnot found"
      "\n  --> returning false by default!\n" << std::endl;
  }
  if (!gtOMRec.isValid()) {
    std::cout << "\nL1GlobalTriggerObjectMapRecord with \n \nnot found"
      "\n  --> returning false by default!\n" << std::endl;
  }

  // L1 decision word
  const DecisionWord dWord = gtRecord->decisionWord();  
  unsigned int numberTriggerBits= dWord.size();

  // just print the L1Bit number and AlgoName in first event
  if ( !initL1){
    initL1=true;
    std::cout << "\n  Number of Trigger bits " << numberTriggerBits << "\n\n";
    std::cout << "\tBit \t L1 Algorithm " << std::endl;

    // get ObjectMaps from ObjectMapRecord
    const std::vector<L1GlobalTriggerObjectMap>& objMapVec =  gtOMRec->gtObjectMap();
    for (std::vector<L1GlobalTriggerObjectMap>::const_iterator itMap = objMapVec.begin();
	 itMap != objMapVec.end(); ++itMap) {

      // Get trigger bits
      int itrig = (*itMap).algoBitNumber();

      // Get trigger names
      algoBitToName[itrig] = (*itMap).algoName();

      std::cout  << "\t" << itrig << "\t" << algoBitToName[itrig] << std::endl;      

      // store the algoNames as bin labels of a histogram
      h_L1AlgoNames->GetXaxis()->SetBinLabel(itrig+1, algoBitToName[itrig].c_str());

    } // end of for loop    
  } // end of initL1

  // save L1 decision for each event 
  for (unsigned int iBit = 0; iBit < numberTriggerBits; ++iBit) {  
    bool accept = dWord[iBit];
    t_L1Decision->push_back(accept);
    // fill the trigger map
    if(debugL1Info_) std::cout << "Bit " << iBit << " " << algoBitToName[iBit] << " " << accept << std::endl;

    if(accept) h_L1AlgoNames->Fill(iBit);
  }

  //===================
  // L1Taus 
  edm::Handle<l1extra::L1JetParticleCollection> l1TauHandle;
  iEvent.getByToken(tok_L1extTauJet_,l1TauHandle);
  l1extra::L1JetParticleCollection::const_iterator itr;
  for(itr = l1TauHandle->begin(); itr != l1TauHandle->end(); ++itr ) {
    t_L1TauJetPt      ->push_back( itr->pt() );
    t_L1TauJetEta     ->push_back( itr->eta() );
    t_L1TauJetPhi     ->push_back( itr->phi() );
    if(debugL1Info_) {
      std::cout << "tauJ p/pt  " << itr->momentum() << " " << itr->pt() 
		<< "  eta/phi " << itr->eta() << " " << itr->phi()
		<< std::endl;
    }
  }

  // L1 Central Jets
  edm::Handle<l1extra::L1JetParticleCollection> l1CenJetHandle;
  iEvent.getByToken(tok_L1extCenJet_,l1CenJetHandle);
  for( itr = l1CenJetHandle->begin();  itr != l1CenJetHandle->end(); ++itr ) {
    t_L1CenJetPt    ->push_back( itr->pt() );
    t_L1CenJetEta   ->push_back( itr->eta() );
    t_L1CenJetPhi   ->push_back( itr->phi() );
    if(debugL1Info_) {
      std::cout << "cenJ p/pt     " << itr->momentum() << " " << itr->pt() 
		<< "  eta/phi " << itr->eta() << " " << itr->phi()
		<< std::endl;
    }
  }
  // L1 Forward Jets
  edm::Handle<l1extra::L1JetParticleCollection> l1FwdJetHandle;
  iEvent.getByToken(tok_L1extFwdJet_,l1FwdJetHandle);
  for( itr = l1FwdJetHandle->begin();  itr != l1FwdJetHandle->end(); ++itr ) {
    t_L1FwdJetPt    ->push_back( itr->pt() );
    t_L1FwdJetEta   ->push_back( itr->eta() );
    t_L1FwdJetPhi   ->push_back( itr->phi() );
    if(debugL1Info_) {
      std::cout << "fwdJ p/pt     " << itr->momentum() << " " << itr->pt() 
		<< "  eta/phi " << itr->eta() << " " << itr->phi()
		<< std::endl;
    }
  }
  // L1 Isolated EM onjects
  l1extra::L1EmParticleCollection::const_iterator itrEm;
  edm::Handle<l1extra::L1EmParticleCollection> l1IsoEmHandle ;
  iEvent.getByToken(tok_L1Em_, l1IsoEmHandle);
  for( itrEm = l1IsoEmHandle->begin();  itrEm != l1IsoEmHandle->end(); ++itrEm ) {
    t_L1IsoEMPt     ->push_back(  itrEm->pt() );
    t_L1IsoEMEta    ->push_back(  itrEm->eta() );
    t_L1IsoEMPhi    ->push_back(  itrEm->phi() );
    if(debugL1Info_) {
      std::cout << "isoEm p/pt    " << itrEm->momentum() << " " << itrEm->pt() 
		<< "  eta/phi " << itrEm->eta() << " " << itrEm->phi()
		<< std::endl;
    }
  }
  // L1 Non-Isolated EM onjects
  edm::Handle<l1extra::L1EmParticleCollection> l1NonIsoEmHandle ;
  iEvent.getByToken(tok_L1extNonIsoEm_, l1NonIsoEmHandle);
  for( itrEm = l1NonIsoEmHandle->begin();  itrEm != l1NonIsoEmHandle->end(); ++itrEm ) {
    t_L1NonIsoEMPt  ->push_back( itrEm->pt() );
    t_L1NonIsoEMEta ->push_back( itrEm->eta() );
    t_L1NonIsoEMPhi ->push_back( itrEm->phi() );
    if(debugL1Info_) {
      std::cout << "nonIsoEm p/pt " << itrEm->momentum() << " " << itrEm->pt() 
		<< "  eta/phi " << itrEm->eta() << " " << itrEm->phi()
		<< std::endl;
    }
  }
  
  // L1 Muons
  l1extra::L1MuonParticleCollection::const_iterator itrMu;
  edm::Handle<l1extra::L1MuonParticleCollection> l1MuHandle ;
  iEvent.getByToken(tok_L1extMusrc_, l1MuHandle);
  for( itrMu = l1MuHandle->begin();  itrMu != l1MuHandle->end(); ++itrMu ) {
    t_L1MuonPt      ->push_back( itrMu->pt() );
    t_L1MuonEta     ->push_back( itrMu->eta() );
    t_L1MuonPhi     ->push_back( itrMu->phi() );
    if(debugL1Info_) {
      std::cout << "l1muon p/pt   " << itrMu->momentum() << " " << itrMu->pt() 
		<< "  eta/phi " << itrMu->eta() << " " << itrMu->phi()
		<< std::endl;
    }
  }
  //=====================================================================
  
  GlobalPoint  posVec, posECAL;
  math::XYZTLorentzVector momVec;
  if (verbosity>0) std::cout << "event number " << iEvent.id().event() <<std::endl;
  if (useHepMC) {
    const HepMC::GenEvent *myGenEvent = hepmc->GetEvent();
    std::vector<spr::propagatedGenTrackID> trackIDs = spr::propagateCALO(myGenEvent, pdt, geo, bField, etaMax, false);
    
    for (unsigned int indx=0; indx<trackIDs.size(); ++indx) {
      int charge = trackIDs[indx].charge;
      HepMC::GenEvent::particle_const_iterator p = trackIDs[indx].trkItr;
      momVec = math::XYZTLorentzVector((*p)->momentum().px(), (*p)->momentum().py(), (*p)->momentum().pz(), (*p)->momentum().e());
      if (verbosity>1) std::cout << "trkIndx " << indx << " pdgid " << trackIDs[indx].pdgId << " charge " << charge <<	" momVec " << momVec << std::endl; 
      // only stable particles avoiding electrons and muons
      if (trackIDs[indx].ok && (std::abs(trackIDs[indx].pdgId)<11 ||
				std::abs(trackIDs[indx].pdgId)>=21)) {
	// consider particles within a phased space	  
	if (momVec.Pt() > ptMin && std::abs(momVec.eta()) < etaMax) { 
	  posVec  = GlobalPoint(0.1*(*p)->production_vertex()->position().x(), 
				0.1*(*p)->production_vertex()->position().y(), 
				0.1*(*p)->production_vertex()->position().z());
	  posECAL = trackIDs[indx].pointECAL;
	  fillTrack (posVec, momVec, posECAL, trackIDs[indx].pdgId, trackIDs[indx].okECAL, true);
	  if (verbosity>1) std::cout << "posECAL " << posECAL << " okECAL " << trackIDs[indx].okECAL << "okHCAL " << trackIDs[indx].okHCAL << std::endl;
	  if (trackIDs[indx].okECAL) {
	    if ( std::abs(charge)>0 ) {
	      spr::eGenSimInfo(trackIDs[indx].detIdECAL, p, trackIDs, geo, caloTopology, 0, 0, isoinfo1x1,   false);
	      spr::eGenSimInfo(trackIDs[indx].detIdECAL, p, trackIDs, geo, caloTopology, 1, 1, isoinfo3x3,   false);
	      spr::eGenSimInfo(trackIDs[indx].detIdECAL, p, trackIDs, geo, caloTopology, 3, 3, isoinfo7x7,   false);
	      spr::eGenSimInfo(trackIDs[indx].detIdECAL, p, trackIDs, geo, caloTopology, 4, 4, isoinfo9x9,   false);
	      spr::eGenSimInfo(trackIDs[indx].detIdECAL, p, trackIDs, geo, caloTopology, 5, 5, isoinfo11x11, false);
	      spr::eGenSimInfo(trackIDs[indx].detIdECAL, p, trackIDs, geo, caloTopology, 7, 7, isoinfo15x15, false);
	      spr::eGenSimInfo(trackIDs[indx].detIdECAL, p, trackIDs, geo, caloTopology,10,10, isoinfo21x21, false);
	      spr::eGenSimInfo(trackIDs[indx].detIdECAL, p, trackIDs, geo, caloTopology,12,12, isoinfo25x25, false);
	      spr::eGenSimInfo(trackIDs[indx].detIdECAL, p, trackIDs, geo, caloTopology,15,15, isoinfo31x31, false);
	      spr::eGenSimInfo(trackIDs[indx].detIdECAL, p, trackIDs, geo, caloTopology, a_mipR, trackIDs[indx].directionECAL, isoinfoR, false);
	      spr::eGenSimInfo(trackIDs[indx].detIdECAL, p, trackIDs, geo, caloTopology, a_neutIsoR, trackIDs[indx].directionECAL, isoinfoIsoR, false);
	      if (trackIDs[indx].okHCAL) {
		spr::hGenSimInfo(trackIDs[indx].detIdHCAL, p, trackIDs, theHBHETopology, 0, 0, isoinfoHC1x1, false);
		spr::hGenSimInfo(trackIDs[indx].detIdHCAL, p, trackIDs, theHBHETopology, 1, 1, isoinfoHC3x3, false);
		spr::hGenSimInfo(trackIDs[indx].detIdHCAL, p, trackIDs, theHBHETopology, 2, 2, isoinfoHC5x5, false);
		spr::hGenSimInfo(trackIDs[indx].detIdHCAL, p, trackIDs, theHBHETopology, 3, 3, isoinfoHC7x7, false);
		spr::hGenSimInfo(trackIDs[indx].detIdHCAL, p, trackIDs, geo, theHBHETopology, a_coneR, trackIDs[indx].directionHCAL, isoinfoHCR, false);
		spr::hGenSimInfo(trackIDs[indx].detIdHCAL, p, trackIDs, geo, theHBHETopology, a_charIsoR, trackIDs[indx].directionHCAL, isoinfoIsoHCR, false);
	      }

	      bool saveTrack = true;
	      if (a_Isolation) saveTrack = (isoinfoR.maxNearP < pCutIsolate);
	      else             saveTrack = (isoinfo7x7.maxNearP < pCutIsolate);
	      if (saveTrack) fillIsolatedTrack(momVec, posECAL, trackIDs[indx].pdgId);
	    }
	  }
	} else { // stabale particles within |eta|=2.5
	  fillTrack (posVec, momVec, posECAL, 0, false, false);
	} 
      }
    }

    unsigned int indx;
    HepMC::GenEvent::particle_const_iterator p;
    for (p=myGenEvent->particles_begin(),indx=0; p!=myGenEvent->particles_end();
	 ++p,++indx) {
      int pdgId  = ((*p)->pdg_id());
      int ix     = particleCode(pdgId);
      if (ix >= 0) {
	double  pp = (*p)->momentum().rho();
	double eta = (*p)->momentum().eta();
	h_pEta[ix]->Fill(pp,eta);
      }
    }
  } else {  // loop over gen particles
    std::vector<spr::propagatedGenParticleID> trackIDs = spr::propagateCALO(genParticles, pdt, geo, bField, etaMax, (verbosity>0));

    for (unsigned int indx=0; indx<trackIDs.size(); ++indx) {
      int charge = trackIDs[indx].charge;
      reco::GenParticleCollection::const_iterator p = trackIDs[indx].trkItr;
      
      momVec = math::XYZTLorentzVector(p->momentum().x(), p->momentum().y(), p->momentum().z(), p->energy());
      if (verbosity>1) std::cout << "trkIndx " << indx << " pdgid " << trackIDs[indx].pdgId << " charge " << charge <<	" momVec " << momVec << std::endl; 
      // only stable particles avoiding electrons and muons
      if (trackIDs[indx].ok && std::abs(trackIDs[indx].pdgId)>21) {	
	// consider particles within a phased space
	if (verbosity>1) std::cout << " pt " << momVec.Pt() << " eta " << momVec.eta() << std::endl;
	if (momVec.Pt() > ptMin && std::abs(momVec.eta()) < etaMax) { 
	  posVec  = GlobalPoint(p->vertex().x(), p->vertex().y(), p->vertex().z());
	  posECAL = trackIDs[indx].pointECAL;
	  if (verbosity>0) std::cout << "posECAL " << posECAL << " okECAL " << trackIDs[indx].okECAL << "okHCAL " << trackIDs[indx].okHCAL << std::endl;
	  fillTrack (posVec, momVec, posECAL, trackIDs[indx].pdgId, trackIDs[indx].okECAL, true);
	  if (trackIDs[indx].okECAL) {
	    if ( std::abs(charge)>0 ) {
	      spr::eGenSimInfo(trackIDs[indx].detIdECAL, p, trackIDs, geo, caloTopology, 0, 0, isoinfo1x1,   verbosity>1);
	      spr::eGenSimInfo(trackIDs[indx].detIdECAL, p, trackIDs, geo, caloTopology, 1, 1, isoinfo3x3,   verbosity>0);
	      spr::eGenSimInfo(trackIDs[indx].detIdECAL, p, trackIDs, geo, caloTopology, 3, 3, isoinfo7x7,   verbosity>1);
	      spr::eGenSimInfo(trackIDs[indx].detIdECAL, p, trackIDs, geo, caloTopology, 4, 4, isoinfo9x9,   verbosity>1);
	      spr::eGenSimInfo(trackIDs[indx].detIdECAL, p, trackIDs, geo, caloTopology, 5, 5, isoinfo11x11, verbosity>1);
	      spr::eGenSimInfo(trackIDs[indx].detIdECAL, p, trackIDs, geo, caloTopology, 7, 7, isoinfo15x15, verbosity>1);
	      spr::eGenSimInfo(trackIDs[indx].detIdECAL, p, trackIDs, geo, caloTopology,10,10, isoinfo21x21, verbosity>1);
	      spr::eGenSimInfo(trackIDs[indx].detIdECAL, p, trackIDs, geo, caloTopology,12,12, isoinfo25x25, verbosity>1);
	      spr::eGenSimInfo(trackIDs[indx].detIdECAL, p, trackIDs, geo, caloTopology,15,15, isoinfo31x31, verbosity>1);
	      spr::eGenSimInfo(trackIDs[indx].detIdECAL, p, trackIDs, geo, caloTopology, a_mipR, trackIDs[indx].directionECAL, isoinfoR, verbosity>1);
	      spr::eGenSimInfo(trackIDs[indx].detIdECAL, p, trackIDs, geo, caloTopology, a_neutIsoR, trackIDs[indx].directionECAL, isoinfoIsoR, verbosity>1);
	      if (trackIDs[indx].okHCAL) {
		spr::hGenSimInfo(trackIDs[indx].detIdHCAL, p, trackIDs, theHBHETopology, 0, 0, isoinfoHC1x1, verbosity>1);
		spr::hGenSimInfo(trackIDs[indx].detIdHCAL, p, trackIDs, theHBHETopology, 1, 1, isoinfoHC3x3, verbosity>1);
		spr::hGenSimInfo(trackIDs[indx].detIdHCAL, p, trackIDs, theHBHETopology, 2, 2, isoinfoHC5x5, verbosity>1);
		spr::hGenSimInfo(trackIDs[indx].detIdHCAL, p, trackIDs, theHBHETopology, 3, 3, isoinfoHC7x7, verbosity>1);
		spr::hGenSimInfo(trackIDs[indx].detIdHCAL, p, trackIDs, geo, theHBHETopology, a_coneR, trackIDs[indx].directionHCAL, isoinfoHCR, verbosity>1);
		spr::hGenSimInfo(trackIDs[indx].detIdHCAL, p, trackIDs, geo, theHBHETopology, a_charIsoR, trackIDs[indx].directionHCAL, isoinfoIsoHCR, verbosity>1);
	      }

	      bool saveTrack = true;
	      if (a_Isolation) saveTrack = (isoinfoIsoR.maxNearP < pCutIsolate);
	      else             saveTrack = (isoinfo7x7.maxNearP < pCutIsolate);
	      if (saveTrack) fillIsolatedTrack(momVec, posECAL, trackIDs[indx].pdgId);
	    }
	  }
	} else { // stabale particles within |eta|=2.5
	  fillTrack (posVec, momVec, posECAL, 0, false, false);
	} 
      }
    } // loop over gen particles

    unsigned int indx;
    reco::GenParticleCollection::const_iterator p;
    for (p=genParticles->begin(),indx=0; p!=genParticles->end(); ++p,++indx) {
      int pdgId  = (p->pdgId());
      int ix     = particleCode(pdgId);
      if (ix >= 0) {
	double  pp = (p->momentum()).R();
	double eta = (p->momentum()).Eta();
	h_pEta[ix]->Fill(pp,eta);
      }
    }

  } 

  //t_nEvtProc->push_back(nEventProc);
  h_NEventProc->SetBinContent(1,nEventProc);
  tree->Fill();
  
} 

void IsolatedGenParticles::beginJob() {
  
  nEventProc=0;
  
  initL1 = false;

  double tempgen_TH[NPBins+1] = { 0.0,  5.0,  12.0, 300.0}; 
  for(int i=0; i<=NPBins; i++)  genPartPBins[i]  = tempgen_TH[i];
  
  double tempgen_Eta[NEtaBins+1] = {0.0, 0.5, 1.1, 1.7, 2.3};
  for(int i=0; i<=NEtaBins; i++) genPartEtaBins[i] = tempgen_Eta[i];
  
  BookHistograms();

}

void IsolatedGenParticles::endJob() {
}

double IsolatedGenParticles::DeltaPhi(double v1, double v2) {
  // Computes the correctly normalized phi difference
  // v1, v2 = phi of object 1 and 2
  
  double pi    = 3.141592654;
  double twopi = 6.283185307;
  
  double diff = std::abs(v2 - v1);
  double corr = twopi - diff;
  if (diff < pi){ return diff;} else { return corr;} 
}

double IsolatedGenParticles::DeltaR(double eta1, double phi1, double eta2, double phi2) {
  double deta = eta1 - eta2;
  double dphi = DeltaPhi(phi1, phi2);
  return std::sqrt(deta*deta + dphi*dphi);
}

double IsolatedGenParticles::DeltaR2(double eta1, double phi1, double eta2, double phi2) {
  double deta = eta1 - eta2;
  double dphi = DeltaPhi(phi1, phi2);
  return deta*deta + dphi*dphi;
}

void IsolatedGenParticles::fillTrack (GlobalPoint & posVec, math::XYZTLorentzVector & momVec, GlobalPoint & posECAL, int pdgId, bool okECAL, bool accept) {

  if (accept) {
    t_isoTrkPAll        ->push_back( momVec.P() );
    t_isoTrkPtAll       ->push_back( momVec.Pt() );
    t_isoTrkPhiAll      ->push_back( momVec.phi() );
    t_isoTrkEtaAll      ->push_back( momVec.eta() );
    t_isoTrkPdgIdAll    ->push_back( pdgId ) ;
    if (okECAL) {
      double phi1 = momVec.phi();
      double phi2 = (posECAL - posVec).phi();
      double dphi = DeltaPhi( phi1, phi2 );
      double deta = momVec.eta() - (posECAL - posVec).eta();
      t_isoTrkDPhiAll ->push_back( dphi );
      t_isoTrkDEtaAll ->push_back( deta );	
    } else {
      t_isoTrkDPhiAll ->push_back( 999.0 );
      t_isoTrkDEtaAll ->push_back( 999.0 );	
    }
  } else {
    t_isoTrkDPhiAll ->push_back( -999.0 );
    t_isoTrkDEtaAll ->push_back( -999.0 );	
  }
}

void IsolatedGenParticles::fillIsolatedTrack(math::XYZTLorentzVector & momVec, GlobalPoint & posECAL, int pdgId) {

  t_isoTrkP           ->push_back(momVec.P());
  t_isoTrkPt          ->push_back(momVec.Pt());
  t_isoTrkEne         ->push_back(momVec.E());
  t_isoTrkEta         ->push_back(momVec.eta());
  t_isoTrkPhi         ->push_back(momVec.phi());
  t_isoTrkEtaEC       ->push_back(posECAL.eta());
  t_isoTrkPhiEC       ->push_back(posECAL.phi());
  t_isoTrkPdgId       ->push_back(pdgId);
  
  t_maxNearP31x31     ->push_back(isoinfo31x31.maxNearP);
  t_cHadronEne31x31   ->push_back(isoinfo31x31.cHadronEne);
  t_cHadronEne31x31_1 ->push_back(isoinfo31x31.cHadronEne_[0]);
  t_cHadronEne31x31_2 ->push_back(isoinfo31x31.cHadronEne_[1]);
  t_cHadronEne31x31_3 ->push_back(isoinfo31x31.cHadronEne_[2]);
  t_nHadronEne31x31   ->push_back(isoinfo31x31.nHadronEne);
  t_photonEne31x31    ->push_back(isoinfo31x31.photonEne);
  t_eleEne31x31       ->push_back(isoinfo31x31.eleEne);
  t_muEne31x31        ->push_back(isoinfo31x31.muEne);
	  
  t_maxNearP25x25     ->push_back(isoinfo25x25.maxNearP);
  t_cHadronEne25x25   ->push_back(isoinfo25x25.cHadronEne);
  t_cHadronEne25x25_1 ->push_back(isoinfo25x25.cHadronEne_[0]);
  t_cHadronEne25x25_2 ->push_back(isoinfo25x25.cHadronEne_[1]);
  t_cHadronEne25x25_3 ->push_back(isoinfo25x25.cHadronEne_[2]);
  t_nHadronEne25x25   ->push_back(isoinfo25x25.nHadronEne);
  t_photonEne25x25    ->push_back(isoinfo25x25.photonEne);
  t_eleEne25x25       ->push_back(isoinfo25x25.eleEne);
  t_muEne25x25        ->push_back(isoinfo25x25.muEne);
	  
  t_maxNearP21x21     ->push_back(isoinfo21x21.maxNearP);
  t_cHadronEne21x21   ->push_back(isoinfo21x21.cHadronEne);
  t_cHadronEne21x21_1 ->push_back(isoinfo21x21.cHadronEne_[0]);
  t_cHadronEne21x21_2 ->push_back(isoinfo21x21.cHadronEne_[1]);
  t_cHadronEne21x21_3 ->push_back(isoinfo21x21.cHadronEne_[2]);
  t_nHadronEne21x21   ->push_back(isoinfo21x21.nHadronEne);
  t_photonEne21x21    ->push_back(isoinfo21x21.photonEne);
  t_eleEne21x21       ->push_back(isoinfo21x21.eleEne);
  t_muEne21x21        ->push_back(isoinfo21x21.muEne);
	  
  t_maxNearP15x15     ->push_back(isoinfo15x15.maxNearP);
  t_cHadronEne15x15   ->push_back(isoinfo15x15.cHadronEne);
  t_cHadronEne15x15_1 ->push_back(isoinfo15x15.cHadronEne_[0]);
  t_cHadronEne15x15_2 ->push_back(isoinfo15x15.cHadronEne_[1]);
  t_cHadronEne15x15_3 ->push_back(isoinfo15x15.cHadronEne_[2]);
  t_nHadronEne15x15   ->push_back(isoinfo15x15.nHadronEne);
  t_photonEne15x15    ->push_back(isoinfo15x15.photonEne);
  t_eleEne15x15       ->push_back(isoinfo15x15.eleEne);
  t_muEne15x15        ->push_back(isoinfo15x15.muEne);
	  
  t_maxNearP11x11     ->push_back(isoinfo11x11.maxNearP);
  t_cHadronEne11x11   ->push_back(isoinfo11x11.cHadronEne);
  t_cHadronEne11x11_1 ->push_back(isoinfo11x11.cHadronEne_[0]);
  t_cHadronEne11x11_2 ->push_back(isoinfo11x11.cHadronEne_[1]);
  t_cHadronEne11x11_3 ->push_back(isoinfo11x11.cHadronEne_[2]);
  t_nHadronEne11x11   ->push_back(isoinfo11x11.nHadronEne);
  t_photonEne11x11    ->push_back(isoinfo11x11.photonEne);
  t_eleEne11x11       ->push_back(isoinfo11x11.eleEne);
  t_muEne11x11        ->push_back(isoinfo11x11.muEne);
	  
  t_maxNearP9x9       ->push_back(isoinfo9x9.maxNearP);
  t_cHadronEne9x9     ->push_back(isoinfo9x9.cHadronEne);
  t_cHadronEne9x9_1   ->push_back(isoinfo9x9.cHadronEne_[0]);
  t_cHadronEne9x9_2   ->push_back(isoinfo9x9.cHadronEne_[1]);
  t_cHadronEne9x9_3   ->push_back(isoinfo9x9.cHadronEne_[2]);
  t_nHadronEne9x9     ->push_back(isoinfo9x9.nHadronEne);
  t_photonEne9x9      ->push_back(isoinfo9x9.photonEne);
  t_eleEne9x9         ->push_back(isoinfo9x9.eleEne);
  t_muEne9x9          ->push_back(isoinfo9x9.muEne);
	    
  t_maxNearP7x7       ->push_back(isoinfo7x7.maxNearP);
  t_cHadronEne7x7     ->push_back(isoinfo7x7.cHadronEne);
  t_cHadronEne7x7_1   ->push_back(isoinfo7x7.cHadronEne_[0]);
  t_cHadronEne7x7_2   ->push_back(isoinfo7x7.cHadronEne_[1]);
  t_cHadronEne7x7_3   ->push_back(isoinfo7x7.cHadronEne_[2]);
  t_nHadronEne7x7     ->push_back(isoinfo7x7.nHadronEne);
  t_photonEne7x7      ->push_back(isoinfo7x7.photonEne);
  t_eleEne7x7         ->push_back(isoinfo7x7.eleEne);
  t_muEne7x7          ->push_back(isoinfo7x7.muEne);
  
  t_maxNearP3x3       ->push_back(isoinfo3x3.maxNearP);
  t_cHadronEne3x3     ->push_back(isoinfo3x3.cHadronEne);
  t_cHadronEne3x3_1   ->push_back(isoinfo3x3.cHadronEne_[0]);
  t_cHadronEne3x3_2   ->push_back(isoinfo3x3.cHadronEne_[1]);
  t_cHadronEne3x3_3   ->push_back(isoinfo3x3.cHadronEne_[2]);
  t_nHadronEne3x3     ->push_back(isoinfo3x3.nHadronEne);
  t_photonEne3x3      ->push_back(isoinfo3x3.photonEne);
  t_eleEne3x3         ->push_back(isoinfo3x3.eleEne);
  t_muEne3x3          ->push_back(isoinfo3x3.muEne);

  t_maxNearP1x1       ->push_back(isoinfo1x1.maxNearP);
  t_cHadronEne1x1     ->push_back(isoinfo1x1.cHadronEne);
  t_cHadronEne1x1_1   ->push_back(isoinfo1x1.cHadronEne_[0]);
  t_cHadronEne1x1_2   ->push_back(isoinfo1x1.cHadronEne_[1]);
  t_cHadronEne1x1_3   ->push_back(isoinfo1x1.cHadronEne_[2]);
  t_nHadronEne1x1     ->push_back(isoinfo1x1.nHadronEne);
  t_photonEne1x1      ->push_back(isoinfo1x1.photonEne);
  t_eleEne1x1         ->push_back(isoinfo1x1.eleEne);
  t_muEne1x1          ->push_back(isoinfo1x1.muEne);

  t_maxNearPHC1x1       ->push_back(isoinfoHC1x1.maxNearP);
  t_cHadronEneHC1x1     ->push_back(isoinfoHC1x1.cHadronEne);
  t_cHadronEneHC1x1_1   ->push_back(isoinfoHC1x1.cHadronEne_[0]);
  t_cHadronEneHC1x1_2   ->push_back(isoinfoHC1x1.cHadronEne_[1]);
  t_cHadronEneHC1x1_3   ->push_back(isoinfoHC1x1.cHadronEne_[2]);
  t_nHadronEneHC1x1     ->push_back(isoinfoHC1x1.nHadronEne);
  t_photonEneHC1x1      ->push_back(isoinfoHC1x1.photonEne);
  t_eleEneHC1x1         ->push_back(isoinfoHC1x1.eleEne);
  t_muEneHC1x1          ->push_back(isoinfoHC1x1.muEne);
  
  t_maxNearPHC3x3       ->push_back(isoinfoHC3x3.maxNearP);
  t_cHadronEneHC3x3     ->push_back(isoinfoHC3x3.cHadronEne);
  t_cHadronEneHC3x3_1   ->push_back(isoinfoHC3x3.cHadronEne_[0]);
  t_cHadronEneHC3x3_2   ->push_back(isoinfoHC3x3.cHadronEne_[1]);
  t_cHadronEneHC3x3_3   ->push_back(isoinfoHC3x3.cHadronEne_[2]);
  t_nHadronEneHC3x3     ->push_back(isoinfoHC3x3.nHadronEne);
  t_photonEneHC3x3      ->push_back(isoinfoHC3x3.photonEne);
  t_eleEneHC3x3         ->push_back(isoinfoHC3x3.eleEne);
  t_muEneHC3x3          ->push_back(isoinfoHC3x3.muEne);

  t_maxNearPHC5x5       ->push_back(isoinfoHC5x5.maxNearP);
  t_cHadronEneHC5x5     ->push_back(isoinfoHC5x5.cHadronEne);
  t_cHadronEneHC5x5_1   ->push_back(isoinfoHC5x5.cHadronEne_[0]);
  t_cHadronEneHC5x5_2   ->push_back(isoinfoHC5x5.cHadronEne_[1]);
  t_cHadronEneHC5x5_3   ->push_back(isoinfoHC5x5.cHadronEne_[2]);
  t_nHadronEneHC5x5     ->push_back(isoinfoHC5x5.nHadronEne);
  t_photonEneHC5x5      ->push_back(isoinfoHC5x5.photonEne);
  t_eleEneHC5x5         ->push_back(isoinfoHC5x5.eleEne);
  t_muEneHC5x5          ->push_back(isoinfoHC5x5.muEne);

  t_maxNearPHC7x7       ->push_back(isoinfoHC7x7.maxNearP);
  t_cHadronEneHC7x7     ->push_back(isoinfoHC7x7.cHadronEne);
  t_cHadronEneHC7x7_1   ->push_back(isoinfoHC7x7.cHadronEne_[0]);
  t_cHadronEneHC7x7_2   ->push_back(isoinfoHC7x7.cHadronEne_[1]);
  t_cHadronEneHC7x7_3   ->push_back(isoinfoHC7x7.cHadronEne_[2]);
  t_nHadronEneHC7x7     ->push_back(isoinfoHC7x7.nHadronEne);
  t_photonEneHC7x7      ->push_back(isoinfoHC7x7.photonEne);
  t_eleEneHC7x7         ->push_back(isoinfoHC7x7.eleEne);
  t_muEneHC7x7          ->push_back(isoinfoHC7x7.muEne);

  t_maxNearPR           ->push_back(isoinfoR.maxNearP);
  t_cHadronEneR         ->push_back(isoinfoR.cHadronEne);
  t_cHadronEneR_1       ->push_back(isoinfoR.cHadronEne_[0]);
  t_cHadronEneR_2       ->push_back(isoinfoR.cHadronEne_[1]);
  t_cHadronEneR_3       ->push_back(isoinfoR.cHadronEne_[2]);
  t_nHadronEneR         ->push_back(isoinfoR.nHadronEne);
  t_photonEneR          ->push_back(isoinfoR.photonEne);
  t_eleEneR             ->push_back(isoinfoR.eleEne);
  t_muEneR              ->push_back(isoinfoR.muEne);

  t_maxNearPIsoR        ->push_back(isoinfoIsoR.maxNearP);
  t_cHadronEneIsoR      ->push_back(isoinfoIsoR.cHadronEne);
  t_cHadronEneIsoR_1    ->push_back(isoinfoIsoR.cHadronEne_[0]);
  t_cHadronEneIsoR_2    ->push_back(isoinfoIsoR.cHadronEne_[1]);
  t_cHadronEneIsoR_3    ->push_back(isoinfoIsoR.cHadronEne_[2]);
  t_nHadronEneIsoR      ->push_back(isoinfoIsoR.nHadronEne);
  t_photonEneIsoR       ->push_back(isoinfoIsoR.photonEne);
  t_eleEneIsoR          ->push_back(isoinfoIsoR.eleEne);
  t_muEneIsoR           ->push_back(isoinfoIsoR.muEne);

  t_maxNearPHCR         ->push_back(isoinfoHCR.maxNearP);
  t_cHadronEneHCR       ->push_back(isoinfoHCR.cHadronEne);
  t_cHadronEneHCR_1     ->push_back(isoinfoHCR.cHadronEne_[0]);
  t_cHadronEneHCR_2     ->push_back(isoinfoHCR.cHadronEne_[1]);
  t_cHadronEneHCR_3     ->push_back(isoinfoHCR.cHadronEne_[2]);
  t_nHadronEneHCR       ->push_back(isoinfoHCR.nHadronEne);
  t_photonEneHCR        ->push_back(isoinfoHCR.photonEne);
  t_eleEneHCR           ->push_back(isoinfoHCR.eleEne);
  t_muEneHCR            ->push_back(isoinfoHCR.muEne);

  t_maxNearPIsoHCR      ->push_back(isoinfoIsoHCR.maxNearP);
  t_cHadronEneIsoHCR    ->push_back(isoinfoIsoHCR.cHadronEne);
  t_cHadronEneIsoHCR_1  ->push_back(isoinfoIsoHCR.cHadronEne_[0]);
  t_cHadronEneIsoHCR_2  ->push_back(isoinfoIsoHCR.cHadronEne_[1]);
  t_cHadronEneIsoHCR_3  ->push_back(isoinfoIsoHCR.cHadronEne_[2]);
  t_nHadronEneIsoHCR    ->push_back(isoinfoIsoHCR.nHadronEne);
  t_photonEneIsoHCR     ->push_back(isoinfoIsoHCR.photonEne);
  t_eleEneIsoHCR        ->push_back(isoinfoIsoHCR.eleEne);
  t_muEneIsoHCR         ->push_back(isoinfoIsoHCR.muEne);
}

void IsolatedGenParticles::BookHistograms(){

  //char hname[100], htit[100];

  h_NEventProc  = fs->make<TH1I>("h_NEventProc",  "h_NEventProc", 2, -0.5, 0.5);
  h_L1AlgoNames = fs->make<TH1I>("h_L1AlgoNames", "h_L1AlgoNames:Bin Labels", 128, -0.5, 127.5);  

  double pBin[PBins+1] = {0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 20.0, 30.0, 40.0, 50.0,
			  60.0, 70.0, 80.0, 90.0, 100.0, 150.0, 200.0, 250.0,
			  300.0, 350.0, 400.0, 450.0, 500.0, 550.0, 600.0,
			  650.0, 700.0, 750.0, 800.0, 850.0, 900.0, 950.0,
			  1000.0};
  double etaBin[EtaBins+1] = {-3.0, -2.9, -2.8, -2.7, -2.6, -2.5, -2.4, -2.3,
			      -2.2, -2.1, -2.0, -1.9, -1.8, -1.7, -1.6, -1.5,
			      -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7,
			      -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,  0.0,  0.1,
  			       0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,
  			       1.0,  1.1,  1.2,  1.3,  1.4,  1.5,  1.6,  1.7,
  			       1.8,  1.9,  2.0,  2.1,  2.2,  2.3,  2.4,  2.5,
			       2.6,  2.7,  2.8,  2.9,  3.0};
  std::string particle[Particles] = {"electron", "positron", "#gamma", "#pi^+",
				     "#pi^-", "K^+", "K^-", "p", "n", "pbar",
				     "nbar", "K^0_L"};
  TFileDirectory dir1     = fs->mkdir( "pEta" );
  char name[20], title[50];
  for (int i=0; i<Particles; ++i) {
    sprintf (name, "pEta%d", i);
    sprintf (title, "#eta vs momentum for %s", particle[i].c_str());
    h_pEta[i] = dir1.make<TH2D>(name, title, PBins, pBin, EtaBins, etaBin);
  }

  // build the tree
  tree = fs->make<TTree>("tree", "tree");

  t_isoTrkPAll        = new std::vector<double>();
  t_isoTrkPtAll       = new std::vector<double>();
  t_isoTrkPhiAll      = new std::vector<double>();
  t_isoTrkEtaAll      = new std::vector<double>();
  t_isoTrkDPhiAll     = new std::vector<double>();
  t_isoTrkDEtaAll     = new std::vector<double>();
  t_isoTrkPdgIdAll    = new std::vector<double>();

  t_isoTrkP           = new std::vector<double>();
  t_isoTrkPt          = new std::vector<double>();
  t_isoTrkEne         = new std::vector<double>();
  t_isoTrkEta         = new std::vector<double>();
  t_isoTrkPhi         = new std::vector<double>();
  t_isoTrkEtaEC       = new std::vector<double>();
  t_isoTrkPhiEC       = new std::vector<double>();
  t_isoTrkPdgId       = new std::vector<double>();
  
  t_maxNearP31x31     = new std::vector<double>();
  t_cHadronEne31x31   = new std::vector<double>();
  t_cHadronEne31x31_1 = new std::vector<double>();
  t_cHadronEne31x31_2 = new std::vector<double>();
  t_cHadronEne31x31_3 = new std::vector<double>();
  t_nHadronEne31x31   = new std::vector<double>();
  t_photonEne31x31    = new std::vector<double>();
  t_eleEne31x31       = new std::vector<double>();
  t_muEne31x31        = new std::vector<double>();

  t_maxNearP25x25     = new std::vector<double>();
  t_cHadronEne25x25   = new std::vector<double>();
  t_cHadronEne25x25_1 = new std::vector<double>();
  t_cHadronEne25x25_2 = new std::vector<double>();
  t_cHadronEne25x25_3 = new std::vector<double>();
  t_nHadronEne25x25   = new std::vector<double>();
  t_photonEne25x25    = new std::vector<double>();
  t_eleEne25x25       = new std::vector<double>();
  t_muEne25x25        = new std::vector<double>();

  t_maxNearP21x21     = new std::vector<double>();
  t_cHadronEne21x21   = new std::vector<double>();
  t_cHadronEne21x21_1 = new std::vector<double>();
  t_cHadronEne21x21_2 = new std::vector<double>();
  t_cHadronEne21x21_3 = new std::vector<double>();
  t_nHadronEne21x21   = new std::vector<double>();
  t_photonEne21x21    = new std::vector<double>();
  t_eleEne21x21       = new std::vector<double>();
  t_muEne21x21        = new std::vector<double>();

  t_maxNearP15x15     = new std::vector<double>();
  t_cHadronEne15x15   = new std::vector<double>();
  t_cHadronEne15x15_1 = new std::vector<double>();
  t_cHadronEne15x15_2 = new std::vector<double>();
  t_cHadronEne15x15_3 = new std::vector<double>();
  t_nHadronEne15x15   = new std::vector<double>();
  t_photonEne15x15    = new std::vector<double>();
  t_eleEne15x15       = new std::vector<double>();
  t_muEne15x15        = new std::vector<double>();

  t_maxNearP11x11     = new std::vector<double>();
  t_cHadronEne11x11   = new std::vector<double>();
  t_cHadronEne11x11_1 = new std::vector<double>();
  t_cHadronEne11x11_2 = new std::vector<double>();
  t_cHadronEne11x11_3 = new std::vector<double>();
  t_nHadronEne11x11   = new std::vector<double>();
  t_photonEne11x11    = new std::vector<double>();
  t_eleEne11x11       = new std::vector<double>();
  t_muEne11x11        = new std::vector<double>();

  t_maxNearP9x9       = new std::vector<double>();
  t_cHadronEne9x9     = new std::vector<double>();
  t_cHadronEne9x9_1   = new std::vector<double>();
  t_cHadronEne9x9_2   = new std::vector<double>();
  t_cHadronEne9x9_3   = new std::vector<double>();
  t_nHadronEne9x9     = new std::vector<double>();
  t_photonEne9x9      = new std::vector<double>();
  t_eleEne9x9         = new std::vector<double>();
  t_muEne9x9          = new std::vector<double>();

  t_maxNearP7x7       = new std::vector<double>();
  t_cHadronEne7x7     = new std::vector<double>();
  t_cHadronEne7x7_1   = new std::vector<double>();
  t_cHadronEne7x7_2   = new std::vector<double>();
  t_cHadronEne7x7_3   = new std::vector<double>();
  t_nHadronEne7x7     = new std::vector<double>();
  t_photonEne7x7      = new std::vector<double>();
  t_eleEne7x7         = new std::vector<double>();
  t_muEne7x7          = new std::vector<double>();

  t_maxNearP3x3       = new std::vector<double>();
  t_cHadronEne3x3     = new std::vector<double>();
  t_cHadronEne3x3_1   = new std::vector<double>();
  t_cHadronEne3x3_2   = new std::vector<double>();
  t_cHadronEne3x3_3   = new std::vector<double>();
  t_nHadronEne3x3     = new std::vector<double>();
  t_photonEne3x3      = new std::vector<double>();
  t_eleEne3x3         = new std::vector<double>();
  t_muEne3x3          = new std::vector<double>();

  t_maxNearP1x1       = new std::vector<double>();
  t_cHadronEne1x1     = new std::vector<double>();
  t_cHadronEne1x1_1   = new std::vector<double>();
  t_cHadronEne1x1_2   = new std::vector<double>();
  t_cHadronEne1x1_3   = new std::vector<double>();
  t_nHadronEne1x1     = new std::vector<double>();
  t_photonEne1x1      = new std::vector<double>();
  t_eleEne1x1         = new std::vector<double>();
  t_muEne1x1          = new std::vector<double>();

  t_maxNearPHC1x1       = new std::vector<double>();
  t_cHadronEneHC1x1     = new std::vector<double>();
  t_cHadronEneHC1x1_1   = new std::vector<double>();
  t_cHadronEneHC1x1_2   = new std::vector<double>();
  t_cHadronEneHC1x1_3   = new std::vector<double>();
  t_nHadronEneHC1x1     = new std::vector<double>();
  t_photonEneHC1x1      = new std::vector<double>();
  t_eleEneHC1x1         = new std::vector<double>();
  t_muEneHC1x1          = new std::vector<double>();

  t_maxNearPHC3x3       = new std::vector<double>();
  t_cHadronEneHC3x3     = new std::vector<double>();
  t_cHadronEneHC3x3_1   = new std::vector<double>();
  t_cHadronEneHC3x3_2   = new std::vector<double>();
  t_cHadronEneHC3x3_3   = new std::vector<double>();
  t_nHadronEneHC3x3     = new std::vector<double>();
  t_photonEneHC3x3      = new std::vector<double>();
  t_eleEneHC3x3         = new std::vector<double>();
  t_muEneHC3x3          = new std::vector<double>();

  t_maxNearPHC5x5       = new std::vector<double>();
  t_cHadronEneHC5x5     = new std::vector<double>();
  t_cHadronEneHC5x5_1   = new std::vector<double>();
  t_cHadronEneHC5x5_2   = new std::vector<double>();
  t_cHadronEneHC5x5_3   = new std::vector<double>();
  t_nHadronEneHC5x5     = new std::vector<double>();
  t_photonEneHC5x5      = new std::vector<double>();
  t_eleEneHC5x5         = new std::vector<double>();
  t_muEneHC5x5          = new std::vector<double>();

  t_maxNearPHC7x7       = new std::vector<double>();
  t_cHadronEneHC7x7     = new std::vector<double>();
  t_cHadronEneHC7x7_1   = new std::vector<double>();
  t_cHadronEneHC7x7_2   = new std::vector<double>();
  t_cHadronEneHC7x7_3   = new std::vector<double>();
  t_nHadronEneHC7x7     = new std::vector<double>();
  t_photonEneHC7x7      = new std::vector<double>();
  t_eleEneHC7x7         = new std::vector<double>();
  t_muEneHC7x7          = new std::vector<double>();

  t_maxNearPR           = new std::vector<double>();
  t_cHadronEneR         = new std::vector<double>();
  t_cHadronEneR_1       = new std::vector<double>();
  t_cHadronEneR_2       = new std::vector<double>();
  t_cHadronEneR_3       = new std::vector<double>();
  t_nHadronEneR         = new std::vector<double>();
  t_photonEneR          = new std::vector<double>();
  t_eleEneR             = new std::vector<double>();
  t_muEneR              = new std::vector<double>();

  t_maxNearPIsoR        = new std::vector<double>();
  t_cHadronEneIsoR      = new std::vector<double>();
  t_cHadronEneIsoR_1    = new std::vector<double>();
  t_cHadronEneIsoR_2    = new std::vector<double>();
  t_cHadronEneIsoR_3    = new std::vector<double>();
  t_nHadronEneIsoR      = new std::vector<double>();
  t_photonEneIsoR       = new std::vector<double>();
  t_eleEneIsoR          = new std::vector<double>();
  t_muEneIsoR           = new std::vector<double>();

  t_maxNearPHCR         = new std::vector<double>();
  t_cHadronEneHCR       = new std::vector<double>();
  t_cHadronEneHCR_1     = new std::vector<double>();
  t_cHadronEneHCR_2     = new std::vector<double>();
  t_cHadronEneHCR_3     = new std::vector<double>();
  t_nHadronEneHCR       = new std::vector<double>();
  t_photonEneHCR        = new std::vector<double>();
  t_eleEneHCR           = new std::vector<double>();
  t_muEneHCR            = new std::vector<double>();

  t_maxNearPIsoHCR      = new std::vector<double>();
  t_cHadronEneIsoHCR    = new std::vector<double>();
  t_cHadronEneIsoHCR_1  = new std::vector<double>();
  t_cHadronEneIsoHCR_2  = new std::vector<double>();
  t_cHadronEneIsoHCR_3  = new std::vector<double>();
  t_nHadronEneIsoHCR    = new std::vector<double>();
  t_photonEneIsoHCR     = new std::vector<double>();
  t_eleEneIsoHCR        = new std::vector<double>();
  t_muEneIsoHCR         = new std::vector<double>();

  //----- L1Trigger 
  t_L1Decision        = new std::vector<int>();
  t_L1CenJetPt        = new std::vector<double>();
  t_L1CenJetEta       = new std::vector<double>();    
  t_L1CenJetPhi       = new std::vector<double>();
  t_L1FwdJetPt        = new std::vector<double>();
  t_L1FwdJetEta       = new std::vector<double>();
  t_L1FwdJetPhi       = new std::vector<double>();
  t_L1TauJetPt        = new std::vector<double>();
  t_L1TauJetEta       = new std::vector<double>();     
  t_L1TauJetPhi       = new std::vector<double>();
  t_L1MuonPt          = new std::vector<double>();
  t_L1MuonEta         = new std::vector<double>();     
  t_L1MuonPhi         = new std::vector<double>();
  t_L1IsoEMPt         = new std::vector<double>();
  t_L1IsoEMEta        = new std::vector<double>();
  t_L1IsoEMPhi        = new std::vector<double>();
  t_L1NonIsoEMPt      = new std::vector<double>();
  t_L1NonIsoEMEta     = new std::vector<double>();
  t_L1NonIsoEMPhi     = new std::vector<double>();
  t_L1METPt           = new std::vector<double>();
  t_L1METEta          = new std::vector<double>();
  t_L1METPhi          = new std::vector<double>();
  
  //tree->Branch("t_nEvtProc",          "vector<int>",    &t_nEvtProc);

  tree->Branch("t_isoTrkPAll",        "vector<double>", &t_isoTrkPAll);
  tree->Branch("t_isoTrkPtAll",       "vector<double>", &t_isoTrkPtAll);
  tree->Branch("t_isoTrkPhiAll",      "vector<double>", &t_isoTrkPhiAll);
  tree->Branch("t_isoTrkEtaAll",      "vector<double>", &t_isoTrkEtaAll);
  tree->Branch("t_isoTrkDPhiAll",     "vector<double>", &t_isoTrkDPhiAll);
  tree->Branch("t_isoTrkDEtaAll",     "vector<double>", &t_isoTrkDEtaAll);
  tree->Branch("t_isoTrkPdgIdAll",    "vector<double>", &t_isoTrkPdgIdAll);

  tree->Branch("t_isoTrkP",           "vector<double>", &t_isoTrkP);
  tree->Branch("t_isoTrkPt",          "vector<double>", &t_isoTrkPt);
  tree->Branch("t_isoTrkEne",         "vector<double>", &t_isoTrkEne);
  tree->Branch("t_isoTrkEta",         "vector<double>", &t_isoTrkEta);
  tree->Branch("t_isoTrkPhi",         "vector<double>", &t_isoTrkPhi);
  tree->Branch("t_isoTrkEtaEC",       "vector<double>", &t_isoTrkEtaEC);
  tree->Branch("t_isoTrkPhiEC",       "vector<double>", &t_isoTrkPhiEC);
  tree->Branch("t_isoTrkPdgId",       "vector<double>", &t_isoTrkPdgId);

  tree->Branch("t_maxNearP31x31",     "vector<double>", &t_maxNearP31x31);
  tree->Branch("t_cHadronEne31x31",   "vector<double>", &t_cHadronEne31x31);
  tree->Branch("t_cHadronEne31x31_1", "vector<double>", &t_cHadronEne31x31_1);
  tree->Branch("t_cHadronEne31x31_2", "vector<double>", &t_cHadronEne31x31_2);
  tree->Branch("t_cHadronEne31x31_3", "vector<double>", &t_cHadronEne31x31_3);
  tree->Branch("t_nHadronEne31x31",   "vector<double>", &t_nHadronEne31x31);
  tree->Branch("t_photonEne31x31",    "vector<double>", &t_photonEne31x31);
  tree->Branch("t_eleEne31x31",       "vector<double>", &t_eleEne31x31);
  tree->Branch("t_muEne31x31",        "vector<double>", &t_muEne31x31);

  tree->Branch("t_maxNearP25x25",     "vector<double>", &t_maxNearP25x25);
  tree->Branch("t_cHadronEne25x25",   "vector<double>", &t_cHadronEne25x25);
  tree->Branch("t_cHadronEne25x25_1", "vector<double>", &t_cHadronEne25x25_1);
  tree->Branch("t_cHadronEne25x25_2", "vector<double>", &t_cHadronEne25x25_2);
  tree->Branch("t_cHadronEne25x25_3", "vector<double>", &t_cHadronEne25x25_3);
  tree->Branch("t_nHadronEne25x25",   "vector<double>", &t_nHadronEne25x25);
  tree->Branch("t_photonEne25x25",    "vector<double>", &t_photonEne25x25);
  tree->Branch("t_eleEne25x25",       "vector<double>", &t_eleEne25x25);
  tree->Branch("t_muEne25x25",        "vector<double>", &t_muEne25x25);
  
  tree->Branch("t_maxNearP21x21",     "vector<double>", &t_maxNearP21x21);
  tree->Branch("t_cHadronEne21x21",   "vector<double>", &t_cHadronEne21x21);
  tree->Branch("t_cHadronEne21x21_1", "vector<double>", &t_cHadronEne21x21_1);
  tree->Branch("t_cHadronEne21x21_2", "vector<double>", &t_cHadronEne21x21_2);
  tree->Branch("t_cHadronEne21x21_3", "vector<double>", &t_cHadronEne21x21_3);
  tree->Branch("t_nHadronEne21x21",   "vector<double>", &t_nHadronEne21x21);
  tree->Branch("t_photonEne21x21",    "vector<double>", &t_photonEne21x21);
  tree->Branch("t_eleEne21x21",       "vector<double>", &t_eleEne21x21);
  tree->Branch("t_muEne21x21",        "vector<double>", &t_muEne21x21);

  tree->Branch("t_maxNearP15x15",     "vector<double>", &t_maxNearP15x15);
  tree->Branch("t_cHadronEne15x15",   "vector<double>", &t_cHadronEne15x15);
  tree->Branch("t_cHadronEne15x15_1", "vector<double>", &t_cHadronEne15x15_1);
  tree->Branch("t_cHadronEne15x15_2", "vector<double>", &t_cHadronEne15x15_2);
  tree->Branch("t_cHadronEne15x15_3", "vector<double>", &t_cHadronEne15x15_3);
  tree->Branch("t_nHadronEne15x15",   "vector<double>", &t_nHadronEne15x15);
  tree->Branch("t_photonEne15x15",    "vector<double>", &t_photonEne15x15);
  tree->Branch("t_eleEne15x15",       "vector<double>", &t_eleEne15x15);
  tree->Branch("t_muEne15x15",        "vector<double>", &t_muEne15x15);

  tree->Branch("t_maxNearP11x11",     "vector<double>", &t_maxNearP11x11);
  tree->Branch("t_cHadronEne11x11",   "vector<double>", &t_cHadronEne11x11);
  tree->Branch("t_cHadronEne11x11_1", "vector<double>", &t_cHadronEne11x11_1);
  tree->Branch("t_cHadronEne11x11_2", "vector<double>", &t_cHadronEne11x11_2);
  tree->Branch("t_cHadronEne11x11_3", "vector<double>", &t_cHadronEne11x11_3);
  tree->Branch("t_nHadronEne11x11",   "vector<double>", &t_nHadronEne11x11);
  tree->Branch("t_photonEne11x11",    "vector<double>", &t_photonEne11x11);
  tree->Branch("t_eleEne11x11",       "vector<double>", &t_eleEne11x11);
  tree->Branch("t_muEne11x11",        "vector<double>", &t_muEne11x11);

  tree->Branch("t_maxNearP9x9",       "vector<double>", &t_maxNearP9x9);
  tree->Branch("t_cHadronEne9x9",     "vector<double>", &t_cHadronEne9x9);
  tree->Branch("t_cHadronEne9x9_1",   "vector<double>", &t_cHadronEne9x9_1);
  tree->Branch("t_cHadronEne9x9_2",   "vector<double>", &t_cHadronEne9x9_2);
  tree->Branch("t_cHadronEne9x9_3",   "vector<double>", &t_cHadronEne9x9_3);
  tree->Branch("t_nHadronEne9x9",     "vector<double>", &t_nHadronEne9x9);
  tree->Branch("t_photonEne9x9",      "vector<double>", &t_photonEne9x9);
  tree->Branch("t_eleEne9x9",         "vector<double>", &t_eleEne9x9);
  tree->Branch("t_muEne9x9",          "vector<double>", &t_muEne9x9);

  tree->Branch("t_maxNearP7x7",       "vector<double>", &t_maxNearP7x7);
  tree->Branch("t_cHadronEne7x7",     "vector<double>", &t_cHadronEne7x7);
  tree->Branch("t_cHadronEne7x7_1",   "vector<double>", &t_cHadronEne7x7_1);
  tree->Branch("t_cHadronEne7x7_2",   "vector<double>", &t_cHadronEne7x7_2);
  tree->Branch("t_cHadronEne7x7_3",   "vector<double>", &t_cHadronEne7x7_3);
  tree->Branch("t_nHadronEne7x7",     "vector<double>", &t_nHadronEne7x7);
  tree->Branch("t_photonEne7x7",      "vector<double>", &t_photonEne7x7);
  tree->Branch("t_eleEne7x7",         "vector<double>", &t_eleEne7x7);
  tree->Branch("t_muEne7x7",          "vector<double>", &t_muEne7x7);

  tree->Branch("t_maxNearP3x3",       "vector<double>", &t_maxNearP3x3);
  tree->Branch("t_cHadronEne3x3",     "vector<double>", &t_cHadronEne3x3);
  tree->Branch("t_cHadronEne3x3_1",   "vector<double>", &t_cHadronEne3x3_1);
  tree->Branch("t_cHadronEne3x3_2",   "vector<double>", &t_cHadronEne3x3_2);
  tree->Branch("t_cHadronEne3x3_3",   "vector<double>", &t_cHadronEne3x3_3);
  tree->Branch("t_nHadronEne3x3",     "vector<double>", &t_nHadronEne3x3);
  tree->Branch("t_photonEne3x3",      "vector<double>", &t_photonEne3x3);
  tree->Branch("t_eleEne3x3",         "vector<double>", &t_eleEne3x3);
  tree->Branch("t_muEne3x3",          "vector<double>", &t_muEne3x3);

  tree->Branch("t_maxNearP1x1",       "vector<double>", &t_maxNearP1x1);
  tree->Branch("t_cHadronEne1x1",     "vector<double>", &t_cHadronEne1x1);
  tree->Branch("t_cHadronEne1x1_1",   "vector<double>", &t_cHadronEne1x1_1);
  tree->Branch("t_cHadronEne1x1_2",   "vector<double>", &t_cHadronEne1x1_2);
  tree->Branch("t_cHadronEne1x1_3",   "vector<double>", &t_cHadronEne1x1_3);
  tree->Branch("t_nHadronEne1x1",     "vector<double>", &t_nHadronEne1x1);
  tree->Branch("t_photonEne1x1",      "vector<double>", &t_photonEne1x1);
  tree->Branch("t_eleEne1x1",         "vector<double>", &t_eleEne1x1);
  tree->Branch("t_muEne1x1",          "vector<double>", &t_muEne1x1);

  tree->Branch("t_maxNearPHC1x1",       "vector<double>", &t_maxNearPHC1x1);
  tree->Branch("t_cHadronEneHC1x1",     "vector<double>", &t_cHadronEneHC1x1);
  tree->Branch("t_cHadronEneHC1x1_1",   "vector<double>", &t_cHadronEneHC1x1_1);
  tree->Branch("t_cHadronEneHC1x1_2",   "vector<double>", &t_cHadronEneHC1x1_2);
  tree->Branch("t_cHadronEneHC1x1_3",   "vector<double>", &t_cHadronEneHC1x1_3);
  tree->Branch("t_nHadronEneHC1x1",     "vector<double>", &t_nHadronEneHC1x1);
  tree->Branch("t_photonEneHC1x1",      "vector<double>", &t_photonEneHC1x1);
  tree->Branch("t_eleEneHC1x1",         "vector<double>", &t_eleEneHC1x1);
  tree->Branch("t_muEneHC1x1",          "vector<double>", &t_muEneHC1x1);

  tree->Branch("t_maxNearPHC3x3",       "vector<double>", &t_maxNearPHC3x3);
  tree->Branch("t_cHadronEneHC3x3",     "vector<double>", &t_cHadronEneHC3x3);
  tree->Branch("t_cHadronEneHC3x3_1",   "vector<double>", &t_cHadronEneHC3x3_1);
  tree->Branch("t_cHadronEneHC3x3_2",   "vector<double>", &t_cHadronEneHC3x3_2);
  tree->Branch("t_cHadronEneHC3x3_3",   "vector<double>", &t_cHadronEneHC3x3_3);
  tree->Branch("t_nHadronEneHC3x3",     "vector<double>", &t_nHadronEneHC3x3);
  tree->Branch("t_photonEneHC3x3",      "vector<double>", &t_photonEneHC3x3);
  tree->Branch("t_eleEneHC3x3",         "vector<double>", &t_eleEneHC3x3);
  tree->Branch("t_muEneHC3x3",          "vector<double>", &t_muEneHC3x3);

  tree->Branch("t_maxNearPHC5x5",       "vector<double>", &t_maxNearPHC5x5);
  tree->Branch("t_cHadronEneHC5x5",     "vector<double>", &t_cHadronEneHC5x5);
  tree->Branch("t_cHadronEneHC5x5_1",   "vector<double>", &t_cHadronEneHC5x5_1);
  tree->Branch("t_cHadronEneHC5x5_2",   "vector<double>", &t_cHadronEneHC5x5_2);
  tree->Branch("t_cHadronEneHC5x5_3",   "vector<double>", &t_cHadronEneHC5x5_3);
  tree->Branch("t_nHadronEneHC5x5",     "vector<double>", &t_nHadronEneHC5x5);
  tree->Branch("t_photonEneHC5x5",      "vector<double>", &t_photonEneHC5x5);
  tree->Branch("t_eleEneHC5x5",         "vector<double>", &t_eleEneHC5x5);
  tree->Branch("t_muEneHC5x5",          "vector<double>", &t_muEneHC5x5);

  tree->Branch("t_maxNearPHC7x7",       "vector<double>", &t_maxNearPHC7x7);
  tree->Branch("t_cHadronEneHC7x7",     "vector<double>", &t_cHadronEneHC7x7);
  tree->Branch("t_cHadronEneHC7x7_1",   "vector<double>", &t_cHadronEneHC7x7_1);
  tree->Branch("t_cHadronEneHC7x7_2",   "vector<double>", &t_cHadronEneHC7x7_2);
  tree->Branch("t_cHadronEneHC7x7_3",   "vector<double>", &t_cHadronEneHC7x7_3);
  tree->Branch("t_nHadronEneHC7x7",     "vector<double>", &t_nHadronEneHC7x7);
  tree->Branch("t_photonEneHC7x7",      "vector<double>", &t_photonEneHC7x7);
  tree->Branch("t_eleEneHC7x7",         "vector<double>", &t_eleEneHC7x7);
  tree->Branch("t_muEneHC7x7",          "vector<double>", &t_muEneHC7x7);

  tree->Branch("t_maxNearPR",       "vector<double>", &t_maxNearPR);
  tree->Branch("t_cHadronEneR",     "vector<double>", &t_cHadronEneR);
  tree->Branch("t_cHadronEneR_1",   "vector<double>", &t_cHadronEneR_1);
  tree->Branch("t_cHadronEneR_2",   "vector<double>", &t_cHadronEneR_2);
  tree->Branch("t_cHadronEneR_3",   "vector<double>", &t_cHadronEneR_3);
  tree->Branch("t_nHadronEneR",     "vector<double>", &t_nHadronEneR);
  tree->Branch("t_photonEneR",      "vector<double>", &t_photonEneR);
  tree->Branch("t_eleEneR",         "vector<double>", &t_eleEneR);
  tree->Branch("t_muEneR",          "vector<double>", &t_muEneR);

  tree->Branch("t_maxNearPIsoR",       "vector<double>", &t_maxNearPIsoR);
  tree->Branch("t_cHadronEneIsoR",     "vector<double>", &t_cHadronEneIsoR);
  tree->Branch("t_cHadronEneIsoR_1",   "vector<double>", &t_cHadronEneIsoR_1);
  tree->Branch("t_cHadronEneIsoR_2",   "vector<double>", &t_cHadronEneIsoR_2);
  tree->Branch("t_cHadronEneIsoR_3",   "vector<double>", &t_cHadronEneIsoR_3);
  tree->Branch("t_nHadronEneIsoR",     "vector<double>", &t_nHadronEneIsoR);
  tree->Branch("t_photonEneIsoR",      "vector<double>", &t_photonEneIsoR);
  tree->Branch("t_eleEneIsoR",         "vector<double>", &t_eleEneIsoR);
  tree->Branch("t_muEneIsoR",          "vector<double>", &t_muEneIsoR);

  tree->Branch("t_maxNearPHCR",       "vector<double>", &t_maxNearPHCR);
  tree->Branch("t_cHadronEneHCR",     "vector<double>", &t_cHadronEneHCR);
  tree->Branch("t_cHadronEneHCR_1",   "vector<double>", &t_cHadronEneHCR_1);
  tree->Branch("t_cHadronEneHCR_2",   "vector<double>", &t_cHadronEneHCR_2);
  tree->Branch("t_cHadronEneHCR_3",   "vector<double>", &t_cHadronEneHCR_3);
  tree->Branch("t_nHadronEneHCR",     "vector<double>", &t_nHadronEneHCR);
  tree->Branch("t_photonEneHCR",      "vector<double>", &t_photonEneHCR);
  tree->Branch("t_eleEneHCR",         "vector<double>", &t_eleEneHCR);
  tree->Branch("t_muEneHCR",          "vector<double>", &t_muEneHCR);

  tree->Branch("t_maxNearPIsoHCR",       "vector<double>", &t_maxNearPIsoHCR);
  tree->Branch("t_cHadronEneIsoHCR",     "vector<double>", &t_cHadronEneIsoHCR);
  tree->Branch("t_cHadronEneIsoHCR_1",   "vector<double>", &t_cHadronEneIsoHCR_1);
  tree->Branch("t_cHadronEneIsoHCR_2",   "vector<double>", &t_cHadronEneIsoHCR_2);
  tree->Branch("t_cHadronEneIsoHCR_3",   "vector<double>", &t_cHadronEneIsoHCR_3);
  tree->Branch("t_nHadronEneIsoHCR",     "vector<double>", &t_nHadronEneIsoHCR);
  tree->Branch("t_photonEneIsoHCR",      "vector<double>", &t_photonEneIsoHCR);
  tree->Branch("t_eleEneIsoHCR",         "vector<double>", &t_eleEneIsoHCR);
  tree->Branch("t_muEneIsoHCR",          "vector<double>", &t_muEneIsoHCR);

  tree->Branch("t_L1Decision",        "vector<int>",    &t_L1Decision);
  tree->Branch("t_L1CenJetPt",        "vector<double>", &t_L1CenJetPt);
  tree->Branch("t_L1CenJetEta",       "vector<double>", &t_L1CenJetEta);
  tree->Branch("t_L1CenJetPhi",       "vector<double>", &t_L1CenJetPhi);
  tree->Branch("t_L1FwdJetPt",        "vector<double>", &t_L1FwdJetPt);
  tree->Branch("t_L1FwdJetEta",       "vector<double>", &t_L1FwdJetEta);
  tree->Branch("t_L1FwdJetPhi",       "vector<double>", &t_L1FwdJetPhi);
  tree->Branch("t_L1TauJetPt",        "vector<double>", &t_L1TauJetPt);
  tree->Branch("t_L1TauJetEta",       "vector<double>", &t_L1TauJetEta);     
  tree->Branch("t_L1TauJetPhi",       "vector<double>", &t_L1TauJetPhi);
  tree->Branch("t_L1MuonPt",          "vector<double>", &t_L1MuonPt);
  tree->Branch("t_L1MuonEta",         "vector<double>", &t_L1MuonEta);
  tree->Branch("t_L1MuonPhi",         "vector<double>", &t_L1MuonPhi);
  tree->Branch("t_L1IsoEMPt",         "vector<double>", &t_L1IsoEMPt);
  tree->Branch("t_L1IsoEMEta",        "vector<double>", &t_L1IsoEMEta);
  tree->Branch("t_L1IsoEMPhi",        "vector<double>", &t_L1IsoEMPhi);
  tree->Branch("t_L1NonIsoEMPt",      "vector<double>", &t_L1NonIsoEMPt);
  tree->Branch("t_L1NonIsoEMEta",     "vector<double>", &t_L1NonIsoEMEta);
  tree->Branch("t_L1NonIsoEMPhi",     "vector<double>", &t_L1NonIsoEMPhi);
  tree->Branch("t_L1METPt",           "vector<double>", &t_L1METPt);
  tree->Branch("t_L1METEta",          "vector<double>", &t_L1METEta);
  tree->Branch("t_L1METPhi",          "vector<double>", &t_L1METPhi);
}


void IsolatedGenParticles::clearTreeVectors() {
  //  t_maxNearP31x31->clear();
  
  //t_nEvtProc          ->clear();

  t_isoTrkPAll        ->clear();
  t_isoTrkPtAll       ->clear();
  t_isoTrkPhiAll      ->clear();
  t_isoTrkEtaAll      ->clear();
  t_isoTrkDPhiAll     ->clear();
  t_isoTrkDEtaAll     ->clear();
  t_isoTrkPdgIdAll    ->clear();

  t_isoTrkP           ->clear();
  t_isoTrkPt          ->clear();
  t_isoTrkEne         ->clear();
  t_isoTrkEta         ->clear();
  t_isoTrkPhi         ->clear();
  t_isoTrkEtaEC       ->clear();
  t_isoTrkPhiEC       ->clear();
  t_isoTrkPdgId       ->clear();

  t_maxNearP31x31     ->clear();
  t_cHadronEne31x31   ->clear();
  t_cHadronEne31x31_1 ->clear();
  t_cHadronEne31x31_2 ->clear();
  t_cHadronEne31x31_3 ->clear();
  t_nHadronEne31x31   ->clear();
  t_photonEne31x31    ->clear();
  t_eleEne31x31       ->clear();
  t_muEne31x31        ->clear();

  t_maxNearP25x25     ->clear();
  t_cHadronEne25x25   ->clear();
  t_cHadronEne25x25_1 ->clear();
  t_cHadronEne25x25_2 ->clear();
  t_cHadronEne25x25_3 ->clear();
  t_nHadronEne25x25   ->clear();
  t_photonEne25x25    ->clear();
  t_eleEne25x25       ->clear();
  t_muEne25x25        ->clear();

  t_maxNearP21x21     ->clear();
  t_cHadronEne21x21   ->clear();
  t_cHadronEne21x21_1 ->clear();
  t_cHadronEne21x21_2 ->clear();
  t_cHadronEne21x21_3 ->clear();
  t_nHadronEne21x21   ->clear();
  t_photonEne21x21    ->clear();
  t_eleEne21x21       ->clear();
  t_muEne21x21        ->clear();

  t_maxNearP15x15     ->clear();
  t_cHadronEne15x15   ->clear();
  t_cHadronEne15x15_1 ->clear();
  t_cHadronEne15x15_2 ->clear();
  t_cHadronEne15x15_3 ->clear();
  t_nHadronEne15x15   ->clear();
  t_photonEne15x15    ->clear();
  t_eleEne15x15       ->clear();
  t_muEne15x15        ->clear();

  t_maxNearP11x11     ->clear();
  t_cHadronEne11x11   ->clear();
  t_cHadronEne11x11_1 ->clear();
  t_cHadronEne11x11_2 ->clear();
  t_cHadronEne11x11_3 ->clear();
  t_nHadronEne11x11   ->clear();
  t_photonEne11x11    ->clear();
  t_eleEne11x11       ->clear();
  t_muEne11x11        ->clear();

  t_maxNearP9x9       ->clear();
  t_cHadronEne9x9     ->clear();
  t_cHadronEne9x9_1   ->clear();
  t_cHadronEne9x9_2   ->clear();
  t_cHadronEne9x9_3   ->clear();
  t_nHadronEne9x9     ->clear();
  t_photonEne9x9      ->clear();
  t_eleEne9x9         ->clear();
  t_muEne9x9          ->clear();

  t_maxNearP7x7       ->clear();
  t_cHadronEne7x7     ->clear();
  t_cHadronEne7x7_1   ->clear();
  t_cHadronEne7x7_2   ->clear();
  t_cHadronEne7x7_3   ->clear();
  t_nHadronEne7x7     ->clear();
  t_photonEne7x7      ->clear();
  t_eleEne7x7         ->clear();
  t_muEne7x7          ->clear();

  t_maxNearP3x3       ->clear();
  t_cHadronEne3x3     ->clear();
  t_cHadronEne3x3_1   ->clear();
  t_cHadronEne3x3_2   ->clear();
  t_cHadronEne3x3_3   ->clear();
  t_nHadronEne3x3     ->clear();
  t_photonEne3x3      ->clear();
  t_eleEne3x3         ->clear();
  t_muEne3x3          ->clear();

  t_maxNearP1x1       ->clear();
  t_cHadronEne1x1     ->clear();
  t_cHadronEne1x1_1   ->clear();
  t_cHadronEne1x1_2   ->clear();
  t_cHadronEne1x1_3   ->clear();
  t_nHadronEne1x1     ->clear();
  t_photonEne1x1      ->clear();
  t_eleEne1x1         ->clear();
  t_muEne1x1          ->clear();

  t_maxNearPHC1x1       ->clear();
  t_cHadronEneHC1x1     ->clear();
  t_cHadronEneHC1x1_1   ->clear();
  t_cHadronEneHC1x1_2   ->clear();
  t_cHadronEneHC1x1_3   ->clear();
  t_nHadronEneHC1x1     ->clear();
  t_photonEneHC1x1      ->clear();
  t_eleEneHC1x1         ->clear();
  t_muEneHC1x1          ->clear();

  t_maxNearPHC3x3       ->clear();
  t_cHadronEneHC3x3     ->clear();
  t_cHadronEneHC3x3_1   ->clear();
  t_cHadronEneHC3x3_2   ->clear();
  t_cHadronEneHC3x3_3   ->clear();
  t_nHadronEneHC3x3     ->clear();
  t_photonEneHC3x3      ->clear();
  t_eleEneHC3x3         ->clear();
  t_muEneHC3x3          ->clear();

  t_maxNearPHC5x5       ->clear();
  t_cHadronEneHC5x5     ->clear();
  t_cHadronEneHC5x5_1   ->clear();
  t_cHadronEneHC5x5_2   ->clear();
  t_cHadronEneHC5x5_3   ->clear();
  t_nHadronEneHC5x5     ->clear();
  t_photonEneHC5x5      ->clear();
  t_eleEneHC5x5         ->clear();
  t_muEneHC5x5          ->clear();

  t_maxNearPHC7x7       ->clear();
  t_cHadronEneHC7x7     ->clear();
  t_cHadronEneHC7x7_1   ->clear();
  t_cHadronEneHC7x7_2   ->clear();
  t_cHadronEneHC7x7_3   ->clear();
  t_nHadronEneHC7x7     ->clear();
  t_photonEneHC7x7      ->clear();
  t_eleEneHC7x7         ->clear();
  t_muEneHC7x7          ->clear();

  t_maxNearPR           ->clear();
  t_cHadronEneR         ->clear();
  t_cHadronEneR_1       ->clear();
  t_cHadronEneR_2       ->clear();
  t_cHadronEneR_3       ->clear();
  t_nHadronEneR         ->clear();
  t_photonEneR          ->clear();
  t_eleEneR             ->clear();
  t_muEneR              ->clear();

  t_maxNearPIsoR        ->clear();
  t_cHadronEneIsoR      ->clear();
  t_cHadronEneIsoR_1    ->clear();
  t_cHadronEneIsoR_2    ->clear();
  t_cHadronEneIsoR_3    ->clear();
  t_nHadronEneIsoR      ->clear();
  t_photonEneIsoR       ->clear();
  t_eleEneIsoR          ->clear();
  t_muEneIsoR           ->clear();

  t_maxNearPHCR         ->clear();
  t_cHadronEneHCR       ->clear();
  t_cHadronEneHCR_1     ->clear();
  t_cHadronEneHCR_2     ->clear();
  t_cHadronEneHCR_3     ->clear();
  t_nHadronEneHCR       ->clear();
  t_photonEneHCR        ->clear();
  t_eleEneHCR           ->clear();
  t_muEneHCR            ->clear();

  t_maxNearPIsoHCR      ->clear();
  t_cHadronEneIsoHCR    ->clear();
  t_cHadronEneIsoHCR_1  ->clear();
  t_cHadronEneIsoHCR_2  ->clear();
  t_cHadronEneIsoHCR_3  ->clear();
  t_nHadronEneIsoHCR    ->clear();
  t_photonEneIsoHCR     ->clear();
  t_eleEneIsoHCR        ->clear();
  t_muEneIsoHCR         ->clear();

  t_L1Decision        ->clear();
  t_L1CenJetPt        ->clear();
  t_L1CenJetEta       ->clear();    
  t_L1CenJetPhi       ->clear();
  t_L1FwdJetPt        ->clear();
  t_L1FwdJetEta       ->clear();
  t_L1FwdJetPhi       ->clear();
  t_L1TauJetPt        ->clear();
  t_L1TauJetEta       ->clear();     
  t_L1TauJetPhi       ->clear();
  t_L1MuonPt          ->clear();
  t_L1MuonEta         ->clear();     
  t_L1MuonPhi         ->clear();
  t_L1IsoEMPt         ->clear();
  t_L1IsoEMEta        ->clear();
  t_L1IsoEMPhi        ->clear();
  t_L1NonIsoEMPt      ->clear();
  t_L1NonIsoEMEta     ->clear();
  t_L1NonIsoEMPhi     ->clear();
  t_L1METPt           ->clear();
  t_L1METEta          ->clear();
  t_L1METPhi          ->clear();
}

int IsolatedGenParticles::particleCode(int pdgId) {
 
 int partID[Particles]={11,-11,21,211,-211,321,-321,2212,2112,-2212,-2112,130};
  int ix = -1;
  for (int ik=0; ik<Particles; ++ik) {
    if (pdgId == partID[ik]) {
      ix = ik; break;
    }
  }
  return ix;
}

//define this as a plug-in
DEFINE_FWK_MODULE(IsolatedGenParticles);
