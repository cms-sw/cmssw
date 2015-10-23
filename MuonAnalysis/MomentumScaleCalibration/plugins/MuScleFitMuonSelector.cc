#include "MuScleFitMuonSelector.h"
#include "DataFormats/MuonReco/interface/CaloMuon.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/Muon.h"

const double MuScleFitMuonSelector::mMu2 = 0.011163612;
const unsigned int MuScleFitMuonSelector::motherPdgIdArray[] = {23, 100553, 100553, 553, 100443, 443};

const reco::Candidate* 
MuScleFitMuonSelector::getStatus1Muon(const reco::Candidate* status3Muon){
  const reco::Candidate* tempMuon = status3Muon;
  //  bool lastCopy = ((reco::GenParticle*)tempMuon)->isLastCopy();                      //  isLastCopy() likely not enough robust
  bool isPromptFinalState = ((reco::GenParticle*)tempMuon)->isPromptFinalState();        //  pre-CMSSW_74X code: int status = tempStatus1Muon->status();
  while(tempMuon == 0 || tempMuon->numberOfDaughters()!=0){
    if ( isPromptFinalState ) break;                                                     //  pre-CMSSW_74X code: if (status == 1) break;
    //std::vector<const reco::Candidate*> daughters;
    for (unsigned int i=0; i<tempMuon->numberOfDaughters(); ++i){
      if ( tempMuon->daughter(i)->pdgId()==tempMuon->pdgId() ){
	tempMuon = tempMuon->daughter(i);
	isPromptFinalState = ((reco::GenParticle*)tempMuon)->isPromptFinalState(); 	 //  pre-CMSSW_74X code: status = tempStatus1Muon->status();
	break;
      }else continue;
    }//for loop
  }//while loop
  
  return tempMuon;
}

const reco::Candidate* 
MuScleFitMuonSelector::getStatus3Muon(const reco::Candidate* status3Muon){
  const reco::Candidate* tempMuon = status3Muon;
  bool lastCopy = ((reco::GenParticle*)tempMuon)->isLastCopyBeforeFSR();        //  pre-CMSSW_74X code: int status = tempStatus1Muon->status();
  while(tempMuon == 0 || tempMuon->numberOfDaughters()!=0){
    if ( lastCopy ) break;                                                      //  pre-CMSSW_74X code: if (status == 3) break;
    //std::vector<const reco::Candidate*> daughters;
    for (unsigned int i=0; i<tempMuon->numberOfDaughters(); ++i){
      if ( tempMuon->daughter(i)->pdgId()==tempMuon->pdgId() ){
	tempMuon = tempMuon->daughter(i);
	lastCopy = ((reco::GenParticle*)tempMuon)->isLastCopyBeforeFSR(); 	//  pre-CMSSW_74X code: status = tempStatus1Muon->status();
	break;
      }else continue;
    }//for loop
  }//while loop
  
  return tempMuon;
}



bool MuScleFitMuonSelector::selGlobalMuon(const pat::Muon* aMuon)
{
  reco::TrackRef iTrack = aMuon->innerTrack();
  const reco::HitPattern& p = iTrack->hitPattern();

  reco::TrackRef gTrack = aMuon->globalTrack();
  const reco::HitPattern& q = gTrack->hitPattern();

  return (//isMuonInAccept(aMuon) &&// no acceptance cuts!
    iTrack->found() > 11 &&
    gTrack->chi2()/gTrack->ndof() < 20.0 &&
    q.numberOfValidMuonHits() > 0 &&
    iTrack->chi2()/iTrack->ndof() < 4.0 &&
    aMuon->muonID("TrackerMuonArbitrated") &&
    aMuon->muonID("TMLastStationAngTight") &&
    p.pixelLayersWithMeasurement() > 1 &&
    fabs(iTrack->dxy()) < 3.0 &&  //should be done w.r.t. PV!
    fabs(iTrack->dz()) < 15.0 //should be done w.r.t. PV!
  );
}

bool MuScleFitMuonSelector::selTrackerMuon(const pat::Muon* aMuon)
{
  reco::TrackRef iTrack = aMuon->innerTrack();
  const reco::HitPattern& p = iTrack->hitPattern();

  return (//isMuonInAccept(aMuon) // no acceptance cuts!
    iTrack->found() > 11 &&
    iTrack->chi2()/iTrack->ndof() < 4.0 &&
    aMuon->muonID("TrackerMuonArbitrated") &&
    aMuon->muonID("TMLastStationAngTight") &&
    p.pixelLayersWithMeasurement() > 1 &&
    fabs(iTrack->dxy()) < 3.0 && //should be done w.r.t. PV!
    fabs(iTrack->dz()) < 15.0 //should be done w.r.t. PV!
  );
}

// Note that at this level we save all the information. Events for which no suitable muon pair is found
// are removed from the tree (together with the gen and sim information) at a later stage.
// It would be better to remove them directly at this point.
void MuScleFitMuonSelector::selectMuons(const edm::Event & event, std::vector<MuScleFitMuon> & muons,
					std::vector<GenMuonPair> & genPair,
					std::vector<std::pair<lorentzVector,lorentzVector> > & simPair,
					MuScleFitPlotter * plotter)
{
  edm::Handle<pat::CompositeCandidateCollection > collAll;
  try {event.getByLabel("onia2MuMuPatTrkTrk",collAll);}
  catch (...) {std::cout << "J/psi not present in event!" << std::endl;}
  std::vector<const pat::Muon*> collMuSel;

  //================onia cuts===========================/

  if( muonType_ <= -1 && PATmuons_) {
    std::vector<const pat::CompositeCandidate*> collSelGG;
    std::vector<const pat::CompositeCandidate*> collSelGT;
    std::vector<const pat::CompositeCandidate*> collSelTT;
    if (collAll.isValid()) {

      for(std::vector<pat::CompositeCandidate>::const_iterator it=collAll->begin();
	  it!=collAll->end();++it) {
      
	const pat::CompositeCandidate* cand = &(*it);	
	// cout << "Now checking candidate of type " << theJpsiCat << " with pt = " << cand->pt() << endl;
	const pat::Muon* muon1 = dynamic_cast<const pat::Muon*>(cand->daughter("muon1"));
	const pat::Muon* muon2 = dynamic_cast<const pat::Muon*>(cand->daughter("muon2"));
      
	if((muon1->charge() * muon2->charge())>0)
	  continue;
	// global + global?
	if (muon1->isGlobalMuon() && muon2->isGlobalMuon() ) {
	  if (selGlobalMuon(muon1) && selGlobalMuon(muon2) && cand->userFloat("vProb") > 0.001 ) {
	    collSelGG.push_back(cand);
	    continue;
	  }
	}
	// global + tracker? (x2)    
	if (muon1->isGlobalMuon() && muon2->isTrackerMuon() ) {
	  if (selGlobalMuon(muon1) &&  selTrackerMuon(muon2) && cand->userFloat("vProb") > 0.001 ) {
	    collSelGT.push_back(cand);
	    continue;
	  }
	}
	if (muon2->isGlobalMuon() && muon1->isTrackerMuon() ) {
	  if (selGlobalMuon(muon2) && selTrackerMuon(muon1) && cand->userFloat("vProb") > 0.001) {
	    collSelGT.push_back(cand);
	    continue;
	  }
	}
	// tracker + tracker?  
	if (muon1->isTrackerMuon() && muon2->isTrackerMuon() ) {
	  if (selTrackerMuon(muon1) && selTrackerMuon(muon2) && cand->userFloat("vProb") > 0.001) {
	    collSelTT.push_back(cand);
	    continue;
	  }
	}
      }
    }
    // Split them in independent collections if using muonType_ == -2, -3 or -4. Take them all if muonType_ == -1.
    std::vector<reco::Track> tracks;
    if(collSelGG.size()){
      //CHECK THAT THEY ARE ORDERED BY PT !!!!!!!!!!!!!!!!!!!!!!!
      const pat::Muon* muon1 = dynamic_cast<const pat::Muon*>(collSelGG[0]->daughter("muon1"));
      const pat::Muon* muon2 = dynamic_cast<const pat::Muon*>(collSelGG[0]->daughter("muon2"));
      if( muonType_ == -1 || muonType_ == -2 ) {
	tracks.push_back(*(muon1->innerTrack()));
	tracks.push_back(*(muon2->innerTrack()));  
	collMuSel.push_back(muon1);
	collMuSel.push_back(muon2);
      }
    }
    else if(!collSelGG.size() && collSelGT.size()){
      //CHECK THAT THEY ARE ORDERED BY PT !!!!!!!!!!!!!!!!!!!!!!!
      const pat::Muon* muon1 = dynamic_cast<const pat::Muon*>(collSelGT[0]->daughter("muon1"));
      const pat::Muon* muon2 = dynamic_cast<const pat::Muon*>(collSelGT[0]->daughter("muon2"));
      if( muonType_ == -1 || muonType_ == -3 ) {
	tracks.push_back(*(muon1->innerTrack()));
	tracks.push_back(*(muon2->innerTrack()));   
 	collMuSel.push_back(muon1);
	collMuSel.push_back(muon2);
     }
    }
    else if(!collSelGG.size() && !collSelGT.size() && collSelTT.size()){
      //CHECK THAT THEY ARE ORDERED BY PT !!!!!!!!!!!!!!!!!!!!!!!
      const pat::Muon* muon1 = dynamic_cast<const pat::Muon*>(collSelTT[0]->daughter("muon1"));
      const pat::Muon* muon2 = dynamic_cast<const pat::Muon*>(collSelTT[0]->daughter("muon2"));
      if( muonType_ == -1 || muonType_ == -4 ) {
	tracks.push_back(*(muon1->innerTrack()));
	tracks.push_back(*(muon2->innerTrack()));   
	collMuSel.push_back(muon1);
	collMuSel.push_back(muon2);
      }
    }
    if (tracks.size() != 2 && tracks.size() != 0){
      std::cout<<"ERROR strange number of muons selected by onia cuts!"<<std::endl;
      abort();
    }
    muons = fillMuonCollection(tracks); 
  }
  else if( (muonType_<4 && muonType_>=0) || muonType_>=10 ) { // Muons (glb,sta,trk)
    std::vector<reco::Track> tracks;
    if( PATmuons_ == true ) {
      edm::Handle<pat::MuonCollection> allMuons;
      event.getByLabel( muonLabel_, allMuons );
      if( muonType_ == 0 ) {
	// Take directly the muon
	muons = fillMuonCollection(*allMuons);
      }
      else {
	for( std::vector<pat::Muon>::const_iterator muon = allMuons->begin(); muon != allMuons->end(); ++muon ) {
	  //std::cout<<"pat muon is global "<<muon->isGlobalMuon()<<std::endl;
	  takeSelectedMuonType(muon, tracks);
	}
	muons = fillMuonCollection(tracks);
      }
    }
    else {
      edm::Handle<reco::MuonCollection> allMuons;
      event.getByLabel (muonLabel_, allMuons);
      if( muonType_ == 0 ) {
	// Take directly the muon
	muons = fillMuonCollection(*allMuons);
      }
      else {
	for( std::vector<reco::Muon>::const_iterator muon = allMuons->begin(); muon != allMuons->end(); ++muon ) {
	  takeSelectedMuonType(muon, tracks);
	}
	muons = fillMuonCollection(tracks);
      }
    }
  }
  else if(muonType_==4){  //CaloMuons
    edm::Handle<reco::CaloMuonCollection> caloMuons;
    event.getByLabel (muonLabel_, caloMuons);
    std::vector<reco::Track> tracks;
    for (std::vector<reco::CaloMuon>::const_iterator muon = caloMuons->begin(); muon != caloMuons->end(); ++muon){
      tracks.push_back(*(muon->track()));
    }
    muons = fillMuonCollection(tracks);
  }

  else if (muonType_==5) { // Inner tracker tracks
    edm::Handle<reco::TrackCollection> tracks;
    event.getByLabel (muonLabel_, tracks);
    muons = fillMuonCollection(*tracks);
  }
  plotter->fillRec(muons);

  // Generation and simulation part
  if( speedup_ == false )
  {
    if( PATmuons_ ) {
      // EM 2015.04.02 temporary fix to run on MINIAODSIM (which contains PAT muons) but not the "onia2MuMuPatTrkTrk" collection   
      // selectGeneratedMuons(collAll, collMuSel, genPair, plotter);
      selectGenSimMuons(event, genPair, simPair, plotter);
    }
    else {
      selectGenSimMuons(event, genPair, simPair, plotter);
    }
  }
}

/// For PATmuons the generator information is read directly from the PAT object
void MuScleFitMuonSelector::selectGeneratedMuons(const edm::Handle<pat::CompositeCandidateCollection> & collAll,
						 const std::vector<const pat::Muon*> & collMuSel,
						 std::vector<GenMuonPair> & genPair,
						 MuScleFitPlotter * plotter)
{
  reco::GenParticleCollection* genPatParticles = new reco::GenParticleCollection;

  //explicitly for JPsi but can be adapted!!!!!
  for(std::vector<pat::CompositeCandidate>::const_iterator it=collAll->begin();
      it!=collAll->end();++it) {
    reco::GenParticleRef genJpsi = it->genParticleRef();
    bool isMatched = (genJpsi.isAvailable() && genJpsi->pdgId() == 443);  
    if (isMatched){
      genPatParticles->push_back(*(const_cast<reco::GenParticle*>(genJpsi.get())));
    }
  }

  if(collMuSel.size()==2) {
    reco::GenParticleRef genMu1 = collMuSel[0]->genParticleRef();
    reco::GenParticleRef genMu2 = collMuSel[1]->genParticleRef();
    bool isMuMatched = (genMu1.isAvailable() && genMu2.isAvailable() && 
			genMu1->pdgId()*genMu2->pdgId() == -169);
    if (isMuMatched) {
      genPatParticles->push_back(*(const_cast<reco::GenParticle*>(genMu1.get())));
      genPatParticles->push_back(*(const_cast<reco::GenParticle*>(genMu2.get())));

      unsigned int motherId = 0;
      if( genMu1->mother() != 0 ) {
	 motherId = genMu1->mother()->pdgId();
      }
      if(genMu1->pdgId()==13)
	genPair.push_back(GenMuonPair(genMu1.get()->p4(), genMu2.get()->p4(), motherId));
      else
	// genPair.push_back(std::make_pair(genMu2.get()->p4(),genMu1.get()->p4()) );
	genPair.push_back(GenMuonPair(genMu2.get()->p4(), genMu1.get()->p4(), motherId));

      plotter->fillGen(const_cast <reco::GenParticleCollection*> (genPatParticles), true);

      if (debug_>0) std::cout << "Found genParticles in PAT" << std::endl;
    }
    else {
      std::cout << "No recomuon selected so no access to generated info"<<std::endl;
      // Fill it in any case, otherwise it will not be in sync with the event number
      // genPair.push_back( std::make_pair( lorentzVector(0.,0.,0.,0.), lorentzVector(0.,0.,0.,0.) ) );    
      genPair.push_back( GenMuonPair(lorentzVector(0.,0.,0.,0.), lorentzVector(0.,0.,0.,0.), 0 ) );    
    }
  }
  else{
    std::cout << "No recomuon selected so no access to generated info"<<std::endl;
    // Fill it in any case, otherwise it will not be in sync with the event number
    // genPair.push_back( std::make_pair( lorentzVector(0.,0.,0.,0.), lorentzVector(0.,0.,0.,0.) ) );
    genPair.push_back( GenMuonPair(lorentzVector(0.,0.,0.,0.), lorentzVector(0.,0.,0.,0.), 0 ) );    
  }
  if(debug_>0) {
    std::cout << "genParticles:" << std::endl;
    // std::cout << genPair.back().first << " , " << genPair.back().second << std::endl;
    std::cout << genPair.back().mu1 << " , " << genPair.back().mu2 << std::endl;
  }
}

void MuScleFitMuonSelector::selectGenSimMuons(const edm::Event & event,
					      std::vector<GenMuonPair> & genPair,
					      std::vector<std::pair<lorentzVector,lorentzVector> > & simPair,
					      MuScleFitPlotter * plotter)
{  
  // Find and store in histograms the generated and simulated resonance and muons
  // ----------------------------------------------------------------------------
  edm::Handle<edm::HepMCProduct> evtMC;
  edm::Handle<reco::GenParticleCollection> genParticles;

  // Fill gen information only in the first loop
  bool ifHepMC=false;

  event.getByLabel( genParticlesName_, evtMC );
  event.getByLabel( genParticlesName_, genParticles );
  if( evtMC.isValid() ) {
    genPair.push_back( findGenMuFromRes(evtMC.product()) );
    plotter->fillGen(evtMC.product(), sherpa_);
    ifHepMC = true;
    if (debug_>0) std::cout << "Found hepMC" << std::endl;
  }
  else if( genParticles.isValid() ) {
    genPair.push_back( findGenMuFromRes(genParticles.product()) );
    plotter->fillGen(genParticles.product());
    if (debug_>0) std::cout << "Found genParticles" << std::endl;
  }
  else {
    std::cout<<"ERROR "<<"non generation info and speedup true!!!!!!!!!!!!"<<std::endl;
    // Fill it in any case, otherwise it will not be in sync with the event number
    // genPair.push_back( std::make_pair( lorentzVector(0.,0.,0.,0.), lorentzVector(0.,0.,0.,0.) ) );
    genPair.push_back( GenMuonPair( lorentzVector(0.,0.,0.,0.), lorentzVector(0.,0.,0.,0.), 0 ) );
  }
  if(debug_>0) {
    std::cout << "genParticles:" << std::endl;
    std::cout << genPair.back().mu1 << " , " << genPair.back().mu2 << std::endl;
  }
  selectSimulatedMuons(event, ifHepMC, evtMC, simPair, plotter);
}

void MuScleFitMuonSelector::selectSimulatedMuons(const edm::Event & event,
						 const bool ifHepMC, edm::Handle<edm::HepMCProduct> evtMC,
						 std::vector<std::pair<lorentzVector,lorentzVector> > & simPair,
						 MuScleFitPlotter * plotter)
{
  edm::Handle<edm::SimTrackContainer> simTracks;
  bool simTracksFound = false;
  event.getByLabel(simTracksCollectionName_, simTracks);
  if( simTracks.isValid() ) {
    plotter->fillSim(simTracks);
    if(ifHepMC) {
      simPair.push_back( findSimMuFromRes(evtMC, simTracks) );
      simTracksFound = true;
      plotter->fillGenSim(evtMC,simTracks);
    }
  }
  else {
    std::cout << "SimTracks not existent" << std::endl;
  }
  if( !simTracksFound ) {
    simPair.push_back( std::make_pair( lorentzVector(0.,0.,0.,0.), lorentzVector(0.,0.,0.,0.) ) );
  }
}

GenMuonPair MuScleFitMuonSelector::findGenMuFromRes( const edm::HepMCProduct* evtMC )
{
  const HepMC::GenEvent* Evt = evtMC->GetEvent();
  GenMuonPair muFromRes;
  //Loop on generated particles
  for (HepMC::GenEvent::particle_const_iterator part=Evt->particles_begin();
       part!=Evt->particles_end(); part++) {
    if (fabs((*part)->pdg_id())==13 && (*part)->status()==1) {
      bool fromRes = false;
      unsigned int motherPdgId = 0;
      for (HepMC::GenVertex::particle_iterator mother = (*part)->production_vertex()->particles_begin(HepMC::ancestors);
	   mother != (*part)->production_vertex()->particles_end(HepMC::ancestors); ++mother) {
        motherPdgId = (*mother)->pdg_id();

        // For sherpa the resonance is not saved. The muons from the resonance can be identified
        // by having as mother a muon of status 3.
        if( sherpa_ ) {
          if( motherPdgId == 13 && (*mother)->status() == 3 ) fromRes = true;
        }
        else {
          for( int ires = 0; ires < 6; ++ires ) {
	    if( motherPdgId == motherPdgIdArray[ires] && resfind_[ires] ) fromRes = true;
          }
        }
      }
      if(fromRes){
	if((*part)->pdg_id()==13) {
	  //   muFromRes.first = (*part)->momentum();
	  muFromRes.mu1 = (lorentzVector((*part)->momentum().px(),(*part)->momentum().py(),
					 (*part)->momentum().pz(),(*part)->momentum().e()));
	}
	else {
	  muFromRes.mu2 = (lorentzVector((*part)->momentum().px(),(*part)->momentum().py(),
					 (*part)->momentum().pz(),(*part)->momentum().e()));
	}
	muFromRes.motherId = motherPdgId; 
      }
    }
  }
  return muFromRes;
}

// std::pair<lorentzVector, lorentzVector> MuScleFitMuonSelector::findGenMuFromRes( const reco::GenParticleCollection* genParticles)
GenMuonPair MuScleFitMuonSelector::findGenMuFromRes( const reco::GenParticleCollection* genParticles)
{
  // std::pair<lorentzVector,lorentzVector> muFromRes;
  GenMuonPair muFromRes;

  //Loop on generated particles
  if( debug_>0 ) std::cout << "Starting loop on " << genParticles->size() << " genParticles" << std::endl;
  for( reco::GenParticleCollection::const_iterator part=genParticles->begin(); part!=genParticles->end(); ++part ) {
    if (debug_>0) std::cout<<"genParticle has pdgId = "<<fabs(part->pdgId())<<" and status = "<<part->status()<<std::endl;
    if (fabs(part->pdgId())==13){// && part->status()==3) {
      bool fromRes = false;
      unsigned int motherPdgId = part->mother()->pdgId();
      if( debug_>0 ) {
	std::cout << "Found a muon with mother: " << motherPdgId << std::endl;
      }
      for( int ires = 0; ires < 6; ++ires ) {
	// if( motherPdgId == motherPdgIdArray[ires] && resfind_[ires] ) fromRes = true; // changed by EM 2015.07.30
	// begin of comment  
	// the list of resonances motherPdgIdArray does not contain the photon (PdgId = 21) while ~1% of the 
	// mu+mu- events in the range [50,120] GeV has a photon as the mother.
	// It needs to be fixed without spoiling the logic of the selection of different resonances
	// e.g. mixing up onia etc.
	// end of comment
	if( ( motherPdgId == motherPdgIdArray[ires] && resfind_[ires] ) || motherPdgId == 21 ) fromRes = true;
      }
      if(fromRes){
	if (debug_>0) std::cout<<"fromRes = true, motherPdgId = "<<motherPdgId<<std::endl;
	const reco::Candidate* status3Muon = &(*part);
	const reco::Candidate* status1Muon = getStatus1Muon(status3Muon);
	if(part->pdgId()==13) {
	  if (status1Muon->p4().pt()!=0) muFromRes.mu1 = MuScleFitMuon(status1Muon->p4(),-1);
	  else muFromRes.mu1 = MuScleFitMuon(status3Muon->p4(),-1);
	  if( debug_>0 ) std::cout << "Found a genMuon - : " << muFromRes.mu1 << std::endl;
	  // 	  muFromRes.first = (lorentzVector(status1Muon->p4().px(),status1Muon->p4().py(),
	  // 					   status1Muon->p4().pz(),status1Muon->p4().e()));
	}
	else {
	  if (status1Muon->p4().pt()!=0) muFromRes.mu2 = MuScleFitMuon(status1Muon->p4(),+1);
	  else muFromRes.mu2 = MuScleFitMuon(status3Muon->p4(),+1);
	  if( debug_>0 ) std::cout << "Found a genMuon + : " << muFromRes.mu2 << std::endl;
	  // 	  muFromRes.second = (lorentzVector(status1Muon->p4().px(),status1Muon->p4().py(),
	  // 					    status1Muon->p4().pz(),status1Muon->p4().e()));
	}
	muFromRes.motherId = motherPdgId; 
      }
    }
  }
  return muFromRes;
}

std::pair<lorentzVector, lorentzVector> MuScleFitMuonSelector::findSimMuFromRes( const edm::Handle<edm::HepMCProduct> & evtMC,
										 const edm::Handle<edm::SimTrackContainer> & simTracks )
{
  //Loop on simulated tracks
  std::pair<lorentzVector, lorentzVector> simMuFromRes;
  for( edm::SimTrackContainer::const_iterator simTrack=simTracks->begin(); simTrack!=simTracks->end(); ++simTrack ) {
    //Chose muons
    if (fabs((*simTrack).type())==13) {
      //If tracks from IP than find mother
      if ((*simTrack).genpartIndex()>0) {
	HepMC::GenParticle* gp = evtMC->GetEvent()->barcode_to_particle ((*simTrack).genpartIndex());
        if( gp != 0 ) {

          for (HepMC::GenVertex::particle_iterator mother = gp->production_vertex()->particles_begin(HepMC::ancestors);
               mother!=gp->production_vertex()->particles_end(HepMC::ancestors); ++mother) {

            bool fromRes = false;
            unsigned int motherPdgId = (*mother)->pdg_id();
            for( int ires = 0; ires < 6; ++ires ) {
              if( ( motherPdgId == motherPdgIdArray[ires] && resfind_[ires] ) || motherPdgId == 21 ) fromRes = true;
            }
            if( fromRes ) {
              if(gp->pdg_id() == 13)
                simMuFromRes.first = lorentzVector(simTrack->momentum().px(),simTrack->momentum().py(),
                                                   simTrack->momentum().pz(),simTrack->momentum().e());
              else
                simMuFromRes.second = lorentzVector(simTrack->momentum().px(),simTrack->momentum().py(),
                                                    simTrack->momentum().pz(),simTrack->momentum().e());
            }
          }
        }
        // else LogDebug("MuScleFitUtils") << "WARNING: no matching genParticle found for simTrack" << std::endl;
      }
    }
  }
  return simMuFromRes;
}
