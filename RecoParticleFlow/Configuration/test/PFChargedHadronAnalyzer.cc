#include "RecoParticleFlow/Configuration/test/PFChargedHadronAnalyzer.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventSetup.h"


using namespace std;
using namespace edm;
using namespace reco;

PFChargedHadronAnalyzer::PFChargedHadronAnalyzer(const edm::ParameterSet& iConfig) {
  
  nCh = std::vector<unsigned int>(10,static_cast<unsigned int>(0));
  nEv = std::vector<unsigned int>(2,static_cast<unsigned int>(0));

  inputTagPFCandidates_ 
    = iConfig.getParameter<InputTag>("PFCandidates");

  inputTagPFSimParticles_ 
    = iConfig.getParameter<InputTag>("PFSimParticles");

  // Smallest track pt
  ptMin_ = iConfig.getParameter<double>("ptMin");

  // Smallest track p
  pMin_ = iConfig.getParameter<double>("pMin");

  // Smallest raw HCAL energy linked to the track
  hcalMin_ = iConfig.getParameter<double>("hcalMin");

  // Largest ECAL energy linked to the track to define a MIP
  ecalMax_ = iConfig.getParameter<double>("ecalMax");

  // Smallest number of pixel hits
  nPixMin_ = iConfig.getParameter<int>("nPixMin");

  // Smallest number of track hits in different eta ranges
  nHitMin_ = iConfig.getParameter< std::vector<int> > ("nHitMin");
  nEtaMin_ = iConfig.getParameter< std::vector<double> > ("nEtaMin");

  verbose_ = 
    iConfig.getUntrackedParameter<bool>("verbose",false);

  LogDebug("PFChargedHadronAnalyzer")
    <<" input collection : "<<inputTagPFCandidates_ ;
   

  // The root tuple
  outputfile_ = iConfig.getParameter<std::string>("rootOutputFile"); 
  tf1 = new TFile(outputfile_.c_str(), "RECREATE");  
  s = new TTree("s"," PFCalibration");

  s->Branch("true",&true_,"true/F");  
  s->Branch("p",&p_,"p/F");  
  s->Branch("ecal",&ecal_,"ecal/F");  
  s->Branch("hcal",&hcal_,"hcal/F");  
  s->Branch("eta",&eta_,"eta/F");  
  s->Branch("phi",&phi_,"phi/F");  

}



PFChargedHadronAnalyzer::~PFChargedHadronAnalyzer() { 

  std::cout << "Total number of events .............. " << nEv[0] << std::endl;
  std::cout << "Number of events with 1 Sim Particle  " << nEv[1] << std::endl;


  std::cout << "Number of PF candidates ............. " << nCh[0] << std::endl;
  std::cout << "Number of PF Charged Hadrons......... " << nCh[1] << std::endl;
  std::cout << " - With pt > " << ptMin_ << " GeV/c ................ " << nCh[2] << std::endl;
  std::cout << " - With E_HCAL > " << hcalMin_ << " GeV .............. " << nCh[3] << std::endl;
  std::cout << " - With only 1 track in the block ... " << nCh[4] << std::endl;
  std::cout << " - With p > " << pMin_ << " GeV/c ................. " << nCh[5] << std::endl;
  std::cout << " - With at least " << nPixMin_ << " pixel hits ....... " << nCh[6] << std::endl;
  std::cout << " - With more than "<< nHitMin_[0] << " track hits ..... " << nCh[7] << std::endl;
  std::cout << " - With E_ECAL < " << ecalMax_ << " GeV ............ " << nCh[8] << std::endl;

  tf1->cd();
  s->Write();
  tf1->Write();
  tf1->Close();  


}



void 
PFChargedHadronAnalyzer::beginRun(const edm::Run& run, 
				  const edm::EventSetup & es) { }


void 
PFChargedHadronAnalyzer::analyze(const Event& iEvent, 
				 const EventSetup& iSetup) {
  
  LogDebug("PFChargedHadronAnalyzer")<<"START event: "<<iEvent.id().event()
			 <<" in run "<<iEvent.id().run()<<endl;
  
  
  // get PFCandidates
  Handle<PFCandidateCollection> pfCandidates;
  iEvent.getByLabel(inputTagPFCandidates_, pfCandidates);
  
  Handle<PFSimParticleCollection> trueParticles;
  bool isSimu = iEvent.getByLabel(inputTagPFSimParticles_,trueParticles);

  if ( isSimu ) { 
    nEv[0]++;
    if ( (*trueParticles).size() != 1 ) return;
    nEv[1]++;
    
    
    // Check if there is a reconstructed track
    bool isCharged = false;
    for( CI ci  = pfCandidates->begin(); 
	 ci!=pfCandidates->end(); ++ci)  {
      const reco::PFCandidate& pfc = *ci;
      // std::cout << "Id = " << pfc.particleId() << std::endl;
      if ( pfc.particleId() < 4 ) { 
	isCharged = true;
	break;
      }
    }
    //std::cout << "isCharged ? " << isCharged << std::endl;
    
    // Case of no reconstructed tracks (and neutral single particles)
    if ( !isCharged || fabs((*trueParticles)[0].charge()) < 1E-10 ) { 
      reco::PFTrajectoryPoint::LayerType ecalEntrance = reco::PFTrajectoryPoint::ECALEntrance;
      const reco::PFTrajectoryPoint& tpatecal = ((*trueParticles)[0]).extrapolatedPoint( ecalEntrance );
      eta_ = tpatecal.positionREP().Eta();
      if ( fabs(eta_) < 1E-10 ) return; 
      phi_ = tpatecal.positionREP().Phi();
      true_ = std::sqrt(tpatecal.momentum().Vect().Mag2());
      p_ = 0.;
      
      ecal_ = 0.;
      hcal_ = 0.;
      for( CI ci  = pfCandidates->begin(); 
	   ci!=pfCandidates->end(); ++ci)  {
	const reco::PFCandidate& pfc = *ci;
	double deta = eta_ - pfc.eta();
	double dphi = phi_ - pfc.phi();
	double dR = std::sqrt(deta*deta+dphi*dphi);
	if ( pfc.particleId() == 4 && dR < 0.04 ) ecal_ += pfc.rawEcalEnergy();
	if ( pfc.particleId() == 5 && dR < 0.2 ) hcal_ += pfc.rawHcalEnergy();
      }
      
      s->Fill();
      
      return;
    }
    
  }
  
  // Case of a reconstructed track.
  // Loop on pfCandidates
  for( CI ci  = pfCandidates->begin(); 
       ci!=pfCandidates->end(); ++ci)  {

    // The pf candidate
    const reco::PFCandidate& pfc = *ci;
    nCh[0]++;

    // Only charged hadrons (no PF muons, no PF electrons)
    if ( pfc.particleId() != 1 ) continue;
    nCh[1]++;

    // Charged hadron minimum pt (the track pt, to an excellent approximation)
    if ( pfc.pt() < ptMin_ ) continue;
    nCh[2]++;

    // At least 1 GeV in HCAL
    double ecalRaw = pfc.rawEcalEnergy();
    double hcalRaw = pfc.rawHcalEnergy();
    if ( ecalRaw + hcalRaw < hcalMin_ ) continue;
    nCh[3]++;

    // Find the corresponding PF block elements
    const PFCandidate::ElementsInBlocks& theElements = pfc.elementsInBlocks();
    if( theElements.empty() ) continue;
    const reco::PFBlockRef blockRef = theElements[0].first;
    PFBlock::LinkData linkData =  blockRef->linkData();
    const edm::OwnVector<reco::PFBlockElement>& elements = blockRef->elements();
    // Check that there is only one track in the block.
    unsigned int nTracks = 0;
    unsigned iTrack = 999;
    for(unsigned iEle=0; iEle<elements.size(); iEle++) {
      // Find the tracks in the block
      PFBlockElement::Type type = elements[iEle].type();
      switch( type ) {
      case PFBlockElement::TRACK:
	iTrack = iEle;
	nTracks++;
	break;
      default:
	continue;
      }
    }
    if ( nTracks != 1 ) continue;
    nCh[4]++;

    // Characteristics of the track
    const reco::PFBlockElementTrack& et =
      dynamic_cast<const reco::PFBlockElementTrack &>( elements[iTrack] );
    double p = et.trackRef()->p();  
    double pt = et.trackRef()->pt(); 
    double eta = et.trackRef()->eta();
    // double phi = et.trackRef()->phi();
    
    // A minimum p and pt
    if ( p < pMin_ || pt < ptMin_ ) continue;
    nCh[5]++;
    
    // Count the number of valid hits (first three iteration only)
    //unsigned int nHits = et.trackRef()->found();
    unsigned int tobN = 0;
    unsigned int tecN = 0;
    unsigned int tibN = 0;
    unsigned int tidN = 0;
    unsigned int pxbN = 0;
    unsigned int pxdN = 0;
    const reco::HitPattern& hp = et.trackRef()->hitPattern();
    switch ( et.trackRef()->algo() ) {
    case TrackBase::initialStep:
    case TrackBase::lowPtTripletStep:
    case TrackBase::pixelPairStep:
    case TrackBase::detachedTripletStep:
      tobN += hp.numberOfValidStripTOBHits();
      tecN += hp.numberOfValidStripTECHits();
      tibN += hp.numberOfValidStripTIBHits();
      tidN += hp.numberOfValidStripTIDHits();
      pxbN += hp.numberOfValidPixelBarrelHits(); 
      pxdN += hp.numberOfValidPixelEndcapHits(); 
      break;
    case TrackBase::mixedTripletStep:
    case TrackBase::pixelLessStep:
    case TrackBase::tobTecStep:
    case TrackBase::jetCoreRegionalStep:
    case TrackBase::muonSeededStepInOut:
    case TrackBase::muonSeededStepOutIn:
    default:
      break;
    }
    int inner = pxbN+pxdN;
    int outer = tibN+tobN+tidN+tecN;
    
    // Number of pixel hits
    if ( inner < nPixMin_ ) continue;
    nCh[6]++;
    
    // Number of tracker hits (eta-dependent cut)
    bool trackerHitOK = false;
    double etaMin = 0.;
    for ( unsigned int ieta=0; ieta<nEtaMin_.size(); ++ieta ) { 
      if ( fabs(eta) < etaMin ) break;
      double etaMax = nEtaMin_[ieta];
      trackerHitOK = 
	fabs(eta)>etaMin && fabs(eta)<etaMax && inner+outer>nHitMin_[ieta]; 
      if ( trackerHitOK ) break;
      etaMin = etaMax;
    }
    if ( !trackerHitOK ) continue;
    nCh[7]++;
    
    // Selects only ECAL MIPs
    if ( ecalRaw > ecalMax_ ) continue;
    nCh[8]++;

    
    /*
    std::cout << "Selected track : p = " << p << "; pt = " << pt 
	      << "; eta/phi = " << eta << " " << phi << std::endl
	      << "PF Ch. hadron  : p = " << pfc.p() << "; pt = " << pfc.pt()
	      << "; eta/phi = " << pfc.eta() << " " << pfc.phi() << std::endl
	      << "Nb of hits (pix/tot) " << inner << " " << inner+outer << std::endl;
    std::cout << "Raw Ecal and HCAL energies : ECAL = " << ecalRaw 
	      << "; HCAL = " << hcalRaw << std::endl;
    */

    // Fill the root-tuple
    p_ = p;
    ecal_ = ecalRaw;
    hcal_ = hcalRaw;
    reco::PFTrajectoryPoint::LayerType ecalEntrance = reco::PFTrajectoryPoint::ECALEntrance;
    const reco::PFTrajectoryPoint& tpatecal = ((*trueParticles)[0]).extrapolatedPoint( ecalEntrance );
    eta_ = tpatecal.positionREP().Eta();
    phi_ = tpatecal.positionREP().Phi();
    true_ = std::sqrt(tpatecal.momentum().Vect().Mag2());
    s->Fill();

  }
}

DEFINE_FWK_MODULE(PFChargedHadronAnalyzer);
