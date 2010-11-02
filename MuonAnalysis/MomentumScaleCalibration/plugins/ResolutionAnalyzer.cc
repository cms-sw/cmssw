#ifndef RESOLUTIONANALYZER_CC
#define RESOLUTIONANALYZER_CC

#include "ResolutionAnalyzer.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/Functions.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
ResolutionAnalyzer::ResolutionAnalyzer(const edm::ParameterSet& iConfig) :
  theMuonLabel_( iConfig.getParameter<edm::InputTag>( "MuonLabel" ) ),
  theMuonType_( iConfig.getParameter<int>( "MuonType" ) ),
  theRootFileName_( iConfig.getUntrackedParameter<string>("OutputFileName") ),
  theCovariancesRootFileName_( iConfig.getUntrackedParameter<string>("InputFileName") ),
  debug_( iConfig.getUntrackedParameter<bool>( "Debug" ) ),
  readCovariances_( iConfig.getUntrackedParameter<bool>( "ReadCovariances" ) )
{
  //now do what ever initialization is needed

  // Initial parameters values
  // -------------------------
  int resolFitType = iConfig.getParameter<int>("ResolFitType");
  MuScleFitUtils::ResolFitType = resolFitType;
  // MuScleFitUtils::resolutionFunction = resolutionFunctionArray[resolFitType];
  MuScleFitUtils::resolutionFunction = resolutionFunctionService( resolFitType );
  // MuScleFitUtils::resolutionFunctionForVec = resolutionFunctionArrayForVec[resolFitType];
  MuScleFitUtils::resolutionFunctionForVec = resolutionFunctionVecService( resolFitType );

  MuScleFitUtils::parResol = iConfig.getParameter<vector<double> >("parResol");

  MuScleFitUtils::resfind = iConfig.getParameter<vector<int> >("ResFind");

  outputFile_ = new TFile(theRootFileName_.c_str(), "RECREATE");
  outputFile_->cd();
  fillHistoMap();

  eventCounter_ = 0;
  resonance_ = iConfig.getUntrackedParameter<bool>( "Resonance" );
}


ResolutionAnalyzer::~ResolutionAnalyzer()
{
  outputFile_->cd();
  writeHistoMap();
  outputFile_->Close();
  cout << "Total analyzed events = " << eventCounter_ << endl;
}


//
// member functions
//

// ------------ method called to for each event  ------------
void ResolutionAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  ++eventCounter_;
  if ( eventCounter_%100 == 0 ) {
    std::cout << "Event number " << eventCounter_ << std::endl;
  }

  Handle<HepMCProduct> evtMC;
  try {
    iEvent.getByLabel ("source", evtMC);
  } catch (...) { 
    cout << "HepMCProduct non existent" << endl;
  }

  Handle<GenParticleCollection> genParticles; 
  try {
    iEvent.getByLabel ("genParticles", genParticles);
    if (debug_>0) cout << "Found genParticles" << endl;
  } catch (...) {
    cout << "GenParticles non existent" << endl;
  }

  Handle<SimTrackContainer> simTracks;
  try {
    iEvent.getByLabel ("g4SimHits",simTracks);
  } catch (...) {
    cout << "SimTracks not existent, not using them" << endl;
  }

  // Take the reco-muons, depending on the type selected in the cfg
  // --------------------------------------------------------------

  vector<reco::LeafCandidate> muons;

  if (theMuonType_==1) { // GlobalMuons
    Handle<reco::MuonCollection> glbMuons;
    iEvent.getByLabel (theMuonLabel_, glbMuons);
    muons = fillMuonCollection(*glbMuons);
  }
  else if (theMuonType_==2) { // StandaloneMuons
    Handle<reco::TrackCollection> saMuons;
    iEvent.getByLabel (theMuonLabel_, saMuons);
    muons = fillMuonCollection(*saMuons);
  }
  else if (theMuonType_==3) { // Tracker tracks
    Handle<reco::TrackCollection> tracks;
    iEvent.getByLabel (theMuonLabel_, tracks);
    muons = fillMuonCollection(*tracks);
  }

  if ( resonance_ ) {

    // Find the best reconstructed resonance
    // -------------------------------------
    reco::Particle::LorentzVector recMu1 = reco::Particle::LorentzVector(0,0,0,0);
    reco::Particle::LorentzVector recMu2 = reco::Particle::LorentzVector(0,0,0,0);
    pair<lorentzVector,lorentzVector> recoRes = MuScleFitUtils::findBestRecoRes(muons);
    if (MuScleFitUtils::ResFound) {
      if (debug_>0) {
        cout <<setprecision(9)<< "Pt after findbestrecores: " << (recoRes.first).Pt() << " " 
             << (recoRes.second).Pt() << endl;
        cout << "recMu1 = " << recMu1 << endl;
        cout << "recMu2 = " << recMu2 << endl;
      }
      recMu1 = recoRes.first;
      recMu2 = recoRes.second;
      if (debug_>0) {
        cout << "after recMu1 = " << recMu1 << endl;
        cout << "after recMu2 = " << recMu2 << endl;
        cout << "mu1.pt = " << recMu1.Pt() << endl;
        cout << "mu2.pt = " << recMu2.Pt() << endl;
      }
    }

    // Histograms with genParticles characteristics
    // --------------------------------------------

    //first is always mu-, second is always mu+
    pair <reco::Particle::LorentzVector, reco::Particle::LorentzVector> genMu = MuScleFitUtils::findGenMuFromRes(evtMC);
  
    reco::Particle::LorentzVector genMother( genMu.first + genMu.second );
  
    mapHisto_["GenMother"]->Fill( genMother );
    mapHisto_["DeltaGenMotherMuons"]->Fill( genMu.first, genMu.second );
    mapHisto_["GenMotherMuons"]->Fill( genMu.first );
    mapHisto_["GenMotherMuons"]->Fill( genMu.second );
  
    // Match the reco muons with the gen and sim tracks
    // ------------------------------------------------
    if(checkDeltaR(genMu.first,recMu1)){
      mapHisto_["PtResolutionGenVSMu"]->Fill(recMu1,(-genMu.first.Pt()+recMu1.Pt())/genMu.first.Pt(),-1);
      mapHisto_["ThetaResolutionGenVSMu"]->Fill(recMu1,(-genMu.first.Theta()+recMu1.Theta()),-1);
      mapHisto_["CotgThetaResolutionGenVSMu"]->Fill(recMu1,(-cos(genMu.first.Theta())/sin(genMu.first.Theta())
                                                            +cos(recMu1.Theta())/sin(recMu1.Theta())),-1);
      mapHisto_["EtaResolutionGenVSMu"]->Fill(recMu1,(-genMu.first.Eta()+recMu1.Eta()),-1);
      // mapHisto_["PhiResolutionGenVSMu"]->Fill(recMu1,(-genMu.first.Phi()+recMu1.Phi()),-1);
      mapHisto_["PhiResolutionGenVSMu"]->Fill(recMu1,MuScleFitUtils::deltaPhiNoFabs(recMu1.Phi(), genMu.first.Phi()),-1);
      recoPtVsgenPt_->Fill(genMu.first.Pt(), recMu1.Pt());
      deltaPtOverPt_->Fill( (recMu1.Pt() - genMu.first.Pt())/genMu.first.Pt() );
      if( fabs(recMu1.Eta()) > 1 && fabs(recMu1.Eta()) < 1.2 ) {
        recoPtVsgenPtEta12_->Fill(genMu.first.Pt(), recMu1.Pt());
        deltaPtOverPtForEta12_->Fill( (recMu1.Pt() - genMu.first.Pt())/genMu.first.Pt() );
      }
    }
    if(checkDeltaR(genMu.second,recMu2)){
      mapHisto_["PtResolutionGenVSMu"]->Fill(recMu2,(-genMu.second.Pt()+recMu2.Pt())/genMu.second.Pt(),+1);
      mapHisto_["ThetaResolutionGenVSMu"]->Fill(recMu2,(-genMu.second.Theta()+recMu2.Theta()),+1);
      mapHisto_["CotgThetaResolutionGenVSMu"]->Fill(recMu2,(-cos(genMu.second.Theta())/sin(genMu.second.Theta())
                                                            +cos(recMu2.Theta())/sin(recMu2.Theta())),+1);
      mapHisto_["EtaResolutionGenVSMu"]->Fill(recMu2,(-genMu.second.Eta()+recMu2.Eta()),+1);
      // mapHisto_["PhiResolutionGenVSMu"]->Fill(recMu2,(-genMu.second.Phi()+recMu2.Phi()),+1);
      mapHisto_["PhiResolutionGenVSMu"]->Fill(recMu2,MuScleFitUtils::deltaPhiNoFabs(recMu2.Phi(), genMu.second.Phi()),+1);
      recoPtVsgenPt_->Fill(genMu.second.Pt(), recMu2.Pt());
      deltaPtOverPt_->Fill( (recMu2.Pt() - genMu.second.Pt())/genMu.second.Pt() );
      if( fabs(recMu2.Eta()) > 1 && fabs(recMu2.Eta()) < 1.2 ) {
        recoPtVsgenPtEta12_->Fill(genMu.second.Pt(), recMu2.Pt());
        deltaPtOverPtForEta12_->Fill( (recMu2.Pt() - genMu.second.Pt())/genMu.second.Pt() );
      }
    }
  
    if( simTracks.isValid() ) {
      pair <reco::Particle::LorentzVector, reco::Particle::LorentzVector> simMu = MuScleFitUtils::findSimMuFromRes(evtMC,simTracks);
      reco::Particle::LorentzVector simResonance( simMu.first+simMu.second );
      mapHisto_["SimResonance"]->Fill( simResonance );
      mapHisto_["DeltaSimResonanceMuons"]->Fill( simMu.first, simMu.second );
      mapHisto_["SimResonanceMuons"]->Fill( simMu.first );
      mapHisto_["SimResonanceMuons"]->Fill( simMu.second );
  
      //first is always mu-, second is always mu+
      if(checkDeltaR(simMu.first,recMu1)){
        mapHisto_["PtResolutionSimVSMu"]->Fill(recMu1,(-simMu.first.Pt()+recMu1.Pt())/simMu.first.Pt(),-1);
        mapHisto_["ThetaResolutionSimVSMu"]->Fill(recMu1,(-simMu.first.Theta()+recMu1.Theta()),-1);
        mapHisto_["CotgThetaResolutionSimVSMu"]->Fill(recMu1,(-cos(simMu.first.Theta())/sin(simMu.first.Theta())
                                                              +cos(recMu1.Theta())/sin(recMu1.Theta())),-1);
        mapHisto_["EtaResolutionSimVSMu"]->Fill(recMu1,(-simMu.first.Eta()+recMu1.Eta()),-1);
        // mapHisto_["PhiResolutionSimVSMu"]->Fill(recMu1,(-simMu.first.Phi()+recMu1.Phi()),-1);
        mapHisto_["PhiResolutionSimVSMu"]->Fill(recMu1,MuScleFitUtils::deltaPhiNoFabs(recMu1.Phi(), simMu.first.Phi()),-1);
      }
      if(checkDeltaR(simMu.second,recMu2)){
        mapHisto_["PtResolutionSimVSMu"]->Fill(recMu2,(-simMu.second.Pt()+recMu2.Pt())/simMu.first.Pt(),+1);
        mapHisto_["ThetaResolutionSimVSMu"]->Fill(recMu2,(-simMu.second.Theta()+recMu2.Theta()),+1);
        mapHisto_["CotgThetaResolutionSimVSMu"]->Fill(recMu2,(-cos(simMu.second.Theta())/sin(simMu.second.Theta())
                                                              +cos(recMu2.Theta())/sin(recMu2.Theta())),+1);
        mapHisto_["EtaResolutionSimVSMu"]->Fill(recMu2,(-simMu.second.Eta()+recMu2.Eta()),+1);
        // mapHisto_["PhiResolutionSimVSMu"]->Fill(recMu2,(-simMu.second.Phi()+recMu2.Phi()),+1);
        mapHisto_["PhiResolutionSimVSMu"]->Fill(recMu2,MuScleFitUtils::deltaPhiNoFabs(recMu2.Phi(), simMu.second.Phi()),+1);
      }
    }
    // Fill the mass resolution histograms
    // -----------------------------------
    // check if the recoMuons match the genMuons
    // if( MuScleFitUtils::ResFound && checkDeltaR(simMu.first,recMu1) && checkDeltaR(simMu.second,recMu2) ) {
    if( MuScleFitUtils::ResFound && checkDeltaR(genMu.first,recMu1) && checkDeltaR(genMu.second,recMu2) ) {

      double recoMass = (recMu1+recMu2).mass();
      double genMass = (genMu.first + genMu.second).mass();
      // first is always mu-, second is always mu+
      mapHisto_["MassResolution"]->Fill(recMu1, -1, genMu.first, recMu2, +1, genMu.second, recoMass, genMass);
  
      // Fill the reconstructed resonance
      reco::Particle::LorentzVector recoResonance( recMu1+recMu2 );
      mapHisto_["RecoResonance"]->Fill( recoResonance );
      mapHisto_["DeltaRecoResonanceMuons"]->Fill( recMu1, recMu2 );
      mapHisto_["RecoResonanceMuons"]->Fill( recMu1 );
      mapHisto_["RecoResonanceMuons"]->Fill( recMu2 );
  
      // Fill the mass resolution (computed from MC), we use the covariance class to compute the variance
      if( genMass != 0 ) {
        // double diffMass = (recoMass - genMass)/genMass;
        double diffMass = recoMass - genMass;
        // Fill if for both muons
        double pt1 = recMu1.pt();
        double eta1 = recMu1.eta();
        double pt2 = recMu2.pt();
        double eta2 = recMu2.eta();
        // This is to avoid nan
        if( diffMass == diffMass ) {
          massResolutionVsPtEta_->Fill(pt1, eta1, diffMass, diffMass);
          massResolutionVsPtEta_->Fill(pt2, eta2, diffMass, diffMass);
        }
        else {
          cout << "Error, there is a nan: recoMass = " << recoMass << ", genMass = " << genMass << endl;
        }
        // Fill with mass resolution from resolution function
        double massRes = MuScleFitUtils::massResolution(recMu1, recMu2, MuScleFitUtils::parResol);
        // The value given by massRes is already divided by the mass, since the derivative functions have mass at the denominator.
        // We alos take the squared value, since var = sigma^2.
        mapHisto_["hFunctionResolMass"]->Fill( recMu1, pow(massRes,2), -1 );
        mapHisto_["hFunctionResolMass"]->Fill( recMu2, pow(massRes,2), +1 );
      }
  
      // Fill resolution functions for the muons (fill the squared value to make it comparable with the variance)
      mapHisto_["hFunctionResolPt"]->Fill( recMu1, MuScleFitUtils::resolutionFunctionForVec->sigmaPt(recMu1.Pt(), recMu1.Eta(), MuScleFitUtils::parResol), -1 );
      mapHisto_["hFunctionResolCotgTheta"]->Fill( recMu1, MuScleFitUtils::resolutionFunctionForVec->sigmaCotgTh(recMu1.Pt(), recMu1.Eta(), MuScleFitUtils::parResol), -1 );
      mapHisto_["hFunctionResolPhi"]->Fill( recMu1, MuScleFitUtils::resolutionFunctionForVec->sigmaPhi(recMu1.Pt(), recMu1.Eta(), MuScleFitUtils::parResol), -1 );
      mapHisto_["hFunctionResolPt"]->Fill( recMu2, MuScleFitUtils::resolutionFunctionForVec->sigmaPt(recMu2.Pt(), recMu2.Eta(), MuScleFitUtils::parResol), +1 );
      mapHisto_["hFunctionResolCotgTheta"]->Fill( recMu2, MuScleFitUtils::resolutionFunctionForVec->sigmaCotgTh(recMu2.Pt(), recMu2.Eta(), MuScleFitUtils::parResol), +1 );
      mapHisto_["hFunctionResolPhi"]->Fill( recMu2, MuScleFitUtils::resolutionFunctionForVec->sigmaPhi(recMu2.Pt(), recMu2.Eta(), MuScleFitUtils::parResol), +1 );

      if( readCovariances_ ) {
        // Compute mass error terms
        // ------------------------
        double mass   = (recMu1+recMu2).mass();
        double pt1    = recMu1.Pt();
        double phi1   = recMu1.Phi();
        double eta1   = recMu1.Eta();
        double theta1 = 2*atan(exp(-eta1));
        double pt2    = recMu2.Pt();
        double phi2   = recMu2.Phi();
        double eta2   = recMu2.Eta();
        double theta2 = 2*atan(exp(-eta2));
        // Derivatives
        double mMu2 = MuScleFitUtils::mMu2;
        double dmdpt1  = (pt1/pow(sin(theta1),2)*sqrt((pow(pt2/sin(theta2),2)+mMu2)/(pow(pt1/sin(theta1),2)+mMu2))- 
                          pt2*(cos(phi1-phi2)+cos(theta1)*cos(theta2)/(sin(theta1)*sin(theta2))))/mass;
        double dmdpt2  = (pt2/pow(sin(theta2),2)*sqrt((pow(pt1/sin(theta1),2)+mMu2)/(pow(pt2/sin(theta2),2)+mMu2))- 
                          pt1*(cos(phi2-phi1)+cos(theta2)*cos(theta1)/(sin(theta2)*sin(theta1))))/mass;
        double dmdphi1 = pt1*pt2/mass*sin(phi1-phi2);
        double dmdphi2 = pt2*pt1/mass*sin(phi2-phi1);
        double dmdcotgth1 = (pt1*pt1*cos(theta1)/sin(theta1)*
                             sqrt((pow(pt2/sin(theta2),2)+mMu2)/(pow(pt1/sin(theta1),2)+mMu2)) - 
                             pt1*pt2*cos(theta2)/sin(theta2))/mass;
        double dmdcotgth2 = (pt2*pt2*cos(theta2)/sin(theta2)*
                             sqrt((pow(pt1/sin(theta1),2)+mMu2)/(pow(pt2/sin(theta2),2)+mMu2)) - 
                             pt2*pt1*cos(theta1)/sin(theta1))/mass;

        // Multiplied by the pt here
        // -------------------------
        double dmdpt[2] = {dmdpt1*recMu1.Pt(), dmdpt2*recMu2.Pt()};
        double dmdphi[2] = {dmdphi1, dmdphi2};
        double dmdcotgth[2] = {dmdcotgth1, dmdcotgth2};

        // Evaluate the single terms in the mass error expression

        reco::Particle::LorentzVector * recMu[2] = { &recMu1, &recMu2 };
        int charge[2] = { -1, +1 };

        double fullMassRes = 0.;
        double massRes1 = 0.;
        double massRes2 = 0.;
        double massRes3 = 0.;
        double massRes4 = 0.;
        double massRes5 = 0.;
        double massRes6 = 0.;

        for( int i=0; i<2; ++i ) {

          double ptVariance = mapHisto_["ReadCovariances"]->Get(*(recMu[i]), "Pt");
          double cotgThetaVariance = mapHisto_["ReadCovariances"]->Get(*(recMu[i]), "CotgTheta");
          double phiVariance = mapHisto_["ReadCovariances"]->Get(*(recMu[i]), "Phi");
          double pt_cotgTheta = mapHisto_["ReadCovariances"]->Get(*(recMu[i]), "Pt-CotgTheta");
          double pt_phi = mapHisto_["ReadCovariances"]->Get(*(recMu[i]), "Pt-Phi");
          double cotgTheta_phi = mapHisto_["ReadCovariances"]->Get(*(recMu[i]), "CotgTheta-Phi");

          double pt1_pt2 = mapHisto_["ReadCovariances"]->Get(*(recMu[i]), "Pt1-Pt2");
          double cotgTheta1_cotgTheta2 = mapHisto_["ReadCovariances"]->Get(*(recMu[i]), "CotgTheta1-CotgTheta2");
          double phi1_phi2 = mapHisto_["ReadCovariances"]->Get(*(recMu[i]), "Phi1-Phi2");
          double pt12_cotgTheta21 = mapHisto_["ReadCovariances"]->Get(*(recMu[i]), "Pt12-CotgTheta21");
          double pt12_phi21 = mapHisto_["ReadCovariances"]->Get(*(recMu[i]), "Pt12-Phi21");
          double cotgTheta12_phi21 = mapHisto_["ReadCovariances"]->Get(*(recMu[i]), "CotgTheta12-Phi21");
  
          // ATTENTION: Pt covariance terms are multiplied by Pt, since DeltaPt/Pt was used to compute them
          mapHisto_["MassResolutionPt"]->Fill( *(recMu[i]), ptVariance*pow(dmdpt[i],2), charge[i] );
          mapHisto_["MassResolutionCotgTheta"]->Fill( *(recMu[i]), cotgThetaVariance*pow(dmdcotgth[i],2), charge[i] );
          mapHisto_["MassResolutionPhi"]->Fill( *(recMu[i]), phiVariance*pow(dmdphi[i],2), charge[i] );
          mapHisto_["MassResolutionPt-CotgTheta"]->Fill( *(recMu[i]), 2*pt_cotgTheta*dmdpt[i]*dmdcotgth[i], charge[i] );
          mapHisto_["MassResolutionPt-Phi"]->Fill( *(recMu[i]), 2*pt_phi*dmdpt[i]*dmdphi[i], charge[i] );
          mapHisto_["MassResolutionCotgTheta-Phi"]->Fill( *(recMu[i]), 2*cotgTheta_phi*dmdcotgth[i]*dmdphi[i], charge[i] );

          mapHisto_["MassResolutionPt1-Pt2"]->Fill( *(recMu[i]), pt1_pt2*dmdpt[0]*dmdpt[1], charge[i] );
          mapHisto_["MassResolutionCotgTheta1-CotgTheta2"]->Fill( *(recMu[i]), cotgTheta1_cotgTheta2*dmdcotgth[0]*dmdcotgth[1], charge[i] );
          mapHisto_["MassResolutionPhi1-Phi2"]->Fill( *(recMu[i]), phi1_phi2*dmdphi[0]*dmdphi[1], charge[i] );
          // This must be filled for both configurations: 12 and 21
          mapHisto_["MassResolutionPt12-CotgTheta21"]->Fill( *(recMu[i]), pt12_cotgTheta21*dmdpt[0]*dmdcotgth[1], charge[i] );
          mapHisto_["MassResolutionPt12-CotgTheta21"]->Fill( *(recMu[i]), pt12_cotgTheta21*dmdpt[1]*dmdcotgth[0], charge[i] );
          mapHisto_["MassResolutionPt12-Phi21"]->Fill( *(recMu[i]), pt12_phi21*dmdpt[0]*dmdphi[1], charge[i] );
          mapHisto_["MassResolutionPt12-Phi21"]->Fill( *(recMu[i]), pt12_phi21*dmdpt[1]*dmdphi[0], charge[i] );
          mapHisto_["MassResolutionCotgTheta12-Phi21"]->Fill( *(recMu[i]), cotgTheta12_phi21*dmdcotgth[0]*dmdphi[1], charge[i] );
          mapHisto_["MassResolutionCotgTheta12-Phi21"]->Fill( *(recMu[i]), cotgTheta12_phi21*dmdcotgth[1]*dmdphi[0], charge[i] );

          // Sigmas for comparison
          mapHisto_["sigmaPtFromVariance"]->Fill( *(recMu[i]), sqrt(ptVariance), charge[i] );
          mapHisto_["sigmaCotgThetaFromVariance"]->Fill( *(recMu[i]), sqrt(cotgThetaVariance), charge[i] );
          mapHisto_["sigmaPhiFromVariance"]->Fill( *(recMu[i]), sqrt(phiVariance), charge[i] );

          // Pt term from function
          mapHisto_["MassResolutionPtFromFunction"]->Fill( *(recMu[i]), ( MuScleFitUtils::resolutionFunctionForVec->sigmaPt((recMu[i])->Pt(), (recMu[i])->Eta(), MuScleFitUtils::parResol) )*pow(dmdpt[i],2), charge[i] );

          if( i == 0 ) {
            fullMassRes =
              ptVariance*pow(dmdpt[i],2) +
              cotgThetaVariance*pow(dmdcotgth[i],2) +
              phiVariance*pow(dmdphi[i],2) +

              // These are worth twice the others since there are: pt1-phi1, phi1-pt1, pt2-phi2, phi2-pt2
              2*pt_cotgTheta*dmdpt[i]*dmdcotgth[i] +
              2*pt_phi*dmdpt[i]*dmdphi[i] +
              2*cotgTheta_phi*dmdcotgth[i]*dmdphi[i] +

              pt1_pt2*dmdpt[0]*dmdpt[1] +
              cotgTheta1_cotgTheta2*dmdcotgth[0]*dmdcotgth[1] +
              phi1_phi2*dmdphi[0]*dmdphi[1] +

              // These are filled twice, because of the two combinations
              pt12_cotgTheta21*dmdpt[0]*dmdcotgth[1] +
              pt12_cotgTheta21*dmdpt[1]*dmdcotgth[0] +
              pt12_phi21*dmdpt[0]*dmdphi[1] +
              pt12_phi21*dmdpt[1]*dmdphi[0] +
              cotgTheta12_phi21*dmdcotgth[0]*dmdphi[1] +
              cotgTheta12_phi21*dmdcotgth[1]*dmdphi[0];

            massRes1 = ptVariance*pow(dmdpt[i],2);
            massRes2 = massRes1 + pt1_pt2*dmdpt[0]*dmdpt[1];
            massRes3 = massRes2 + cotgThetaVariance*pow(dmdcotgth[i],2) +
              phiVariance*pow(dmdphi[i],2);
            massRes4 = massRes3 + 2*pt_cotgTheta*dmdpt[i]*dmdcotgth[i] +
              2*pt_phi*dmdpt[i]*dmdphi[i] +
              2*cotgTheta_phi*dmdcotgth[i]*dmdphi[i];
            massRes5 = massRes4 + cotgTheta1_cotgTheta2*dmdcotgth[0]*dmdcotgth[1] +
              phi1_phi2*dmdphi[0]*dmdphi[1];
            massRes6 = massRes5 + pt12_cotgTheta21*dmdpt[0]*dmdcotgth[1] +
              pt12_cotgTheta21*dmdpt[1]*dmdcotgth[0] +
              pt12_phi21*dmdpt[0]*dmdphi[1] +
              pt12_phi21*dmdpt[1]*dmdphi[0] +
              cotgTheta12_phi21*dmdcotgth[0]*dmdphi[1] +
              cotgTheta12_phi21*dmdcotgth[1]*dmdphi[0];
          }
          else {
            fullMassRes +=
              ptVariance*pow(dmdpt[i],2) +
              cotgThetaVariance*pow(dmdcotgth[i],2) +
              phiVariance*pow(dmdphi[i],2) +

              // These are worth twice the others since there are: pt1-phi1, phi1-pt1, pt2-phi2, phi2-pt2
              2*pt_cotgTheta*dmdpt[i]*dmdcotgth[i] +
              2*pt_phi*dmdpt[i]*dmdphi[i] +
              2*cotgTheta_phi*dmdcotgth[i]*dmdphi[i] +

              pt1_pt2*dmdpt[0]*dmdpt[1] +
              cotgTheta1_cotgTheta2*dmdcotgth[0]*dmdcotgth[1] +
              phi1_phi2*dmdphi[0]*dmdphi[1] +

              // These are filled twice, because of the two combinations
              pt12_cotgTheta21*dmdpt[0]*dmdcotgth[1] +
              pt12_cotgTheta21*dmdpt[1]*dmdcotgth[0] +
              pt12_phi21*dmdpt[0]*dmdphi[1] +
              pt12_phi21*dmdpt[1]*dmdphi[0] +
              cotgTheta12_phi21*dmdcotgth[0]*dmdphi[1] +
              cotgTheta12_phi21*dmdcotgth[1]*dmdphi[0];

            massRes1 += ptVariance*pow(dmdpt[i],2);
            massRes2 += ptVariance*pow(dmdpt[i],2) +
              pt1_pt2*dmdpt[0]*dmdpt[1];
            massRes3 += ptVariance*pow(dmdpt[i],2) +
              pt1_pt2*dmdpt[0]*dmdpt[1] +
              cotgThetaVariance*pow(dmdcotgth[i],2) +
              phiVariance*pow(dmdphi[i],2);
            massRes4 += ptVariance*pow(dmdpt[i],2) +
              pt1_pt2*dmdpt[0]*dmdpt[1] +
              cotgThetaVariance*pow(dmdcotgth[i],2) +
              phiVariance*pow(dmdphi[i],2) +
              2*pt_cotgTheta*dmdpt[i]*dmdcotgth[i] +
              2*pt_phi*dmdpt[i]*dmdphi[i] +
              2*cotgTheta_phi*dmdcotgth[i]*dmdphi[i];
            massRes5 += ptVariance*pow(dmdpt[i],2) +
              pt1_pt2*dmdpt[0]*dmdpt[1] +
              cotgThetaVariance*pow(dmdcotgth[i],2) +
              phiVariance*pow(dmdphi[i],2) +
              2*pt_cotgTheta*dmdpt[i]*dmdcotgth[i] +
              2*pt_phi*dmdpt[i]*dmdphi[i] +
              2*cotgTheta_phi*dmdcotgth[i]*dmdphi[i] +
              cotgTheta1_cotgTheta2*dmdcotgth[0]*dmdcotgth[1] +
              phi1_phi2*dmdphi[0]*dmdphi[1];
            massRes6 += ptVariance*pow(dmdpt[i],2) +
              pt1_pt2*dmdpt[0]*dmdpt[1] +
              cotgThetaVariance*pow(dmdcotgth[i],2) +
              phiVariance*pow(dmdphi[i],2) +
              2*pt_cotgTheta*dmdpt[i]*dmdcotgth[i] +
              2*pt_phi*dmdpt[i]*dmdphi[i] +
              2*cotgTheta_phi*dmdcotgth[i]*dmdphi[i] +
              cotgTheta1_cotgTheta2*dmdcotgth[0]*dmdcotgth[1] +
              phi1_phi2*dmdphi[0]*dmdphi[1] +
              pt12_cotgTheta21*dmdpt[0]*dmdcotgth[1] +
              pt12_cotgTheta21*dmdpt[1]*dmdcotgth[0] +
              pt12_phi21*dmdpt[0]*dmdphi[1] +
              pt12_phi21*dmdpt[1]*dmdphi[0] +
              cotgTheta12_phi21*dmdcotgth[0]*dmdphi[1] +
              cotgTheta12_phi21*dmdcotgth[1]*dmdphi[0];
          }
          // Derivatives
          mapHisto_["DerivativePt"]->Fill( *(recMu[i]), dmdpt[i], charge[i] );
          mapHisto_["DerivativeCotgTheta"]->Fill( *(recMu[i]), dmdcotgth[i], charge[i] );
          mapHisto_["DerivativePhi"]->Fill( *(recMu[i]), dmdphi[i], charge[i] );
        }
        // Fill the complete resolution function with covariance terms
        mapHisto_["FullMassResolution"]->Fill( *(recMu[0]), fullMassRes, charge[0]);
        mapHisto_["FullMassResolution"]->Fill( *(recMu[1]), fullMassRes, charge[1]);

        mapHisto_["MassRes1"]->Fill( *(recMu[0]), massRes1, charge[0] );
        mapHisto_["MassRes1"]->Fill( *(recMu[1]), massRes1, charge[1] );
        mapHisto_["MassRes2"]->Fill( *(recMu[0]), massRes2, charge[0] );
        mapHisto_["MassRes2"]->Fill( *(recMu[1]), massRes2, charge[1] );
        mapHisto_["MassRes3"]->Fill( *(recMu[0]), massRes3, charge[0] );
        mapHisto_["MassRes3"]->Fill( *(recMu[1]), massRes3, charge[1] );
        mapHisto_["MassRes4"]->Fill( *(recMu[0]), massRes4, charge[0] );
        mapHisto_["MassRes4"]->Fill( *(recMu[1]), massRes4, charge[1] );
        mapHisto_["MassRes5"]->Fill( *(recMu[0]), massRes5, charge[0] );
        mapHisto_["MassRes5"]->Fill( *(recMu[1]), massRes5, charge[1] );
        mapHisto_["MassRes6"]->Fill( *(recMu[0]), massRes6, charge[0] );
        mapHisto_["MassRes6"]->Fill( *(recMu[1]), massRes6, charge[1] );
      }
      else {
        // Fill the covariances histograms
        mapHisto_["Covariances"]->Fill(recMu1, genMu.first, recMu2, genMu.second);
      }
    }
  } // end if resonance
  else {

    // Loop on the recMuons
    vector<reco::LeafCandidate>::const_iterator recMuon = muons.begin();
    for ( ; recMuon!=muons.end(); ++recMuon ) {  
      int charge = recMuon->charge();

      lorentzVector recMu(recMuon->p4());

      // Find the matching MC muon
      const HepMC::GenEvent* Evt = evtMC->GetEvent();
      //Loop on generated particles
      map<double, lorentzVector> genAssocMap;
      HepMC::GenEvent::particle_const_iterator part = Evt->particles_begin();
      for( ; part!=Evt->particles_end(); ++part ) {
        if (fabs((*part)->pdg_id())==13 && (*part)->status()==1) {
          lorentzVector genMu = (lorentzVector((*part)->momentum().px(),(*part)->momentum().py(),
                                               (*part)->momentum().pz(),(*part)->momentum().e()));

          double deltaR = sqrt(MuScleFitUtils::deltaPhi(recMu.Phi(),genMu.Phi()) * MuScleFitUtils::deltaPhi(recMu.Phi(),genMu.Phi()) +
                               ((recMu.Eta()-genMu.Eta()) * (recMu.Eta()-genMu.Eta())));

          // 13 for the muon (-1) and -13 for the antimuon (+1), thus pdg*charge = -13.
          // Only in this case we consider it matching.
          if( ((*part)->pdg_id())*charge == -13 ) genAssocMap.insert(make_pair(deltaR, genMu));
        }
      }
      // Take the closest in deltaR
      lorentzVector genMu(genAssocMap.begin()->second);

      // Histograms with genParticles characteristics
      // --------------------------------------------

      if(checkDeltaR(genMu,recMu)){
        mapHisto_["PtResolutionGenVSMu"]->Fill(genMu,(-genMu.Pt()+recMu.Pt())/genMu.Pt(),charge);
        mapHisto_["ThetaResolutionGenVSMu"]->Fill(genMu,(-genMu.Theta()+recMu.Theta()),charge);
        mapHisto_["CotgThetaResolutionGenVSMu"]->Fill(genMu,(-cos(genMu.Theta())/sin(genMu.Theta())
                                                             +cos(recMu.Theta())/sin(recMu.Theta())),charge);
        mapHisto_["EtaResolutionGenVSMu"]->Fill(genMu,(-genMu.Eta()+recMu.Eta()),charge);
        mapHisto_["PhiResolutionGenVSMu"]->Fill(genMu,MuScleFitUtils::deltaPhiNoFabs(recMu.Phi(), genMu.Phi()),charge);
      }

      // Find the matching simMu
      if( simTracks.isValid() ) {
        map<double, lorentzVector> simAssocMap;
        for ( vector<SimTrack>::const_iterator simMuon=simTracks->begin(); simMuon!=simTracks->end(); ++simMuon ) {
          lorentzVector simMu = lorentzVector(simMuon->momentum().px(),simMuon->momentum().py(),
                                              simMuon->momentum().pz(),simMuon->momentum().e());

          double deltaR = sqrt(MuScleFitUtils::deltaPhi(recMu.Phi(),simMu.Phi()) * MuScleFitUtils::deltaPhi(recMu.Phi(),simMu.Phi()) +
                               ((recMu.Eta()-simMu.Eta()) * (recMu.Eta()-simMu.Eta())));

          if( simMuon->charge()*charge == 1 ) simAssocMap.insert(make_pair(deltaR, simMu));
        }
        lorentzVector simMu(genAssocMap.begin()->second);

        //first is always mu-, second is always mu+
        if(checkDeltaR(simMu,recMu)) {
          mapHisto_["PtResolutionSimVSMu"]->Fill(simMu,(-simMu.Pt()+recMu.Pt())/simMu.Pt(),charge);
          mapHisto_["ThetaResolutionSimVSMu"]->Fill(simMu,(-simMu.Theta()+recMu.Theta()),charge);
          mapHisto_["CotgThetaResolutionSimVSMu"]->Fill(simMu,(-cos(simMu.Theta())/sin(simMu.Theta())
                                                               +cos(recMu.Theta())/sin(recMu.Theta())),charge);
          mapHisto_["EtaResolutionSimVSMu"]->Fill(simMu,(-simMu.Eta()+recMu.Eta()),charge);
          mapHisto_["PhiResolutionSimVSMu"]->Fill(simMu,MuScleFitUtils::deltaPhiNoFabs(recMu.Phi(), simMu.Phi()),charge);
        }
      }
    }
  }

}

void ResolutionAnalyzer::fillHistoMap() {

  outputFile_->cd();

  // Resonances
  // If no Z is required, use a smaller mass range.
  double minMass = 0.;
  double maxMass = 200.;
  if( MuScleFitUtils::resfind[0] != 1 ) {
    maxMass = 30.;
  }
  mapHisto_["GenMother"]               = new HParticle(outputFile_, "GenMother", minMass, maxMass);
  mapHisto_["SimResonance"]            = new HParticle(outputFile_, "SimResonance", minMass, maxMass);
  mapHisto_["RecoResonance"]           = new HParticle(outputFile_, "RecoResonance", minMass, maxMass);
  
  // Resonance muons
  mapHisto_["GenMotherMuons"]          = new HParticle(outputFile_, "GenMotherMuons", minMass, 1.);
  mapHisto_["SimResonanceMuons"]       = new HParticle(outputFile_, "SimResonanceMuons", minMass, 1.);
  mapHisto_["RecoResonanceMuons"]      = new HParticle(outputFile_, "RecoResonanceMuons", minMass, 1.);
  
  // Deltas between resonance muons
  mapHisto_["DeltaGenMotherMuons"]     = new HDelta (outputFile_, "DeltaGenMotherMuons");
  mapHisto_["DeltaSimResonanceMuons"]  = new HDelta (outputFile_, "DeltaSimResonanceMuons");
  mapHisto_["DeltaRecoResonanceMuons"] = new HDelta (outputFile_, "DeltaRecoResonanceMuons");
  
  //   //Reconstructed muon kinematics
  //   //-----------------------------
  //   mapHisto_["hRecBestMu"]             = new HParticle         ("hRecBestMu");
  //   mapHisto_["hRecBestMu_Acc"]         = new HParticle         ("hRecBestMu_Acc"); 
  //   mapHisto_["hDeltaRecBestMu"]        = new HDelta            ("hDeltaRecBestMu");
  
  //   mapHisto_["hRecBestRes"]            = new HParticle         ("hRecBestRes");
  //   mapHisto_["hRecBestRes_Acc"]        = new HParticle         ("hRecBestRes_Acc"); 
  //   mapHisto_["hRecBestResVSMu"]        = new HMassVSPart       ("hRecBestResVSMu");
  
  //Resolution VS muon kinematic
  //----------------------------
  mapHisto_["PtResolutionGenVSMu"]        = new HResolutionVSPart (outputFile_, "PtResolutionGenVSMu");
  mapHisto_["PtResolutionSimVSMu"]        = new HResolutionVSPart (outputFile_, "PtResolutionSimVSMu");
  mapHisto_["EtaResolutionGenVSMu"]       = new HResolutionVSPart (outputFile_, "EtaResolutionGenVSMu");
  mapHisto_["EtaResolutionSimVSMu"]       = new HResolutionVSPart (outputFile_, "EtaResolutionSimVSMu");
  mapHisto_["ThetaResolutionGenVSMu"]     = new HResolutionVSPart (outputFile_, "ThetaResolutionGenVSMu");
  mapHisto_["ThetaResolutionSimVSMu"]     = new HResolutionVSPart (outputFile_, "ThetaResolutionSimVSMu");
  mapHisto_["CotgThetaResolutionGenVSMu"] = new HResolutionVSPart (outputFile_, "CotgThetaResolutionGenVSMu", -0.02, 0.02, -0.02, 0.02);
  mapHisto_["CotgThetaResolutionSimVSMu"] = new HResolutionVSPart (outputFile_, "CotgThetaResolutionSimVSMu");
  mapHisto_["PhiResolutionGenVSMu"]       = new HResolutionVSPart (outputFile_, "PhiResolutionGenVSMu", -0.002, 0.002, -0.002, 0.002);
  mapHisto_["PhiResolutionSimVSMu"]       = new HResolutionVSPart (outputFile_, "PhiResolutionSimVSMu");

  // Covariances between muons kinematic quantities
  // ----------------------------------------------
  double ptMax = 200.;
  if (TString(outputFile_->GetName()).Contains("Y")) ptMax = 40.;

  // Mass resolution
  // ---------------
  mapHisto_["MassResolution"] = new HMassResolutionVSPart (outputFile_,"MassResolution");
  
  //  mapHisto_["hResolRecoMassVSGenMassVSPt"] = new HResolutionVSPart
  
  // Mass resolution vs (pt, eta) of the muons from MC
  massResolutionVsPtEta_ = new HCovarianceVSxy ( "Mass", "Mass", 100, 0., ptMax, 60, -3, 3 );
  // Mass resolution vs (pt, eta) of the muons from function
  recoPtVsgenPt_ = new TH2D("recoPtVsgenPt", "recoPtVsgenPt", 100, 0, ptMax, 100, 0, ptMax);
  recoPtVsgenPtEta12_ = new TH2D("recoPtVsgenPtEta12", "recoPtVsgenPtEta12", 100, 0, ptMax, 100, 0, ptMax);
  deltaPtOverPt_ = new TH1D("deltaPtOverPt", "deltaPtOverPt", 100, -0.1, 0.1);
  deltaPtOverPtForEta12_ = new TH1D("deltaPtOverPtForEta12", "deltaPtOverPtForEta12", 100, -0.1, 0.1);

  // Muons resolutions from resolution functions
  // -------------------------------------------
  mapHisto_["hFunctionResolMass"]      = new HFunctionResolution (outputFile_, "hFunctionResolMass", ptMax);
  mapHisto_["hFunctionResolPt"]        = new HFunctionResolution (outputFile_, "hFunctionResolPt", ptMax);
  mapHisto_["hFunctionResolCotgTheta"] = new HFunctionResolution (outputFile_, "hFunctionResolCotgTheta", ptMax);
  mapHisto_["hFunctionResolPhi"]       = new HFunctionResolution (outputFile_, "hFunctionResolPhi", ptMax);

  if( readCovariances_ ) {
    // Covariances read from file. Used to compare the terms in the expression of mass error
    mapHisto_["ReadCovariances"] = new HCovarianceVSParts ( theCovariancesRootFileName_, "Covariance" );

    double ptMax = 40.;

    // Variances
    mapHisto_["MassResolutionPt"]                    = new HFunctionResolutionVarianceCheck(outputFile_,"functionResolMassPt", ptMax);
    mapHisto_["MassResolutionCotgTheta"]             = new HFunctionResolutionVarianceCheck(outputFile_,"functionResolMassCotgTheta", ptMax);
    mapHisto_["MassResolutionPhi"]                   = new HFunctionResolutionVarianceCheck(outputFile_,"functionResolMassPhi", ptMax);
    // Covariances
    mapHisto_["MassResolutionPt-CotgTheta"]          = new HFunctionResolution(outputFile_,"functionResolMassPt-CotgTheta", ptMax);
    mapHisto_["MassResolutionPt-Phi"]                = new HFunctionResolution(outputFile_,"functionResolMassPt-Phi", ptMax);
    mapHisto_["MassResolutionCotgTheta-Phi"]         = new HFunctionResolution(outputFile_,"functionResolMassCotgTheta-Phi", ptMax);
    mapHisto_["MassResolutionPt1-Pt2"]               = new HFunctionResolution(outputFile_,"functionResolMassPt1-Pt2", ptMax);
    mapHisto_["MassResolutionCotgTheta1-CotgTheta2"] = new HFunctionResolution(outputFile_,"functionResolMassCotgTheta1-CotgTheta2", ptMax);
    mapHisto_["MassResolutionPhi1-Phi2"]             = new HFunctionResolution(outputFile_,"functionResolMassPhi1-Phi2", ptMax);
    mapHisto_["MassResolutionPt12-CotgTheta21"]      = new HFunctionResolution(outputFile_,"functionResolMassPt12-CotgTheta21", ptMax);
    mapHisto_["MassResolutionPt12-Phi21"]            = new HFunctionResolution(outputFile_,"functionResolMassPt12-Phi21", ptMax);
    mapHisto_["MassResolutionCotgTheta12-Phi21"]     = new HFunctionResolution(outputFile_,"functionResolMassCotgTheta12-Phi21", ptMax);

    mapHisto_["sigmaPtFromVariance"]                 = new HFunctionResolution(outputFile_,"sigmaPtFromVariance", ptMax);
    mapHisto_["sigmaCotgThetaFromVariance"]          = new HFunctionResolution(outputFile_,"sigmaCotgThetaFromVariance", ptMax);
    mapHisto_["sigmaPhiFromVariance"]                = new HFunctionResolution(outputFile_,"sigmaPhiFromVariance", ptMax);

    // Derivatives
    mapHisto_["DerivativePt"]                        = new HFunctionResolution(outputFile_,"derivativePt", ptMax);
    mapHisto_["DerivativeCotgTheta"]                 = new HFunctionResolution(outputFile_,"derivativeCotgTheta", ptMax);
    mapHisto_["DerivativePhi"]                       = new HFunctionResolution(outputFile_,"derivativePhi", ptMax);

    // Pt term from function
    mapHisto_["MassResolutionPtFromFunction"]        = new HFunctionResolutionVarianceCheck(outputFile_,"functionResolMassPtFromFunction", ptMax);

    mapHisto_["FullMassResolution"]                  = new HFunctionResolution(outputFile_, "fullMassResolution", ptMax);
    mapHisto_["MassRes1"]                            = new HFunctionResolution(outputFile_, "massRes1", ptMax);
    mapHisto_["MassRes2"]                            = new HFunctionResolution(outputFile_, "massRes2", ptMax);
    mapHisto_["MassRes3"]                            = new HFunctionResolution(outputFile_, "massRes3", ptMax);
    mapHisto_["MassRes4"]                            = new HFunctionResolution(outputFile_, "massRes4", ptMax);
    mapHisto_["MassRes5"]                            = new HFunctionResolution(outputFile_, "massRes5", ptMax);
    mapHisto_["MassRes6"]                            = new HFunctionResolution(outputFile_, "massRes6", ptMax);
  }
  else {
    mapHisto_["Covariances"] = new HCovarianceVSParts ( outputFile_, "Covariance", ptMax );
  }
}

void ResolutionAnalyzer::writeHistoMap() {
  for (map<string, Histograms*>::const_iterator histo=mapHisto_.begin(); 
       histo!=mapHisto_.end(); histo++) {
    (*histo).second->Write();
  }
  outputFile_->cd();
  massResolutionVsPtEta_->Write();
  recoPtVsgenPt_->Write();
  recoPtVsgenPtEta12_->Write();
  deltaPtOverPt_->Write();
  deltaPtOverPtForEta12_->Write();
}

bool ResolutionAnalyzer::checkDeltaR(const reco::Particle::LorentzVector & genMu, const reco::Particle::LorentzVector & recMu){
  //first is always mu-, second is always mu+
  double deltaR = sqrt(MuScleFitUtils::deltaPhi(recMu.Phi(),genMu.Phi()) * MuScleFitUtils::deltaPhi(recMu.Phi(),genMu.Phi()) +
                       ((recMu.Eta()-genMu.Eta()) * (recMu.Eta()-genMu.Eta())));
  if(deltaR<0.01)
    return true;
  else if (debug_>0)
    cout<<"Reco muon "<<recMu<<" with eta "<<recMu.Eta()<<" and phi "<<recMu.Phi()<<endl
	<<" DOES NOT MATCH with generated muon from resonance: "<<endl
	<<genMu<<" with eta "<<genMu.Eta()<<" and phi "<<genMu.Phi()<<endl;
  return false;
}

//define this as a plug-in
DEFINE_FWK_MODULE(ResolutionAnalyzer);

#endif // RESOLUTIONANALYZER_CC
