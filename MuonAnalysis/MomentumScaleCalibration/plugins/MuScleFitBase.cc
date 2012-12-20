#ifndef MUSCLEFITBASE_C
#define MUSCLEFITBASE_C

#include "MuScleFitBase.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

void MuScleFitBase::fillHistoMap(TFile* outputFile, unsigned int iLoop) {
  //Reconstructed muon kinematics
  //-----------------------------
  outputFile->cd();
  // If no Z is required, use a smaller mass range.
  double minMass = 0.;
  double maxMass = 200.;
  double maxPt = 100.;
  double yMaxEta = 4.;
  double yMaxPt = 2.;
  if( MuScleFitUtils::resfind[0] != 1 ) {
    maxMass = 20.;
    maxPt = 20.;
    yMaxEta = 0.2;
    yMaxPt = 0.2;
    // If running on standalone muons we need to expand the window range
    if( theMuonType_ == 2 ) {
      yMaxEta = 20.;
    }
  }

  LogDebug("MuScleFitBase") << "Creating new histograms" << std::endl;

  mapHisto_["hRecBestMu"]      = new HParticle ("hRecBestMu", minMass, maxMass, maxPt);
  mapHisto_["hRecBestMuVSEta"] = new HPartVSEta ("hRecBestMuVSEta", minMass, maxMass, maxPt);
  //mapHisto_["hRecBestMuVSPhi"] = new HPartVSPhi ("hRecBestMuVSPhi"); 
  //mapHisto_["hRecBestMu_Acc"]  = new HParticle ("hRecBestMu_Acc", minMass, maxMass);
  mapHisto_["hDeltaRecBestMu"] = new HDelta ("hDeltaRecBestMu");

  mapHisto_["hRecBestRes"]          = new HParticle   ("hRecBestRes", minMass, maxMass, maxPt);
  mapHisto_["hRecBestResAllEvents"] = new HParticle   ("hRecBestResAllEvents", minMass, maxMass, maxPt);
  //mapHisto_["hRecBestRes_Acc"] = new HParticle   ("hRecBestRes_Acc", minMass, maxMass);
  // If not finding Z, use a smaller mass window
  mapHisto_["hRecBestResVSMu"] = new HMassVSPart ("hRecBestResVSMu", minMass, maxMass, maxPt);
  mapHisto_["hRecBestResVSRes"] = new HMassVSPart ("hRecBestResVSRes", minMass, maxMass, maxPt);
  //Generated Mass versus pt
  mapHisto_["hGenResVSMu"] = new HMassVSPart ("hGenResVSMu", minMass, maxMass, maxPt);
  // Likelihood values VS muon variables
  // -------------------------------------
  mapHisto_["hLikeVSMu"]       = new HLikelihoodVSPart ("hLikeVSMu");
  mapHisto_["hLikeVSMuMinus"]  = new HLikelihoodVSPart ("hLikeVSMuMinus");
  mapHisto_["hLikeVSMuPlus"]   = new HLikelihoodVSPart ("hLikeVSMuPlus");

  //Resolution VS muon kinematic
  //----------------------------
  mapHisto_["hResolMassVSMu"]         = new HResolutionVSPart( outputFile, "hResolMassVSMu", maxPt, 0., yMaxEta, 0., yMaxPt, true );
  mapHisto_["hFunctionResolMassVSMu"] = new HResolutionVSPart( outputFile, "hFunctionResolMassVSMu", maxPt, 0, 0.1, 0, 0.1, true );
  mapHisto_["hResolPtGenVSMu"]        = new HResolutionVSPart( outputFile, "hResolPtGenVSMu", maxPt, -0.1, 0.1, -0.1, 0.1 );
  mapHisto_["hResolPtSimVSMu"]        = new HResolutionVSPart( outputFile, "hResolPtSimVSMu", maxPt, -0.1, 0.1, -0.1, 0.1 );
  mapHisto_["hResolEtaGenVSMu"]       = new HResolutionVSPart( outputFile, "hResolEtaGenVSMu", maxPt, -0.02, 0.02, -0.02, 0.02 );
  mapHisto_["hResolEtaSimVSMu"]       = new HResolutionVSPart( outputFile, "hResolEtaSimVSMu", maxPt, -0.02, 0.02, -0.02, 0.02 );
  mapHisto_["hResolThetaGenVSMu"]     = new HResolutionVSPart( outputFile, "hResolThetaGenVSMu", maxPt, -0.02, 0.02, -0.02, 0.02 );
  mapHisto_["hResolThetaSimVSMu"]     = new HResolutionVSPart( outputFile, "hResolThetaSimVSMu", maxPt, -0.02, 0.02, -0.02, 0.02 );
  mapHisto_["hResolCotgThetaGenVSMu"] = new HResolutionVSPart( outputFile, "hResolCotgThetaGenVSMu", maxPt, -0.02, 0.02, -0.02, 0.02 );
  mapHisto_["hResolCotgThetaSimVSMu"] = new HResolutionVSPart( outputFile, "hResolCotgThetaSimVSMu", maxPt, -0.02, 0.02, -0.02, 0.02 );
  mapHisto_["hResolPhiGenVSMu"]       = new HResolutionVSPart( outputFile, "hResolPhiGenVSMu", maxPt, -0.02, 0.02, -0.02, 0.02 );
  mapHisto_["hResolPhiSimVSMu"]       = new HResolutionVSPart( outputFile, "hResolPhiSimVSMu", maxPt, -0.02, 0.02, -0.02, 0.02 );

  if( MuScleFitUtils::debugMassResol_ ) {
    mapHisto_["hdMdPt1"] = new HResolutionVSPart( outputFile, "hdMdPt1", maxPt, 0, 100, -3.2, 3.2, true );
    mapHisto_["hdMdPt2"] = new HResolutionVSPart( outputFile, "hdMdPt2", maxPt, 0, 100, -3.2, 3.2, true );
    mapHisto_["hdMdPhi1"] = new HResolutionVSPart( outputFile, "hdMdPhi1", maxPt, 0, 100, -3.2, 3.2, true );
    mapHisto_["hdMdPhi2"] = new HResolutionVSPart( outputFile, "hdMdPhi2", maxPt, 0, 100, -3.2, 3.2, true );
    mapHisto_["hdMdCotgTh1"] = new HResolutionVSPart( outputFile, "hdMdCotgTh1", maxPt, 0, 100, -3.2, 3.2, true );
    mapHisto_["hdMdCotgTh2"] = new HResolutionVSPart( outputFile, "hdMdCotgTh2", maxPt, 0, 100, -3.2, 3.2, true );
  }

  HTH2D * recoGenHisto = new HTH2D(outputFile, "hPtRecoVsPtGen", "Pt reco vs Pt gen", "hPtRecoVsPtGen", 120, 0., 120., 120, 0, 120.);
  (*recoGenHisto)->SetXTitle("Pt gen (GeV)");
  (*recoGenHisto)->SetYTitle("Pt reco (GeV)");
  mapHisto_["hPtRecoVsPtGen"] = recoGenHisto;
  HTH2D * recoSimHisto = new HTH2D(outputFile, "hPtRecoVsPtSim", "Pt reco vs Pt sim", "hPtRecoVsPtSim", 120, 0., 120., 120, 0, 120.);
  (*recoSimHisto)->SetXTitle("Pt sim (GeV)");
  (*recoSimHisto)->SetYTitle("Pt reco (GeV)");
  mapHisto_["hPtRecoVsPtSim"] = recoSimHisto;
  // Resolutions from resolution functions
  // -------------------------------------
  mapHisto_["hFunctionResolPt"]        = new HFunctionResolution( outputFile, "hFunctionResolPt", maxPt );
  mapHisto_["hFunctionResolCotgTheta"] = new HFunctionResolution( outputFile, "hFunctionResolCotgTheta", maxPt );
  mapHisto_["hFunctionResolPhi"]       = new HFunctionResolution( outputFile, "hFunctionResolPhi", maxPt );

  // Mass probability histograms
  // ---------------------------
  // The word "profile" is added to the title automatically
  mapHisto_["hMass_P"]      = new HTProfile( outputFile, "Mass_P", "Mass probability", 4000, 0., 200., 0., 50. );
  mapHisto_["hMass_fine_P"] = new HTProfile( outputFile, "Mass_fine_P", "Mass probability", 4000, 0., 20., 0., 50. );
  mapHisto_["hMass_Probability"]      = new HTH1D( outputFile, "Mass_Probability", "Mass probability", 4000, 0., 200.);
  mapHisto_["hMass_fine_Probability"] = new HTH1D( outputFile, "Mass_fine_Probability", "Mass probability", 4000, 0., 20.);
  mapHisto_["hMassProbVsMu"] = new HMassVSPartProfile( "hMassProbVsMu", minMass, maxMass, maxPt );
  mapHisto_["hMassProbVsRes"] = new HMassVSPartProfile( "hMassProbVsRes", minMass, maxMass, maxPt );
  mapHisto_["hMassProbVsMu_fine"] = new HMassVSPartProfile( "hMassProbVsMu_fine", minMass, maxMass, maxPt );
  mapHisto_["hMassProbVsRes_fine"] = new HMassVSPartProfile( "hMassProbVsRes_fine", minMass, maxMass, maxPt );

  // (M_reco-M_gen)/M_gen vs (pt, eta) of the muons from MC
  mapHisto_["hDeltaMassOverGenMassVsPt"] = new HTH2D( outputFile, "DeltaMassOverGenMassVsPt", "DeltaMassOverGenMassVsPt", "DeltaMassOverGenMass", 200, 0, maxPt, 200, -0.2, 0.2 );
  mapHisto_["hDeltaMassOverGenMassVsEta"] = new HTH2D( outputFile, "DeltaMassOverGenMassVsEta", "DeltaMassOverGenMassVsEta", "DeltaMassOverGenMass", 200, -3., 3., 200, -0.2, 0.2 );

  // Square of mass resolution vs (pt, eta) of the muons from MC
  // EM 2012.12.19  mapHisto_["hMassResolutionVsPtEta"] = new HCovarianceVSxy( "Mass", "Mass", 100, 0., maxPt, 60, -3, 3, outputFile->mkdir("MassCovariance") );
  // Mass resolution vs (pt, eta) from resolution function
  mapHisto_["hFunctionResolMass"] = new HFunctionResolution( outputFile, "hFunctionResolMass", maxPt );
}

void MuScleFitBase::clearHistoMap() {
  for (std::map<std::string, Histograms*>::const_iterator histo=mapHisto_.begin();
       histo!=mapHisto_.end(); histo++) {
    delete (*histo).second;
  }
}

void MuScleFitBase::writeHistoMap( const unsigned int iLoop ) {
  for (std::map<std::string, Histograms*>::const_iterator histo=mapHisto_.begin();
       histo!=mapHisto_.end(); histo++) {
    // This is to avoid writing into subdirs. Need a workaround.
    theFiles_[iLoop]->cd();
    (*histo).second->Write();
  }
}

void MuScleFitBase::readProbabilityDistributionsFromFile()
{
  TH2D * GLZ[24];
  TH2D * GL[6];
  TFile * ProbsFile;
  if( probabilitiesFile_ != "" ) {
    ProbsFile = new TFile (probabilitiesFile_.c_str());
    std::cout << "[MuScleFit-Constructor]: Reading TH2D probabilities from " << probabilitiesFile_ << std::endl;
  }
  else {
    // edm::FileInPath file("MuonAnalysis/MomentumScaleCalibration/test/Probs_new_1000_CTEQ.root");
    // edm::FileInPath file("MuonAnalysis/MomentumScaleCalibration/test/Probs_new_Horace_CTEQ_1000.root");
    // edm::FileInPath file("MuonAnalysis/MomentumScaleCalibration/test/Probs_merge.root");
    edm::FileInPath file(probabilitiesFileInPath_.c_str());
    ProbsFile = new TFile (file.fullPath().c_str());
    std::cout << "[MuScleFit-Constructor]: Reading TH2D probabilities from " << probabilitiesFileInPath_ << std::endl;
  }


  ProbsFile->cd();
  if( theMuonType_!=2 && MuScleFitUtils::resfind[0]) {
    for ( int i=0; i<24; i++ ) {
      char nameh[6];
      sprintf (nameh,"GLZ%d",i);
      GLZ[i] = dynamic_cast<TH2D*>(ProbsFile->Get(nameh));
    }
  }
  if( theMuonType_==2 && MuScleFitUtils::resfind[0]) 
    GL[0] = dynamic_cast<TH2D*> (ProbsFile->Get("GL0"));
  if(MuScleFitUtils::resfind[1]) 
    GL[1] = dynamic_cast<TH2D*> (ProbsFile->Get("GL1"));
  if(MuScleFitUtils::resfind[2]) 
    GL[2] = dynamic_cast<TH2D*> (ProbsFile->Get("GL2"));
  if(MuScleFitUtils::resfind[3]) 
    GL[3] = dynamic_cast<TH2D*> (ProbsFile->Get("GL3"));
  if(MuScleFitUtils::resfind[4]) 
    GL[4] = dynamic_cast<TH2D*> (ProbsFile->Get("GL4"));
  if(MuScleFitUtils::resfind[5]) 
    GL[5] = dynamic_cast<TH2D*> (ProbsFile->Get("GL5"));

  // Read the limits for M and Sigma axis for each pdf
  // Note: we assume all the Z histograms to have the same limits
  // x is mass, y is sigma
  if(MuScleFitUtils::resfind[0] && theMuonType_!=2) {
    MuScleFitUtils::ResHalfWidth[0] = (GLZ[0]->GetXaxis()->GetXmax() - GLZ[0]->GetXaxis()->GetXmin())/2.;
    MuScleFitUtils::ResMaxSigma[0] = (GLZ[0]->GetYaxis()->GetXmax() - GLZ[0]->GetYaxis()->GetXmin());
    MuScleFitUtils::ResMinMass[0] = (GLZ[0]->GetXaxis()->GetXmin());
  }
  if(MuScleFitUtils::resfind[0] && theMuonType_==2) {
    MuScleFitUtils::ResHalfWidth[0] = (GL[0]->GetXaxis()->GetXmax() - GL[0]->GetXaxis()->GetXmin())/2.;
    MuScleFitUtils::ResMaxSigma[0] = (GL[0]->GetYaxis()->GetXmax() - GL[0]->GetYaxis()->GetXmin());
    MuScleFitUtils::ResMinMass[0] = (GL[0]->GetXaxis()->GetXmin());
  }
  for( int i=1; i<6; ++i ) {
    if(MuScleFitUtils::resfind[i]){
      MuScleFitUtils::ResHalfWidth[i] = (GL[i]->GetXaxis()->GetXmax() - GL[i]->GetXaxis()->GetXmin())/2.;
      MuScleFitUtils::ResMaxSigma[i] = (GL[i]->GetYaxis()->GetXmax() - GL[i]->GetYaxis()->GetXmin());
      MuScleFitUtils::ResMinMass[i] = (GL[i]->GetXaxis()->GetXmin());
     // if( debug_>2 ) {
      std::cout << "MuScleFitUtils::ResHalfWidth["<<i<<"] = " << MuScleFitUtils::ResHalfWidth[i] << std::endl;
      std::cout << "MuScleFitUtils::ResMaxSigma["<<i<<"] = " << MuScleFitUtils::ResMaxSigma[i] << std::endl;
      // }
    }
  }

  // Extract normalization for mass slice in Y bins of Z
  // ---------------------------------------------------
  if(MuScleFitUtils::resfind[0] && theMuonType_!=2) {
    for (int iY=0; iY<24; iY++) {
      int nBinsX = GLZ[iY]->GetNbinsX();
      int nBinsY = GLZ[iY]->GetNbinsY();
      if( nBinsX != MuScleFitUtils::nbins+1 || nBinsY != MuScleFitUtils::nbins+1 ) {
	std::cout << "Error: for histogram \"" << GLZ[iY]->GetName() << "\" bins are not " << MuScleFitUtils::nbins << std::endl;
	std::cout<< "nBinsX = " << nBinsX << ", nBinsY = " << nBinsY << std::endl;
	exit(1);
      }
      for (int iy=0; iy<=MuScleFitUtils::nbins; iy++) {
	MuScleFitUtils::GLZNorm[iY][iy] = 0.;
	for (int ix=0; ix<=MuScleFitUtils::nbins; ix++) {
	  MuScleFitUtils::GLZValue[iY][ix][iy] = GLZ[iY]->GetBinContent(ix+1, iy+1);
	  MuScleFitUtils::GLZNorm[iY][iy] += MuScleFitUtils::GLZValue[iY][ix][iy]*(2*MuScleFitUtils::ResHalfWidth[0])/MuScleFitUtils::nbins;
	}
      }
    }
  }

  if(MuScleFitUtils::resfind[0] && theMuonType_==2){
      int nBinsX = GL[0]->GetNbinsX();
      int nBinsY = GL[0]->GetNbinsY();
      if( nBinsX != MuScleFitUtils::nbins+1 || nBinsY != MuScleFitUtils::nbins+1 ) {
	std::cout << "Error: for histogram \"" << GL[0]->GetName() << "\" bins are not " << MuScleFitUtils::nbins << std::endl;
	std::cout<< "nBinsX = " << nBinsX << ", nBinsY = " << nBinsY << std::endl;
	exit(1);
      }

      for (int iy=0; iy<=MuScleFitUtils::nbins; iy++) {
	MuScleFitUtils::GLNorm[0][iy] = 0.;
	for (int ix=0; ix<=MuScleFitUtils::nbins; ix++) {
	  MuScleFitUtils::GLValue[0][ix][iy] = GL[0]->GetBinContent(ix+1, iy+1);
	  // N.B. approximation: we should compute the integral of the function used to compute the probability (linear
	  // interpolation of the mass points). This computation could be troublesome because the points have a steep
	  // variation near the mass peak and the normal integral is not precise in these conditions.
	  // Furthermore it is slow.
	  MuScleFitUtils::GLNorm[0][iy] += MuScleFitUtils::GLValue[0][ix][iy]*(2*MuScleFitUtils::ResHalfWidth[0])/MuScleFitUtils::nbins;
	}
      }
    }  
  // Extract normalization for each mass slice
  // -----------------------------------------
  for (int ires=1; ires<6; ires++) {
    if(MuScleFitUtils::resfind[ires]){
      int nBinsX = GL[ires]->GetNbinsX();
      int nBinsY = GL[ires]->GetNbinsY();
      if( nBinsX != MuScleFitUtils::nbins+1 || nBinsY != MuScleFitUtils::nbins+1 ) {
	std::cout << "Error: for histogram \"" << GL[ires]->GetName() << "\" bins are not " << MuScleFitUtils::nbins << std::endl;
	std::cout<< "nBinsX = " << nBinsX << ", nBinsY = " << nBinsY << std::endl;
	exit(1);
      }

      for (int iy=0; iy<=MuScleFitUtils::nbins; iy++) {
	MuScleFitUtils::GLNorm[ires][iy] = 0.;
	for (int ix=0; ix<=MuScleFitUtils::nbins; ix++) {
	  MuScleFitUtils::GLValue[ires][ix][iy] = GL[ires]->GetBinContent(ix+1, iy+1);
	  // N.B. approximation: we should compute the integral of the function used to compute the probability (linear
	  // interpolation of the mass points). This computation could be troublesome because the points have a steep
	  // variation near the mass peak and the normal integral is not precise in these conditions.
	  // Furthermore it is slow.
	  MuScleFitUtils::GLNorm[ires][iy] += MuScleFitUtils::GLValue[ires][ix][iy]*(2*MuScleFitUtils::ResHalfWidth[ires])/MuScleFitUtils::nbins;
	}
      }
    }
  }
  // Free all the memory for the probability histograms.
  if(MuScleFitUtils::resfind[0] && theMuonType_!=2) {
    for ( int i=0; i<24; i++ ) {
      delete GLZ[i];
    }
  }
  if(MuScleFitUtils::resfind[0] && theMuonType_==2)
    delete GL[0];
  for (int ires=1; ires<6; ires++) {
    if(MuScleFitUtils::resfind[ires])
      delete GL[ires];
  }
  delete ProbsFile;
}

#endif
