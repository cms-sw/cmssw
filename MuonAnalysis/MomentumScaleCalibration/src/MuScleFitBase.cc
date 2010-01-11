#ifndef MUSCLEFITBASE_C
#define MUSCLEFITBASE_C

#include "MuonAnalysis/MomentumScaleCalibration/interface/MuScleFitBase.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

void MuScleFitBase::fillHistoMap(TFile* outputFile, unsigned int iLoop) {
  //Reconstructed muon kinematics
  //-----------------------------
  outputFile->cd();
  // If no Z is required, use a smaller mass range.
  double minMass = 0.;
  double maxMass = 200.;
  double maxPt = 100.;
  if( MuScleFitUtils::resfind[0] != 1 ) {
    maxMass = 20.;
    maxPt = 20.;
  }

  LogDebug("MuScleFitBase") << "Creating new histograms" << endl;

  mapHisto_["hRecBestMu"]      = new HParticle ("hRecBestMu", minMass, maxMass, maxPt);
  mapHisto_["hRecBestMuVSEta"] = new HPartVSEta ("hRecBestMuVSEta", minMass, maxMass, maxPt);
  //mapHisto_["hRecBestMu_Acc"]  = new HParticle ("hRecBestMu_Acc", minMass, maxMass);
  mapHisto_["hDeltaRecBestMu"] = new HDelta ("hDeltaRecBestMu");

  mapHisto_["hRecBestRes"]          = new HParticle   ("hRecBestRes", minMass, maxMass, maxPt);
  mapHisto_["hRecBestResAllEvents"] = new HParticle   ("hRecBestResAllEvents", minMass, maxMass, maxPt);
  //mapHisto_["hRecBestRes_Acc"] = new HParticle   ("hRecBestRes_Acc", minMass, maxMass);
  // If not finding Z, use a smaller mass window
  mapHisto_["hRecBestResVSMu"] = new HMassVSPart ("hRecBestResVSMu", minMass, maxMass, maxPt);
  //Generated Mass versus pt
  mapHisto_["hGenResVSMu"] = new HMassVSPart ("hGenResVSMu", minMass, maxMass, maxPt);
  // Likelihood values VS muon variables
  // -------------------------------------
  mapHisto_["hLikeVSMu"]       = new HLikelihoodVSPart ("hLikeVSMu");
  mapHisto_["hLikeVSMuMinus"]  = new HLikelihoodVSPart ("hLikeVSMuMinus");
  mapHisto_["hLikeVSMuPlus"]   = new HLikelihoodVSPart ("hLikeVSMuPlus");

  //Resolution VS muon kinematic
  //----------------------------
  mapHisto_["hResolMassVSMu"]         = new HResolutionVSPart( outputFile, "hResolMassVSMu", maxPt, 0, 4 );
  mapHisto_["hFunctionResolMassVSMu"] = new HResolutionVSPart( outputFile, "hFunctionResolMassVSMu", maxPt, 0, 0.1, 0, 0.1, true );
  mapHisto_["hResolPtGenVSMu"]        = new HResolutionVSPart( outputFile, "hResolPtGenVSMu", maxPt );
  mapHisto_["hResolPtSimVSMu"]        = new HResolutionVSPart( outputFile, "hResolPtSimVSMu", maxPt );
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

  HTH2D * recoGenHisto = new HTH2D(outputFile, "hPtRecoVsPtGen", "Pt reco vs Pt gen", 120, 0., 120., 120, 0, 120.);
  (*recoGenHisto)->SetXTitle("Pt gen (GeV)");
  (*recoGenHisto)->SetYTitle("Pt reco (GeV)");
  mapHisto_["hPtRecoVsPtGen"] = recoGenHisto;
  HTH2D * recoSimHisto = new HTH2D(outputFile, "hPtRecoVsPtSim", "Pt reco vs Pt sim", 120, 0., 120., 120, 0, 120.);
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
  mapHisto_["hMass_P"]      = new HTProfile( outputFile, "Mass_P", "Mass probability", 4000, 0., 200., 0., 1. );
  mapHisto_["hMass_fine_P"] = new HTProfile( outputFile, "Mass_fine_P", "Mass probability", 4000, 0., 20., 0., 1. );
  mapHisto_["hMass_Probability"]      = new HTH1D( outputFile, "Mass_Probability", "Mass probability", 4000, 0., 200.);
  mapHisto_["hMass_fine_Probability"] = new HTH1D( outputFile, "Mass_fine_Probability", "Mass probability", 4000, 0., 20.);

  // (M_reco-M_gen)/M_gen vs (pt, eta) of the muons from MC
  mapHisto_["hDeltaMassOverGenMassVsPt"] = new HTH2D( outputFile, "DeltaMassOverGenMassVsPt", "DeltaMassOverGenMassVsPt", 200, 0, maxPt, 200, -1., 1. );
  mapHisto_["hDeltaMassOverGenMassVsEta"] = new HTH2D( outputFile, "DeltaMassOverGenMassVsEta", "DeltaMassOverGenMassVsEta", 60, -3.2, 3.2, 200, -1., 1. );

  // Square of mass resolution vs (pt, eta) of the muons from MC
  mapHisto_["hMassResolutionVsPtEta"] = new HCovarianceVSxy( "Mass", "Mass", 100, 0., maxPt, 60, -3, 3, outputFile->mkdir("MassCovariance") );
  // Mass resolution vs (pt, eta) from resolution function
  mapHisto_["hFunctionResolMass"] = new HFunctionResolution( outputFile, "hFunctionResolMass", maxPt );
}

void MuScleFitBase::clearHistoMap() {
  for (map<string, Histograms*>::const_iterator histo=mapHisto_.begin();
       histo!=mapHisto_.end(); histo++) {
    delete (*histo).second;
  }
}

void MuScleFitBase::writeHistoMap( const unsigned int iLoop ) {
  for (map<string, Histograms*>::const_iterator histo=mapHisto_.begin();
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
  if ( theMuonType_!=2 ) {
    if( probabilitiesFile_ != "" ) {
      ProbsFile = new TFile (probabilitiesFile_.c_str());
      cout << "[MuScleFit-Constructor]: Reading TH2D probabilities from " << probabilitiesFile_ << endl;
    }
    else {
      // edm::FileInPath file("MuonAnalysis/MomentumScaleCalibration/test/Probs_new_1000_CTEQ.root");
      // edm::FileInPath file("MuonAnalysis/MomentumScaleCalibration/test/Probs_new_Horace_CTEQ_1000.root");
      // edm::FileInPath file("MuonAnalysis/MomentumScaleCalibration/test/Probs_merge.root");
      edm::FileInPath file(probabilitiesFileInPath_.c_str());
      ProbsFile = new TFile (file.fullPath().c_str());
      cout << "[MuScleFit-Constructor]: Reading TH2D probabilities from " << probabilitiesFileInPath_ << endl;
    }
  }
  else {
    edm::FileInPath fileSM("MuonAnalysis/MomentumScaleCalibration/test/Probs_SM_1000.root");
    ProbsFile = new TFile (fileSM.fullPath().c_str()); // NNBB need to reset this if MuScleFitUtils::nbins changes
    // ProbsFile = new TFile ("Probs_SM_1000.root"); // NNBB need to reset this if MuScleFitUtils::nbins changes
    cout << "[MuScleFit-Constructor]: Reading TH2D probabilities from Probs_new_SM_1000.root file" << endl;
  }
  ProbsFile->cd();
  if( theMuonType_!=2 ) {
    for ( int i=0; i<24; i++ ) {
      char nameh[6];
      sprintf (nameh,"GLZ%d",i);
      GLZ[i] = dynamic_cast<TH2D*>(ProbsFile->Get(nameh));
    }
  }
  GL[0] = dynamic_cast<TH2D*> (ProbsFile->Get("GL0"));
  GL[1] = dynamic_cast<TH2D*> (ProbsFile->Get("GL1"));
  GL[2] = dynamic_cast<TH2D*> (ProbsFile->Get("GL2"));
  GL[3] = dynamic_cast<TH2D*> (ProbsFile->Get("GL3"));
  GL[4] = dynamic_cast<TH2D*> (ProbsFile->Get("GL4"));
  GL[5] = dynamic_cast<TH2D*> (ProbsFile->Get("GL5"));

  // Read the limits for M and Sigma axis for each pdf
  // Note: we assume all the Z histograms to have the same limits
  // x is mass, y is sigma
  // We also take ResMass as the mean value between the two axis
  if( theMuonType_!=2 ) {
    MuScleFitUtils::ResHalfWidth[0] = (GLZ[0]->GetXaxis()->GetXmax() - GLZ[0]->GetXaxis()->GetXmin())/2.;
    MuScleFitUtils::ResMaxSigma[0] = (GLZ[0]->GetYaxis()->GetXmax() - GLZ[0]->GetYaxis()->GetXmin());
    MuScleFitUtils::ResMass[0] = (GLZ[0]->GetXaxis()->GetXmax() + GLZ[0]->GetXaxis()->GetXmin())/2.;
  }
  else {
    MuScleFitUtils::ResHalfWidth[0] = (GL[0]->GetXaxis()->GetXmax() - GL[0]->GetXaxis()->GetXmin())/2.;
    MuScleFitUtils::ResMaxSigma[0] = (GL[0]->GetYaxis()->GetXmax() - GL[0]->GetYaxis()->GetXmin());
    MuScleFitUtils::ResMass[0] = (GL[0]->GetXaxis()->GetXmax() + GL[0]->GetXaxis()->GetXmin())/2.;
  }
  for( int i=1; i<6; ++i ) {
    MuScleFitUtils::ResHalfWidth[i] = (GL[i]->GetXaxis()->GetXmax() - GL[i]->GetXaxis()->GetXmin())/2.;
    MuScleFitUtils::ResMaxSigma[i] = (GL[i]->GetYaxis()->GetXmax() - GL[i]->GetYaxis()->GetXmin());
    MuScleFitUtils::ResMass[i] = (GL[i]->GetXaxis()->GetXmax() + GL[i]->GetXaxis()->GetXmin())/2.;
  }
  // if( debug_>2 ) {
  for( int i=0; i<6; ++i ) {
    cout << "MuScleFitUtils::ResMass["<<i<<"] = " << MuScleFitUtils::ResMass[i] << endl;
    cout << "MuScleFitUtils::ResHalfWidth["<<i<<"] = " << MuScleFitUtils::ResHalfWidth[i] << endl;
    cout << "MuScleFitUtils::ResMaxSigma["<<i<<"] = " << MuScleFitUtils::ResMaxSigma[i] << endl;
  }
  // }

  // Extract normalization for mass slice in Y bins of Z
  // ---------------------------------------------------
  for (int iY=0; iY<24; iY++) {
    int nBinsX = GLZ[iY]->GetNbinsX();
    int nBinsY = GLZ[iY]->GetNbinsY();
    if( nBinsX != MuScleFitUtils::nbins+1 || nBinsY != MuScleFitUtils::nbins+1 ) {
      cout << "Error: for histogram \"" << GLZ[iY]->GetName() << "\" bins are not " << MuScleFitUtils::nbins << endl;
      cout<< "nBinsX = " << nBinsX << ", nBinsY = " << nBinsY << endl;
      exit(1);
    }
    // double sigmaMin = GLZ[iY]->GetYaxis()->GetXmin();
    for (int iy=0; iy<=MuScleFitUtils::nbins; iy++) {
      MuScleFitUtils::GLZNorm[iY][iy] = 1.;
      for (int ix=0; ix<=MuScleFitUtils::nbins; ix++) {
        MuScleFitUtils::GLZValue[iY][ix][iy] = GLZ[iY]->GetBinContent (ix+1, iy+1);
        MuScleFitUtils::GLZNorm[iY][iy] += MuScleFitUtils::GLZValue[iY][ix][iy]*(2*MuScleFitUtils::ResHalfWidth[0])/MuScleFitUtils::nbins;
        // MuScleFitUtils::GLZNorm[iY][iy] += MuScleFitUtils::GLZValue[iY][ix][iy];
      }

//      double sigma = sigmaMin + (double(iy)+0.1)*(MuScleFitUtils::ResMaxSigma[0])/double(nBinsY);
//      double minMass = MuScleFitUtils::ResMass[0] - MuScleFitUtils::ResHalfWidth[0];
//      double maxMass = MuScleFitUtils::ResMass[0] + MuScleFitUtils::ResHalfWidth[0];
//      TF1 * probForIntegral = new TF1("probForIntegral", ProbForIntegral(sigma, 0, iY, true), minMass, maxMass, 0);
//      MuScleFitUtils::GLZNorm[iY][iy] = probForIntegral->Integral(minMass, maxMass);
//
//      cout << "GLZNorm["<<iY<<"]["<<iy<<"] = " << MuScleFitUtils::GLZNorm[iY][iy] << endl;
//
//      double tempNorm = 0.;
//      for (int ix=0; ix<=MuScleFitUtils::nbins; ix++) {
//        // MuScleFitUtils::GLZValue[iY][ix][iy] = GLZ[iY]->GetBinContent (ix+1, iy+1);
//        tempNorm += MuScleFitUtils::GLZValue[iY][ix][iy];
//      }
//      cout << "Old normalization = " << tempNorm << endl;


//      if (debug_>2) cout << "GLZValue[" << iY << "][500][" << iy << "] = "
//                        << MuScleFitUtils::GLZValue[iY][500][iy]
//                        << " GLZNorm[" << iY << "][" << iy << "] = "
//                        << MuScleFitUtils::GLZNorm[iY][iy] << endl;
    }
  }
  // Extract normalization for each mass slice
  // -----------------------------------------
  for (int ires=0; ires<6; ires++) {
    int nBinsX = GL[ires]->GetNbinsX();
    int nBinsY = GL[ires]->GetNbinsY();
    if( nBinsX != MuScleFitUtils::nbins+1 || nBinsY != MuScleFitUtils::nbins+1 ) {
      cout << "Error: for histogram \"" << GLZ[ires]->GetName() << "\" bins are not " << MuScleFitUtils::nbins << endl;
      cout<< "nBinsX = " << nBinsX << ", nBinsY = " << nBinsY << endl;
      exit(1);
    }
    // double sigmaMin = GL[ires]->GetYaxis()->GetXmin();

    for (int iy=0; iy<=MuScleFitUtils::nbins; iy++) {

//      MuScleFitUtils::GLNorm[ires][iy] = 1.;
//      for (int ix=0; ix<=MuScleFitUtils::nbins; ix++) {
//        MuScleFitUtils::GLValue[ires][ix][iy] = GL[ires]->GetBinContent (ix+1, iy+1);
//      }
//
//      double sigma = sigmaMin + (double(iy)+0.1)*(MuScleFitUtils::ResMaxSigma[ires])/double(nBinsY);
//      double minMass = MuScleFitUtils::ResMass[ires] - MuScleFitUtils::ResHalfWidth[ires];
//      double maxMass = MuScleFitUtils::ResMass[ires] + MuScleFitUtils::ResHalfWidth[ires];
//
//      TF1 * probForIntegral = new TF1("probForIntegral", ProbForIntegral(sigma, ires, ires, false), minMass, maxMass, 0);
//      MuScleFitUtils::GLNorm[ires][iy] = probForIntegral->Integral(minMass, maxMass);
//
//      cout << "GLNorm["<<ires<<"]["<<iy<<"] = " << MuScleFitUtils::GLNorm[ires][iy] << endl;
//
//      double tempNorm = 0.;
//      for (int ix=0; ix<=MuScleFitUtils::nbins; ix++) {
//        tempNorm += MuScleFitUtils::GLValue[ires][ix][iy];
//      }
//      cout << "Old normalization = " << tempNorm << endl;

      MuScleFitUtils::GLNorm[ires][iy] = 0.;
//      double tempSum = 0.;
//      double tempSumNorm = 0.;
      for (int ix=0; ix<=MuScleFitUtils::nbins; ix++) {
        MuScleFitUtils::GLValue[ires][ix][iy] = GL[ires]->GetBinContent (ix+1, iy+1);
        // N.B. approximation: we should compute the integral of the function used to compute the probability (linear
        // interpolation of the mass points). This computation could be troublesome because the points have a steep
        // variation near the mass peak and the normal integral is not precise in these conditions.
        // Furthermore it is slow.
//        tempSum += MuScleFitUtils::GLValue[ires][ix][iy];
//        tempSumNorm += MuScleFitUtils::GLValue[ires][ix][iy]*(2*MuScleFitUtils::ResHalfWidth[ires]);
        MuScleFitUtils::GLNorm[ires][iy] += MuScleFitUtils::GLValue[ires][ix][iy]*(2*MuScleFitUtils::ResHalfWidth[ires])/MuScleFitUtils::nbins;
        // MuScleFitUtils::GLNorm[ires][iy] += MuScleFitUtils::GLValue[ires][ix][iy];
      }
//      cout << "tempSum["<<ires<<"]["<<iy<<"] = " << tempSum << endl;
//      cout << "tempSumNorm["<<ires<<"]["<<iy<<"] = " << tempSumNorm << endl;
//      cout << "MuScleFitUtils::GLNorm["<<ires<<"]["<<iy<<"] = " << MuScleFitUtils::GLNorm[ires][iy] << endl;

      //      if (debug_>2) cout << "GLValue[" << ires << "][500][" << iy << "] = "
//                        << MuScleFitUtils::GLValue[ires][500][iy]
//                        << " GLNorm[" << ires << "][" << iy << "] = "
//                        << MuScleFitUtils::GLNorm[ires][iy] << endl;
//      MuScleFitUtils::GLNorm[ires][iy] *= (2*MuScleFitUtils::ResHalfWidth[ires])/(double)MuScleFitUtils::nbins;
    }
  }
  // Free all the memory for the probability histograms.
  for ( int i=0; i<24; i++ ) {
    delete GLZ[i];
  }
  delete GL[0];
  delete GL[1];
  delete GL[2];
  delete GL[3];
  delete GL[4];
  delete GL[5];
  delete ProbsFile;
}

#endif
