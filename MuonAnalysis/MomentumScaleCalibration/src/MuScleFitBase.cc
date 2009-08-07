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
  if( MuScleFitUtils::resfind[0] != 1 ) {
    maxMass = 20.;
  }

  LogDebug("MuScleFitBase") << "Creating new histograms" << endl;

  mapHisto_["hRecBestMu"]      = new HParticle ("hRecBestMu", minMass, maxMass);
  mapHisto_["hRecBestMuVSEta"] = new HPartVSEta ("hRecBestMuVSEta");
  //mapHisto_["hRecBestMu_Acc"]  = new HParticle ("hRecBestMu_Acc", minMass, maxMass); 
  mapHisto_["hDeltaRecBestMu"] = new HDelta ("hDeltaRecBestMu");

  mapHisto_["hRecBestRes"]          = new HParticle   ("hRecBestRes", minMass, maxMass);
  mapHisto_["hRecBestResAllEvents"] = new HParticle   ("hRecBestResAllEvents", minMass, maxMass);
  //mapHisto_["hRecBestRes_Acc"] = new HParticle   ("hRecBestRes_Acc", minMass, maxMass); 
  // If not finding Z, use a smaller mass window
  mapHisto_["hRecBestResVSMu"] = new HMassVSPart ("hRecBestResVSMu", minMass, maxMass);

  // Likelihood values VS muon variables
  // -------------------------------------
  mapHisto_["hLikeVSMu"]       = new HLikelihoodVSPart ("hLikeVSMu");
  mapHisto_["hLikeVSMuMinus"]  = new HLikelihoodVSPart ("hLikeVSMuMinus");
  mapHisto_["hLikeVSMuPlus"]   = new HLikelihoodVSPart ("hLikeVSMuPlus");

  //Resolution VS muon kinematic
  //----------------------------
  mapHisto_["hResolMassVSMu"]         = new HResolutionVSPart( outputFile, "hResolMassVSMu" );
  mapHisto_["hResolPtGenVSMu"]        = new HResolutionVSPart( outputFile, "hResolPtGenVSMu" );
  mapHisto_["hResolPtSimVSMu"]        = new HResolutionVSPart( outputFile, "hResolPtSimVSMu" );
  mapHisto_["hResolEtaGenVSMu"]       = new HResolutionVSPart( outputFile, "hResolEtaGenVSMu" );
  mapHisto_["hResolEtaSimVSMu"]       = new HResolutionVSPart( outputFile, "hResolEtaSimVSMu" );
  mapHisto_["hResolThetaGenVSMu"]     = new HResolutionVSPart( outputFile, "hResolThetaGenVSMu" );
  mapHisto_["hResolThetaSimVSMu"]     = new HResolutionVSPart( outputFile, "hResolThetaSimVSMu" );
  mapHisto_["hResolCotgThetaGenVSMu"] = new HResolutionVSPart( outputFile, "hResolCotgThetaGenVSMu" );
  mapHisto_["hResolCotgThetaSimVSMu"] = new HResolutionVSPart( outputFile, "hResolCotgThetaSimVSMu" );
  mapHisto_["hResolPhiGenVSMu"]       = new HResolutionVSPart( outputFile, "hResolPhiGenVSMu" );
  mapHisto_["hResolPhiSimVSMu"]       = new HResolutionVSPart( outputFile, "hResolPhiSimVSMu" );

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
  mapHisto_["hFunctionResolPt"]        = new HFunctionResolution( outputFile, "hFunctionResolPt" );
  mapHisto_["hFunctionResolCotgTheta"] = new HFunctionResolution( outputFile, "hFunctionResolCotgTheta" );
  mapHisto_["hFunctionResolPhi"]       = new HFunctionResolution( outputFile, "hFunctionResolPhi" );

  // Mass probability histograms
  // ---------------------------
  mapHisto_["hMass_P"]      = new HTProfile( outputFile, "Mass_P", "Mass probability", 4000, 0., 200., 0., 1. );
  mapHisto_["hMass_fine_P"] = new HTProfile( outputFile, "Mass_fine_P", "Mass probability", 4000, 0., 20., 0., 1. );

  double ptMax = 40.;
  // Mass resolution vs (pt, eta) of the muons from MC
  mapHisto_["hMassResolutionVsPtEta"] = new HCovarianceVSxy( "Mass", "Mass", 100, 0., ptMax, 60, -3, 3, outputFile->mkdir("MassCovariance") );
  // Mass resolution vs (pt, eta) from resolution function
  mapHisto_["hFunctionResolMass"] = new HFunctionResolution( outputFile, "hFunctionResolMass", ptMax );
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
    //edm::FileInPath file("MuonAnalysis/MomentumScaleCalibration/test/Probs_new_1000_CTEQ.root");
    edm::FileInPath file("MuonAnalysis/MomentumScaleCalibration/test/Probs_new_Horace_CTEQ_1000.root");
    ProbsFile = new TFile (file.fullPath().c_str()); // NNBB need to reset this if MuScleFitUtils::nbins changes
    // ProbsFile = new TFile ("Probs_new_1000_CTEQ.root"); // NNBB need to reset this if MuScleFitUtils::nbins changes
    //cout << "[MuScleFit-Constructor]: Reading TH2D probabilities from Probs_new_1000_CTEQ.root file" << endl;
    cout << "[MuScleFit-Constructor]: Reading TH2D probabilities from Probs_new_Horace_CTEQ_1000.root file" << endl;
  } else {
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
  if( theMuonType_!=2 ) {
    MuScleFitUtils::ResHalfWidth[0] = (GLZ[0]->GetXaxis()->GetXmax() - GLZ[0]->GetXaxis()->GetXmin())/2;
    MuScleFitUtils::ResMaxSigma[0] = (GLZ[0]->GetYaxis()->GetXmax() - GLZ[0]->GetYaxis()->GetXmin());
  }
  else {
    MuScleFitUtils::ResHalfWidth[0] = (GL[0]->GetXaxis()->GetXmax() - GL[0]->GetXaxis()->GetXmin())/2;
    MuScleFitUtils::ResMaxSigma[0] = (GL[0]->GetYaxis()->GetXmax() - GL[0]->GetYaxis()->GetXmin());
  }
  for( int i=1; i<6; ++i ) {
    MuScleFitUtils::ResHalfWidth[i] = (GL[i]->GetXaxis()->GetXmax() - GL[i]->GetXaxis()->GetXmin())/2;
    MuScleFitUtils::ResMaxSigma[i] = (GL[i]->GetYaxis()->GetXmax() - GL[i]->GetYaxis()->GetXmin());
  }
  // if( debug_>2 ) {
    for( int i=0; i<6; ++i ) {
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
    for (int iy=0; iy<=MuScleFitUtils::nbins; iy++) {
      MuScleFitUtils::GLZNorm[iY][iy] = 0.;
      for (int ix=0; ix<=MuScleFitUtils::nbins; ix++) {
        MuScleFitUtils::GLZValue[iY][ix][iy] = GLZ[iY]->GetBinContent (ix+1, iy+1);
        MuScleFitUtils::GLZNorm[iY][iy] += MuScleFitUtils::GLZValue[iY][ix][iy];
      }
      if (debug_>2) cout << "GLZValue[" << iY << "][500][" << iy << "] = " 
                        << MuScleFitUtils::GLZValue[iY][500][iy] 
                        << " GLZNorm[" << iY << "][" << iy << "] = " 
                        << MuScleFitUtils::GLZNorm[iY][iy] << endl;
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
    for (int iy=0; iy<=MuScleFitUtils::nbins; iy++) {
      MuScleFitUtils::GLNorm[ires][iy] = 0.;
      for (int ix=0; ix<=MuScleFitUtils::nbins; ix++) {
        MuScleFitUtils::GLValue[ires][ix][iy] = GL[ires]->GetBinContent (ix+1, iy+1);
        MuScleFitUtils::GLNorm[ires][iy] += MuScleFitUtils::GLValue[ires][ix][iy];
      }
      if (debug_>2) cout << "GLValue[" << ires << "][500][" << iy << "] = " 
                        << MuScleFitUtils::GLValue[ires][500][iy] 
                        << " GLNorm[" << ires << "][" << iy << "] = " 
                        << MuScleFitUtils::GLNorm[ires][iy] << endl;
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
//  ProbsFile->Close();
  delete ProbsFile;
}

// void MuScleFitBase::readProbabilityDistributions( const edm::EventSetup & eventSetup )
// {

//   edm::ESHandle<MuScleFitLikelihoodPdf> likelihoodPdf;
//   eventSetup.get<MuScleFitLikelihoodPdfRcd>().get(likelihoodPdf);
//   string smSuffix = "";

//   // Should read different histograms in the two cases
//   if ( theMuonType_ == 2 ) {
//     smSuffix = "SM";
//     cout << "Error: Not yet implemented..." << endl;
//     exit(1);
//   }

//   edm::LogInfo("MuScleFit") << "[MuScleFit::readProbabilityDistributions] End Reading MuScleFitLikelihoodPdfRcd" << endl;
//   vector<PhysicsTools::Calibration::HistogramD2D>::const_iterator histo = likelihoodPdf->histograms.begin();
//   vector<string>::const_iterator name = likelihoodPdf->names.begin();
//   vector<int>::const_iterator xBins = likelihoodPdf->xBins.begin();
//   vector<int>::const_iterator yBins = likelihoodPdf->yBins.begin();
//   int ires = 0;
//   int iY = 0;
//   for( ; histo != likelihoodPdf->histograms.end(); ++histo, ++name, ++xBins, ++yBins ) {
//     int nBinsX = *xBins;
//     int nBinsY = *yBins;
//     if( nBinsX != MuScleFitUtils::nbins+1 || nBinsY != MuScleFitUtils::nbins+1 ) {
//       cout << "Error: for histogram \"" << *name << "\" bins are not " << MuScleFitUtils::nbins << endl; 
//       cout<< "nBinsX = " << nBinsX << ", nBinsY = " << nBinsY << endl;
//       exit(1);
//     }

//     // cout << "name = " << *name << endl;

//     // To separate the Z histograms from the other resonances we use tha names.
//     if( name->find("GLZ") != string::npos ) {
//       // ATTENTION: they are expected to be ordered

//       // cout << "For iY = " << iY << " the histogram is \"" << *name << "\"" << endl;

//       // Extract normalization for mass slice in Y bins of Z
//       // ---------------------------------------------------
//       for(int iy=1; iy<=nBinsY; iy++){
//         MuScleFitUtils::GLZNorm[iY][iy] = 0.;
//         for(int ix=1; ix<=nBinsX; ix++){
//           MuScleFitUtils::GLZValue[iY][ix][iy] = histo->binContent (ix+1, iy+1);
//           MuScleFitUtils::GLZNorm[iY][iy] += MuScleFitUtils::GLZValue[iY][ix][iy];
//         }
//         if (debug_>2) cout << "GLZValue[" << iY << "][500][" << iy << "] = " 
//                           << MuScleFitUtils::GLZValue[iY][500][iy] 
//                           << " GLZNorm[" << iY << "][" << iy << "] = " 
//                           << MuScleFitUtils::GLZNorm[iY][iy] << endl;
//       }
//       // increase the histogram counter
//       ++iY;
//     }
//     else {
//       // ATTENTION: they are expected to be ordered

//       // Extract normalization for each mass slice
//       // -----------------------------------------

//       // cout << "For ires = " << ires << " the histogram is \"" << *name << "\"" << endl;

//       // The histograms are filled like the root TH2D from which they are taken,
//       // meaning that bin = 0 is the underflow and nBins+1 is the overflow.
//       // We start from 1 and loop up to the last bin, excluding under/overflow.
//       for(int iy=1; iy<=nBinsY; iy++){
//         MuScleFitUtils::GLNorm[ires][iy] = 0.;
//         for(int ix=1; ix<=nBinsX; ix++){
//           MuScleFitUtils::GLValue[ires][ix][iy] = histo->binContent (ix+1, iy+1);
//           MuScleFitUtils::GLNorm[ires][iy] += MuScleFitUtils::GLValue[ires][ix][iy];
//         }
//         if (debug_>2) cout << "GLValue[" << ires << "][500][" << iy << "] = " 
//                           << MuScleFitUtils::GLValue[ires][500][iy] 
//                           << " GLNorm[" << ires << "][" << iy << "] = " 
//                           << MuScleFitUtils::GLNorm[ires][iy] << endl;
//       }
//       // increase the histogram counter
//       ++ires;
//     }
//   }
// }



#endif
