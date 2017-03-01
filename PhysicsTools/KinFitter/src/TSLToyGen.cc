#include <iostream>
#include "PhysicsTools/KinFitter/interface/TSLToyGen.h"
#include "TMatrixD.h" 
#include "PhysicsTools/KinFitter/interface/TFitConstraintM.h"
#include "PhysicsTools/KinFitter/interface/TFitConstraintEp.h"
#include "PhysicsTools/KinFitter/interface/TAbsFitParticle.h"
#include "TH1.h"
#include "TMath.h"
#include "TRandom.h"
#include "TString.h"



TSLToyGen::TSLToyGen( const TAbsFitParticle* bReco, const TAbsFitParticle* lepton, 
		      const TAbsFitParticle* X, const TAbsFitParticle* neutrino):
  _inimeasParticles(0),
  _iniunmeasParticles(0),
  _measParticles(0),
  _unmeasParticles(0),
  _Y4S(0., 0., 0.)

{
  
  // Clone input particles
  _iniBreco = bReco->clone( bReco->GetName() + (TString) "INI" );
  _breco = bReco->clone( bReco->GetName() + (TString) "SMEAR" );
  _iniLepton = lepton->clone( lepton->GetName() + (TString) "INI" );
  _lepton = lepton->clone( lepton->GetName() + (TString) "SMEAR" );
  _iniX = X->clone( X->GetName() + (TString) "INI" );
  _X = X->clone( X->GetName() + (TString) "SMEAR" );
  _iniNeutrino = neutrino->clone( neutrino->GetName() +(TString)  "INI" );
  _neutrino = neutrino->clone( neutrino->GetName() + (TString) "SMEAR" );

  _printPartIni = false;
  _printConsIni = false;
  _printSmearedPartBefore = false;
  _printConsBefore = false;
  _printConsAfter = false;
  _printPartAfter = false;
  _withMassConstraint = false;
  _withMPDGCons = false;
  _doCheckConstraintsTruth = true;
}

TSLToyGen::~TSLToyGen() {

  delete _iniBreco;
  delete _iniLepton;
  delete _iniX;
  delete _iniNeutrino;

  delete _breco;
  delete _lepton;
  delete _X;
  delete _neutrino;

}

Bool_t TSLToyGen::doToyExperiments( Int_t nbExperiments ) {

  // define fitter
  TKinFitter fitter;

  std::vector<TAbsFitParticle*> ParVec(0);
  ParVec.push_back(_breco);
  ParVec.push_back(_lepton);
  ParVec.push_back(_X);
  ParVec.push_back(_neutrino);

  fitter.addMeasParticle(_breco);
  _inimeasParticles.push_back( _iniBreco );
  _measParticles.push_back( _breco );
  fitter.addMeasParticle(_lepton);
  _inimeasParticles.push_back( _iniLepton );
  _measParticles.push_back( _lepton );
  fitter.addMeasParticle(_X);
  _inimeasParticles.push_back( _iniX );
  _measParticles.push_back( _X );
  fitter.addUnmeasParticle(_neutrino);
  _iniunmeasParticles.push_back( _iniNeutrino );
  _iniunmeasParticles.push_back( _neutrino );

 // Calculate Y4S
  _Y4S.SetXYZ(0., 0., 0.);
  for (unsigned int p = 0;  p < _inimeasParticles.size(); p++) {
    _Y4S += _inimeasParticles[p]->getIni4Vec()->Vect();
  }
  _Y4S += _iniNeutrino->getIni4Vec()->Vect();
  //_Y4S.SetXYZ(-0.1212, -0.0033, 5.8784);
  Double_t EY4S = TMath::Sqrt( _Y4S.Mag2() + 10.58*10.58 );
  //  std::cout << "_Y4S : " <<_Y4S.x() << " / " << _Y4S.y() << " / " << _Y4S.z() << " / " <<EY4S<< std::endl;

  TFitConstraintEp pXCons( "pX", "pX", &ParVec, TFitConstraintEp::pX, _Y4S.x() );
  TFitConstraintEp pYCons( "pY", "pY", &ParVec, TFitConstraintEp::pY, _Y4S.y() );
  TFitConstraintEp pZCons( "pZ", "pZ", &ParVec, TFitConstraintEp::pZ, _Y4S.z() );
  TFitConstraintEp ECons( "E", "E", &ParVec, TFitConstraintEp::E, EY4S );
  TFitConstraintM MCons( "MassConstraint", "Mass-Constraint", 0, 0 ,0);
  MCons.addParticle1( _breco );
  MCons.addParticles2( _lepton, _neutrino, _X );
  TFitConstraintM MPDGCons( "MPDGCons", "MPDGCons", 0, 0 , 5.279 );
  MPDGCons.addParticles1( _lepton, _neutrino, _X );
//   TFitConstraintE EBCons( "EBXlnuCons", "EBXlnuCons", 0, 0 );
//   EBCons.addParticle1( _breco );
//   EBCons.addParticles2( _lepton, _neutrino, _X );

  fitter.addConstraint(&pXCons);
  fitter.addConstraint(&pYCons);
  fitter.addConstraint(&pZCons);
  fitter.addConstraint(&ECons);
  if (_withMassConstraint)
    fitter.addConstraint(&MCons);
  if(_withMPDGCons)
     fitter.addConstraint(&MPDGCons);
  // fitter.addConstraint(&EBCons);

  fitter.setMaxNbIter( 50 );
  fitter.setMaxDeltaS( 5e-5 );
  fitter.setMaxF( 1e-4 );
  fitter.setVerbosity(0);

  if( _printPartIni ) {
    std::cout << std::endl
	      << "----------------------------------" << std::endl;
    std::cout << "--- PRINTING INITIAL PARTICLES ---" << std::endl;
    std::cout << "----------------------------------" << std::endl ;
    _iniBreco->print();
    _iniLepton->print();
    _iniX->print();
    _iniNeutrino->print();
    std::cout << std::endl << std::endl;
  }
 
  if( _printConsIni ) {
    std::cout << std::endl
	      << "-------------------------------------------------" << std::endl;
    std::cout << "INITIAL CONSTRAINTS " << std::endl ;
    std::cout << "-------------------------------------------------" << std::endl;
    std::cout << "     M: " << MCons.getCurrentValue()
	      << "  MPDG: " << MPDGCons.getCurrentValue()
	      << "    px: " << pXCons.getCurrentValue()  
	      << "    py: " << pYCons.getCurrentValue()
	      << "    pz: " << pZCons.getCurrentValue() 
	      << "     E: " << ECons.getCurrentValue() << std::endl << std::endl;
  }

  // Check initial constraints
  if (  _doCheckConstraintsTruth ) {
    if (fitter.getF() > fitter.getMaxF()) {
      //std::cout << "Initial constraints are not fulfilled." << std::endl;
      return false;
    }
  }
  
  // create histograms
  createHists();

  // perform pseudo experiments
  for (int i = 0; i < nbExperiments; i++) {

    smearParticles();

    if( _printSmearedPartBefore ) {
      std::cout <<  std::endl  
		<< "-------------------------------------------------------" << std::endl ;
      std::cout << "--- PRINTING SMEARED PARTICLES BEFORE FIT FOR experiment # " <<i+1 << std::endl;
      std::cout << "-------------------------------------------------------" << std::endl;
      _breco->print();
      _lepton->print();
      _X->print();
      _neutrino->print();
    }
    
  
    if( _printConsBefore ) {
      std::cout << std::endl
		<< "-------------------------------------------------" << std::endl;
      std::cout << "INITIAL (SMEARED) CONSTRAINTS FOR experiment # "<< i+1 << std::endl ;
      std::cout << "-------------------------------------------------" << std::endl;
      std::cout << "     M: " << MCons.getCurrentValue() 
		<< "    px: " << pXCons.getCurrentValue()  
		<< "    py: " << pYCons.getCurrentValue()
		<< "    pz: " << pZCons.getCurrentValue() 
		<< "     E: " << ECons.getCurrentValue() << std::endl << std::endl;
    }
    
    fitter.fit();
    
    if( _printConsAfter) {
      std::cout << std::endl
		<< "-------------------------------------------------" << std::endl;
      std::cout << " CONSTRAINTS AFTER FIT FOR experiment # "<< i+1 << std::endl ;
      std::cout << "-------------------------------------------------" << std::endl;
      std::cout << "     M: " << MCons.getCurrentValue() 
		<< "  MPDG: " << MPDGCons.getCurrentValue()
		<< "    px: " << pXCons.getCurrentValue()  
		<< "    py: " << pYCons.getCurrentValue()
		<< "    pz: " << pZCons.getCurrentValue() 
		<< "     E: " << ECons.getCurrentValue() << std::endl << std::endl;
    }
    
    if( _printPartAfter ) {
      std::cout <<  std::endl  
		<< "--------------------------------------------------------" << std::endl ;
      std::cout << "--- PRINTING PARTICLES AFTER FIT FOR experiment # "<< i+1 << std::endl;
      std::cout << "--------------------------------------------------------" << std::endl;
      _breco->print();
      _lepton->print();
      _X->print();
      _neutrino->print();
    }
    
    _histStatus->Fill( fitter.getStatus() );
    _histNIter->Fill( fitter.getNbIter() );
    if ( fitter.getStatus() == 0 ) {
      _histPChi2->Fill( TMath::Prob( fitter.getS(), fitter.getNDF() ) );
      _histChi2->Fill( fitter.getS());
      fillPull1();
      fillPull2();
      fillPar();
      fillM();
    }

    if (i % 176 == 0) {
      std::cout << "\r";  
      std::cout <<" ------ "<< (Double_t) i/nbExperiments*100. << " % PROCESSED ------";
      std::cout.flush();
    }    
  }
  
  return true;

}

void TSLToyGen::smearParticles() {
  // Smear measured particles

  for (unsigned int p = 0; p < _measParticles.size(); p++) {

    TAbsFitParticle* particle =  _measParticles[p];
    TAbsFitParticle* iniParticle =  _inimeasParticles[p];
    TMatrixD parIni( *(iniParticle->getParIni()) );
    const TMatrixD* covM = iniParticle->getCovMatrix();
    for (int m = 0; m < iniParticle->getNPar(); m++) {
      parIni(m, 0) += gRandom->Gaus(0., TMath::Sqrt( (*covM)(m,m) ) );
    }				       
    TLorentzVector* ini4Vec = iniParticle->calc4Vec( &parIni );
    particle->setIni4Vec( ini4Vec );
    delete ini4Vec;
    TLorentzVector vectrue( *_inimeasParticles[p]->getIni4Vec() );
    TMatrixD* partrue = _measParticles[p]->transform( vectrue );
    //_measParticles[p]->setParIni(partrue);
    delete partrue;
  }

  // Calculate neutrino
  TVector3 nuP3 = _Y4S;
  for (unsigned int p = 0;  p < _measParticles.size(); p++) {
    nuP3 -= _measParticles[p]->getCurr4Vec()->Vect();
  }
  TLorentzVector ini4VecNeutrino  ;
  ini4VecNeutrino.SetXYZM( nuP3.X(), nuP3.Y(), nuP3.Z(), 0. );
  _neutrino->setIni4Vec( &ini4VecNeutrino );

}

void TSLToyGen::fillPull1() {

  Int_t histindex = 0;
  for (unsigned int p = 0; p < _measParticles.size(); p++) {

    //const TAbsFitParticle* particle = _measParticles[p];
    TLorentzVector vectrue( *_inimeasParticles[p]->getIni4Vec() );
    TMatrixD* partrue = _measParticles[p]->transform( vectrue );
    const TMatrixD* parfit = _measParticles[p]->getParCurr();

    TMatrixD parpull( *parfit );
    parpull -= (*partrue);
    const TMatrixD* covMatrixFit = _measParticles[p]->getCovMatrixFit();
    for (int i = 0; i < parpull.GetNrows(); i++) {

      ((TH1D*) _histsDiff1[histindex])->Fill( parpull(i, 0) );
      parpull(i, 0) /= TMath::Sqrt( (*covMatrixFit)(i, i) );
      ((TH1D*) _histsPull1[histindex])->Fill( parpull(i, 0) );
      ((TH1D*) _histsError1[histindex])->Fill( TMath::Sqrt( (*covMatrixFit)(i, i) ) );
      histindex++;

    }
    delete partrue;

  }

}

void TSLToyGen::fillPull2() {

  Int_t histindex = 0;
  for (unsigned int p = 0; p <  _measParticles.size(); p++) {

    const TMatrixD* pull = _measParticles[p]->getPull();
    const TMatrixD* VDeltaY = _measParticles[p]->getCovMatrixDeltaY();
    TMatrixD pardiff( *(_measParticles[p]->getParCurr()) );
    pardiff -= *(_measParticles[p]->getParIni());
    for (int i = 0; i < pull->GetNrows(); i++) {
      ( (TH1D*) _histsPull2[histindex])->Fill( (*pull)(i, 0) );
      ( (TH1D*) _histsError2[histindex])->Fill( TMath::Sqrt( (*VDeltaY)(i, i) ) );
      ( (TH1D*) _histsDiff2[histindex])->Fill( pardiff(i, 0) );
      histindex++;
    }
  }
  
}

void TSLToyGen::fillPar() {

  Int_t histindex = 0;
  for (unsigned int p = 0; p <  _measParticles.size(); p++) {

    const TMatrixD* partrue = _inimeasParticles[p]->getParIni();
    const TMatrixD* parsmear = _measParticles[p]->getParIni();
    const TMatrixD* parfit = _measParticles[p]->getParCurr();
    for (int i = 0; i < partrue->GetNrows(); i++) {
      ( (TH1D*) _histsParTrue[histindex])->Fill( (*partrue)(i, 0) );
      ( (TH1D*) _histsParSmear[histindex])->Fill( (*parsmear)(i, 0) );
      ( (TH1D*) _histsParFit[histindex])->Fill( (*parfit)(i, 0) );
      histindex++;
    }
  }

}

void TSLToyGen::fillM() {

  _histMBrecoTrue->Fill( _iniBreco->getIni4Vec()->M() );
  _histMBrecoSmear->Fill( _breco->getIni4Vec()->M() );
  _histMBrecoFit->Fill( _breco->getCurr4Vec()->M() );

  _histMXTrue->Fill( _iniX->getIni4Vec()->M() );
  _histMXSmear->Fill( _X->getIni4Vec()->M() );
  _histMXFit->Fill( _X->getCurr4Vec()->M() );

  TLorentzVector xlnutrue =  *(_iniLepton->getIni4Vec());
  xlnutrue += *(_iniX->getIni4Vec());
  xlnutrue += *(_iniNeutrino->getIni4Vec());
  _histMXlnuTrue->Fill( xlnutrue.M() );

  TLorentzVector xlnusmear =  *(_lepton->getIni4Vec());
  xlnusmear += *(_X->getIni4Vec());
  xlnusmear += *(_neutrino->getIni4Vec());
  _histMXlnuSmear->Fill( xlnusmear.M() );

  TLorentzVector xlnufit =  *(_lepton->getCurr4Vec());
  xlnufit += *(_X->getCurr4Vec());
  xlnufit += *(_neutrino->getCurr4Vec());
  _histMXlnuFit->Fill( xlnufit.M() );

}

void  TSLToyGen::createHists() {

  _histStatus = new TH1D( "hStatus", "Status of the Fit", 16, -1, 15);
  _histNIter = new TH1D( "hNIter", "Number of iterations", 100, 0, 100);
  _histPChi2 = new TH1D( "hPChi2", "Chi2 probability", 100, 0., 1.);
  _histChi2 = new TH1D(  "hChi2", "Chi2 ", 200, 0., 20.);
  _histMBrecoTrue = new TH1D( "histMBrecoTrue", "histMBrecoTrue", 2000, 4., 6.);
  _histMBrecoSmear = new TH1D( "histMBrecoSmear", "histMBrecoSmear", 2000, 4., 6.);
  _histMBrecoFit = new TH1D( "histMBrecoFit", "histMBrecoFit", 2000, 4., 6.);
  _histMXTrue = new TH1D( "histMXTrue", "histMXTrue", 600, 0., 6.);
  _histMXSmear = new TH1D( "histMXSmear", "histMXSmear", 600, 0., 6.);
  _histMXFit = new TH1D( "histMXFit", "histMXFit", 600, 0., 6.);
  _histMXlnuTrue = new TH1D( "histMXlnuTrue", "histMXlnuTrue", 3000, 4., 7.);
  _histMXlnuSmear = new TH1D( "histMXlnuSmear", "histMXlnuSmear", 500, 3., 8.);
  _histMXlnuFit = new TH1D( "histMXlnuFit", "histMXlnuFit", 3000, 4., 7.);

  _histsParTrue.Clear();
  _histsParSmear.Clear();
  _histsParFit.Clear();
  _histsPull1.Clear();
  _histsError1.Clear();
  _histsDiff1.Clear();
  _histsPull2.Clear();
  _histsError2.Clear();
  _histsDiff2.Clear();

  TObjArray histarrays;
  histarrays.Add( &_histsParTrue );
  histarrays.Add( &_histsParSmear );
  histarrays.Add( &_histsParFit );
  histarrays.Add( &_histsPull1 );
  histarrays.Add( &_histsError1 );
  histarrays.Add( &_histsDiff1 );
  histarrays.Add( &_histsPull2 );
  histarrays.Add( &_histsError2 );
  histarrays.Add( &_histsDiff2 );


  TString histnames[] = {"hParTrue", "hParSmear", "hParFit", "hPull1", "hError1", "hDiff1", "hPull2", "hError2", "hDiff2" };

  TArrayD arrmin( histarrays.GetEntries() );
  TArrayD arrmax( histarrays.GetEntries() );
  arrmin[0] = 0.;   arrmax[0] = 2.;   // Parameters
  arrmin[1] = 0.;   arrmax[1] = 2.;
  arrmin[2] = 0.;   arrmax[2] = 2.;

  arrmin[3] = -3.;   arrmax[3] = 3.;   // Pull1
  arrmin[4] = 0.;   arrmax[4] = .2;    //Error1
  arrmin[5] = -.5;   arrmax[5] = .5;   //Diff1
  arrmin[6] = -3.;   arrmax[6] = 3.;   // Pull2
  arrmin[7] = 0.;   arrmax[7] = 0.2;   //Error2
  arrmin[8] = -.5;   arrmax[8] = .5;   //Diff2

  for (unsigned int p = 0; p <  _measParticles.size(); p++) {

    const TAbsFitParticle* particle = _measParticles[p];
    //    const TMatrixD* covMatrix = particle->getCovMatrix();

    for (int i = 0; i < particle->getNPar(); i++) {
      for (int h = 0; h < histarrays.GetEntries(); h++ ) {
	TString name = histnames[h] + (TString) particle->GetName();
	name += i;
	if ( h < 3) {
	  const TMatrixD* parfit = _measParticles[p]->getParCurr();
	  arrmin[h] = (*parfit)(i,0)*0.5;
	  arrmax[h] = (*parfit)(i,0)*1.5;
	}
	TH1D* newhisto =  new TH1D( name, name, 100, arrmin[h], arrmax[h]) ; 
	((TObjArray*) histarrays[h])->Add( newhisto );

	//	newhisto->SetCanExtend(TH1::kAllAxes);
      }
    }
  }
}
