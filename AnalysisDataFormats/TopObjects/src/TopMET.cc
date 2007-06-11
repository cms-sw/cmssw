// -*- C++ -*-
//
// Package:     TopMET
// Class  :     TopMET
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Wed May 10 11:48:25 CEST 2006
// $Id: TopMET.cc,v 1.1 2007/05/22 16:36:50 heyninck Exp $
//

// system include files

// user include files
#include "AnalysisDataFormats/TopObjects/interface/TopMET.h"
#include "TMath.h"

TopMET::TopMET() {
	iscomplex_ = false;
}

TopMET::TopMET(METType aMet): TopObject<METType>(aMet) {
	iscomplex_ = false;
}

TopMET::~TopMET(){ }

void 		TopMET::setGenMET(reco::Particle gm)     { genMET = gm; }
void 		TopMET::setFitMET(TopParticle fm)   { fitMET = fm; }

reco::Particle	TopMET::getGenMET() const	    { return genMET; }
TopParticle	TopMET::getFitMET() const	    { return fitMET; }

//_________________________________________________
TopParticle TopMET::getPz(TopParticle lepton, int type) {

	// fixme: check you have initialized MET first
	
	double M_W  = 80.4;
	double M_mu =  0.10566;
	double emu = lepton.energy();
	double pxmu = lepton.px();
	double pymu = lepton.py();
	double pzmu = lepton.pz();
	double pxnu = fitMET.px();
	double pynu = fitMET.py();
	double pznu = 0.;

	double a = M_W*M_W - M_mu*M_mu + 2.0*pxmu*pxnu + 2.0*pymu*pynu;
	double A = 4.0*(emu*emu - pzmu*pzmu);
	double B = -4.0*a*pzmu;
	double C = 4.0*emu*emu*(pxnu*pxnu + pynu*pynu) - a*a;

	double tmproot = B*B - 4.0*A*C;

	if (tmproot<0) {
		iscomplex_= true;
		if (type==0) pznu = - B/(2*A); // take real part of complex roots

		//std::cout<< "complex sol. tmproot=" << tmproot << " pznu=" << pznu << std::endl;
	}
	else {
		iscomplex_ = false;
		double tmpsol1 = (-B + TMath::Sqrt(tmproot))/(2.0*A);
		double tmpsol2 = (-B - TMath::Sqrt(tmproot))/(2.0*A);
		
		if (type == 0 ) {
			// two real roots, pick one with pz closest to muon
			if (TMath::Abs(tmpsol2-pzmu) < TMath::Abs(tmpsol1-pzmu)) { pznu = tmpsol2;}
			else pznu = tmpsol1;
		}
	}

	Particle neutrino;
	neutrino.setP4( LorentzVector(pxnu, pynu, pznu, TMath::Sqrt(pxnu*pxnu + pynu*pynu + pznu*pznu ))) ;
	
	return neutrino;

	
}
