//
// Author:  Steven Lowette
// Created: Thu May  3 10:37:17 PDT 2007
//
// $Id: TopMET.cc,v 1.2 2007/06/11 21:00:37 yumiceva Exp $
//

#include "AnalysisDataFormats/TopObjects/interface/TopMET.h"

#include "TMath.h"


/// default constructor
TopMET::TopMET() : isComplex_(false) {
}


/// constructor from TopMETType
TopMET::TopMET(const TopMETType & aMET) : TopObject<TopMETType>(aMET), isComplex_(false) {
}


/// destructor
TopMET::~TopMET() {
}


/// return the generated MET from neutrinos
reco::Particle	TopMET::getGenMET() const {
  return (genMET_.size() > 0 ?
    genMET_.front() :
    reco::Particle(0, reco::Particle::LorentzVector(0, 0, 0, 0), reco::Particle::Point(0,0,0))
  );
}


/// return the fitted MET
TopParticle TopMET::getFitMET() const {
  return (fitMET_.size() > 0 ?
    fitMET_.front() :
    TopParticle()
  );
}


/// reconstruct the MET Pz from the W mass constraint
TopParticle TopMET::getPz(const TopParticle & lepton, int type) {

// FIXME: currently the isComplex_ in the current class is changed, and a
// particle is returned, so the information on complex or not is seperated
// we should probably return a TopMET, having the isComplex_ set there.
// Whether we should actually change the current object or give a MET back
// is another issue I need to think about; it is related to the fact that
// we should seperate algorithmic code from dataformats; probably we should
// move this code to a helper class in the TopEventProducers directory.

	// fixme: check you have initialized MET first
	
	double M_W  = 80.4;
	double M_mu =  0.10566;
	double emu = lepton.energy();
	double pxmu = lepton.px();
	double pymu = lepton.py();
	double pzmu = lepton.pz();
	double pxnu = this->px();
	double pynu = this->py();
	double pznu = 0.;

	double a = M_W*M_W - M_mu*M_mu + 2.0*pxmu*pxnu + 2.0*pymu*pynu;
	double A = 4.0*(emu*emu - pzmu*pzmu);
	double B = -4.0*a*pzmu;
	double C = 4.0*emu*emu*(pxnu*pxnu + pynu*pynu) - a*a;

	double tmproot = B*B - 4.0*A*C;

	if (tmproot<0) {
		isComplex_= true;
		if (type==0) pznu = - B/(2*A); // take real part of complex roots

		//std::cout<< "complex sol. tmproot=" << tmproot << " pznu=" << pznu << std::endl;
	}
	else {
		isComplex_ = false;
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


/// return whether the p_z solution from the W mass constraint is complex
bool TopMET::isSolutionComplex() const {
  return isComplex_;
}


/// method to set the generated MET
void TopMET::setGenMET(const reco::Particle & gm) {
  genMET_.clear();
  genMET_.push_back(gm);
}


/// method to set the fitted MET
void TopMET::setFitMET(const TopParticle & fm) {
  fitMET_.clear();
  fitMET_.push_back(fm);
}
