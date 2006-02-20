#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonExtra.h"
#include <algorithm>
using namespace std;
using namespace reco;

Muon::Muon( float chi2, unsigned short ndof, int found, int invalid, int lost,
	    const Parameters & p, const Covariance & c ) :
  TrackBase( chi2, ndof, found, invalid, lost, p, c ) {
}

Muon::Muon( float chi2, unsigned short ndof, int found, int invalid, int lost,
	    int q, const Point & v, const Vector & p, 
	    const PosMomError & err ) :
  TrackBase( chi2, ndof, found, invalid, lost, q, v, p, err ) {
}

const Muon::Point & Muon::outerPosition() const { 
  return extra_->outerPosition(); 
}

const Muon::Vector & Muon::outerMomentum() const { 
  return extra_->outerMomentum(); 
}

bool Muon::outerOk() const { 
  return extra_->outerOk(); 
}

recHit_iterator Muon::recHitsBegin() const { 
  return extra_->recHitsBegin(); 
}

recHit_iterator Muon::recHitsEnd() const { 
  return extra_->recHitsEnd(); 
}

size_t Muon::recHitsSize() const { 
  return extra_->recHitsSize(); 
}

double Muon::outerPx() const { 
  return extra_->outerPx(); 
}

double Muon::outerPy() const { 
  return extra_->outerPy(); 
}

double Muon::outerPz() const { 
  return extra_->outerPz(); 
}

double Muon::outerX() const { 
  return extra_->outerX(); 
}

double Muon::outerY() const { 
  return extra_->outerY(); 
}

double Muon::outerZ() const { 
  return extra_->outerZ(); 
}

double Muon::outerP() const { 
  return extra_->outerP(); 
}

double Muon::outerPt() const { 
  return extra_->outerPt(); 
}

double Muon::outerPhi() const { 
  return extra_->outerPhi(); 
}

double Muon::outerEta() const { 
  return extra_->outerEta(); 
}

double Muon::outerTheta() const { 
  return extra_->outerTheta(); 
}

double Muon::outerRadius() const { 
  return extra_->outerRadius(); 
}

const TrackRef & Muon::trackerSegment() const {
  return extra_->trackerSegment();
}

const TrackRef & Muon::muonSegment() const {
  return extra_->muonSegment();
}


