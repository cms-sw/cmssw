#include "DataFormats/BTauReco/interface/CombinedBTagVertex.h"

using namespace std;

void reco::CombinedBTagVertex::CombinedBTagVertex::print() const
{
  cout << "****** print CombinedBTagVertex from extended bTag information (combined bTag) " << endl;
  cout << "chi2                         " << chi2()       << endl;
  cout << "ndof                         " << ndof()       << endl;
  cout << "nTracks                      " << nTracks()    << endl; 
  cout << "mass                         " << mass()       << endl;   
  cout << "isV0                         " << isV0()       << endl;     
  cout << "fracPV                       " << fracPV()     << endl;    
  cout << "flightDistanced2             " << flightDistance2D().value() << endl;
  cout << "flightDistanced3             " << flightDistance3D().value() << endl;
}

reco::CombinedBTagVertex::CombinedBTagVertex( const reco::Vertex & vertex,
        const GlobalVector & trackVector, double mass,
        bool isV0, double fracPV, const Measurement1D & d2, const Measurement1D & d3 ) : 
  vertex_(vertex), trackVector_ ( trackVector ), mass_ ( mass ), isV0_ ( isV0 ), 
  fracPV_ ( fracPV ), d2_(d2), d3_(d3), isValid_(true)
{}

reco::CombinedBTagVertex::CombinedBTagVertex() : mass_(0.), isV0_(false),
  fracPV_(0.), isValid_(false)
{
}

void reco::CombinedBTagVertex::setFlightDistance2D ( const Measurement1D & d )
{
  d2_=d;
}

void reco::CombinedBTagVertex::setFlightDistance3D ( const Measurement1D & d )
{
  d3_=d;
}

double reco::CombinedBTagVertex::chi2() const
{
  return vertex_.chi2();
}

double reco::CombinedBTagVertex::ndof() const
{
  return vertex_.ndof();
}

int reco::CombinedBTagVertex::nTracks() const
{
  return vertex_.tracksSize();
}

double reco::CombinedBTagVertex::mass() const
{
  return mass_;
}

const reco::Vertex & reco::CombinedBTagVertex::vertex() const
{
  return vertex_;
}

const GlobalVector & reco::CombinedBTagVertex::trackVector() const
{
  return trackVector_;
}

bool reco::CombinedBTagVertex::isV0() const
{
  return isV0_;
}

double reco::CombinedBTagVertex::fracPV() const
{
  return fracPV_;
}

Measurement1D reco::CombinedBTagVertex::flightDistance2D() const
{
  return d2_;
}

Measurement1D reco::CombinedBTagVertex::flightDistance3D() const
{
  return d3_;
}

bool reco::CombinedBTagVertex::isValid() const
{
  return isValid_;
}
