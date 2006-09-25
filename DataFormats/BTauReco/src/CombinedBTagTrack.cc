#include "DataFormats/BTauReco/interface/CombinedBTagTrack.h"
#include <limits>

using namespace std;

namespace {
  typedef std::numeric_limits<double> num;
}

reco::CombinedBTagTrack::CombinedBTagTrack::CombinedBTagTrack() :
  usedInSVX_ ( false ), rapidity_ ( num::quiet_NaN() ), 
  d0Sign_ ( num::quiet_NaN() ),
  ip2D_ ( Measurement1D() ),
  ip3D_ ( Measurement1D() ),
  aboveCharmMass_ ( false ), isValid_ ( false )
{} 

reco::CombinedBTagTrack::CombinedBTagTrack::CombinedBTagTrack(
           const reco::TrackRef & ref, bool usedInSVX, double rapidity,
           double d0Sign, double jetDistance,
           const Measurement1D & ip2d,
           const Measurement1D & ip3d,
           bool aboveCharmMass ) :
  trackRef_(ref),usedInSVX_(usedInSVX),rapidity_(rapidity),d0Sign_(d0Sign),
  jetDistance_(jetDistance),
  ip2D_ ( ip2d ), ip3D_ ( ip3d ), aboveCharmMass_(aboveCharmMass), isValid_(true)
{}

reco::CombinedBTagTrack::CombinedBTagTrack::CombinedBTagTrack(
           const reco::TrackRef & ref,
           double d0Sign, double jetDistance,
           const Measurement1D & ip2d,
           const Measurement1D & ip3d ) :
  trackRef_(ref),usedInSVX_(false),rapidity_( num::quiet_NaN() ),d0Sign_(d0Sign),
  jetDistance_(jetDistance),
  ip2D_ ( ip2d ), ip3D_ ( ip3d ), aboveCharmMass_( false ), isValid_(true)
{}


void reco::CombinedBTagTrack::CombinedBTagTrack::print() const
{
  cout << "*** printing trackData for combined b-tag info " << endl;
  cout << "    usedInSVX        " << usedInSVX()        << endl;
  cout << "    aboveCharmMass   " << aboveCharmMass()   << endl;
  cout << "    pt               " << pt()               << endl;
  cout << "    rapidity         " << rapidity()         << endl;
  cout << "    eta              " << eta()              << endl;
  cout << "    d0               " << d0()               << endl;
  cout << "    d0Sign           " << d0Sign()           << endl;
  cout << "    d0Error          " << d0Error()          << endl;
  cout << "    jetDistance      " << jetDistance()      << endl;
  cout << "    nHitsTotal       " << nHitsTotal()       << endl;
  cout << "    nHitsPixel       " << nHitsPixel()       << endl;
  cout << "    firstHitPixel    " << firstHitPixel()    << endl;
  cout << "    chi2             " << chi2()             << endl;
  cout << "    ip2D             " << ip2D().value()     << endl;
  cout << "    ip3D             " << ip3D().value()     << endl;
}

double reco::CombinedBTagTrack::chi2() const
{
  return trackRef_->chi2();
}

double reco::CombinedBTagTrack::pt() const
{
  return trackRef_->pt();
}

double reco::CombinedBTagTrack::eta() const
{
  return trackRef_->eta();
}

double reco::CombinedBTagTrack::d0() const
{
  return trackRef_->d0();
}

double reco::CombinedBTagTrack::d0Error() const
{
  return trackRef_->d0Error();
}

int reco::CombinedBTagTrack::nHitsTotal() const
{
  return trackRef_->recHitsSize();
}

int reco::CombinedBTagTrack::nHitsPixel() const
{
  return trackRef_->hitPattern().numberOfValidPixelHits();
}

bool reco::CombinedBTagTrack::firstHitPixel() const
{
  return trackRef_->hitPattern().hasValidHitInFirstPixelBarrel();
}

bool reco::CombinedBTagTrack::isValid() const
{
  return isValid_;
} 

const reco::TrackRef & reco::CombinedBTagTrack::trackRef() const
{
  return trackRef_;
}

double reco::CombinedBTagTrack::rapidity() const
{
  return rapidity_;
}

double reco::CombinedBTagTrack::d0Sign() const
{
  return d0Sign_;
}

double reco::CombinedBTagTrack::jetDistance() const
{
  return jetDistance_;
}

Measurement1D reco::CombinedBTagTrack::ip2D() const
{
  return ip2D_;
}

Measurement1D reco::CombinedBTagTrack::ip3D() const
{
  return ip3D_;
}

bool reco::CombinedBTagTrack::aboveCharmMass() const
{
  return aboveCharmMass_;
}

bool reco::CombinedBTagTrack::usedInSVX() const
{
  return usedInSVX_;
}

void reco::CombinedBTagTrack::setUsedInSVX( bool s )
{
  usedInSVX_=s;
}

void reco::CombinedBTagTrack::setAboveCharmMass( bool s )
{
  aboveCharmMass_=s;
}

void reco::CombinedBTagTrack::setRapidity ( double r )
{
  rapidity_=r;
}
