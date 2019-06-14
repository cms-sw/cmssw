#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/ParticleFlowReco/interface/PFTrajectoryPoint.h"

#include <iomanip>

using namespace reco;
using namespace std;

PFBlockElementTrack::PFBlockElementTrack(const PFRecTrackRef& ref)
    : PFBlockElement(TRACK), trackRefPF_(ref), trackRef_(ref->trackRef()), trackType_(0) {
  if (ref.isNull())
    throw cms::Exception("NullRef") << " PFBlockElementTrack constructed from a null reference to PFRecTrack.";

  const reco::PFTrajectoryPoint& atECAL = ref->extrapolatedPoint(reco::PFTrajectoryPoint::ECALEntrance);

  if (atECAL.isValid())
    positionAtECALEntrance_.SetCoordinates(atECAL.position().x(), atECAL.position().y(), atECAL.position().z());
  // if the position at ecal entrance is invalid,
  // positionAtECALEntrance_ is initialized by default to 0,0,0

  setTrackType(DEFAULT, true);
}

void PFBlockElementTrack::Dump(ostream& out, const char* tab) const {
  if (!out)
    return;

  if (!trackRef_.isNull()) {
    double charge = trackRef_->charge();
    double pt = trackRef_->pt();
    double p = trackRef_->p();
    string s = "  at vertex";
    double tracketa = trackRef_->eta();
    double trackphi = trackRef_->phi();

    // COLIN
    // the following lines rely on the presence of the PFRecTrack,
    // which for most people is not there (PFRecTracks are transient)
    // commented these lines out to remove the spurious error message
    // for the missing PFRecTrack product
    //     const reco::PFTrajectoryPoint& atECAL
    //       = trackRefPF_->extrapolatedPoint( reco::PFTrajectoryPoint::ECALShowerMax );
    //     // check if  reach ecal Shower max
    //     if( atECAL.isValid() ) {
    //       s = "  at ECAL shower max";
    //       tracketa = atECAL.position().Eta();
    //       trackphi = atECAL.position().Phi();
    //     }

    out << setprecision(0);
    out << tab << setw(7) << "charge=" << setw(3) << charge;
    out << setprecision(3);
    out << setiosflags(ios::right);
    out << setiosflags(ios::fixed);
    out << ", pT =" << setw(7) << pt;
    out << ", p =" << setw(7) << p;
    out << " (eta,phi)= (";
    out << tracketa << ",";
    out << trackphi << ")" << s;

    out << resetiosflags(ios::right | ios::fixed);
  }
}
