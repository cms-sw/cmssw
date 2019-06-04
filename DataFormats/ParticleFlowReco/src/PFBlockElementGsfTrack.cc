#include "DataFormats/ParticleFlowReco/interface/PFBlockElementGsfTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/ParticleFlowReco/interface/PFTrajectoryPoint.h"

#include <iomanip>

using namespace reco;
using namespace std;

PFBlockElementGsfTrack::PFBlockElementGsfTrack(const GsfPFRecTrackRef& gsfref,
                                               const math::XYZTLorentzVector& Pin,
                                               const math::XYZTLorentzVector& Pout)
    : PFBlockElement(GSF),
      GsftrackRefPF_(gsfref),
      GsftrackRef_(gsfref->gsfTrackRef()),
      Pin_(Pin),
      Pout_(Pout),
      trackType_(0) {
  if (gsfref.isNull())
    throw cms::Exception("NullRef") << " PFBlockElementGsfTrack constructed from a null reference to PFGsfRecTrack.";
  const reco::PFTrajectoryPoint& atECAL = gsfref->extrapolatedPoint(reco::PFTrajectoryPoint::ECALEntrance);
  if (atECAL.isValid())
    positionAtECALEntrance_.SetCoordinates(atECAL.position().x(), atECAL.position().y(), atECAL.position().z());

  setTrackType(DEFAULT, true);
}

void PFBlockElementGsfTrack::Dump(ostream& out, const char* tab) const {
  if (!out)
    return;

  if (!GsftrackRefPF_.isNull()) {
    //    double charge = trackPF().charge;
    double charge = GsftrackRefPF_->charge();
    math::XYZTLorentzVector pin = Pin_;
    math::XYZTLorentzVector pout = Pout_;
    double ptin = pin.pt();
    double etain = pin.eta();
    double phiin = pin.phi();
    double ptout = pout.pt();
    double etaout = pout.eta();
    double phiout = pout.phi();
    out << setprecision(0);
    out << tab << setw(7) << "charge=" << setw(3) << charge;
    out << setprecision(3);
    out << setiosflags(ios::right);
    out << setiosflags(ios::fixed);
    out << ", Inner pT  =" << setw(7) << ptin;
    out << " Inner (eta,phi)= (";
    out << etain << ",";
    out << phiin << ")";
    out << ", Outer pT  =" << setw(7) << ptout;
    out << " Outer (eta,phi)= (";
    out << etaout << ",";
    out << phiout << ")";
    out << resetiosflags(ios::right | ios::fixed);
  }
}
