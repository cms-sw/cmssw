#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"
#include "DataFormats/Math/interface/normalizedPhi.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "TMath.h"

using BVector2D = Basic2DVector<double>;
using Vector2D = Basic2DVector<double>::MathVector;
namespace {
  const Vector2D one2D = BVector2D(1.0, 1.0).v;
  const Vector2D fivepercent2D = BVector2D(0.05, 0.05).v;
}  // namespace

// to enable debugs
//#define PFLOW_DEBUG

double LinkByRecHit::testTrackAndClusterByRecHit(const reco::PFRecTrack& track,
                                                 const reco::PFCluster& cluster,
                                                 bool isBrem,
                                                 bool debug) {
#ifdef PFLOW_DEBUG
  if (debug)
    std::cout << "entering test link by rechit function" << std::endl;
#endif

  //cluster position
  auto clustereta = cluster.positionREP().Eta();
  auto clusterphi = cluster.positionREP().Phi();
  auto clusterZ = cluster.position().Z();

  bool barrel = false;
  bool hcal = false;
  // double distance = 999999.9;
  double horesolscale = 1.0;

  //track extrapolation
  const reco::PFTrajectoryPoint& atVertex = track.extrapolatedPoint(reco::PFTrajectoryPoint::ClosestApproach);
  const reco::PFTrajectoryPoint& atECAL = track.extrapolatedPoint(reco::PFTrajectoryPoint::ECALShowerMax);

  //track at calo's
  double tracketa = 999999.9;
  double trackphi = 999999.9;
  double track_X = 999999.9;
  double track_Y = 999999.9;
  double track_Z = 999999.9;
  double dHEta = 0.;
  double dHPhi = 0.;

  // Quantities at vertex
  double trackPt = isBrem ? 999. : sqrt(atVertex.momentum().Vect().Perp2());
  // double trackEta = isBrem ? 999. : atVertex.momentum().Vect().Eta();

  switch (cluster.layer()) {
    case PFLayer::ECAL_BARREL:
      barrel = true;
      [[fallthrough]];
    case PFLayer::ECAL_ENDCAP:
#ifdef PFLOW_DEBUG
      if (debug)
        std::cout << "Fetching Ecal Resolution Maps" << std::endl;
#endif
      // did not reach ecal, cannot be associated with a cluster.
      if (!atECAL.isValid())
        return -1.;

      tracketa = atECAL.positionREP().Eta();
      trackphi = atECAL.positionREP().Phi();
      track_X = atECAL.position().X();
      track_Y = atECAL.position().Y();
      track_Z = atECAL.position().Z();

      /*    distance 
      = std::sqrt( (track_X-clusterX)*(track_X-clusterX)
		  +(track_Y-clusterY)*(track_Y-clusterY)
		  +(track_Z-clusterZ)*(track_Z-clusterZ)
		   );
*/
      break;

    case PFLayer::HCAL_BARREL1:
      barrel = true;
      [[fallthrough]];
    case PFLayer::HCAL_ENDCAP:
#ifdef PFLOW_DEBUG
      if (debug)
        std::cout << "Fetching Hcal Resolution Maps" << std::endl;
#endif
      if (isBrem) {
        return -1.;
      } else {
        hcal = true;
        const reco::PFTrajectoryPoint& atHCAL = track.extrapolatedPoint(reco::PFTrajectoryPoint::HCALEntrance);
        const reco::PFTrajectoryPoint& atHCALExit = track.extrapolatedPoint(reco::PFTrajectoryPoint::HCALExit);
        // did not reach hcal, cannot be associated with a cluster.
        if (!atHCAL.isValid())
          return -1.;

        // The link is computed between 0 and ~1 interaction length in HCAL
        dHEta = atHCALExit.positionREP().Eta() - atHCAL.positionREP().Eta();
        dHPhi = atHCALExit.positionREP().Phi() - atHCAL.positionREP().Phi();
        if (dHPhi > M_PI)
          dHPhi = dHPhi - 2. * M_PI;
        else if (dHPhi < -M_PI)
          dHPhi = dHPhi + 2. * M_PI;
        tracketa = atHCAL.positionREP().Eta() + 0.1 * dHEta;
        trackphi = atHCAL.positionREP().Phi() + 0.1 * dHPhi;
        track_X = atHCAL.position().X();
        track_Y = atHCAL.position().Y();
        track_Z = atHCAL.position().Z();
        /*      distance 
	= -std::sqrt( (track_X-clusterX)*(track_X-clusterX)
		     +(track_Y-clusterY)*(track_Y-clusterY)
		     +(track_Z-clusterZ)*(track_Z-clusterZ)
		     );
*/
      }
      break;

    case PFLayer::HCAL_BARREL2:
      barrel = true;
#ifdef PFLOW_DEBUG
      if (debug)
        std::cout << "Fetching HO Resolution Maps" << std::endl;
#endif
      if (isBrem) {
        return -1.;
      } else {
        hcal = true;
        horesolscale = 1.15;
        const reco::PFTrajectoryPoint& atHO = track.extrapolatedPoint(reco::PFTrajectoryPoint::HOLayer);
        // did not reach ho, cannot be associated with a cluster.
        if (!atHO.isValid())
          return -1.;

        //
        tracketa = atHO.positionREP().Eta();
        trackphi = atHO.positionREP().Phi();
        track_X = atHO.position().X();
        track_Y = atHO.position().Y();
        track_Z = atHO.position().Z();

        // Is this check really useful ?
        if (fabs(track_Z) > 700.25)
          return -1.;

/*      distance 
	= std::sqrt( (track_X-clusterX)*(track_X-clusterX)
		      +(track_Y-clusterY)*(track_Y-clusterY)
		      +(track_Z-clusterZ)*(track_Z-clusterZ)
		      );
*/
#ifdef PFLOW_DEBUG
/*      if( debug ) {
	std::cout <<"dist "<<distance<<" "<<cluster.energy()<<" "<<cluster.layer()<<" "
		  <<track_X<<" "<<clusterX<<" "
		  <<track_Y<<" "<<clusterY<<" "
		  <<track_Z<<" "<<clusterZ<<std::endl;
      }
*/
#endif
      }
      break;

    case PFLayer::PS1:
      [[fallthrough]];
    case PFLayer::PS2:
      //Note Alex: Nothing implemented for the
      //PreShower (No resolution maps yet)
#ifdef PFLOW_DEBUG
      if (debug)
        std::cout << "No link by rechit possible for pre-shower yet!" << std::endl;
#endif
      return -1.;
    default:
      return -1.;
  }

  // Check that, if the cluster is in the endcap,
  // 0) the track indeed points to the endcap at vertex (DISABLED)
  // 1) the track extrapolation is in the endcap too !
  // 2) the track is in the same end-cap !
  // PJ - 10-May-09
  if (!barrel) {
    // if ( fabs(trackEta) < 1.0 ) return -1;
    if (!hcal && fabs(track_Z) < 300.)
      return -1.;
    if (track_Z * clusterZ < 0.)
      return -1.;
  }
  // Check that, if the cluster is in the barrel,
  // 1) the track is in the barrel too !
  if (barrel) {
    if (!hcal && fabs(track_Z) > 300.)
      return -1.;
  }

  double dist = LinkByRecHit::computeDist(clustereta, clusterphi, tracketa, trackphi);

#ifdef PFLOW_DEBUG
  if (debug)
    std::cout << "test link by rechit " << dist << " " << std::endl;
  if (debug) {
    std::cout << " clustereta " << clustereta << " clusterphi " << clusterphi << " tracketa " << tracketa
              << " trackphi " << trackphi << std::endl;
  }
#endif

  //Testing if Track can be linked by rechit to a cluster.
  //A cluster can be linked to a track if the extrapolated position
  //of the track to the ECAL ShowerMax/HCAL entrance falls within
  //the boundaries of any cell that belongs to this cluster.

  const std::vector<reco::PFRecHitFraction>& fracs = cluster.recHitFractions();

  bool linkedbyrechit = false;
  //loop rechits
  for (unsigned int rhit = 0; rhit < fracs.size(); ++rhit) {
    const reco::PFRecHitRef& rh = fracs[rhit].recHitRef();
    double fraction = fracs[rhit].fraction();
    if (fraction < 1E-4)
      continue;
    if (rh.isNull())
      continue;

    //getting rechit center position
    const auto& rechit_cluster = *rh;
    const auto& posxyz = rechit_cluster.position();
    const auto& posrep = rechit_cluster.positionREP();

    //getting rechit corners
    const auto& cornersxyz = rechit_cluster.getCornersXYZ();
    const auto& corners = rechit_cluster.getCornersREP();

    if (barrel || hcal) {  // barrel case matching in eta/phi
                           // (and HCAL endcap too!)

      //rechit size determination
      // blown up by 50% (HCAL) to 100% (ECAL) to include cracks & gaps
      // also blown up to account for multiple scattering at low pt.
      double rhsizeEta = std::abs(corners[3].eta() - corners[1].eta());
      double rhsizePhi = std::abs(corners[3].phi() - corners[1].phi());
      if (rhsizePhi > M_PI)
        rhsizePhi = 2. * M_PI - rhsizePhi;
      if (hcal) {
        const double mult = horesolscale * (1.50 + 0.5 / fracs.size());
        rhsizeEta = rhsizeEta * mult + 0.2 * std::abs(dHEta);
        rhsizePhi = rhsizePhi * mult + 0.2 * fabs(dHPhi);

      } else {
        const double mult = 2.00 + 1.0 / (fracs.size() * std::min(1., 0.5 * trackPt));
        rhsizeEta *= mult;
        rhsizePhi *= mult;
      }

#ifdef PFLOW_DEBUG
      if (debug) {
        std::cout << rhit << " Hcal RecHit=" << posrep.Eta() << " " << posrep.Phi() << " " << rechit_cluster.energy()
                  << std::endl;
        for (unsigned jc = 0; jc < 4; ++jc)
          std::cout << "corners " << jc << " " << corners[jc].eta() << " " << corners[jc].phi() << std::endl;

        std::cout << "RecHit SizeEta=" << rhsizeEta << " SizePhi=" << rhsizePhi << std::endl;
      }
#endif

      //distance track-rechit center
      // const math::XYZPoint& posxyz
      // = rechit_cluster.position();
      double deta = fabs(posrep.eta() - tracketa);
      double dphi = fabs(posrep.phi() - trackphi);
      if (dphi > M_PI)
        dphi = 2. * M_PI - dphi;

#ifdef PFLOW_DEBUG
      if (debug) {
        std::cout << "distance=" << deta << " " << dphi << " ";
        if (deta < (0.5 * rhsizeEta) && dphi < (0.5 * rhsizePhi))
          std::cout << " link here !" << std::endl;
        else
          std::cout << std::endl;
      }
#endif

      if (deta < (0.5 * rhsizeEta) && dphi < (0.5 * rhsizePhi)) {
        linkedbyrechit = true;
        break;
      }
    } else {  //ECAL & PS endcap case, matching in X,Y

#ifdef PFLOW_DEBUG
      if (debug) {
        const auto& posxyz = rechit_cluster.position();

        std::cout << "RH " << posxyz.x() << " " << posxyz.y() << std::endl;

        std::cout << "TRACK " << track_X << " " << track_Y << std::endl;
      }
#endif

      double x[5];
      double y[5];

      for (unsigned jc = 0; jc < 4; ++jc) {
        const auto& cornerposxyz = cornersxyz[jc];
        const double mult = (1.00 + 0.50 / (fracs.size() * std::min(1., 0.5 * trackPt)));
        x[3 - jc] = cornerposxyz.x() + (cornerposxyz.x() - posxyz.x()) * mult;
        y[3 - jc] = cornerposxyz.y() + (cornerposxyz.y() - posxyz.y()) * mult;

#ifdef PFLOW_DEBUG
        if (debug) {
          std::cout << "corners " << jc << " " << cornerposxyz.x() << " " << cornerposxyz.y() << std::endl;
        }
#endif
      }  //loop corners

      //need to close the polygon in order to
      //use the TMath::IsInside fonction from root lib
      x[4] = x[0];
      y[4] = y[0];

      //Check if the extrapolation point of the track falls
      //within the rechit boundaries
      bool isinside = TMath::IsInside(track_X, track_Y, 5, x, y);

      if (isinside) {
        linkedbyrechit = true;
        break;
      }
    }  //

  }  //loop rechits

  if (linkedbyrechit) {
#ifdef PFLOW_DEBUG
    if (debug)
      std::cout << "Track and Cluster LINKED BY RECHIT" << std::endl;
#endif
    /*    
    //if ( distance > 40. || distance < -100. ) 
    double clusterr = std::sqrt(clusterX*clusterX+clusterY*clusterY);
    double trackr = std::sqrt(track_X*track_X+track_Y*track_Y);
    if ( distance > 40. ) 
    std::cout << "Distance = " << distance 
    << ", Barrel/Hcal/Brem ? " << barrel << " " << hcal << " " << isBrem << std::endl
    << " Cluster " << clusterr << " " << clusterZ << " " << clusterphi << " " << clustereta << std::endl
    << " Track   " << trackr << " " << track_Z << " " << trackphi << " " << tracketa << std::endl;
    if ( !barrel && fabs(trackEta) < 1.0 ) { 
      double clusterr = std::sqrt(clusterX*clusterX+clusterY*clusterY);
      double trackr = std::sqrt(track_X*track_X+track_Y*track_Y);
      std::cout << "TrackEta/Pt = " << trackEta << " " << trackPt << ", distance = " << distance << std::endl 
		<< ", Barrel/Hcal/Brem ? " << barrel << " " << hcal << " " << isBrem << std::endl
		<< " Cluster " << clusterr << " " << clusterZ << " " << clusterphi << " " << clustereta << std::endl
		<< " Track   " << trackr << " " << track_Z << " " << trackphi << " " << tracketa << " " << trackEta << " " << trackPt << std::endl;
    } 
    */
    return dist;
  } else {
    return -1.;
  }
}

double LinkByRecHit::testECALAndPSByRecHit(const reco::PFCluster& clusterECAL,
                                           const reco::PFCluster& clusterPS,
                                           bool debug) {
  // 0.19 <-> strip_pitch
  // 6.1  <-> strip_length
  static const double resPSpitch = 0.19 / sqrt(12.);
  static const double resPSlength = 6.1 / sqrt(12.);

  // Check that clusterECAL is in ECAL endcap and that clusterPS is a preshower cluster
  if (clusterECAL.layer() != PFLayer::ECAL_ENDCAP ||
      (clusterPS.layer() != PFLayer::PS1 && clusterPS.layer() != PFLayer::PS2))
    return -1.;

#ifdef PFLOW_DEBUG
  if (debug)
    std::cout << "entering test link by rechit function for ECAL and PS" << std::endl;
#endif

  //ECAL cluster position
  double zECAL = clusterECAL.position().Z();
  double xECAL = clusterECAL.position().X();
  double yECAL = clusterECAL.position().Y();

  // PS cluster position, extrapolated to ECAL
  double zPS = clusterPS.position().Z();
  double xPS = clusterPS.position().X();  //* zECAL/zPS;
  double yPS = clusterPS.position().Y();  //* zECAL/zPS;
                                          // MDN jan09 : check that zEcal and zPs have the same sign
  if (zECAL * zPS < 0.)
    return -1.;
  double deltaX = 0.;
  double deltaY = 0.;
  double sqr12 = std::sqrt(12.);
  switch (clusterPS.layer()) {
    case PFLayer::PS1:
      // vertical strips, measure x with pitch precision
      deltaX = resPSpitch * sqr12;
      deltaY = resPSlength * sqr12;
      break;
    case PFLayer::PS2:
      // horizontal strips, measure y with pitch precision
      deltaY = resPSpitch * sqr12;
      deltaX = resPSlength * sqr12;
      break;
    default:
      break;
  }

  auto deltaXY = BVector2D(deltaX, deltaY).v * 0.5;
  // Get the rechits
  auto zCorr = zPS / zECAL;
  const std::vector<reco::PFRecHitFraction>& fracs = clusterECAL.recHitFractions();
  bool linkedbyrechit = false;
  //loop rechits
  for (unsigned int rhit = 0; rhit < fracs.size(); ++rhit) {
    const auto& rh = fracs[rhit].recHitRef();
    double fraction = fracs[rhit].fraction();
    if (fraction < 1E-4)
      continue;
    if (rh.isNull())
      continue;

    //getting rechit center position
    const reco::PFRecHit& rechit_cluster = *rh;

    //getting rechit corners
    auto const& corners = rechit_cluster.getCornersXYZ();

    auto posxy = BVector2D(rechit_cluster.position().xy()).v * zCorr;
#ifdef PFLOW_DEBUG
    if (debug) {
      std::cout << "Ecal rechit " << posxy.x() << " " << posxy.y() << std::endl;
      std::cout << "PS cluster  " << xPS << " " << yPS << std::endl;
    }
#endif

    double x[5];
    double y[5];
    for (unsigned jc = 0; jc < 4; ++jc) {
      // corner position projected onto the preshower
      Vector2D cornerpos = BVector2D(corners[jc].basicVector().xy()).v * zCorr;
      auto dist = (cornerpos - posxy);
      auto adist =
          BVector2D(std::abs(dist[0]), std::abs(dist[1])).v;  // all this beacuse icc does not support vector extension
      // Inflate the size by the size of the PS strips, and by 5% to include ECAL cracks.
      auto xy = cornerpos + (dist * (fivepercent2D + one2D / adist) * deltaXY);
      /*
      Vector2D xy(
		  cornerpos.x() + (cornerpos.x()-posxy.x()) * (0.05 +1.0/std::abs((cornerpos.x()-posxy.x()))*deltaXY.x()),
		  cornerpos.y() + (cornerpos.y()-posxy.y()) * (0.05 +1.0/std::abs((cornerpos.y()-posxy.y()))*deltaXY.y())
		  );
      */
      x[3 - jc] = xy[0];
      y[3 - jc] = xy[1];

#ifdef PFLOW_DEBUG
      if (debug) {
        std::cout << "corners " << jc << " " << cornerpos.x() << " " << x[3 - jc] << " " << cornerpos.y() << " "
                  << y[3 - jc] << std::endl;
      }
#endif
    }  //loop corners

    //need to close the polygon in order to
    //use the TMath::IsInside fonction from root lib
    x[4] = x[0];
    y[4] = y[0];

    //Check if the extrapolation point of the track falls
    //within the rechit boundaries
    bool isinside = TMath::IsInside(xPS, yPS, 5, x, y);

    if (isinside) {
      linkedbyrechit = true;
      break;
    }

  }  //loop rechits

  if (linkedbyrechit) {
#ifdef PFLOW_DEBUG
    if (debug)
      std::cout << "Cluster PS and Cluster ECAL LINKED BY RECHIT" << std::endl;
#endif
    constexpr double scale = 1. / 1000.;
    double dist = computeDist(xECAL * scale, yECAL * scale, xPS * scale, yPS * scale, false);
    return dist;
  } else {
    return -1.;
  }
}

double LinkByRecHit::testHFEMAndHFHADByRecHit(const reco::PFCluster& clusterHFEM,
                                              const reco::PFCluster& clusterHFHAD,
                                              bool debug) {
  const auto& posxyzEM = clusterHFEM.position();
  const auto& posxyzHAD = clusterHFHAD.position();

  double dX = posxyzEM.X() - posxyzHAD.X();
  double dY = posxyzEM.Y() - posxyzHAD.Y();
  double sameZ = posxyzEM.Z() * posxyzHAD.Z();

  if (sameZ < 0)
    return -1.;

  double dist2 = dX * dX + dY * dY;

  if (dist2 < 0.1) {
    // less than one mm
    double dist = sqrt(dist2);
    return dist;
    ;
  } else
    return -1.;
}

double LinkByRecHit::computeDist(double eta1, double phi1, double eta2, double phi2, bool etaPhi) {
  auto phicor = etaPhi ? normalizedPhi(phi1 - phi2) : phi1 - phi2;
  auto etadiff = eta1 - eta2;

  // double chi2 =
  //  (eta1 - eta2)*(eta1 - eta2) / ( reta1*reta1+ reta2*reta2 ) +
  //  phicor*phicor / ( rphi1*rphi1+ rphi2*rphi2 );

  return std::sqrt(etadiff * etadiff + phicor * phicor);
}
