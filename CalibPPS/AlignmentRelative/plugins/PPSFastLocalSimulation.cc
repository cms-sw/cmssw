/****************************************************************************
* Authors:
*  Jan Kašpar (jan.kaspar@gmail.com)
****************************************************************************/

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CLHEP/Random/RandomEngine.h"
#include "CLHEP/Random/RandGauss.h"
#include "CLHEP/Random/RandExponential.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/CTPPSReco/interface/TotemRPRecHit.h"
#include "DataFormats/CTPPSReco/interface/CTPPSDiamondRecHit.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelRecHit.h"

#include "Geometry/VeryForwardRPTopology/interface/RPTopology.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"
#include "Geometry/Records/interface/VeryForwardMisalignedGeometryRecord.h"

#include "CalibPPS/AlignmentRelative/interface/Utilities.h"

#include "TMath.h"
#include "TMatrixD.h"
#include "TVectorD.h"

/**
 *\brief Fast (no G4) proton simulation in within one station.
 * Uses misaligned geometry.
 */
class PPSFastLocalSimulation : public edm::stream::EDProducer<> {
public:
  PPSFastLocalSimulation(const edm::ParameterSet &);
  ~PPSFastLocalSimulation() override;

protected:
  /// verbosity level
  unsigned int verbosity_;

  /// whether a HepMC description of the proton shall be saved in the event
  bool makeHepMC_;

  /// whether the hits of the proton shall be calculated and saved
  bool makeHits_;

  /// the list of RPs to simulate
  std::vector<unsigned int> RPs_;

  /// number of particles to generate per event
  unsigned int particlesPerEvent_;

  /// particle energy and momentum
  double particle_E_, particle_p_;

  /// the "origin" of tracks, in mm
  double z0_;

  /// whether measurement values shall be rounded to the nearest strip
  bool roundToPitch_;

  /// in mm
  double pitchStrips_, pitchDiamonds_, pitchPixels_;

  /// size of insensitive margin at sensor's edge facing the beam, in mm
  double insensitiveMarginStrips_;

  struct Distribution {
    enum Type { dtBox, dtGauss, dtGaussLimit } type_;
    double x_mean_, x_width_, x_min_, x_max_;
    double y_mean_, y_width_, y_min_, y_max_;

    Distribution(const edm::ParameterSet &);

    void Generate(CLHEP::HepRandomEngine &rndEng, double &x, double &y);
  };

  /// position parameters in mm
  Distribution position_dist_;

  /// angular parameters in rad
  Distribution angular_dist_;

  //---------- internal parameters ----------

  /// v position of strip 0, in mm
  double stripZeroPosition_;

  edm::ESGetToken<CTPPSGeometry, VeryForwardMisalignedGeometryRecord> esTokenGeometry_;

  void GenerateTrack(unsigned int pi,
                     CLHEP::HepRandomEngine &rndEng,
                     HepMC::GenEvent *gEv,
                     std::unique_ptr<edm::DetSetVector<TotemRPRecHit>> &stripHitColl,
                     std::unique_ptr<edm::DetSetVector<CTPPSDiamondRecHit>> &diamondHitColl,
                     std::unique_ptr<edm::DetSetVector<CTPPSPixelRecHit>> &pixelHitColl,
                     const CTPPSGeometry &geometry);

  //---------- framework methods ----------

  void produce(edm::Event &, const edm::EventSetup &) override;
};

//----------------------------------------------------------------------------------------------------

using namespace edm;
using namespace std;
using namespace CLHEP;
using namespace HepMC;

//----------------------------------------------------------------------------------------------------

PPSFastLocalSimulation::Distribution::Distribution(const edm::ParameterSet &ps) {
  // get type
  string typeName = ps.getParameter<string>("type");
  if (!typeName.compare("box"))
    type_ = dtBox;
  else if (!typeName.compare("gauss"))
    type_ = dtGauss;
  else if (!typeName.compare("gauss-limit"))
    type_ = dtGaussLimit;
  else
    throw cms::Exception("PPS") << "Unknown distribution type `" << typeName << "'.";

  x_mean_ = ps.getParameter<double>("x_mean");
  x_width_ = ps.getParameter<double>("x_width");
  x_min_ = ps.getParameter<double>("x_min");
  x_max_ = ps.getParameter<double>("x_max");

  y_mean_ = ps.getParameter<double>("y_mean");
  y_width_ = ps.getParameter<double>("y_width");
  y_min_ = ps.getParameter<double>("y_min");
  y_max_ = ps.getParameter<double>("y_max");
}

//----------------------------------------------------------------------------------------------------

void PPSFastLocalSimulation::Distribution::Generate(CLHEP::HepRandomEngine &rndEng, double &x, double &y) {
  switch (type_) {
    case dtBox:
      x = x_mean_ + x_width_ * (rndEng.flat() - 0.5);
      y = y_mean_ + y_width_ * (rndEng.flat() - 0.5);
      break;

    case dtGauss:
      x = x_mean_ + RandGauss::shoot(&rndEng) * x_width_;
      y = y_mean_ + RandGauss::shoot(&rndEng) * y_width_;
      break;

    case dtGaussLimit: {
      const double u_x = rndEng.flat(), u_y = rndEng.flat();

      const double cdf_x_min = (1. + TMath::Erf((x_min_ - x_mean_) / x_width_ / sqrt(2.))) / 2.;
      const double cdf_x_max = (1. + TMath::Erf((x_max_ - x_mean_) / x_width_ / sqrt(2.))) / 2.;
      const double a_x = cdf_x_max - cdf_x_min, b_x = cdf_x_min;

      const double cdf_y_min = (1. + TMath::Erf((y_min_ - y_mean_) / y_width_ / sqrt(2.))) / 2.;
      const double cdf_y_max = (1. + TMath::Erf((y_max_ - y_mean_) / y_width_ / sqrt(2.))) / 2.;
      const double a_y = cdf_y_max - cdf_y_min, b_y = cdf_y_min;

      x = x_mean_ + x_width_ * sqrt(2.) * TMath::ErfInverse(2. * (a_x * u_x + b_x) - 1.);
      y = y_mean_ + y_width_ * sqrt(2.) * TMath::ErfInverse(2. * (a_y * u_y + b_y) - 1.);
    }

    break;

    default:
      x = y = 0.;
  }
}

//----------------------------------------------------------------------------------------------------

PPSFastLocalSimulation::PPSFastLocalSimulation(const edm::ParameterSet &ps)
    : verbosity_(ps.getUntrackedParameter<unsigned int>("verbosity", 0)),

      makeHepMC_(ps.getParameter<bool>("makeHepMC")),
      makeHits_(ps.getParameter<bool>("makeHits")),

      RPs_(ps.getParameter<vector<unsigned int>>("RPs")),

      particlesPerEvent_(ps.getParameter<unsigned int>("particlesPerEvent")),
      particle_E_(ps.getParameter<double>("particle_E")),
      particle_p_(ps.getParameter<double>("particle_p")),
      z0_(ps.getParameter<double>("z0")),

      roundToPitch_(ps.getParameter<bool>("roundToPitch")),
      pitchStrips_(ps.getParameter<double>("pitchStrips")),
      pitchDiamonds_(ps.getParameter<double>("pitchDiamonds")),
      pitchPixels_(ps.getParameter<double>("pitchPixels")),

      insensitiveMarginStrips_(ps.getParameter<double>("insensitiveMarginStrips")),

      position_dist_(ps.getParameterSet("position_distribution")),
      angular_dist_(ps.getParameterSet("angular_distribution")),

      esTokenGeometry_(esConsumes()) {
  // v position of strip 0
  stripZeroPosition_ = RPTopology::last_strip_to_border_dist_ + (RPTopology::no_of_strips_ - 1) * RPTopology::pitch_ -
                       RPTopology::y_width_ / 2.;

  // register the output
  if (makeHepMC_)
    produces<HepMCProduct>();

  if (makeHits_) {
    produces<DetSetVector<TotemRPRecHit>>();
    produces<DetSetVector<CTPPSDiamondRecHit>>();
    produces<DetSetVector<CTPPSPixelRecHit>>();
  }
}

//----------------------------------------------------------------------------------------------------

PPSFastLocalSimulation::~PPSFastLocalSimulation() {}

//----------------------------------------------------------------------------------------------------

void PPSFastLocalSimulation::GenerateTrack(unsigned int idx,
                                           CLHEP::HepRandomEngine &rndEng,
                                           HepMC::GenEvent *gEv,
                                           unique_ptr<edm::DetSetVector<TotemRPRecHit>> &stripHitColl,
                                           unique_ptr<edm::DetSetVector<CTPPSDiamondRecHit>> &diamondHitColl,
                                           unique_ptr<edm::DetSetVector<CTPPSPixelRecHit>> &pixelHitColl,
                                           const CTPPSGeometry &geometry) {
  // generate track
  double bx = 0., by = 0., ax = 0., ay = 0.;
  position_dist_.Generate(rndEng, bx, by);
  angular_dist_.Generate(rndEng, ax, ay);

  if (verbosity_ > 5)
    printf("\tax = %.3f mrad, bx = %.3f mm, ay = %.3f mrad, by = %.3f mm, z0 = %.3f m\n",
           ax * 1E3,
           bx,
           ay * 1E3,
           by,
           z0_ * 1E-3);

  // add HepMC track description
  if (makeHepMC_) {
    GenVertex *gVx = new GenVertex(HepMC::FourVector(bx, by, z0_, 0.));
    gEv->add_vertex(gVx);

    GenParticle *gPe;
    double az = sqrt(1. - ax * ax - ay * ay);
    gPe = new GenParticle(HepMC::FourVector(particle_p_ * ax, particle_p_ * ay, particle_p_ * az, particle_E_),
                          2212,
                          1);  // add a proton in final state
    gPe->suggest_barcode(idx + 1);
    gVx->add_particle_out(gPe);
  }

  if (makeHits_) {
    // check all sensors known to geometry
    for (CTPPSGeometry::mapType::const_iterator it = geometry.beginSensor(); it != geometry.endSensor(); ++it) {
      // get RP decimal id
      CTPPSDetId detId(it->first);
      unsigned int decRPId = detId.arm() * 100 + detId.station() * 10 + detId.rp();

      // stop if the RP is not selected
      if (find(RPs_.begin(), RPs_.end(), decRPId) == RPs_.end())
        continue;

      // keep only 1 diamond channel to represent 1 plane
      if (detId.subdetId() == CTPPSDetId::sdTimingDiamond) {
        CTPPSDiamondDetId channelId(it->first);
        if (channelId.channel() != 0)
          continue;
      }

      if (verbosity_ > 5) {
        printf("        ");
        printId(it->first);
        printf(": ");
      }

      // determine the track impact point (in global coordinates)
      // !! this assumes that local axes (1, 0, 0) and (0, 1, 0) describe the sensor surface
      const auto gl_o = geometry.localToGlobal(detId, CTPPSGeometry::Vector(0, 0, 0));
      const auto gl_a1 = geometry.localToGlobal(detId, CTPPSGeometry::Vector(1, 0, 0)) - gl_o;
      const auto gl_a2 = geometry.localToGlobal(detId, CTPPSGeometry::Vector(0, 1, 0)) - gl_o;

      TMatrixD A(3, 3);
      TVectorD B(3);
      A(0, 0) = ax;
      A(0, 1) = -gl_a1.x();
      A(0, 2) = -gl_a2.x();
      B(0) = gl_o.x() - bx;
      A(1, 0) = ay;
      A(1, 1) = -gl_a1.y();
      A(1, 2) = -gl_a2.y();
      B(1) = gl_o.y() - by;
      A(2, 0) = 1.;
      A(2, 1) = -gl_a1.z();
      A(2, 2) = -gl_a2.z();
      B(2) = gl_o.z() - z0_;
      TMatrixD Ai(3, 3);
      Ai = A.Invert();
      TVectorD P(3);
      P = Ai * B;

      double de_z = P(0);
      CTPPSGeometry::Vector h_glo(ax * de_z + bx, ay * de_z + by, de_z + z0_);

      // hit in local coordinates
      CTPPSGeometry::Vector h_loc = geometry.globalToLocal(detId, h_glo);

      // strips
      if (detId.subdetId() == CTPPSDetId::sdTrackingStrip) {
        double u = h_loc.x();
        double v = h_loc.y();

        if (verbosity_ > 5)
          printf("            u=%+8.4f, v=%+8.4f", u, v);

        // is it within detector?
        if (!RPTopology::IsHit(u, v, insensitiveMarginStrips_)) {
          if (verbosity_ > 5)
            printf(" | no hit\n");
          continue;
        }

        // round the measurement
        if (roundToPitch_) {
          double m = stripZeroPosition_ - v;
          signed int strip = (int)floor(m / pitchStrips_ + 0.5);

          v = stripZeroPosition_ - pitchStrips_ * strip;

          if (verbosity_ > 5)
            printf(" | strip=%+4i", strip);
        }

        double sigma = pitchStrips_ / sqrt(12.);

        if (verbosity_ > 5)
          printf(" | m=%+8.4f, sigma=%+8.4f\n", v, sigma);

        DetSet<TotemRPRecHit> &hits = stripHitColl->find_or_insert(detId);
        hits.emplace_back(v, sigma);
      }

      // diamonds
      if (detId.subdetId() == CTPPSDetId::sdTimingDiamond) {
        if (roundToPitch_) {
          h_loc.SetX(pitchDiamonds_ * floor(h_loc.x() / pitchDiamonds_ + 0.5));
        }

        if (verbosity_ > 5)
          printf("            m = %.3f\n", h_loc.x());

        const double width = pitchDiamonds_;

        DetSet<CTPPSDiamondRecHit> &hits = diamondHitColl->find_or_insert(detId);
        hits.emplace_back(h_loc.x(), width, 0., 0., 0., 0., 0., 0., 0., 0, HPTDCErrorFlags(), false);
      }

      // pixels
      if (detId.subdetId() == CTPPSDetId::sdTrackingPixel) {
        if (roundToPitch_) {
          h_loc.SetX(pitchPixels_ * floor(h_loc.x() / pitchPixels_ + 0.5));
          h_loc.SetY(pitchPixels_ * floor(h_loc.y() / pitchPixels_ + 0.5));
        }

        if (verbosity_ > 5)
          printf("            m1 = %.3f, m2 = %.3f\n", h_loc.x(), h_loc.y());

        const double sigma = pitchPixels_ / sqrt(12.);

        const LocalPoint lp(h_loc.x(), h_loc.y(), h_loc.z());
        const LocalError le(sigma, 0., sigma);

        DetSet<CTPPSPixelRecHit> &hits = pixelHitColl->find_or_insert(detId);
        hits.emplace_back(lp, le);
      }
    }
  }
}

//----------------------------------------------------------------------------------------------------

void PPSFastLocalSimulation::produce(edm::Event &event, const edm::EventSetup &es) {
  if (verbosity_ > 2)
    printf(">> PPSFastLocalSimulation::produce > event %llu\n", event.id().event());

  Service<edm::RandomNumberGenerator> rng;
  HepRandomEngine &rndEng = rng->getEngine(event.streamID());

  if (verbosity_ > 5)
    printf("\tseed = %li\n", rndEng.getSeed());

  // get geometry
  auto const &geometry = es.getData(esTokenGeometry_);

  // initialize products
  GenEvent *gEv = new GenEvent();
  gEv->set_event_number(event.id().event());

  unique_ptr<DetSetVector<TotemRPRecHit>> stripHitColl(new DetSetVector<TotemRPRecHit>());
  unique_ptr<DetSetVector<CTPPSDiamondRecHit>> diamondHitColl(new DetSetVector<CTPPSDiamondRecHit>());
  unique_ptr<DetSetVector<CTPPSPixelRecHit>> pixelHitColl(new DetSetVector<CTPPSPixelRecHit>());

  // run particle loop
  for (unsigned int pi = 0; pi < particlesPerEvent_; pi++) {
    if (verbosity_ > 5)
      printf("    generating track %u\n", pi);

    GenerateTrack(pi, rndEng, gEv, stripHitColl, diamondHitColl, pixelHitColl, geometry);
  }

  // save products
  if (makeHepMC_) {
    unique_ptr<HepMCProduct> hepMCoutput(new HepMCProduct());
    hepMCoutput->addHepMCData(gEv);
    event.put(move(hepMCoutput));
  }

  if (makeHits_) {
    event.put(move(stripHitColl));
    event.put(move(diamondHitColl));
    event.put(move(pixelHitColl));
  }
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(PPSFastLocalSimulation);
