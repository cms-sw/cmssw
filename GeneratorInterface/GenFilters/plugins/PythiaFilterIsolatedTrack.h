#ifndef PythiaFilterIsolatedTrack_h
#define PythiaFilterIsolatedTrack_h

/** \class PythiaFilterIsolatedTrack
 *
 *  PythiaFilterGammaJet filter implements generator-level preselections 
 *  for ChargedHadron+jet like events to be used in jet energy calibration.
 *  Ported from fortran code written by V.Konoplianikov.
 * 
 * \author O.Kodolova, SINP
 *
 * Cleaned up code
 * \author J.P. Chou, Brown University
 *
 ************************************************************/

// system include files
#include <atomic>
#include <memory>

// Root objects
#include "TH1F.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

namespace PythiaFilterIsoTracks {
  struct Counters {
    Counters() : nAll_(0), nGood_(0) {}
    mutable std::atomic<unsigned int> nAll_, nGood_;
  };
}  // namespace PythiaFilterIsoTracks

class PythiaFilterIsolatedTrack : public edm::stream::EDFilter<edm::GlobalCache<PythiaFilterIsoTracks::Counters> > {
public:
  explicit PythiaFilterIsolatedTrack(const edm::ParameterSet&, const PythiaFilterIsoTracks::Counters* count);
  ~PythiaFilterIsolatedTrack() override;

  static std::unique_ptr<PythiaFilterIsoTracks::Counters> initializeGlobalCache(edm::ParameterSet const&) {
    return std::make_unique<PythiaFilterIsoTracks::Counters>();
  }

  bool filter(edm::Event&, edm::EventSetup const&) override;
  void endStream() override;
  static void globalEndJob(const PythiaFilterIsoTracks::Counters* counters);

  // helper functions
  std::pair<double, double> GetEtaPhiAtEcal(double etaIP, double phiIP, double pT, int charge, double vtxZ);
  double getDistInCM(double eta1, double phi1, double eta2, double phi2);

private:
  // parameters
  const edm::EDGetTokenT<edm::HepMCProduct> token_;  // token to get the generated particles
  const double maxSeedEta_;                          // maximum eta of the isolated track seed
  const double minSeedEta_;                          // minimum eta of the isolated track seed
  const double minSeedMom_;                          // minimum momentum of the isolated track seed
  const double minIsolTrackMom_;                     // minimum prohibited momentum of a nearby track
  const double isolCone_;                            // cone size (in mm) around the seed to consider a track "nearby"
  const bool onlyHadrons_;                           // select only isolated hadrons

  unsigned int nAll_, nGood_;
  double ecDist_;  //distance to ECAL andcap from IP (cm)
  double ecRad_;   //radius of ECAL barrel (cm)
};
#endif
