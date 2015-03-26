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
 * Cleaned up code, and added pixel efficiency functionality
 * \author J.P. Chou, Brown University
 *
 ************************************************************/

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

namespace edm {
  class HepMCProduct;
}

class PythiaFilterIsolatedTrack : public edm::EDFilter {
   public:
      explicit PythiaFilterIsolatedTrack(const edm::ParameterSet&);
      ~PythiaFilterIsolatedTrack();

      virtual bool filter(edm::Event&, const edm::EventSetup&);


      // helper functions
      static std::pair<double,double> GetEtaPhiAtEcal(double etaIP, double phiIP, double pT, int charge, double vtxZ);
      static double getDistInCM(double eta1, double phi1, double eta2, double phi2);

   private:
      
      // parameters
      edm::EDGetTokenT<edm::HepMCProduct> token_; // token to get the generated particles
      double MaxSeedEta_;       // maximum eta of the isolated track seed
      double MinSeedMom_;       // minimum momentum of the isolated track seed
      double MinIsolTrackMom_;  // minimum prohibited momentum of a nearby track
      double IsolCone_;         // cone size (in mm) around the seed to consider a track "nearby"
      double PixelEfficiency_;  // efficiency to reconstruct a pixel track (used to throw out nearby tracks, randomly)

      // to get a random number
      edm::Service<edm::RandomNumberGenerator> rng_;
};
#endif
