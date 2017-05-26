#ifndef IOMC_ParticleGuns_FlatRandomXiGunProducer_h
#define IOMC_ParticleGuns_FlatRandomXiGunProducer_h

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "CLHEP/Random/RandGauss.h"
#include "CLHEP/Random/RandFlat.h"

#include "HepPDT/ParticleDataTable.hh"

namespace edm
{
  class FlatRandomXiGunProducer : public edm::stream::EDProducer<> {
    public:
      explicit FlatRandomXiGunProducer( const edm::ParameterSet& );
      ~FlatRandomXiGunProducer();

      static void fillDescriptions( edm::ConfigurationDescriptions& descriptions );

    private:
      virtual void beginRun( const edm::Run&, const edm::EventSetup& ) override;
      virtual void produce( edm::Event&, const edm::EventSetup& ) override;
      virtual void endRun( const edm::Run&, const edm::EventSetup& ) override;

      HepMC::FourVector shoot( CLHEP::HepRandomEngine*, double );

      edm::ESHandle<HepPDT::ParticleDataTable> pdgTable_;

      edm::ParameterSet partGunParams_;
      std::vector<int> partIds_;
      double sqrtS_;
      edm::ParameterSet beamConditions_;

      double minXi_, maxXi_;
      double minPhi_, maxPhi_;

      double thetaPhys_;
      double vertexSize_;
      double beamDivergence_;

      bool simulateVertexX_, simulateVertexY_;
      bool simulateScatteringAngleX_, simulateScatteringAngleY_;
      bool simulateBeamDivergence_;
  };
}

#endif
