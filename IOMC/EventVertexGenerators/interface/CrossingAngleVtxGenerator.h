#ifndef IOMC_EventVertexGenerators_CrossingAngleVtxGenerator_h
#define IOMC_EventVertexGenerators_CrossingAngleVtxGenerator_h

#include <memory>

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "DataFormats/Provenance/interface/Provenance.h"
#include <CLHEP/Random/RandGauss.h>

namespace HepMC {
  class FourVector;
  class GenParticle;
}
namespace CLHEP { class HepRandomEngine; }

class CrossingAngleVtxGenerator : public edm::stream::EDProducer<>
{
  public:
    explicit CrossingAngleVtxGenerator( const edm::ParameterSet& );
    ~CrossingAngleVtxGenerator();

    void produce( edm::Event&, const edm::EventSetup& ) override;

  private :
    std::shared_ptr<HepMC::FourVector> vertexPosition() const;
    void rotateParticle( HepMC::GenParticle* ) const;
 
    edm::EDGetTokenT<edm::HepMCProduct> sourceToken_;
    double scatteringAngle_;
    double vertexSize_;
    double beamDivergence_;
    double halfCrossingAngleSector45_, halfCrossingAngleSector56_;
    bool simulateVertexX_, simulateVertexY_;
    bool simulateScatteringAngleX_, simulateScatteringAngleY_;
    bool simulateBeamDivergence_;
    CLHEP::HepRandomEngine* rnd_;
};

#endif
