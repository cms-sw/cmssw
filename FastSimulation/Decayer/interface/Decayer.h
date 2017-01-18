#ifndef FASTSIM_DECAYER_H
#define FASTSIM_DECAYER_H

#include <memory>
#include <vector>

namespace gen {
  class P8RndmEngine;
}

namespace CLHEP {
  class HepRandomEngine;
}

namespace Pythia8 {
  class Pythia;
}

namespace fastsim
{
    class Particle;
    class Decayer 
    {
    public:
	
	Decayer();
	~Decayer();
	void decay(const Particle & particle,std::vector<std::unique_ptr<Particle> > & secondaries,CLHEP::HepRandomEngine & engine) const;
	
    private:
	
	std::unique_ptr<Pythia8::Pythia> pythia_; 
	std::unique_ptr<gen::P8RndmEngine> pythiaRandomEngine_;
    };
}
#endif
