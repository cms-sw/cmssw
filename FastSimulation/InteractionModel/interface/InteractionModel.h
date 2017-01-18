#ifndef FASTSIM_INTERACTIONMODEL
#define FASTSIM_INTERACTIONMODEL

#include "string"
#include "vector"
#include "memory"

namespace edm
{
    class Event;
    class ProducerBase;
}

class RandomEngineAndDistribution;

namespace fastsim
{
    class Layer;
    class Particle;
    class InteractionModel 
    {
    public:
	InteractionModel(std::string name)
	    : name_(name){}
	virtual ~InteractionModel(){;}
	virtual void interact(Particle & particle,const Layer & layer,std::vector<std::unique_ptr<Particle> > & secondaries,const RandomEngineAndDistribution & random) = 0;
	virtual void registerProducts(edm::ProducerBase & producer) const{;}
	virtual void storeProducts(edm::Event & iEvent) {;}
	const std::string getName(){return name_;}
 	friend std::ostream& operator << (std::ostream& o , const InteractionModel & model); 
   private:
	const std::string name_;
    };
    std::ostream& operator << (std::ostream& os , const InteractionModel & interactionModel);

}

#endif
