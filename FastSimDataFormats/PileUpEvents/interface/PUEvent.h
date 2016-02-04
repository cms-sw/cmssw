#ifndef FastSimulation_PileUpProducer_PUEvent_h
#define FastSimulation_PileUpProducer_PUEvent_h

#include <vector>

class PUEvent {
  
  
 public:
  PUEvent() {}
  void reset() {
    PUParticles_.clear();
    PUMinBiasEvts_.clear();
  }

  class PUParticle {
  public:
    PUParticle() : px(0.), py(0.), pz(0.),mass(0.),id(0) {}
    float px; 
    float py;
    float pz;
    float mass;
    int id;
  };   


  class PUMinBiasEvt {
  public:
    PUMinBiasEvt() : first(0), size(0) {}
    unsigned first; 
    unsigned size;
  };   


  void addPUParticle( const PUParticle& ptc ) {
    PUParticles_.push_back(ptc);
  }

  void addPUMinBiasEvt( const PUMinBiasEvt& idx ) {
    PUMinBiasEvts_.push_back(idx);
  }

  const std::vector<PUEvent::PUParticle>& thePUParticles() 
    {return PUParticles_;}

  const std::vector<PUEvent::PUMinBiasEvt>& thePUMinBiasEvts() 
    {return PUMinBiasEvts_;}

  const unsigned nParticles() const { return PUParticles_.size(); }

  const unsigned nMinBias() const { return PUMinBiasEvts_.size(); }

 private:

  std::vector<PUEvent::PUParticle> PUParticles_;
  std::vector<PUEvent::PUMinBiasEvt> PUMinBiasEvts_;
  
};

#endif
