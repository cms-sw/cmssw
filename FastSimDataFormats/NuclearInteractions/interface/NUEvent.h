#ifndef FastSimulation_MaterialEffects_NUEvent_h
#define FastSimulation_MaterialEffects_NUEvent_h

#include <vector>

class NUEvent {
  
  
 public:
  NUEvent() {}
  void reset() {
    NUParticles_.clear();
    NUInteractions_.clear();
  }

  class NUParticle {
  public:
    NUParticle() : px(0.), py(0.), pz(0.),mass(0.),id(0) {}
    float px; 
    float py;
    float pz;
    float mass;
    int id;
  };   


  class NUInteraction {
  public:
    NUInteraction() : first(0), last(0) {}
    unsigned first; 
    unsigned last;
  };   


  void addNUParticle( const NUParticle& ptc ) {
    NUParticles_.push_back(ptc);
  }

  void addNUInteraction( const NUInteraction& idx ) {
    NUInteractions_.push_back(idx);
  }

  const std::vector<NUEvent::NUParticle>& theNUParticles() 
    {return NUParticles_;}

  const std::vector<NUEvent::NUInteraction>& theNUInteractions() 
    {return NUInteractions_;}

  const unsigned nParticles() const { return NUParticles_.size(); }

  const unsigned nInteractions() const { return NUInteractions_.size(); }

 private:

  std::vector<NUEvent::NUParticle> NUParticles_;
  std::vector<NUEvent::NUInteraction> NUInteractions_;
  
};

#endif
