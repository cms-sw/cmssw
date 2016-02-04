#ifndef FastSimulation_MaterialEffects_NUEvent_h
#define FastSimulation_MaterialEffects_NUEvent_h

#include <vector>

class NUEvent {
  
  
 public:
  NUEvent() : NParticles_(0), NInteractions_(0) {}
  virtual ~NUEvent() {}
  void reset() {
    NUParticles_.clear();
    NUInteractions_.clear();
    NParticles_ = 0;
    NInteractions_ = 0;  
  }

  class NUParticle {
  public:
    NUParticle() : px(0.), py(0.), pz(0.),mass(0.),id(0) {}
    virtual ~NUParticle() {}
    float px; 
    float py;
    float pz;
    float mass;
    int id;
  };   


  class NUInteraction {
  public:
    NUInteraction() : first(0), last(0) {}
    virtual ~NUInteraction() {}
    unsigned first; 
    unsigned last;
  };   


  void addNUParticle( const NUParticle& ptc ) {
    NUParticles_.push_back(ptc);
    ++NParticles_;
  }

  void addNUInteraction( const NUInteraction& idx ) {
    NUInteractions_.push_back(idx);
    ++NInteractions_;
  }

  const std::vector<NUEvent::NUParticle>& theNUParticles() 
    {return NUParticles_;}

  const std::vector<NUEvent::NUInteraction>& theNUInteractions() 
    {return NUInteractions_;}

  const unsigned nParticles() const { return NParticles_; }

  const unsigned nInteractions() const { return NInteractions_; }

 private:

  std::vector<NUEvent::NUParticle> NUParticles_;
  std::vector<NUEvent::NUInteraction> NUInteractions_;
  unsigned NParticles_;
  unsigned NInteractions_;
  
};

#endif
