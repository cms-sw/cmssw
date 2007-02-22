#ifndef Demo_PFRootEvent_EventColin_h
#define Demo_PFRootEvent_EventColin_h

#include <vector>

class EventColin {
  
  
 public:
  EventColin() {}
  virtual ~EventColin() {}
  void reset() {
    particles_.clear();
    clusters_.clear();
    clustersIsland_.clear();
    jetsMC_.clear();
    jetsEHT_.clear();
    jetsPF_.clear();
  }

  class Particle {
  public:
    Particle() : eta(0),phi(0),e(0) {}
    virtual ~Particle() {}
    double eta; 
    double phi;
    double e;
  };   

  class Cluster {
  public:
    Cluster() : eta(0),phi(0),e(0),layer(0),type(0) {}
    virtual ~Cluster() {}
    double eta; 
    double phi;
    double e;
    int layer;
    int type;
  };   

  class Jets {
  public:
    Jets() : eta(0),phi(0),et(0),e(0) {}
    virtual ~Jets() {}
    double eta;
    double phi;
    double et;
    double e;
  };

  void addParticle( const Particle& ptc ) {
    particles_.push_back(ptc);
  }

  void addCluster( const Cluster& ptc ) {
    clusters_.push_back(ptc);
  }

  void addClusterIsland( const Cluster& ptc ) {
    clustersIsland_.push_back(ptc);
  }

  void addJetsMC( const Jets & jets ) {
    jetsMC_.push_back(jets);
  }

  void addJetsEHT( const Jets & jets ) {
    jetsEHT_.push_back(jets);
  }

  void addJetsPF( const Jets & jets ) {
    jetsPF_.push_back(jets);
  }

  const std::vector<EventColin::Particle>& particles() 
    {return particles_;}
  const std::vector<EventColin::Cluster>& clusters() 
    {return clusters_;}
  const std::vector<EventColin::Cluster>& clustersIsland() 
    {return clustersIsland_;}

 private:


  std::vector<EventColin::Particle> particles_;
  std::vector<EventColin::Cluster>  clusters_;
  std::vector<EventColin::Cluster>  clustersIsland_;
  std::vector<EventColin::Jets>     jetsMC_;
  std::vector<EventColin::Jets>     jetsEHT_;
  std::vector<EventColin::Jets>     jetsPF_;
  
};

#endif
