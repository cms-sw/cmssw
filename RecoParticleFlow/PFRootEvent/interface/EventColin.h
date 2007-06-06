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


  void addParticle( const Particle& ptc ) {
    particles_.push_back(ptc);
  }

  void addCluster( const Cluster& ptc ) {
    clusters_.push_back(ptc);
  }

  void addClusterIsland( const Cluster& ptc ) {
    clustersIsland_.push_back(ptc);
  }

  const std::vector<EventColin::Particle>& particles() 
    {return particles_;}
  const std::vector<EventColin::Cluster>& clusters() 
    {return clusters_;}
  const std::vector<EventColin::Cluster>& clustersIsland() 
    {return clustersIsland_;}

 private:


  std::vector<EventColin::Particle> particles_;
  std::vector<EventColin::Cluster> clusters_;
  std::vector<EventColin::Cluster> clustersIsland_;
  
};

#endif
