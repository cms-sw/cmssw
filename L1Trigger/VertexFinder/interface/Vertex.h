#ifndef __L1Trigger_VertexFinder_Vertex_h__
#define __L1Trigger_VertexFinder_Vertex_h__


#include <vector>

#include "L1Trigger/VertexFinder/interface/Settings.h"
#include "L1Trigger/VertexFinder/interface/TP.h"



namespace l1tVertexFinder {

class Vertex {

public:
  // Fill useful info about tracking particle.
  Vertex(){}

  Vertex(double vz):vz_(vz){}

  ~Vertex(){}

  /// Tracking Particles in vertex    
  const std::vector<TP>& tracks()    const { return    tracks_;    }
  /// Number of tracks originating from this vertex
  unsigned int      numTracks() const { return  tracks_.size();}
  /// Assign TP to this vertex
  void              insert(TP tp)     { tracks_.push_back(tp);}
  /// Compute vertex parameters
  void              computeParameters();
  /// Sum ot fitted tracks transverse momentum
  double            pT()        const {return pT_;}
  /// Vertex z0 position
  double            z0()        const {return z0_;}
  /// Vertex z0 width
  double            z0width()   const {return z0width_;}
  /// Vertex z position
  double            vz()        const {return vz_;}
  /// Vertec MET
  double            met()       const {return met_;}
private:

  double            vz_;
  double            z0_;
  double            z0width_;
  double            pT_;
  double            met_;
  double            metX_;
  double            metY_;


  std::vector<TP>   tracks_;

  
};

} // end namespace l1tVertexFinder

#endif
