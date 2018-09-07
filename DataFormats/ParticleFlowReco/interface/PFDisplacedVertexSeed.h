#ifndef DataFormat_ParticleFlowReco_PFDisplacedVertexSeed_h
#define DataFormat_ParticleFlowReco_PFDisplacedVertexSeed_h 

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include <vector>
#include <iostream>



namespace reco {

  
  /// \brief Block of elements
  /*!
    \author Gouzevitch Maxime
    \date November 2009

    A DisplacedVertexSeed is an intermediate format, usually not persistent,
    used by PFDisplacedVertexFinder to keep the information for vertex fit.
    This format is produced after parsing of PFDisplacedVertexCandidate which
    by construction may contain many seeds. This format contains:
    - a set of Refs to tracks.
    - a Point which indicated the approximative position of the vertex.
  */
  
  class PFDisplacedVertexSeed {

  public:

    /// Default constructor
    PFDisplacedVertexSeed();

    /// Add a track Reference to the current Seed
    /// If the track reference is already in the collection, it is ignored
    void addElement(TrackBaseRef);

    /// Reserve space for elements
    void reserveElements(size_t);

    /// Add a track Ref to the Seed and recalculate the seedPoint with a new dcaPoint
    /// A weight different from 1 may be assign to the new DCA point 
    void updateSeedPoint(const GlobalPoint& dcaPoint, const TrackBaseRef, 
			 const TrackBaseRef, double weight = 1);

    /// Merge two Seeds if their seed Points are close enough
    void mergeWith(const PFDisplacedVertexSeed& displacedVertex);

    /// Check if it is a new Seed
    bool isEmpty() const {return (elements_.empty());}

    /// \return vector of unique references to tracks 
    const std::vector <TrackBaseRef>& elements() const 
      {return elements_;}

    const double nTracks() const {return elements_.size();}

    /// \return the seedPoint for the vertex fitting
    const GlobalPoint& seedPoint() const {return seedPoint_;}

    /// \return the total weight
    const double totalWeight() const {return totalWeight_;}

    /// cout function
    void Dump(std::ostream& out = std::cout) const;


  private:


    friend std::ostream& operator<<( std::ostream& out, const PFDisplacedVertexSeed& co );    

    /// --------- MEMBERS ---------- ///

    /// Set of tracks refs associated to the seed
    std::vector< TrackBaseRef>     elements_;
    /// Seed point which indicated the approximative position of the vertex.
    GlobalPoint                   seedPoint_;
    /// Total weight of the points used to calculate the seed point. 
    /// Necessary for UpdateSeed Point function
    float                         totalWeight_;

  };
}

#endif


  
