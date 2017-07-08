#ifndef DataFormat_ParticleFlowReco_PFDisplacedVertexSeed_h
#define DataFormat_ParticleFlowReco_PFDisplacedVertexSeed_h 

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include <set>
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

    /// -------- Useful Types -------- ///

    typedef std::set< reco::TrackBaseRef >::iterator IEset;

    /// A function necessary to use a set format to store the tracks Refs.
    /// The set is the most appropriate in that case to avoid the double counting 
    /// of the tracks durring the build up procedure.
    /// The position of the tracks in the Collection
    /// is used as a classification parameter.
    struct Compare{
      bool operator()(const TrackBaseRef& s1, const TrackBaseRef& s2) const
      {return s1.key() < s2.key();}
    };

    /// Default constructor
    PFDisplacedVertexSeed();

    /// Add a track Reference to the current Seed
    void addElement(TrackBaseRef);

    /// Add a track Ref to the Seed and recalculate the seedPoint with a new dcaPoint
    /// A weight different from 1 may be assign to the new DCA point 
    void updateSeedPoint(const GlobalPoint& dcaPoint, const TrackBaseRef, 
			 const TrackBaseRef, double weight = 1);

    /// Merge two Seeds if their seed Points are close enough
    void mergeWith(const PFDisplacedVertexSeed& displacedVertex);

    /// Check if it is a new Seed
    bool isEmpty() const {return (elements_.empty());}

    /// \return set of references to tracks 
    const std::set < TrackBaseRef, Compare >& elements() const 
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
    std::set < TrackBaseRef , Compare >     elements_;
    /// Seed point which indicated the approximative position of the vertex.
    GlobalPoint                   seedPoint_;
    /// Total weight of the points used to calculate the seed point. 
    /// Necessary for UpdateSeed Point function
    float                         totalWeight_;

  };
}

#endif


  
