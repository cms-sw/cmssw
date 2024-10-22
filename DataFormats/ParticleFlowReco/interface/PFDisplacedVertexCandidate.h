#ifndef DataFormat_ParticleFlowReco_PFDisplacedVertexCandidate_h
#define DataFormat_ParticleFlowReco_PFDisplacedVertexCandidate_h

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include <vector>
#include <map>
#include <iostream>

namespace reco {

  /// \brief A block of tracks linked together
  /*!
    \author Gouzevitch Maxime
    \date November 2009

    A DisplacedVertexCandidate is a format produced by 
    the DisplacedVertexCandidateFinder to hold a Collection
    of Refs to the tracks linked together by DCA. It contains: 
    - a set of reco::Tracks candidates for a common displaced vertex.
    - a set of links between these reco::Tracks.
    - a Seed point calculated as average from DCA Points.
    ie. points Pm in the middle between two points P1 and P2 belonging 
    to tracks 1 and 2:  P1--Pm--P2
  */

  class PFDisplacedVertexCandidate {
  public:
    /// A structure dedicated to describe the link between two tracks
    /// containing a distance value and a Point where this distance
    /// is measurted.
    struct VertexLink {
      VertexLink() : distance_(-1), dcaPoint_(0, 0, 0), test_(0) {}
      VertexLink(float d, GlobalPoint p, char t) : distance_(d), dcaPoint_(p), test_(t) {}
      float distance_;
      GlobalPoint dcaPoint_;
      char test_;
    };

    /// Test used for the track linkind. For the moment only DCA is used,
    /// but other are possibles like distance between inner hits.
    enum VertexLinkTest { LINKTEST_DCA, LINKTEST_DUMMY, LINKTEST_ALL };

    typedef std::map<unsigned int, VertexLink> VertexLinkData;

    /// A type to provide the information about the position of DCA Points
    /// or values of DCA.
    typedef std::map<float, std::pair<int, int> > DistMap;
    typedef std::vector<float> DistVector;

    /// Default constructor
    PFDisplacedVertexCandidate();

    /// add a track Reference to the current Candidate
    void addElement(const TrackBaseRef);

    /// set a link between elements of indices i1 and i2, of "distance" dist
    /// the link is set in the linkData vector provided as an argument.
    /// As indicated by the 'const' statement, 'this' is not modified.
    void setLink(unsigned i1,
                 unsigned i2,
                 const float dist,
                 const GlobalPoint& dcaPoint,
                 const VertexLinkTest test = LINKTEST_DCA);

    /// associate 2 elements
    void associatedElements(const unsigned i,
                            const VertexLinkData& vertexLinkData,
                            std::multimap<float, unsigned>& sortedAssociates,
                            const VertexLinkTest test = LINKTEST_DCA) const;

    /// -------- Provide useful information -------- ///

    /// \return the map of Radius^2 to DCA Points
    DistMap r2Map() const;

    /// \return the vector of Radius^2 to DCA Points
    /// useful for FWLite
    DistVector r2Vector() const;

    /// \return the vector of DCA useful for DCA
    DistVector distVector() const;

    /// \return DCA point between two tracks
    const GlobalPoint dcaPoint(unsigned ie1, unsigned ie2) const;

    /// A Vertex Candidate is valid if it has at least two tracks
    bool isValid() const { return elements_.size() > 1; }

    /// \return the reference to a given tracks
    const TrackBaseRef& tref(unsigned ie) const { return elements_[ie]; }

    /// \return the vector of Refs to tracks
    const std::vector<TrackBaseRef>& elements() const { return elements_; }

    /// \return the number of tracks associated to the candidate
    unsigned nTracks() const { return elements_.size(); }

    /// \return the map of link data
    const VertexLinkData& vertexLinkData() const { return vertexLinkData_; }

    /// cout function
    void Dump(std::ostream& out = std::cout) const;

  private:
    /// -------- Internal tools -------- ///

    /// \return distance of link between two tracks
    const float dist(unsigned ie1, unsigned ie2) const;

    /// test if a link between two tracks is valid: value_link =! -1
    bool testLink(unsigned ie1, unsigned ie2) const;

    /// cout function
    friend std::ostream& operator<<(std::ostream&, const PFDisplacedVertexCandidate&);

    /// -------- Storage of the information -------- ///

    /// Those are the tools from PFBlockAlgo
    /// \return size of linkData_, calculated from the number of elements
    unsigned vertexLinkDataSize() const;

    /// makes the correspondance between a 2d element matrix and
    /// the 1D vector which is the most compact way to store the matrix
    bool matrix2vector(unsigned i, unsigned j, unsigned& index) const;

    /// -------- MEMBERS -------- ///

    /// vector of refs to the associated tracks
    std::vector<TrackBaseRef> elements_;

    /// map of links between tracks
    VertexLinkData vertexLinkData_;
  };
}  // namespace reco

#endif
