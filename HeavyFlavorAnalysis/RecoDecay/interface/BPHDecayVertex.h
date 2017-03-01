#ifndef HeavyFlavorAnalysis_RecoDecay_BPHDecayVertex_h
#define HeavyFlavorAnalysis_RecoDecay_BPHDecayVertex_h
/** \class BPHDecayVertex
 *
 *  Description: 
 *     Mid-level base class to reconstruct decay vertex
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHDecayMomentum.h"

namespace edm {
  class EventSetup;
}

namespace reco {
  class TransientTrack;
  class Vertex;
}

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "DataFormats/VertexReco/interface/Vertex.h"

//---------------
// C++ Headers --
//---------------
#include <vector>
#include <map>
#include <string>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHDecayVertex: public virtual BPHDecayMomentum {

 public:

  /** Constructor is protected
   *  this object can exist only as part of a derived class
   */

  /** Destructor
   */
  virtual ~BPHDecayVertex();

  /** Operations
   */

  /// check for valid reconstructed vertex
  virtual bool validTracks() const;
  virtual bool validVertex() const;

  /// get reconstructed vertex
  virtual const reco::Vertex& vertex() const;

  /// get list of Tracks
  const std::vector<const reco::Track*>& tracks() const;

  /// get Track for a daughter
  const reco::Track* getTrack( const reco::Candidate* cand ) const;

  /// get list of TransientTracks
  const std::vector<reco::TransientTrack>& transientTracks() const;

  /// get TransientTrack for a daughter
  reco::TransientTrack* getTransientTrack( const reco::Candidate* cand ) const;

  /// retrieve track search list
  const std::string& getTrackSearchList( const reco::Candidate* cand ) const;

 protected:

  // constructor
  BPHDecayVertex( const edm::EventSetup* es );
  // pointer used to retrieve informations from other bases
  BPHDecayVertex( const BPHDecayVertex* ptr,
                  const edm::EventSetup* es );

  /// add a simple particle giving it a name and specifying an option list 
  /// to search for the associated track
  virtual void addV( const std::string& name,
                     const reco::Candidate* daug, 
                     const std::string& searchList,
                     double mass );
  /// add a previously reconstructed particle giving it a name
  virtual void addV( const std::string& name,
                     const BPHRecoConstCandPtr& comp );

  // utility function used to cash reconstruction results
  virtual void setNotUpdated() const;

 private:

  // EventSetup needed to build TransientTrack
  const edm::EventSetup* evSetup;

  // map linking particles to associated track search list
  std::map<const reco::Candidate*,std::string> searchMap;

  // reconstruction results cache
  mutable bool oldTracks;
  mutable bool oldVertex;
  mutable bool validTks;
  mutable std::vector<const    reco::Track*> rTracks;
  mutable std::vector<reco::TransientTrack> trTracks;
  mutable std::map<const reco::Candidate*,const    reco::Track*> tkMap;
  mutable std::map<const reco::Candidate*,reco::TransientTrack*> ttMap;
  mutable reco::Vertex fittedVertex;

  // create TransientTrack and fit vertex
  virtual void tTracks() const;
  virtual void fitVertex() const;

};


#endif

