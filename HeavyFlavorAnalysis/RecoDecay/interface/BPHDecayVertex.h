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

class BPHEventSetupWrapper;

namespace edm {
  class EventSetup;
}

namespace reco {
  class TransientTrack;
  class Vertex;
}  // namespace reco

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "RecoVertex/VertexPrimitives/interface/VertexFitter.h"

//---------------
// C++ Headers --
//---------------
#include <vector>
#include <map>
#include <string>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHDecayVertex : public virtual BPHDecayMomentum {
public:
  /** Constructors are protected
   *  this object can exist only as part of a derived class
   */
  // deleted copy constructor and assignment operator
  BPHDecayVertex(const BPHDecayVertex& x) = delete;
  BPHDecayVertex& operator=(const BPHDecayVertex& x) = delete;

  /** Destructor
   */
  ~BPHDecayVertex() override;

  /** Operations
   */

  /// check for valid reconstructed vertex
  virtual bool validTracks() const;
  virtual bool validVertex() const;

  /// get reconstructed vertex
  virtual const reco::Vertex& vertex(VertexFitter<5>* fitter = nullptr,
                                     const reco::BeamSpot* bs = nullptr,
                                     const GlobalPoint* priorPos = nullptr,
                                     const GlobalError* priorError = nullptr) const;

  /// get list of Tracks
  const std::vector<const reco::Track*>& tracks() const;

  /// get Track for a daughter
  const reco::Track* getTrack(const reco::Candidate* cand) const;

  /// get Track mode for a daughter
  char getTMode(const reco::Candidate* cand) const;

  /// get list of TransientTracks
  const std::vector<reco::TransientTrack>& transientTracks() const;

  /// get TransientTrack for a daughter
  reco::TransientTrack* getTransientTrack(const reco::Candidate* cand) const;

  /// retrieve EventSetup
  const BPHEventSetupWrapper* getEventSetup() const;

  /// retrieve track search list
  const std::string& getTrackSearchList(const reco::Candidate* cand) const;

protected:
  // constructor
  BPHDecayVertex(const BPHEventSetupWrapper* es, int daugNum = 2, int compNum = 2);
  // pointer used to retrieve informations from other bases
  BPHDecayVertex(const BPHDecayVertex* ptr, const BPHEventSetupWrapper* es);

  // add a simple particle giving it a name and specifying an option list
  // to search for the associated track
  virtual void addV(const std::string& name, const reco::Candidate* daug, const std::string& searchList, double mass);
  // add a previously reconstructed particle giving it a name
  virtual void addV(const std::string& name, const BPHRecoConstCandPtr& comp);

  // utility function used to cash reconstruction results
  void setNotUpdated() const override;

private:
  // EventSetup needed to build TransientTrack
  const BPHEventSetupWrapper* evSetup;

  // map linking particles to associated track search list
  std::map<const reco::Candidate*, std::string> searchMap;

  // reconstruction results cache
  mutable bool oldTracks;
  mutable bool oldTTracks;
  mutable bool oldVertex;
  mutable bool validTks;
  mutable std::vector<const reco::Track*> rTracks;
  mutable std::vector<reco::TransientTrack> trTracks;
  mutable std::map<const reco::Candidate*, const reco::Track*> tkMap;
  mutable std::map<const reco::Candidate*, char> tmMap;
  mutable std::map<const reco::Candidate*, reco::TransientTrack*> ttMap;
  mutable reco::Vertex fittedVertex;
  mutable VertexFitter<5>* savedFitter;
  mutable const reco::BeamSpot* savedBS;
  mutable const GlobalPoint* savedPP;
  mutable const GlobalError* savedPE;

  // create TransientTrack and fit vertex
  virtual void fTracks() const;
  virtual void fTTracks() const;
  virtual void fitVertex(VertexFitter<5>* fitter,
                         const reco::BeamSpot* bs,
                         const GlobalPoint* priorPos,
                         const GlobalError* priorError) const;
};

#endif
