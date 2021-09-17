#ifndef HeavyFlavorAnalysis_RecoDecay_BPHRecoSelect_h
#define HeavyFlavorAnalysis_RecoDecay_BPHRecoSelect_h
/** \class BPHRecoSelect
 *
 *  Description: 
 *     Base class for daughter particle selection
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"

namespace reco {
  class Candidate;
}

//---------------
// C++ Headers --
//---------------
#include <string>
#include <map>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHRecoSelect {
public:
  /** Constructor
   */
  BPHRecoSelect() {}

  // deleted copy constructor and assignment operator
  BPHRecoSelect(const BPHRecoSelect& x) = delete;
  BPHRecoSelect& operator=(const BPHRecoSelect& x) = delete;

  /** Destructor
   */
  virtual ~BPHRecoSelect() {}

  using AcceptArg = reco::Candidate;

  /** Operations
   */
  /// accept function
  /// pointers to other particles in the decays can be obtained
  /// by the function "get" giving the particle name (passing the pointer
  /// to the builder)
  virtual bool accept(const reco::Candidate& cand) const = 0;
  virtual bool accept(const reco::Candidate& cand, const BPHRecoBuilder* build) const { return accept(cand); }

protected:
  // function to get other particles pointers
  const reco::Candidate* get(const std::string& name, const BPHRecoBuilder* build) const {
    if (build == nullptr)
      return nullptr;
    std::map<std::string, const reco::Candidate*>& cMap = build->daugMap;
    std::map<std::string, const reco::Candidate*>::iterator iter = cMap.find(name);
    return (iter != cMap.end() ? iter->second : nullptr);
  }
};

#endif
