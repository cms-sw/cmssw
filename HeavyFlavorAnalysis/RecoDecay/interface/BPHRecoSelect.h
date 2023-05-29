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
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
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
  virtual ~BPHRecoSelect() = default;

  using AcceptArg = reco::Candidate;

  /** Operations
   */
  /// accept function
  /// pointers to other particles in the decays can be obtained
  /// by the function "get" giving the particle name (passing the pointer
  /// to the builder)
  virtual bool accept(const reco::Candidate& cand) const = 0;
  virtual bool accept(const reco::Candidate& cand, const BPHRecoBuilder* builder) const { return accept(cand); }
};

#endif
