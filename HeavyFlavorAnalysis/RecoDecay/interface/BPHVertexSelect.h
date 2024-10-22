#ifndef HeavyFlavorAnalysis_RecoDecay_BPHVertexSelect_h
#define HeavyFlavorAnalysis_RecoDecay_BPHVertexSelect_h
/** \class BPHVertexSelect
 *
 *  Description: 
 *     Base class for candidate selection at vertex reconstruction level
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
class BPHDecayVertex;
class BPHRecoBuilder;

//---------------
// C++ Headers --
//---------------

//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHVertexSelect {
public:
  /** Constructor
   */
  BPHVertexSelect() {}

  // deleted copy constructor and assignment operator
  BPHVertexSelect(const BPHVertexSelect& x) = delete;
  BPHVertexSelect& operator=(const BPHVertexSelect& x) = delete;

  /** Destructor
   */
  virtual ~BPHVertexSelect() = default;

  using AcceptArg = BPHDecayVertex;

  /** Operations
   */
  /// accept function
  virtual bool accept(const BPHDecayVertex& cand) const = 0;
  virtual bool accept(const BPHDecayVertex& cand, const BPHRecoBuilder* builder) const { return accept(cand); }
};

#endif
