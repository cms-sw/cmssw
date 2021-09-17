#ifndef HeavyFlavorAnalysis_RecoDecay_BPHFitSelect_h
#define HeavyFlavorAnalysis_RecoDecay_BPHFitSelect_h
/** \class BPHFitSelect
 *
 *  Description: 
 *     Base class for candidate selection at kinematic fit level
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
class BPHKinematicFit;

//---------------
// C++ Headers --
//---------------

//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHFitSelect {
public:
  /** Constructor
   */
  BPHFitSelect() {}

  // deleted copy constructor and assignment operator
  BPHFitSelect(const BPHFitSelect& x) = delete;
  BPHFitSelect& operator=(const BPHFitSelect& x) = delete;

  /** Destructor
   */
  virtual ~BPHFitSelect() {}

  using AcceptArg = BPHKinematicFit;

  /** Operations
   */
  /// accept function
  virtual bool accept(const BPHKinematicFit& cand) const = 0;
};

#endif
