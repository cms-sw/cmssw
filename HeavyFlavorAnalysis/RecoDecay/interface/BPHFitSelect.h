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
  BPHFitSelect();

  /** Destructor
   */
  virtual ~BPHFitSelect();

  /** Operations
   */
  /// accept function
  virtual bool accept( const BPHKinematicFit& cand ) const = 0;

 private:

  // private copy and assigment constructors
  BPHFitSelect           ( const BPHFitSelect& x );
  BPHFitSelect& operator=( const BPHFitSelect& x );

};


#endif

