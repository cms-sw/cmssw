#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHMassCuts_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHMassCuts_h
/** \class BPHMassCuts
 *
 *  Description: 
 *     Base class for candidate selection by invariant mass:
 *     only the mass cuts are handled here, actual selection
 *     (at momentum sum or kinemtic fit level) are to be implemented
 *     in derived classes
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

//---------------
// C++ Headers --
//---------------

//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHMassCuts {
public:
  /** Constructor
   */
  BPHMassCuts(double minMass, double maxMass) : mMin(minMass), mMax(maxMass) {}

  // deleted copy constructor and assignment operator
  BPHMassCuts(const BPHMassCuts& x) = delete;
  BPHMassCuts& operator=(const BPHMassCuts& x) = delete;

  /** Destructor
   */
  virtual ~BPHMassCuts() = default;

  /** Operations
   */
  /// set mass cuts
  void setMassMin(double m) {
    mMin = m;
    return;
  }
  void setMassMax(double m) {
    mMax = m;
    return;
  }

  /// get current mass cuts
  double getMassMin() const { return mMin; }
  double getMassMax() const { return mMax; }

protected:
  double mMin;
  double mMax;
};

#endif
