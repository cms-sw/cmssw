#ifndef BPHMassCuts_H
#define BPHMassCuts_H
/** \class BPHMassCuts
 *
 *  Description: 
 *     Base class for candidate selection by invariant mass:
 *     only the mass cuts are handled here, actual selection
 *     (at momentum sum or kinemtic fit level) are to be implemented
 *     in derived classes
 *
 *  $Date: 2016-05-03 14:47:26 $
 *  $Revision: 1.1 $
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
  BPHMassCuts( double minMass, double maxMass ): mMin( minMass ),
                                                 mMax( maxMass ) {}

  /** Destructor
   */
  virtual ~BPHMassCuts() {}

  /** Operations
   */
  /// set mass cuts
  void setMassMin( double m ) { mMin = m; return; }
  void setMassMax( double m ) { mMax = m; return; }

  /// get current mass cuts
  double getMassMin() const { return mMin; }
  double getMassMax() const { return mMax; }

 protected:

  // private copy and assigment constructors
  BPHMassCuts           ( const BPHMassCuts& x );
  BPHMassCuts& operator=( const BPHMassCuts& x );

  double mMin;
  double mMax;

};


#endif // BPHMassCuts_H

