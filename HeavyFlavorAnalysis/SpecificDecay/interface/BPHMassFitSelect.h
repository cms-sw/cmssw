#ifndef BPHMassFitSelect_H
#define BPHMassFitSelect_H
/** \class BPHMassFitSelect
 *
 *  Description: 
 *     Class for candidate selection by invariant mass (at kinematic fit level)
 *
 *  $Date: 2016-08-11 15:49:00 $
 *  $Revision: 1.1 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHFitSelect.h"
class BPHKinematicFit;

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
class MultiTrackKinematicConstraint;
class KinematicConstraint;

//---------------
// C++ Headers --
//---------------
#include <string>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHMassFitSelect: public BPHFitSelect {

 public:

  /** Constructor
   */
  BPHMassFitSelect( double minMass, double maxMass );
  BPHMassFitSelect( const std::string& name, double mass, double sigma,
                    double minMass, double maxMass );
  BPHMassFitSelect( const std::string& name, double mass,
                    double minMass, double maxMass );
  BPHMassFitSelect( const std::string& name, KinematicConstraint* c,
                    double minMass, double maxMass );
  BPHMassFitSelect( const std::string& name, MultiTrackKinematicConstraint* c,
                    double minMass, double maxMass );

  /** Destructor
   */
  virtual ~BPHMassFitSelect();

  /** Operations
   */
  /// accept particle
  virtual bool accept( const BPHKinematicFit& cand ) const;

  /// set mass cuts
  void setMassMin( double m );
  void setMassMax( double m );

  /// get current mass cuts
  double getMassMin() const;
  double getMassMax() const;

 private:

  // private copy and assigment constructors
  BPHMassFitSelect           ( const BPHMassFitSelect& x );
  BPHMassFitSelect& operator=( const BPHMassFitSelect& x );

  const std::string cName;
  double cMass;
  double cSigma;
            KinematicConstraint*   kc;
  MultiTrackKinematicConstraint* mtkc;

  double mMin;
  double mMax;

};


#endif // BPHMassFitSelect_H

