#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHMassFitSelect_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHMassFitSelect_h
/** \class BPHMassFitSelect
 *
 *  Description: 
 *     Class for candidate selection by invariant mass (at kinematic fit level)
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHFitSelect.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHMassCuts.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHKinematicFit.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicConstraint.h"
#include "RecoVertex/KinematicFitPrimitives/interface/MultiTrackKinematicConstraint.h"

//---------------
// C++ Headers --
//---------------
#include <string>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHMassFitSelect: public BPHFitSelect, public BPHMassCuts {

 public:

  /** Constructor
   */
  BPHMassFitSelect( double minMass, double maxMass ):
   BPHMassCuts( minMass, maxMass ) { setFitConstraint(); }

  BPHMassFitSelect( const std::string& name, double mass, double sigma,
                    double minMass, double maxMass ):
   BPHMassCuts( minMass, maxMass ) { setFitConstraint( name, mass, sigma ); }

  BPHMassFitSelect( const std::string& name, double mass,
                    double minMass, double maxMass ):
   BPHMassCuts( minMass, maxMass ) { setFitConstraint( name, mass ); }

  BPHMassFitSelect( const std::string& name, KinematicConstraint* c,
                    double minMass, double maxMass ):
   BPHMassCuts( minMass, maxMass ) { setFitConstraint( name, c ); }

  BPHMassFitSelect( const std::string& name, MultiTrackKinematicConstraint* c,
                    double minMass, double maxMass ):
   BPHMassCuts( minMass, maxMass ) { setFitConstraint( name, c ); }

  /** Destructor
   */
  virtual ~BPHMassFitSelect() {
  }

  /** Operations
   */
  /// select particle
  virtual bool accept( const BPHKinematicFit& cand ) const {
    switch ( type ) {
    default:
    case none: break;
    case mcss: cand.kinematicTree( cName, cMass, cSigma ); break;
    case mcst: cand.kinematicTree( cName, cMass )        ; break;
    case   kf: cand.kinematicTree( cName,   kc          ); break;
    case mtkf: cand.kinematicTree( cName, mtkc          ); break;
    }
    double mass = cand.p4().mass();
    return ( ( mass > mMin ) && ( mass < mMax ) );
  }

  /// set fit constraint
  void setFitConstraint() {
    type = none;
    cName  = ""  ;
    cMass  = -1.0;
    cSigma = -1.0;
      kc   =  0  ;
    mtkc   =  0  ;
  }
  void setFitConstraint( const std::string& name, double mass ) {
    type = mcst;
    cName  = name ;
    cMass  = mass ;
    cSigma = -1.0 ;
      kc   =  0   ;
    mtkc   =  0   ;
  }
  void setFitConstraint( const std::string& name, double mass, double sigma ) {
    type = mcss;
    cName  = name ;
    cMass  = mass ;
    cSigma = sigma;
      kc   =  0   ;
    mtkc   =  0   ;
  }
  void setFitConstraint( const std::string& name, KinematicConstraint* c ) {
    type = kf;
    cName  = name ;
    cMass  = -1.0 ;
    cSigma = -1.0 ;
      kc   =  c   ;
    mtkc   =  0   ;
  }
  void setFitConstraint( const std::string& name, 
                                        MultiTrackKinematicConstraint* c ) {
    type = mtkf;
    cName  = name ;
    cMass  = -1.0 ;
    cSigma = -1.0 ;
      kc   =  0   ;
    mtkc   =  c   ;
  }

  /// get fit constraint
  const std::string& getConstrainedName() const { return cName; }
  double getMass () const { return cMass;  }
  double getSigma() const { return cSigma; }
            KinematicConstraint* getKC          () const { return   kc; }
  MultiTrackKinematicConstraint* getMultiTrackKC() const { return mtkc; }

 private:

  // private copy and assigment constructors
  BPHMassFitSelect           ( const BPHMassFitSelect& x );
  BPHMassFitSelect& operator=( const BPHMassFitSelect& x );

  enum fit_type { none, mcss, mcst, kf, mtkf };

  fit_type type;
  std::string cName;
  double cMass;
  double cSigma;
            KinematicConstraint*   kc;
  MultiTrackKinematicConstraint* mtkc;

};


#endif

