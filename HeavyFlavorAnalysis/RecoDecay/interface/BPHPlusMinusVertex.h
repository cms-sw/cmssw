#ifndef HeavyFlavorAnalysis_RecoDecay_BPHPlusMinusVertex_h
#define HeavyFlavorAnalysis_RecoDecay_BPHPlusMinusVertex_h
/** \class BPHPlusMinusVertex
 *
 *  Description: 
 *     class for reconstructed decay vertices to opposite charge
 *     particle pairs
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHDecayVertex.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHPlusMinusCandidatePtr.h"
#include "TrackingTools/PatternTools/interface/ClosestApproachInRPhi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class BPHEventSetupWrapper;

//---------------
// C++ Headers --
//---------------

//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHPlusMinusVertex : public virtual BPHDecayVertex {
public:
  /** Constructor is protected
   *  this object can exist only as part of a derived class
   */
  // deleted copy constructor and assignment operator
  BPHPlusMinusVertex(const BPHPlusMinusVertex& x) = delete;
  BPHPlusMinusVertex& operator=(const BPHPlusMinusVertex& x) = delete;

  /** Destructor
   */
  ~BPHPlusMinusVertex() override;

  /** Operations
   */
  /// compute distance of closest approach
  virtual const ClosestApproachInRPhi& cAppInRPhi() const;

protected:
  BPHPlusMinusVertex(const BPHEventSetupWrapper* es);

  // utility functions to check/enforce the number of decay particles at 2
  template <class T>
  static bool chkName(const T& cont, const std::string& name, const std::string& msg);
  template <class T>
  static bool chkSize(const T& cont, const std::string& msg);
  bool chkSize(const std::string& msg) const;

  // utility function used to cash reconstruction results
  void setNotUpdated() const override;

private:
  // reconstruction results cache
  mutable bool oldA;
  mutable ClosestApproachInRPhi* inRPhi;

  // compute closest approach distance and cache it
  virtual void computeApp() const;
};

template <class T>
bool BPHPlusMinusVertex::chkName(const T& cont, const std::string& name, const std::string& msg) {
  if (cont.find(name) != cont.end())
    return true;
  edm::LogPrint("ParticleNotFound") << msg << ", " << name << " not found";
  return false;
}

template <class T>
bool BPHPlusMinusVertex::chkSize(const T& cont, const std::string& msg) {
  int n = cont.size();
  if (n == 2)
    return true;
  edm::LogPrint("WrongDataSize") << msg << ", size = " << n;
  return false;
}

#endif
