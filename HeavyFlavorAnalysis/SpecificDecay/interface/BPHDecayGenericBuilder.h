#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHDecayGenericBuilder_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHDecayGenericBuilder_h
/** \class BPHDecayGenericBuilder
 *
 *  Description: 
 *     Class to build a generic decay applying selections to the
 *     reconstructed particle
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
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHMassSelect.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHChi2Select.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHMassFitSelect.h"

#include "FWCore/Framework/interface/Event.h"

//---------------
// C++ Headers --
//---------------
#include <string>
#include <vector>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHDecayGenericBuilder {
public:
  /** Constructor
   */
  BPHDecayGenericBuilder(const edm::EventSetup& es, BPHMassFitSelect* mfs = nullptr);

  // deleted copy constructor and assignment operator
  BPHDecayGenericBuilder(const BPHDecayGenericBuilder& x) = delete;
  BPHDecayGenericBuilder& operator=(const BPHDecayGenericBuilder& x) = delete;

  /** Destructor
   */
  virtual ~BPHDecayGenericBuilder();

  /** Operations
   */
  /// set cuts
  void setMassMin(double m);
  void setMassMax(double m);
  void setMassRange(double mMin, double mMax);
  void setProbMin(double p);
  void setMassFitMin(double m);
  void setMassFitMax(double m);
  void setMassFitRange(double mMin, double mMax);

  /// get current cuts
  double getMassMin() const { return massSel->getMassMin(); }
  double getMassMax() const { return massSel->getMassMax(); }
  double getProbMin() const { return chi2Sel->getProbMin(); }
  double getMassFitMin() const { return mFitSel->getMassMin(); }
  double getMassFitMax() const { return mFitSel->getMassMax(); }

  /// track min p difference
  void setMinPDiff(double mpd) { minPDiff = mpd; }
  double getMinPDiff() { return minPDiff; }

protected:
  const edm::EventSetup* evSetup;

  BPHMassSelect* massSel;
  BPHChi2Select* chi2Sel;
  BPHMassFitSelect* mFitSel;

  double minPDiff;
  bool updated;
};

#endif
