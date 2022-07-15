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
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayGenericBuilderBase.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHMassSelect.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHChi2Select.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHMassFitSelect.h"

#include "FWCore/Framework/interface/EventSetup.h"

class BPHEventSetupWrapper;

//---------------
// C++ Headers --
//---------------
#include <vector>
#include <iostream>

//              ---------------------
//              -- Class Interface --
//              ---------------------

template <class ProdType>
class BPHDecayGenericBuilder : public virtual BPHDecayGenericBuilderBase {
public:
  typedef typename ProdType::const_pointer prod_ptr;

  /** Constructor
   */
  BPHDecayGenericBuilder(const BPHEventSetupWrapper& es, BPHMassFitSelect* mfs) : BPHDecayGenericBuilderBase(es, mfs) {}

  // deleted copy constructor and assignment operator
  BPHDecayGenericBuilder(const BPHDecayGenericBuilder& x) = delete;
  BPHDecayGenericBuilder& operator=(const BPHDecayGenericBuilder& x) = delete;

  /** Destructor
   */
  ~BPHDecayGenericBuilder() override = default;

  /** Operations
   */
  /// build candidates
  virtual std::vector<prod_ptr> build() {
    if (outdated) {
      recList.clear();
      fillRecList();
      outdated = false;
    }
    return recList;
  }

protected:
  BPHDecayGenericBuilder() {}

  std::vector<prod_ptr> recList;
};

#endif
