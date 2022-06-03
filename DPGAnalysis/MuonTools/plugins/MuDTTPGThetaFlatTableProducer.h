#ifndef Mu_MuDTTPGThetaFlatTableProducer_h
#define Mu_MuDTTPGThetaFlatTableProducer_h

/** \class MuDTTPGThetaFlatTableProducer MuDTTPGThetaFlatTableProducer.h DPGAnalysis/MuonTools/plugins/MuDTTPGThetaFlatTableProducer.h
 *  
 * Helper class : the Phase-1 local trigger FlatTableProducer for TwinMux and BMTF in (the DataFormat is the same)
 *
 * \author C. Battilana (INFN BO)
 *
 *
 */

#include "DPGAnalysis/MuonTools/src/MuBaseFlatTableProducer.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"

class MuDTTPGThetaFlatTableProducer : public MuBaseFlatTableProducer {
public:
  enum class TriggerTag { TM_IN = 0, BMTF_IN };

  /// Constructor
  MuDTTPGThetaFlatTableProducer(const edm::ParameterSet&);

  /// Fill descriptors
  static void fillDescriptions(edm::ConfigurationDescriptions&);

protected:
  /// Fill tree branches for a given events
  void fillTable(edm::Event&) final;

private:
  /// Enum to activate "flavour-by-flavour"
  /// changes in the filling logic
  TriggerTag m_tag;

  /// The trigger-primitive token
  nano_mu::EDTokenHandle<L1MuDTChambThContainer> m_token;

  /// Helper function translating config parameter into TriggerTag
  TriggerTag getTag(const edm::ParameterSet&);
};

#endif
