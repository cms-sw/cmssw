#ifndef Mu_MuDTTPGPhiFlatTableProducer_h
#define Mu_MuDTTPGPhiFlatTableProducer_h

/** \class MuDTTPGPhiFlatTableProducer MuDTTPGPhiFlatTableProducer.h DPGAnalysis/MuonTools/plugins/MuDTTPGPhiFlatTableProducer.h
 *  
 * Helper class : the Phase-1 local trigger FlatTableProducer for TwinMux in/out and BMTF in (the DataFormat is the same)
 *
 * \author C. Battilana (INFN BO)
 *
 *
 */

#include "DPGAnalysis/MuonTools/src/MuBaseFlatTableProducer.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"

class MuDTTPGPhiFlatTableProducer : public MuBaseFlatTableProducer {
public:
  enum class TriggerTag { TM_IN = 0, TM_OUT, BMTF_IN };

  /// Constructor
  MuDTTPGPhiFlatTableProducer(const edm::ParameterSet&);

  /// Fill descriptors
  static void fillDescriptions(edm::ConfigurationDescriptions&);

protected:
  /// Fill tree branches for a given event
  void fillTable(edm::Event&) final;

  /// Get info from the ES by run
  void getFromES(const edm::Run&, const edm::EventSetup&) final;

private:
  /// Enum to activate "flavour-by-flavour"
  /// changes in the filling logic
  TriggerTag m_tag;

  /// The trigger-primitive token
  nano_mu::EDTokenHandle<L1MuDTChambPhContainer> m_token;

  /// The class to perform DT local trigger coordinate conversions
  nano_mu::DTTrigGeomUtils m_trigGeomUtils;

  /// Helper function translating config parameter into TriggerTag
  TriggerTag getTag(const edm::ParameterSet&);
};

#endif
