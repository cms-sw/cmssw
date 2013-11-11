#ifndef HLTRapGapFilter_h
#define HLTRapGapFilter_h

/** \class HLTRapGapFilter
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"

//
// class decleration
//

class HLTRapGapFilter : public HLTFilter {

   public:
      explicit HLTRapGapFilter(const edm::ParameterSet&);
      ~HLTRapGapFilter();
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

   private:
      edm::EDGetTokenT<reco::CaloJetCollection> m_theJetToken;
      edm::InputTag inputTag_; // input tag identifying jets
      double absEtaMin_;
      double absEtaMax_;
      double caloThresh_;
};

#endif //HLTRapGapFilter_h
