#ifndef HLTHPDFilter_h
#define HLTHPDFilter_h

/** \class HLTHPDFilter
 *
 *  \author Fedor Ratnikov (UMd)
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace edm {
   class ConfigurationDescriptions;
}

class HLTHPDFilter : public edm::EDFilter {

   public:
      explicit HLTHPDFilter(const edm::ParameterSet&);
      ~HLTHPDFilter();
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag mInputTag; // input tag for HCAL HBHE digis
      double mEnergyThreshold;
      double mHPDSpikeEnergyThreshold;
      double mHPDSpikeIsolationEnergyThreshold;
      double mRBXSpikeEnergyThreshold;
      double mRBXSpikeUnbalanceThreshold;
};

#endif //HLTHPDFilter_h
 
