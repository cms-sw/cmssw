#ifndef RecoEcal_EgammaClusterProducers_SCTestAnalyzer_h_
#define RecoEcal_EgammaClusterProducers_SCTestAnalyzer_h_

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoEcal/EgammaClusterProducers/interface/IslandClusterAlgo.h"


//


class SCTestAnalyzer : public edm::EDAnalyzer {
   public:
      explicit SCTestAnalyzer(const edm::ParameterSet&);
      ~SCTestAnalyzer();


      virtual void analyze(const edm::Event&, const edm::EventSetup&);
   private:

};


#endif
