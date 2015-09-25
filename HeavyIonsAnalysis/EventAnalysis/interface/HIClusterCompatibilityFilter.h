#ifndef HIRun2015Ana_HIClusterCompatibilityFilter
#define HIRun2015Ana_HIClusterCompatibilityFilter

// Derived from HLTrigger/special/src/HLTPixelClusterShapeFilter.cc
// at version 7_5_0_pre3
//
//
// Author of Derived Filter:  Eric Appelt
//         Created:  Wed Apr 29, 2015
//
//


#include <iostream>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include <DataFormats/HeavyIonEvent/interface/ClusterCompatibility.h>


class HIClusterCompatibilityFilter : public edm::EDFilter {
  public:
    explicit HIClusterCompatibilityFilter(const edm::ParameterSet&);
    ~HIClusterCompatibilityFilter();

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    static double determineQuality(const reco::ClusterCompatibility & cc, 
                                   double minZ, double maxZ);

  private:
    virtual void beginJob() override;
    virtual bool filter(edm::Event&, const edm::EventSetup&) override;
    virtual void endJob() override;

    edm::EDGetTokenT<reco::ClusterCompatibility> cluscomSrc_;
  
    double              minZ_;          // beginning z-vertex position
    double              maxZ_;          // end z-vertex position

    std::vector<double> clusterPars_;   //pixel cluster polynomial pars for vertex compatibility cut
    int                 nhitsTrunc_;    //maximum pixel clusters to apply compatibility check
    double              clusterTrunc_;  //maximum vertex compatibility value for event rejection
};

#endif
