///step03
/// \class l1t::Stage1Layer2EGammaAlgorithm
///
/// Description: interface for MP firmware
///
/// Implementation:
///
/// \authors: Jim Brooke - University of Bristol
///           Kalanand Mishra - Fermilab
///

//

#ifndef Stage1Layer2EGammaAlgorithm_h
#define Stage1Layer2EGammaAlgorithm_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/L1TCalorimeter/interface/CaloStage1Cluster.h"
#include "DataFormats/L1TCalorimeter/interface/CaloRegion.h"

#include "DataFormats/L1Trigger/interface/EGamma.h"

#include "L1Trigger/L1TCalorimeter/interface/L1GObject.h"
#include "DataFormats/L1TCalorimeter/interface/CaloEmCand.h"

#include <vector>


namespace l1t {

  class Stage1Layer2EGammaAlgorithm {
  public:
    virtual void processEvent(const std::vector<l1t::CaloStage1Cluster> & clusters,
			      const std::vector<l1t::CaloRegion> & regions,
			      std::vector<l1t::EGamma> & egammas,
			      std::list<L1GObject> & rlxEGList,
			      std::list<L1GObject> & isoEGList) = 0;

    virtual ~Stage1Layer2EGammaAlgorithm(){};


    unsigned int egtSeed;

    double puLevel;
    double relativeIsolationCut;
    double relativeJetIsolationCut;
  };

}

#endif
