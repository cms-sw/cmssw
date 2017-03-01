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
#include "DataFormats/L1TCalorimeter/interface/CaloRegion.h"

#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Jet.h"

#include "L1Trigger/L1TCalorimeter/interface/L1GObject.h"
#include "DataFormats/L1TCalorimeter/interface/CaloEmCand.h"

#include <vector>


namespace l1t {

  class Stage1Layer2EGammaAlgorithm {
  public:
    virtual void processEvent(const std::vector<l1t::CaloEmCand> & EMCands,
			      const std::vector<l1t::CaloRegion> & regions,
			      const std::vector<l1t::Jet> * jets,
			      std::vector<l1t::EGamma>* egammas) = 0;

    virtual ~Stage1Layer2EGammaAlgorithm(){};

  private:
    // double Isolation(int ieta, int iphi,
    // 		     const std::vector<l1t::CaloRegion> & regions)  const;
    // double HoverE(int et, int ieta, int iphi,
    // 		  const std::vector<l1t::CaloRegion> & regions)  const;
    // unsigned int egtSeed;
    // double relativeIsolationCut;
    // double HoverECut;
  };

}

#endif
