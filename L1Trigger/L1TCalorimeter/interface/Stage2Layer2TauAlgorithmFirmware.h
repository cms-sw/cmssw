///
/// Description: Firmware headers
///
/// Implementation:
///    Concrete firmware implementations
///
/// \author: Jim Brooke - University of Bristol
///

//
//

#ifndef Stage2Layer2TauAlgorithmFirmware_H
#define Stage2Layer2TauAlgorithmFirmware_H

#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2TauAlgorithm.h"
#include "CondFormats/L1TObjects/interface/CaloParams.h"

namespace l1t {

  // Imp1 is for v1 and v2
  class Stage2Layer2TauAlgorithmFirmwareImp1 : public Stage2Layer2TauAlgorithm {
  public:
    Stage2Layer2TauAlgorithmFirmwareImp1(CaloParams* params); //const CaloMainProcessorParams & dbPars);
    virtual ~Stage2Layer2TauAlgorithmFirmwareImp1();
    virtual void processEvent(const std::vector<CaloCluster> & clusters,
    				 		  const std::vector<CaloTower>& towers,
			      			  std::vector<Tau> & taus);
    
  private:
    void merging(const std::vector<l1t::CaloCluster>& clusters,  const std::vector<l1t::CaloTower>& towers, std::vector<l1t::Tau>& taus);

    // isolation
    int isoCalTauHwFootPrint(const l1t::CaloCluster&,const std::vector<l1t::CaloTower>&);

	//calibration
    void loadCalibrationLuts();

    double calibratedPt(int hwPtEm, int hwPtHad, int ieta);
    
    // parameters
    CaloParams* params_;
    std::vector<std::vector<float> >coefficients_;

    float offsetBarrelEH_;
    float offsetBarrelH_;
    float offsetEndcapsEH_;
    float offsetEndcapsH_;
	
	unsigned int isoLutIndex(int Et, unsigned int nrTowers);
  };
  
}

#endif
