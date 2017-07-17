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
#include "L1Trigger/L1TCalorimeter/interface/CaloStage2Nav.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"


namespace l1t {

  // Imp1 is for v1 and v2
  class Stage2Layer2TauAlgorithmFirmwareImp1 : public Stage2Layer2TauAlgorithm {
  public:
    Stage2Layer2TauAlgorithmFirmwareImp1(CaloParamsHelper* params); //const CaloMainProcessorParams & dbPars);
    virtual ~Stage2Layer2TauAlgorithmFirmwareImp1();
    virtual void processEvent(const std::vector<CaloCluster> & clusters,
                  const std::vector<CaloTower>& towers,
                  std::vector<Tau> & taus);
  private:
    void merging(const std::vector<l1t::CaloCluster>& clusters,  const std::vector<l1t::CaloTower>& towers, std::vector<l1t::Tau>& taus);
    void dosorting(std::vector<l1t::Tau>& taus);

    // isolation
    int isoCalTauHwFootPrint(const l1t::CaloCluster&,const std::vector<l1t::CaloTower>&);

    //calibration
    void loadCalibrationLuts();

    // double calibratedPt(int hwPtEm, int hwPtHad, int ieta);

    // parameters
    CaloParamsHelper* params_;
    std::vector<std::vector<float> >coefficients_;

    float offsetBarrelEH_;
    float offsetBarrelH_;
    float offsetEndcapsEH_;
    float offsetEndcapsH_;
    unsigned int isoLutIndex(int Et, int hweta, unsigned int nrTowers);
    unsigned int trimMainLutIndex (int neighPos, bool isWe);
    static bool compareTowers (l1t::CaloTower TT1, l1t::CaloTower TT2); // implements operator < for TT
    bool is3x3Maximum (const l1t::CaloTower& tower, const std::vector<CaloTower>& towers, l1t::CaloStage2Nav& caloNav); // is maximum in the 3x3 window? (recompute jet flag)
    std::vector<l1t::CaloCluster*> makeSecClusters (const std::vector<l1t::CaloTower>& towers, std::vector<int> & sites, const l1t::CaloCluster& mainCluster, l1t::CaloStage2Nav& caloNav); // make the secondary clusters fr merging (need to be deleted later)
    unsigned int calibLutIndex (int ieta, int Et, int hasEM, int isMerged);
    int calibratedPt(const l1t::CaloCluster& clus, const std::vector<l1t::CaloTower>& towers, int hwPt, bool isMerged);

  };

}

#endif
