//
// F.Ratnikov (UMd), Dec. 14, 2005
//
#ifndef HcalDbHardcodeIn_h
#define HcalDbHardcodeIn_h

#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "CondFormats/HcalObjects/interface/HcalPedestal.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidth.h"
#include "CondFormats/HcalObjects/interface/HcalGain.h"
#include "CondFormats/HcalObjects/interface/HcalGainWidth.h"
#include "CondFormats/HcalObjects/interface/HcalQIECoder.h"
#include "CondFormats/HcalObjects/interface/HcalQIEShape.h"
#include "CondFormats/HcalObjects/interface/HcalQIEType.h"
#include "CondFormats/HcalObjects/interface/HcalCalibrationQIECoder.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
#include "CondFormats/HcalObjects/interface/HcalFrontEndMap.h"
#include "DataFormats/HcalDetId/interface/HcalDcsDetId.h"
#include "CondFormats/HcalObjects/interface/HcalDcsMap.h"
#include "CondFormats/HcalObjects/interface/HcalRecoParam.h"
#include "CondFormats/HcalObjects/interface/HcalTimingParam.h"
#include "CondFormats/HcalObjects/interface/HcalMCParam.h"
#include "CondFormats/HcalObjects/interface/HcalSiPMParameter.h"
#include "CondFormats/HcalObjects/interface/HcalSiPMCharacteristics.h"
#include "CondFormats/HcalObjects/interface/HcalTPParameters.h"
#include "CondFormats/HcalObjects/interface/HcalTPChannelParameters.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalHardcodeParameters.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

#include <vector>
#include <map>
#include <utility>

/**

   \class HcalDbHardcode
   \brief Hardcode implementation of some conditions data
   \author Fedor Ratnikov
   
*/
class HcalDbHardcode {
  public:
    //constructor
    HcalDbHardcode();
    
    //destructor
    virtual ~HcalDbHardcode() {}
    
    //setters
    void setHB(HcalHardcodeParameters p) { theHBParameters_ = p; setHB_ = true; }
    void setHE(HcalHardcodeParameters p) { theHEParameters_ = p; setHE_ = true; }
    void setHF(HcalHardcodeParameters p) { theHFParameters_ = p; setHF_ = true; }
    void setHO(HcalHardcodeParameters p) { theHOParameters_ = p; setHO_ = true; }
    void setHBUpgrade(HcalHardcodeParameters p) { theHBUpgradeParameters_ = p; setHBUpgrade_ = true; }
    void setHEUpgrade(HcalHardcodeParameters p) { theHEUpgradeParameters_ = p; setHEUpgrade_ = true; }
    void setHFUpgrade(HcalHardcodeParameters p) { theHFUpgradeParameters_ = p; setHFUpgrade_ = true; }
    void useHBUpgrade(bool b) { useHBUpgrade_ = b; }
    void useHEUpgrade(bool b) { useHEUpgrade_ = b; }
    void useHOUpgrade(bool b) { useHOUpgrade_ = b; }
    void useHFUpgrade(bool b) { useHFUpgrade_ = b; }
    void testHFQIE10(bool b) { testHFQIE10_ = b; }
    void setSiPMCharacteristics(std::vector<edm::ParameterSet> vps) { theSiPMCharacteristics_ = vps; }
    void setKillHE(bool b) { killHE_ = b; } 
    
    //getters
    const bool useHBUpgrade() const { return useHBUpgrade_; }
    const bool useHEUpgrade() const { return useHEUpgrade_; }
    const bool useHOUpgrade() const { return useHOUpgrade_; }
    const bool useHFUpgrade() const { return useHFUpgrade_; }
    const HcalHardcodeParameters& getParameters(HcalGenericDetId fId);
    const int getGainIndex(HcalGenericDetId fId);
    const bool killHE() const { return killHE_; }
    HcalPedestal makePedestal (HcalGenericDetId fId, bool fSmear = false);
    HcalPedestalWidth makePedestalWidth (HcalGenericDetId fId);
    HcalGain makeGain (HcalGenericDetId fId, bool fSmear = false);
    HcalGainWidth makeGainWidth (HcalGenericDetId fId);
    HcalQIECoder makeQIECoder (HcalGenericDetId fId);
    HcalCalibrationQIECoder makeCalibrationQIECoder (HcalGenericDetId fId);
    HcalQIEShape makeQIEShape ();
    HcalQIEType makeQIEType (HcalGenericDetId fId);
    HcalRecoParam makeRecoParam (HcalGenericDetId fId);
    HcalMCParam makeMCParam (HcalGenericDetId fId);
    HcalTimingParam makeTimingParam (HcalGenericDetId fId);
    void makeHardcodeMap(HcalElectronicsMap& emap, const std::vector<HcalGenericDetId>& cells);
    void makeHardcodeDcsMap(HcalDcsMap& dcs_map);
    void makeHardcodeFrontEndMap(HcalFrontEndMap& emap, const std::vector<HcalGenericDetId>& cells);
    HcalSiPMParameter makeHardcodeSiPMParameter (HcalGenericDetId fId, const HcalTopology* topo);
    void makeHardcodeSiPMCharacteristics (HcalSiPMCharacteristics& sipm);
    HcalTPChannelParameter makeHardcodeTPChannelParameter (HcalGenericDetId fId);
    void makeHardcodeTPParameters (HcalTPParameters& tppar);
    int getLayersInDepth(int ieta, int depth, const HcalTopology* topo);
    
  private:
    //member variables
    HcalHardcodeParameters theDefaultParameters_;
    HcalHardcodeParameters theHBParameters_, theHEParameters_, theHFParameters_, theHOParameters_;
    HcalHardcodeParameters theHBUpgradeParameters_, theHEUpgradeParameters_, theHFUpgradeParameters_;
    bool setHB_, setHE_, setHF_, setHO_, setHBUpgrade_, setHEUpgrade_, setHFUpgrade_, killHE_;
    bool useHBUpgrade_, useHEUpgrade_, useHOUpgrade_, useHFUpgrade_, testHFQIE10_;
    std::vector<edm::ParameterSet> theSiPMCharacteristics_;
    std::map<std::pair<int,int>,int> theLayersInDepths_;
};

#endif
