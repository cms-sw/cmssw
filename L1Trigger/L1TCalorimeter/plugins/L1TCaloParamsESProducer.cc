///
/// \class L1TCaloParamsESProducer
///
/// Description: Produces configuration parameters for the fictitious Yellow trigger.
///
/// Implementation:
///    Dummy producer for L1 calo upgrade configuration parameters
///
/// \author: Jim Brooke, University of Bristol
///

//
//




// system include files
#include <memory>
#include <iostream>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducts.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "CondFormats/L1TObjects/interface/CaloParams.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"
#include "CondFormats/DataRecord/interface/L1TCaloParamsRcd.h"

using namespace std;

//
// class declaration
//

using namespace l1t;

class L1TCaloParamsESProducer : public edm::ESProducer {
public:
  L1TCaloParamsESProducer(const edm::ParameterSet&);
  ~L1TCaloParamsESProducer();

  typedef std::shared_ptr<CaloParams> ReturnType;

  ReturnType produce(const L1TCaloParamsRcd&);

private:
  CaloParams  m_params ;
  std::string m_label;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
L1TCaloParamsESProducer::L1TCaloParamsESProducer(const edm::ParameterSet& conf)
{

  //the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this);
  //setWhatProduced(this, conf.getParameter<std::string>("label"));

  CaloParamsHelper m_params_helper;

  // towers
  m_params_helper.setTowerLsbH(conf.getParameter<double>("towerLsbH"));
  m_params_helper.setTowerLsbE(conf.getParameter<double>("towerLsbE"));
  m_params_helper.setTowerLsbSum(conf.getParameter<double>("towerLsbSum"));
  m_params_helper.setTowerNBitsH(conf.getParameter<int>("towerNBitsH"));
  m_params_helper.setTowerNBitsE(conf.getParameter<int>("towerNBitsE"));
  m_params_helper.setTowerNBitsSum(conf.getParameter<int>("towerNBitsSum"));
  m_params_helper.setTowerNBitsRatio(conf.getParameter<int>("towerNBitsRatio"));
  m_params_helper.setTowerEncoding(conf.getParameter<bool>("towerEncoding"));

  // regions
  m_params_helper.setRegionLsb(conf.getParameter<double>("regionLsb"));
  m_params_helper.setRegionPUSType(conf.getParameter<std::string>("regionPUSType"));
  m_params_helper.setRegionPUSParams(conf.getParameter<std::vector<double> >("regionPUSParams"));

  // EG
  m_params_helper.setEgEtaCut(conf.getParameter<int>("egEtaCut"));

  m_params_helper.setEgLsb(conf.getParameter<double>("egLsb"));
  m_params_helper.setEgSeedThreshold(conf.getParameter<double>("egSeedThreshold"));
  m_params_helper.setEgNeighbourThreshold(conf.getParameter<double>("egNeighbourThreshold"));
  m_params_helper.setEgHcalThreshold(conf.getParameter<double>("egHcalThreshold"));

  edm::FileInPath egTrimmingLUTFile = conf.getParameter<edm::FileInPath>("egTrimmingLUTFile");
  std::ifstream egTrimmingLUTStream(egTrimmingLUTFile.fullPath());
  auto egTrimmingLUT = std::make_shared<LUT>(egTrimmingLUTStream);
  m_params_helper.setEgTrimmingLUT(*egTrimmingLUT);

  m_params_helper.setEgMaxHcalEt(conf.getParameter<double>("egMaxHcalEt"));
  m_params_helper.setEgMaxPtHOverE(conf.getParameter<double>("egMaxPtHOverE"));
  m_params_helper.setEgHOverEcutBarrel(conf.getParameter<int>("egHOverEcutBarrel"));
  m_params_helper.setEgHOverEcutEndcap(conf.getParameter<int>("egHOverEcutEndcap"));
  m_params_helper.setEgMinPtJetIsolation(conf.getParameter<int>("egMinPtJetIsolation"));
  m_params_helper.setEgMaxPtJetIsolation(conf.getParameter<int>("egMaxPtJetIsolation"));
  m_params_helper.setEgMinPtHOverEIsolation(conf.getParameter<int>("egMinPtHOverEIsolation"));
  m_params_helper.setEgMaxPtHOverEIsolation(conf.getParameter<int>("egMaxPtHOverEIsolation"));
  m_params_helper.setEgBypassEGVetos(conf.getParameter<unsigned>("egBypassEGVetos"));


  edm::FileInPath egMaxHOverELUTFile = conf.getParameter<edm::FileInPath>("egMaxHOverELUTFile");
  std::ifstream egMaxHOverELUTStream(egMaxHOverELUTFile.fullPath());
  auto egMaxHOverELUT = std::make_shared<LUT>(egMaxHOverELUTStream);
  m_params_helper.setEgMaxHOverELUT(*egMaxHOverELUT);

  edm::FileInPath egCompressShapesLUTFile = conf.getParameter<edm::FileInPath>("egCompressShapesLUTFile");
  std::ifstream egCompressShapesLUTStream(egCompressShapesLUTFile.fullPath());
  auto egCompressShapesLUT = std::make_shared<LUT>(egCompressShapesLUTStream);
  m_params_helper.setEgCompressShapesLUT(*egCompressShapesLUT);

  m_params_helper.setEgShapeIdType(conf.getParameter<std::string>("egShapeIdType"));
  m_params_helper.setEgShapeIdVersion(conf.getParameter<unsigned>("egShapeIdVersion"));
  edm::FileInPath egShapeIdLUTFile = conf.getParameter<edm::FileInPath>("egShapeIdLUTFile");
  std::ifstream egShapeIdLUTStream(egShapeIdLUTFile.fullPath());
  auto egShapeIdLUT = std::make_shared<LUT>(egShapeIdLUTStream);
  m_params_helper.setEgShapeIdLUT(*egShapeIdLUT);

  m_params_helper.setEgPUSType(conf.getParameter<std::string>("egPUSType"));

  m_params_helper.setEgIsolationType(conf.getParameter<std::string>("egIsolationType"));
  edm::FileInPath egIsoLUTFile = conf.getParameter<edm::FileInPath>("egIsoLUTFile");
  std::ifstream egIsoLUTStream(egIsoLUTFile.fullPath());
  auto egIsoLUT = std::make_shared<LUT>(egIsoLUTStream);
  m_params_helper.setEgIsolationLUT(*egIsoLUT);

  //edm::FileInPath egIsoLUTFileBarrel = conf.getParameter<edm::FileInPath>("egIsoLUTFileBarrel");
  //std::ifstream egIsoLUTBarrelStream(egIsoLUTFileBarrel.fullPath());
  //auto egIsoLUTBarrel = std::make_shared<LUT>(egIsoLUTBarrelStream);
  //m_params_helper.setEgIsolationLUTBarrel(egIsoLUTBarrel);

  //edm::FileInPath egIsoLUTFileEndcaps = conf.getParameter<edm::FileInPath>("egIsoLUTFileEndcaps");
  //std::ifstream egIsoLUTEndcapsStream(egIsoLUTFileEndcaps.fullPath());
  //auto egIsoLUTEndcaps = std::make_shared<LUT>(egIsoLUTEndcapsStream);
  //m_params_helper.setEgIsolationLUTEndcaps(egIsoLUTEndcaps);


  m_params_helper.setEgIsoAreaNrTowersEta(conf.getParameter<unsigned int>("egIsoAreaNrTowersEta"));
  m_params_helper.setEgIsoAreaNrTowersPhi(conf.getParameter<unsigned int>("egIsoAreaNrTowersPhi"));
  m_params_helper.setEgIsoVetoNrTowersPhi(conf.getParameter<unsigned int>("egIsoVetoNrTowersPhi"));
  //m_params_helper.setEgIsoPUEstTowerGranularity(conf.getParameter<unsigned int>("egIsoPUEstTowerGranularity"));
  //m_params_helper.setEgIsoMaxEtaAbsForTowerSum(conf.getParameter<unsigned int>("egIsoMaxEtaAbsForTowerSum"));
  //m_params_helper.setEgIsoMaxEtaAbsForIsoSum(conf.getParameter<unsigned int>("egIsoMaxEtaAbsForIsoSum"));
  m_params_helper.setEgPUSParams(conf.getParameter<std::vector<double>>("egPUSParams"));

  m_params_helper.setEgCalibrationType(conf.getParameter<std::string>("egCalibrationType"));
  m_params_helper.setEgCalibrationVersion(conf.getParameter<unsigned>("egCalibrationVersion"));
  edm::FileInPath egCalibrationLUTFile = conf.getParameter<edm::FileInPath>("egCalibrationLUTFile");
  std::ifstream egCalibrationLUTStream(egCalibrationLUTFile.fullPath());
  auto egCalibrationLUT = std::make_shared<LUT>(egCalibrationLUTStream);
  m_params_helper.setEgCalibrationLUT(*egCalibrationLUT);

  // tau
  m_params_helper.setTauRegionMask(conf.getParameter<int>("tauRegionMask"));
  m_params_helper.setTauLsb(conf.getParameter<double>("tauLsb"));
  m_params_helper.setTauSeedThreshold(conf.getParameter<double>("tauSeedThreshold"));
  m_params_helper.setTauNeighbourThreshold(conf.getParameter<double>("tauNeighbourThreshold"));
  m_params_helper.setTauMaxPtTauVeto(conf.getParameter<double>("tauMaxPtTauVeto"));
  m_params_helper.setTauMinPtJetIsolationB(conf.getParameter<double>("tauMinPtJetIsolationB"));
  m_params_helper.setTauPUSType(conf.getParameter<std::string>("tauPUSType"));
  m_params_helper.setTauMaxJetIsolationB(conf.getParameter<double>("tauMaxJetIsolationB"));
  m_params_helper.setTauMaxJetIsolationA(conf.getParameter<double>("tauMaxJetIsolationA"));
  m_params_helper.setTauIsoAreaNrTowersEta(conf.getParameter<unsigned int>("tauIsoAreaNrTowersEta"));
  m_params_helper.setTauIsoAreaNrTowersPhi(conf.getParameter<unsigned int>("tauIsoAreaNrTowersPhi"));
  m_params_helper.setTauIsoVetoNrTowersPhi(conf.getParameter<unsigned int>("tauIsoVetoNrTowersPhi"));

  edm::FileInPath tauIsoLUTFile = conf.getParameter<edm::FileInPath>("tauIsoLUTFile");
  std::ifstream tauIsoLUTStream(tauIsoLUTFile.fullPath());
  auto tauIsoLUT = std::make_shared<LUT>(tauIsoLUTStream);
  m_params_helper.setTauIsolationLUT(*tauIsoLUT);

  edm::FileInPath tauIsoLUTFile2 = conf.getParameter<edm::FileInPath>("tauIsoLUTFile2");
  std::ifstream tauIsoLUTStream2(tauIsoLUTFile2.fullPath());
  std::shared_ptr<LUT> tauIsoLUT2( new LUT(tauIsoLUTStream2) );
  m_params_helper.setTauIsolationLUT2(*tauIsoLUT2);

  edm::FileInPath tauCalibrationLUTFile = conf.getParameter<edm::FileInPath>("tauCalibrationLUTFile");
  std::ifstream tauCalibrationLUTStream(tauCalibrationLUTFile.fullPath());
  auto tauCalibrationLUT = std::make_shared<LUT>(tauCalibrationLUTStream);
  m_params_helper.setTauCalibrationLUT(*tauCalibrationLUT);

  edm::FileInPath tauCompressLUTFile = conf.getParameter<edm::FileInPath>("tauCompressLUTFile");
  std::ifstream tauCompressLUTStream(tauCompressLUTFile.fullPath());
  auto tauCompressLUT = std::make_shared<LUT>(tauCompressLUTStream);
  m_params_helper.setTauCompressLUT(*tauCompressLUT);

  edm::FileInPath tauEtToHFRingEtLUTFile = conf.getParameter<edm::FileInPath>("tauEtToHFRingEtLUTFile");
  std::ifstream tauEtToHFRingEtLUTStream(tauEtToHFRingEtLUTFile.fullPath());
  auto tauEtToHFRingEtLUT = std::make_shared<LUT>(tauEtToHFRingEtLUTStream);
  m_params_helper.setTauEtToHFRingEtLUT(*tauEtToHFRingEtLUT);

  m_params_helper.setIsoTauEtaMin(conf.getParameter<int> ("isoTauEtaMin"));
  m_params_helper.setIsoTauEtaMax(conf.getParameter<int> ("isoTauEtaMax"));

  m_params_helper.setTauPUSParams(conf.getParameter<std::vector<double>>("tauPUSParams"));

  // jets
  m_params_helper.setJetLsb(conf.getParameter<double>("jetLsb"));
  m_params_helper.setJetSeedThreshold(conf.getParameter<double>("jetSeedThreshold"));
  m_params_helper.setJetNeighbourThreshold(conf.getParameter<double>("jetNeighbourThreshold"));
  m_params_helper.setJetRegionMask(conf.getParameter<int>("jetRegionMask"));
  m_params_helper.setJetPUSType(conf.getParameter<std::string>("jetPUSType"));
  m_params_helper.setJetBypassPUS(conf.getParameter<unsigned>("jetBypassPUS"));
  m_params_helper.setJetCalibrationType(conf.getParameter<std::string>("jetCalibrationType"));
  m_params_helper.setJetCalibrationParams(conf.getParameter<std::vector<double> >("jetCalibrationParams"));
  edm::FileInPath jetCalibrationLUTFile = conf.getParameter<edm::FileInPath>("jetCalibrationLUTFile");
  std::ifstream jetCalibrationLUTStream(jetCalibrationLUTFile.fullPath());
  auto jetCalibrationLUT = std::make_shared<LUT>(jetCalibrationLUTStream);
  m_params_helper.setJetCalibrationLUT(*jetCalibrationLUT);
  edm::FileInPath jetCompressEtaLUTFile = conf.getParameter<edm::FileInPath>("jetCompressEtaLUTFile");
  std::ifstream jetCompressEtaLUTStream(jetCompressEtaLUTFile.fullPath());
  std::shared_ptr<LUT> jetCompressEtaLUT( new LUT(jetCompressEtaLUTStream) );
  m_params_helper.setJetCompressEtaLUT(*jetCompressEtaLUT);
  edm::FileInPath jetCompressPtLUTFile = conf.getParameter<edm::FileInPath>("jetCompressPtLUTFile");
  std::ifstream jetCompressPtLUTStream(jetCompressPtLUTFile.fullPath());
  std::shared_ptr<LUT> jetCompressPtLUT( new LUT(jetCompressPtLUTStream) );
  m_params_helper.setJetCompressPtLUT(*jetCompressPtLUT);

  // sums
  m_params_helper.setEtSumLsb(conf.getParameter<double>("etSumLsb"));

  std::vector<int> etSumEtaMin = conf.getParameter<std::vector<int> >("etSumEtaMin");
  std::vector<int> etSumEtaMax = conf.getParameter<std::vector<int> >("etSumEtaMax");
  std::vector<double> etSumEtThreshold = conf.getParameter<std::vector<double> >("etSumEtThreshold");
  m_params_helper.setEtSumBypassPUS(conf.getParameter<unsigned>("etSumBypassPUS"));
  m_params_helper.setEtSumPUSType(conf.getParameter<std::string>("etSumPUSType"));
  m_params_helper.setEtSumXCalibrationType(conf.getParameter<std::string>("etSumXCalibrationType"));
  m_params_helper.setEtSumYCalibrationType(conf.getParameter<std::string>("etSumYCalibrationType"));
  m_params_helper.setEtSumEttCalibrationType(conf.getParameter<std::string>("etSumEttCalibrationType"));
  m_params_helper.setEtSumEcalSumCalibrationType(conf.getParameter<std::string>("etSumEcalSumCalibrationType"));

  if ((etSumEtaMin.size() == etSumEtaMax.size()) &&  (etSumEtaMin.size() == etSumEtThreshold.size())) {
    for (unsigned i=0; i<etSumEtaMin.size(); ++i) {
      m_params_helper.setEtSumEtaMin(i, etSumEtaMin.at(i));
      m_params_helper.setEtSumEtaMax(i, etSumEtaMax.at(i));
      m_params_helper.setEtSumEtThreshold(i, etSumEtThreshold.at(i));
    }
  }
  else {
    edm::LogError("l1t|calo") << "Inconsistent number of EtSum parameters" << std::endl;
  }

  edm::FileInPath etSumPUSLUTFile = conf.getParameter<edm::FileInPath>("etSumPUSLUTFile");
  std::ifstream etSumPUSLUTStream(etSumPUSLUTFile.fullPath());
  std::shared_ptr<LUT> etSumPUSLUT( new LUT(etSumPUSLUTStream) );
  m_params_helper.setEtSumPUSLUT(*etSumPUSLUT);

  edm::FileInPath etSumXCalibrationLUTFile = conf.getParameter<edm::FileInPath>("etSumXCalibrationLUTFile");
  std::ifstream etSumXCalibrationLUTStream(etSumXCalibrationLUTFile.fullPath());
  std::shared_ptr<LUT> etSumXCalibrationLUT( new LUT(etSumXCalibrationLUTStream) );
  m_params_helper.setEtSumXCalibrationLUT(*etSumXCalibrationLUT);
  
  edm::FileInPath etSumYCalibrationLUTFile = conf.getParameter<edm::FileInPath>("etSumYCalibrationLUTFile");
  std::ifstream etSumYCalibrationLUTStream(etSumYCalibrationLUTFile.fullPath());
  std::shared_ptr<LUT> etSumYCalibrationLUT( new LUT(etSumYCalibrationLUTStream) );
  m_params_helper.setEtSumYCalibrationLUT(*etSumYCalibrationLUT);

  edm::FileInPath etSumEttCalibrationLUTFile = conf.getParameter<edm::FileInPath>("etSumEttCalibrationLUTFile");
  std::ifstream etSumEttCalibrationLUTStream(etSumEttCalibrationLUTFile.fullPath());
  std::shared_ptr<LUT> etSumEttCalibrationLUT( new LUT(etSumEttCalibrationLUTStream) );
  m_params_helper.setEtSumEttCalibrationLUT(*etSumEttCalibrationLUT);

  edm::FileInPath etSumEcalSumCalibrationLUTFile = conf.getParameter<edm::FileInPath>("etSumEcalSumCalibrationLUTFile");
  std::ifstream etSumEcalSumCalibrationLUTStream(etSumEcalSumCalibrationLUTFile.fullPath());
  std::shared_ptr<LUT> etSumEcalSumCalibrationLUT( new LUT(etSumEcalSumCalibrationLUTStream) );
  m_params_helper.setEtSumEcalSumCalibrationLUT(*etSumEcalSumCalibrationLUT);

  // HI centrality trigger
  edm::FileInPath centralityLUTFile = conf.getParameter<edm::FileInPath>("centralityLUTFile");
  std::ifstream centralityLUTStream(centralityLUTFile.fullPath());
  auto centralityLUT = std::make_shared<LUT>(centralityLUTStream);
  m_params_helper.setCentralityLUT(*centralityLUT);
  m_params_helper.setCentralityRegionMask(conf.getParameter<int>("centralityRegionMask"));
  std::vector<int> minbiasThresholds = conf.getParameter<std::vector<int> >("minimumBiasThresholds");
  if(minbiasThresholds.size() == 4) {
    m_params_helper.setMinimumBiasThresholds(minbiasThresholds);
  } else {
    edm::LogError("l1t|calo") << "Incorrect number of minimum bias thresholds set.";
  }

  // HI Q2 trigger
  edm::FileInPath q2LUTFile = conf.getParameter<edm::FileInPath>("q2LUTFile");
  std::ifstream q2LUTStream(q2LUTFile.fullPath());
  auto q2LUT = std::make_shared<LUT>(q2LUTStream);
  m_params_helper.setQ2LUT(*q2LUT);

  // Layer 1 LUT specification
  m_params_helper.setLayer1ECalScaleFactors(conf.getParameter<std::vector<double>>("layer1ECalScaleFactors"));
  m_params_helper.setLayer1HCalScaleFactors(conf.getParameter<std::vector<double>>("layer1HCalScaleFactors"));
  m_params_helper.setLayer1HFScaleFactors  (conf.getParameter<std::vector<double>>("layer1HFScaleFactors"));

  m_params_helper.setLayer1ECalScaleETBins(conf.getParameter<std::vector<int>>("layer1ECalScaleETBins"));
  m_params_helper.setLayer1HCalScaleETBins(conf.getParameter<std::vector<int>>("layer1HCalScaleETBins"));
  m_params_helper.setLayer1HFScaleETBins  (conf.getParameter<std::vector<int>>("layer1HFScaleETBins"));

  m_params_helper.setLayer1ECalScalePhiBins(conf.getParameter<std::vector<unsigned>>("layer1ECalScalePhiBins"));
  m_params_helper.setLayer1HCalScalePhiBins(conf.getParameter<std::vector<unsigned>>("layer1HCalScalePhiBins"));
  m_params_helper.setLayer1HFScalePhiBins  (conf.getParameter<std::vector<unsigned>>("layer1HFScalePhiBins"));

  m_params = (CaloParams)m_params_helper;
}


L1TCaloParamsESProducer::~L1TCaloParamsESProducer()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
L1TCaloParamsESProducer::ReturnType
L1TCaloParamsESProducer::produce(const L1TCaloParamsRcd& iRecord)
{
   using namespace edm::es;
   return std::make_shared<CaloParams>(m_params);
}



//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TCaloParamsESProducer);
