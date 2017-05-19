///
/// \class L1TCaloStage2ParamsESProducer
///
/// Description: Produces configuration parameters for stage 2 trigger
///
/// Implementation:
///    L1TCaloParamsESProducer is for stage 1
///
/// \author: L1 Offline Software
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
#include "CondFormats/DataRecord/interface/L1TCaloStage2ParamsRcd.h"

using namespace std;

//
// class declaration
//

using namespace l1t;

class L1TCaloStage2ParamsESProducer : public edm::ESProducer {
public:
  L1TCaloStage2ParamsESProducer(const edm::ParameterSet&);
  ~L1TCaloStage2ParamsESProducer();

  typedef std::shared_ptr<CaloParams> ReturnType;

  ReturnType produce(const L1TCaloStage2ParamsRcd&);

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
L1TCaloStage2ParamsESProducer::L1TCaloStage2ParamsESProducer(const edm::ParameterSet& conf)
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
  std::shared_ptr<LUT> egTrimmingLUT( new LUT(egTrimmingLUTStream) );
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
  std::shared_ptr<LUT> egMaxHOverELUT( new LUT(egMaxHOverELUTStream) );
  m_params_helper.setEgMaxHOverELUT(*egMaxHOverELUT);

  edm::FileInPath egCompressShapesLUTFile = conf.getParameter<edm::FileInPath>("egCompressShapesLUTFile");
  std::ifstream egCompressShapesLUTStream(egCompressShapesLUTFile.fullPath());
  std::shared_ptr<LUT> egCompressShapesLUT( new LUT(egCompressShapesLUTStream) );
  m_params_helper.setEgCompressShapesLUT(*egCompressShapesLUT);

  m_params_helper.setEgShapeIdType(conf.getParameter<std::string>("egShapeIdType"));
  m_params_helper.setEgShapeIdVersion(conf.getParameter<unsigned>("egShapeIdVersion"));
  edm::FileInPath egShapeIdLUTFile = conf.getParameter<edm::FileInPath>("egShapeIdLUTFile");
  std::ifstream egShapeIdLUTStream(egShapeIdLUTFile.fullPath());
  std::shared_ptr<LUT> egShapeIdLUT( new LUT(egShapeIdLUTStream) );
  m_params_helper.setEgShapeIdLUT(*egShapeIdLUT);

  m_params_helper.setEgPUSType(conf.getParameter<std::string>("egPUSType"));

  m_params_helper.setEgIsolationType(conf.getParameter<std::string>("egIsolationType"));
  edm::FileInPath egIsoLUTFile = conf.getParameter<edm::FileInPath>("egIsoLUTFile");
  std::ifstream egIsoLUTStream(egIsoLUTFile.fullPath());
  std::shared_ptr<LUT> egIsoLUT( new LUT(egIsoLUTStream) );
  m_params_helper.setEgIsolationLUT(*egIsoLUT);

  //edm::FileInPath egIsoLUTFileBarrel = conf.getParameter<edm::FileInPath>("egIsoLUTFileBarrel");
  //std::ifstream egIsoLUTBarrelStream(egIsoLUTFileBarrel.fullPath());
  //std::shared_ptr<LUT> egIsoLUTBarrel( new LUT(egIsoLUTBarrelStream) );
  //m_params_helper.setEgIsolationLUTBarrel(egIsoLUTBarrel);

  //edm::FileInPath egIsoLUTFileEndcaps = conf.getParameter<edm::FileInPath>("egIsoLUTFileEndcaps");
  //std::ifstream egIsoLUTEndcapsStream(egIsoLUTFileEndcaps.fullPath());
  //std::shared_ptr<LUT> egIsoLUTEndcaps( new LUT(egIsoLUTEndcapsStream) );
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
  std::shared_ptr<LUT> egCalibrationLUT( new LUT(egCalibrationLUTStream) );
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
  std::shared_ptr<LUT> tauIsoLUT( new LUT(tauIsoLUTStream) );
  m_params_helper.setTauIsolationLUT(*tauIsoLUT);

  edm::FileInPath tauIsoLUTFile2 = conf.getParameter<edm::FileInPath>("tauIsoLUTFile2");
  std::ifstream tauIsoLUTStream2(tauIsoLUTFile2.fullPath());
  std::shared_ptr<LUT> tauIsoLUT2( new LUT(tauIsoLUTStream2) );
  m_params_helper.setTauIsolationLUT2(*tauIsoLUT2);

  edm::FileInPath tauCalibrationLUTFile = conf.getParameter<edm::FileInPath>("tauCalibrationLUTFile");
  std::ifstream tauCalibrationLUTStream(tauCalibrationLUTFile.fullPath());
  std::shared_ptr<LUT> tauCalibrationLUT( new LUT(tauCalibrationLUTStream) );
  m_params_helper.setTauCalibrationLUT(*tauCalibrationLUT);

  edm::FileInPath tauCompressLUTFile = conf.getParameter<edm::FileInPath>("tauCompressLUTFile");
  std::ifstream tauCompressLUTStream(tauCompressLUTFile.fullPath());
  std::shared_ptr<LUT> tauCompressLUT( new LUT(tauCompressLUTStream) );
  m_params_helper.setTauCompressLUT(*tauCompressLUT);

  edm::FileInPath tauEtToHFRingEtLUTFile = conf.getParameter<edm::FileInPath>("tauEtToHFRingEtLUTFile");
  std::ifstream tauEtToHFRingEtLUTStream(tauEtToHFRingEtLUTFile.fullPath());
  std::shared_ptr<LUT> tauEtToHFRingEtLUT( new LUT(tauEtToHFRingEtLUTStream) );
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
  std::shared_ptr<LUT> jetCalibrationLUT( new LUT(jetCalibrationLUTStream) );
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

  edm::FileInPath etSumXPUSLUTFile = conf.getParameter<edm::FileInPath>("etSumXPUSLUTFile");
  std::ifstream etSumXPUSLUTStream(etSumXPUSLUTFile.fullPath());
  std::shared_ptr<LUT> etSumXPUSLUT( new LUT(etSumXPUSLUTStream) );
  m_params_helper.setEtSumXPUSLUT(*etSumXPUSLUT);
  
  edm::FileInPath etSumYPUSLUTFile = conf.getParameter<edm::FileInPath>("etSumYPUSLUTFile");
  std::ifstream etSumYPUSLUTStream(etSumYPUSLUTFile.fullPath());
  std::shared_ptr<LUT> etSumYPUSLUT( new LUT(etSumYPUSLUTStream) );
  m_params_helper.setEtSumYPUSLUT(*etSumYPUSLUT);

  edm::FileInPath etSumEttPUSLUTFile = conf.getParameter<edm::FileInPath>("etSumEttPUSLUTFile");
  std::ifstream etSumEttPUSLUTStream(etSumEttPUSLUTFile.fullPath());
  std::shared_ptr<LUT> etSumEttPUSLUT( new LUT(etSumEttPUSLUTStream) );
  m_params_helper.setEtSumEttPUSLUT(*etSumEttPUSLUT);

  edm::FileInPath etSumEcalSumPUSLUTFile = conf.getParameter<edm::FileInPath>("etSumEcalSumPUSLUTFile");
  std::ifstream etSumEcalSumPUSLUTStream(etSumEcalSumPUSLUTFile.fullPath());
  std::shared_ptr<LUT> etSumEcalSumPUSLUT( new LUT(etSumEcalSumPUSLUTStream) );
  m_params_helper.setEtSumEcalSumPUSLUT(*etSumEcalSumPUSLUT);

  // HI centrality trigger
  edm::FileInPath centralityLUTFile = conf.getParameter<edm::FileInPath>("centralityLUTFile");
  std::ifstream centralityLUTStream(centralityLUTFile.fullPath());
  std::shared_ptr<LUT> centralityLUT( new LUT(centralityLUTStream) );
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
  std::shared_ptr<LUT> q2LUT( new LUT(q2LUTStream) );
  m_params_helper.setQ2LUT(*q2LUT);

  // Layer 1 LUT specification
  m_params_helper.setLayer1ECalScaleFactors(conf.getParameter<std::vector<double>>("layer1ECalScaleFactors"));
  m_params_helper.setLayer1HCalScaleFactors(conf.getParameter<std::vector<double>>("layer1HCalScaleFactors"));
  m_params_helper.setLayer1HFScaleFactors  (conf.getParameter<std::vector<double>>("layer1HFScaleFactors"));

  m_params_helper.setLayer1ECalScaleETBins(conf.getParameter<std::vector<int>>("layer1ECalScaleETBins"));
  m_params_helper.setLayer1HCalScaleETBins(conf.getParameter<std::vector<int>>("layer1HCalScaleETBins"));
  m_params_helper.setLayer1HFScaleETBins  (conf.getParameter<std::vector<int>>("layer1HFScaleETBins"));

  m_params = (CaloParams)m_params_helper;
}


L1TCaloStage2ParamsESProducer::~L1TCaloStage2ParamsESProducer()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
L1TCaloStage2ParamsESProducer::ReturnType
L1TCaloStage2ParamsESProducer::produce(const L1TCaloStage2ParamsRcd& iRecord)
{
   using namespace edm::es;
   std::shared_ptr<CaloParams> pCaloParams ;

   pCaloParams = std::make_shared< CaloParams >(m_params);
   return pCaloParams;
}



//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TCaloStage2ParamsESProducer);
