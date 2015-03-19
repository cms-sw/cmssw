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
#include "boost/shared_ptr.hpp"
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

  typedef boost::shared_ptr<CaloParams> ReturnType;

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

  // towers
  m_params.setTowerLsbH(conf.getParameter<double>("towerLsbH"));
  m_params.setTowerLsbE(conf.getParameter<double>("towerLsbE"));
  m_params.setTowerLsbSum(conf.getParameter<double>("towerLsbSum"));
  m_params.setTowerNBitsH(conf.getParameter<int>("towerNBitsH"));
  m_params.setTowerNBitsE(conf.getParameter<int>("towerNBitsE"));
  m_params.setTowerNBitsSum(conf.getParameter<int>("towerNBitsSum"));
  m_params.setTowerNBitsRatio(conf.getParameter<int>("towerNBitsRatio"));
  m_params.setTowerEncoding(conf.getParameter<bool>("towerEncoding"));

  // regions
  m_params.setRegionLsb(conf.getParameter<double>("regionLsb"));
  m_params.setRegionPUSType(conf.getParameter<std::string>("regionPUSType"));
  m_params.setRegionPUSParams(conf.getParameter<std::vector<double> >("regionPUSParams"));

  // EG
  m_params.setEgLsb(conf.getParameter<double>("egLsb"));
  m_params.setEgSeedThreshold(conf.getParameter<double>("egSeedThreshold"));
  m_params.setEgNeighbourThreshold(conf.getParameter<double>("egNeighbourThreshold"));
  m_params.setEgHcalThreshold(conf.getParameter<double>("egHcalThreshold"));

  edm::FileInPath egTrimmingLUTFile = conf.getParameter<edm::FileInPath>("egTrimmingLUTFile");
  std::ifstream egTrimmingLUTStream(egTrimmingLUTFile.fullPath());
  std::shared_ptr<LUT> egTrimmingLUT( new LUT(egTrimmingLUTStream) );
  m_params.setEgTrimmingLUT(*egTrimmingLUT);

  m_params.setEgMaxHcalEt(conf.getParameter<double>("egMaxHcalEt"));
  m_params.setEgMaxPtHOverE(conf.getParameter<double>("egMaxPtHOverE"));
  m_params.setEgMinPtJetIsolation(conf.getParameter<int>("egMinPtJetIsolation"));
  m_params.setEgMaxPtJetIsolation(conf.getParameter<int>("egMaxPtJetIsolation"));
  m_params.setEgMinPtHOverEIsolation(conf.getParameter<int>("egMinPtHOverEIsolation"));
  m_params.setEgMaxPtHOverEIsolation(conf.getParameter<int>("egMaxPtHOverEIsolation"));


  edm::FileInPath egMaxHOverELUTFile = conf.getParameter<edm::FileInPath>("egMaxHOverELUTFile");
  std::ifstream egMaxHOverELUTStream(egMaxHOverELUTFile.fullPath());
  std::shared_ptr<LUT> egMaxHOverELUT( new LUT(egMaxHOverELUTStream) );
  m_params.setEgMaxHOverELUT(*egMaxHOverELUT);

  edm::FileInPath egCompressShapesLUTFile = conf.getParameter<edm::FileInPath>("egCompressShapesLUTFile");
  std::ifstream egCompressShapesLUTStream(egCompressShapesLUTFile.fullPath());
  std::shared_ptr<LUT> egCompressShapesLUT( new LUT(egCompressShapesLUTStream) );
  m_params.setEgCompressShapesLUT(*egCompressShapesLUT);

  edm::FileInPath egShapeIdLUTFile = conf.getParameter<edm::FileInPath>("egShapeIdLUTFile");
  std::ifstream egShapeIdLUTStream(egShapeIdLUTFile.fullPath());
  std::shared_ptr<LUT> egShapeIdLUT( new LUT(egShapeIdLUTStream) );
  m_params.setEgShapeIdLUT(*egShapeIdLUT);

  m_params.setEgPUSType(conf.getParameter<std::string>("egPUSType"));

  edm::FileInPath egIsoLUTFile = conf.getParameter<edm::FileInPath>("egIsoLUTFile");
  std::ifstream egIsoLUTStream(egIsoLUTFile.fullPath());
  std::shared_ptr<LUT> egIsoLUT( new LUT(egIsoLUTStream) );
  m_params.setEgIsolationLUT(*egIsoLUT);

  //edm::FileInPath egIsoLUTFileBarrel = conf.getParameter<edm::FileInPath>("egIsoLUTFileBarrel");
  //std::ifstream egIsoLUTBarrelStream(egIsoLUTFileBarrel.fullPath());
  //std::shared_ptr<LUT> egIsoLUTBarrel( new LUT(egIsoLUTBarrelStream) );
  //m_params.setEgIsolationLUTBarrel(egIsoLUTBarrel);

  //edm::FileInPath egIsoLUTFileEndcaps = conf.getParameter<edm::FileInPath>("egIsoLUTFileEndcaps");
  //std::ifstream egIsoLUTEndcapsStream(egIsoLUTFileEndcaps.fullPath());
  //std::shared_ptr<LUT> egIsoLUTEndcaps( new LUT(egIsoLUTEndcapsStream) );
  //m_params.setEgIsolationLUTEndcaps(egIsoLUTEndcaps);


  m_params.setEgIsoAreaNrTowersEta(conf.getParameter<unsigned int>("egIsoAreaNrTowersEta"));
  m_params.setEgIsoAreaNrTowersPhi(conf.getParameter<unsigned int>("egIsoAreaNrTowersPhi"));
  m_params.setEgIsoVetoNrTowersPhi(conf.getParameter<unsigned int>("egIsoVetoNrTowersPhi"));
  //m_params.setEgIsoPUEstTowerGranularity(conf.getParameter<unsigned int>("egIsoPUEstTowerGranularity"));
  //m_params.setEgIsoMaxEtaAbsForTowerSum(conf.getParameter<unsigned int>("egIsoMaxEtaAbsForTowerSum"));
  //m_params.setEgIsoMaxEtaAbsForIsoSum(conf.getParameter<unsigned int>("egIsoMaxEtaAbsForIsoSum"));
  m_params.setEgPUSParams(conf.getParameter<std::vector<double>>("egPUSParams"));

  edm::FileInPath egCalibrationLUTFile = conf.getParameter<edm::FileInPath>("egCalibrationLUTFile");
  std::ifstream egCalibrationLUTStream(egCalibrationLUTFile.fullPath());
  std::shared_ptr<LUT> egCalibrationLUT( new LUT(egCalibrationLUTStream) );
  m_params.setEgCalibrationLUT(*egCalibrationLUT);

  // tau
  m_params.setTauLsb(conf.getParameter<double>("tauLsb"));
  m_params.setTauSeedThreshold(conf.getParameter<double>("tauSeedThreshold"));
  m_params.setTauNeighbourThreshold(conf.getParameter<double>("tauNeighbourThreshold"));
  m_params.setTauMaxPtTauVeto(conf.getParameter<double>("tauMaxPtTauVeto"));
  m_params.setTauMinPtJetIsolationB(conf.getParameter<double>("tauMinPtJetIsolationB"));
  m_params.setTauPUSType(conf.getParameter<std::string>("tauPUSType"));
  m_params.setTauMaxJetIsolationB(conf.getParameter<double>("tauMaxJetIsolationB"));
  m_params.setTauMaxJetIsolationA(conf.getParameter<double>("tauMaxJetIsolationA"));
  m_params.setTauIsoAreaNrTowersEta(conf.getParameter<unsigned int>("tauIsoAreaNrTowersEta"));
  m_params.setTauIsoAreaNrTowersPhi(conf.getParameter<unsigned int>("tauIsoAreaNrTowersPhi"));
  m_params.setTauIsoVetoNrTowersPhi(conf.getParameter<unsigned int>("tauIsoVetoNrTowersPhi"));

  edm::FileInPath tauIsoLUTFile = conf.getParameter<edm::FileInPath>("tauIsoLUTFile");
  std::ifstream tauIsoLUTStream(tauIsoLUTFile.fullPath());
  std::shared_ptr<LUT> tauIsoLUT( new LUT(tauIsoLUTStream) );
  m_params.setTauIsolationLUT(*tauIsoLUT);

  edm::FileInPath tauCalibrationLUTFile = conf.getParameter<edm::FileInPath>("tauCalibrationLUTFile");
  std::ifstream tauCalibrationLUTStream(tauCalibrationLUTFile.fullPath());
  std::shared_ptr<LUT> tauCalibrationLUT( new LUT(tauCalibrationLUTStream) );
  m_params.setTauCalibrationLUT(*tauCalibrationLUT);

  edm::FileInPath tauEtToHFRingEtLUTFile = conf.getParameter<edm::FileInPath>("tauEtToHFRingEtLUTFile");
  std::ifstream tauEtToHFRingEtLUTStream(tauEtToHFRingEtLUTFile.fullPath());
  std::shared_ptr<LUT> tauEtToHFRingEtLUT( new LUT(tauEtToHFRingEtLUTStream) );
  m_params.setTauEtToHFRingEtLUT(*tauEtToHFRingEtLUT);

  m_params.setIsoTauEtaMin(conf.getParameter<int> ("isoTauEtaMin"));
  m_params.setIsoTauEtaMax(conf.getParameter<int> ("isoTauEtaMax"));

  m_params.setTauPUSParams(conf.getParameter<std::vector<double>>("tauPUSParams"));

  // jets
  m_params.setJetLsb(conf.getParameter<double>("jetLsb"));
  m_params.setJetSeedThreshold(conf.getParameter<double>("jetSeedThreshold"));
  m_params.setJetNeighbourThreshold(conf.getParameter<double>("jetNeighbourThreshold"));
  m_params.setJetPUSType(conf.getParameter<std::string>("jetPUSType"));
  m_params.setJetCalibrationType(conf.getParameter<std::string>("jetCalibrationType"));
  m_params.setJetCalibrationParams(conf.getParameter<std::vector<double> >("jetCalibrationParams"));
  edm::FileInPath jetCalibrationLUTFile = conf.getParameter<edm::FileInPath>("jetCalibrationLUTFile");
  std::ifstream jetCalibrationLUTStream(jetCalibrationLUTFile.fullPath());
  std::shared_ptr<LUT> jetCalibrationLUT( new LUT(jetCalibrationLUTStream) );
  m_params.setJetCalibrationLUT(*jetCalibrationLUT);

  // sums
  m_params.setEtSumLsb(conf.getParameter<double>("etSumLsb"));

  std::vector<int> etSumEtaMin = conf.getParameter<std::vector<int> >("etSumEtaMin");
  std::vector<int> etSumEtaMax = conf.getParameter<std::vector<int> >("etSumEtaMax");
  std::vector<double> etSumEtThreshold = conf.getParameter<std::vector<double> >("etSumEtThreshold");

  if ((etSumEtaMin.size() == etSumEtaMax.size()) &&  (etSumEtaMin.size() == etSumEtThreshold.size())) {
    for (unsigned i=0; i<etSumEtaMin.size(); ++i) {
      m_params.setEtSumEtaMin(i, etSumEtaMin.at(i));
      m_params.setEtSumEtaMax(i, etSumEtaMax.at(i));
      m_params.setEtSumEtThreshold(i, etSumEtThreshold.at(i));
    }
  }
  else {
    edm::LogError("l1t|calo") << "Inconsistent number of EtSum parameters" << std::endl;
  }

  // HI centrality trigger
  edm::FileInPath centralityLUTFile = conf.getParameter<edm::FileInPath>("centralityLUTFile");
  std::ifstream centralityLUTStream(centralityLUTFile.fullPath());
  std::shared_ptr<LUT> centralityLUT( new LUT(centralityLUTStream) );
  m_params.setCentralityLUT(*centralityLUT);

  // HI Q2 trigger
  edm::FileInPath q2LUTFile = conf.getParameter<edm::FileInPath>("q2LUTFile");
  std::ifstream q2LUTStream(q2LUTFile.fullPath());
  std::shared_ptr<LUT> q2LUT( new LUT(q2LUTStream) );
  m_params.setQ2LUT(*q2LUT);


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
   boost::shared_ptr<CaloParams> pCaloParams ;

   pCaloParams = boost::shared_ptr< CaloParams >(new CaloParams(m_params));
   return pCaloParams;
}



//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TCaloParamsESProducer);
