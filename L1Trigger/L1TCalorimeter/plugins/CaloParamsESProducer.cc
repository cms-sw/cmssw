///
/// \class l1t::CaloParamsESProducer
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

#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "CondFormats/L1TObjects/interface/CaloParams.h"
#include "CondFormats/DataRecord/interface/L1TCaloParamsRcd.h"

using namespace std;

//
// class declaration
//

namespace l1t {

class CaloParamsESProducer : public edm::ESProducer {
public:
  CaloParamsESProducer(const edm::ParameterSet&);
  ~CaloParamsESProducer();
  
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
CaloParamsESProducer::CaloParamsESProducer(const edm::ParameterSet& conf)
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
  m_params.setEgMaxHcalEt(conf.getParameter<double>("egMaxHcalEt"));
  m_params.setEgEtToRemoveHECut(conf.getParameter<double>("egEtToRemoveHECut"));
  m_params.setEgMaxHOverE(conf.getParameter<double>("egMaxHOverE"));  
  m_params.setEgIsoPUSType(conf.getParameter<std::string>("egIsoPUSType"));
  
  edm::FileInPath egIsoLUTFile = conf.getParameter<edm::FileInPath>("egIsoLUTFile");
  std::ifstream egIsoLUTStream(egIsoLUTFile.fullPath());
  std::shared_ptr<l1t::LUT> egIsoLUT( new l1t::LUT(egIsoLUTStream) );
  m_params.setEgIsolationLUT(egIsoLUT);

  m_params.setEgIsoAreaNrTowersEta(conf.getParameter<unsigned int>("egIsoAreaNrTowersEta"));
  m_params.setEgIsoAreaNrTowersPhi(conf.getParameter<unsigned int>("egIsoAreaNrTowersPhi"));
  m_params.setEgIsoVetoNrTowersPhi(conf.getParameter<unsigned int>("egIsoVetoNrTowersPhi"));
  m_params.setEgIsoPUEstTowerGranularity(conf.getParameter<unsigned int>("egIsoPUEstTowerGranularity"));
  m_params.setEgIsoMaxEtaAbsForTowerSum(conf.getParameter<unsigned int>("egIsoMaxEtaAbsForTowerSum"));
  m_params.setEgIsoMaxEtaAbsForIsoSum(conf.getParameter<unsigned int>("egIsoMaxEtaAbsForIsoSum"));
  
  // tau
  m_params.setTauLsb(conf.getParameter<double>("tauLsb"));
  m_params.setTauSeedThreshold(conf.getParameter<double>("tauSeedThreshold"));
  m_params.setTauNeighbourThreshold(conf.getParameter<double>("tauNeighbourThreshold"));
  m_params.setTauIsoPUSType(conf.getParameter<std::string>("tauIsoPUSType"));

  edm::FileInPath tauIsoLUTFile = conf.getParameter<edm::FileInPath>("tauIsoLUTFile");
  std::ifstream tauIsoLUTStream(tauIsoLUTFile.fullPath());
  std::shared_ptr<l1t::LUT> tauIsoLUT( new l1t::LUT(tauIsoLUTStream) );
  m_params.setTauIsolationLUT(tauIsoLUT);

  // jets
  m_params.setJetLsb(conf.getParameter<double>("jetLsb"));
  m_params.setJetSeedThreshold(conf.getParameter<double>("jetSeedThreshold"));
  m_params.setJetNeighbourThreshold(conf.getParameter<double>("jetNeighbourThreshold"));
  m_params.setJetPUSType(conf.getParameter<std::string>("jetPUSType"));
  m_params.setJetCalibrationType(conf.getParameter<std::string>("jetCalibrationType"));
  m_params.setJetCalibrationParams(conf.getParameter<std::vector<double> >("jetCalibrationParams"));
  
  // sums
  m_params.setEtSumLsb(conf.getParameter<double>("etSumLsb"));
  m_params.setEtSumEtaMin(0, conf.getParameter<int>("ettEtaMin"));
  m_params.setEtSumEtaMax(0, conf.getParameter<int>("ettEtaMax"));
  m_params.setEtSumEtThreshold(0, conf.getParameter<double>("ettEtThreshold"));
  m_params.setEtSumEtaMin(1, conf.getParameter<int>("httEtaMin"));
  m_params.setEtSumEtaMax(1, conf.getParameter<int>("httEtaMax"));
  m_params.setEtSumEtThreshold(1, conf.getParameter<double>("httEtThreshold"));
  m_params.setEtSumEtaMin(2, conf.getParameter<int>("metEtaMin"));
  m_params.setEtSumEtaMax(2, conf.getParameter<int>("metEtaMax"));
  m_params.setEtSumEtThreshold(2, conf.getParameter<double>("metEtThreshold"));
  m_params.setEtSumEtaMin(3, conf.getParameter<int>("mhtEtaMin"));
  m_params.setEtSumEtaMax(3, conf.getParameter<int>("mhtEtaMax"));
  m_params.setEtSumEtThreshold(3, conf.getParameter<double>("mhtEtThreshold"));
  
}


CaloParamsESProducer::~CaloParamsESProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
CaloParamsESProducer::ReturnType
CaloParamsESProducer::produce(const L1TCaloParamsRcd& iRecord)
{
   using namespace edm::es;
   boost::shared_ptr<CaloParams> pCaloParams ;

   pCaloParams = boost::shared_ptr< CaloParams >(new CaloParams(m_params));
   return pCaloParams;
}

}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(l1t::CaloParamsESProducer);
