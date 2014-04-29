///
/// \class l1t::
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

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/ESProducts.h"

#include "CondFormats/L1TObjects/interface/CaloParams.h"
#include "CondFormats/DataRecord/interface/L1TCaloParamsRcd.h"

using namespace std;

//
// class declaration
//

namespace l1t {

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
  m_params.setTowerEncoding(conf.getParameter<bool>("towerCompression"));

  // regions
  m_params.setRegionPUSType(conf.getParameter<std::string>("regionPUSType"));
  m_params.setRegionPUSParams(conf.getParameter<std::vector<double> >("regionPUSParams"));
    
  // EG
  m_params.setEgSeedThreshold(conf.getParameter<double>("egSeedThreshold"));
  m_params.setEgNeighbourThreshold(conf.getParameter<double>("egNeighbourThreshold"));
  m_params.setEgMaxHcalEt(conf.getParameter<double>("egMaxHcalEt"));
  m_params.setEgMaxHOverE(conf.getParameter<double>("egMaxHOverE"));
  m_params.setEgIsoPUSType(conf.getParameter<std::string>("egIsoPUSType"));
  //  m_params.setEgIsolationLUT(lut);

  // tau
  m_params.setTauSeedThreshold(conf.getParameter<double>("tauSeedThreshold"));
  m_params.setTauNeighbourThreshold(conf.getParameter<double>("tauNeighbourThreshold"));
  m_params.setTauIsoPUSType(conf.getParameter<std::string>("tauIsoPUSType"));
  //  m_params.setTauIsolationLUT(lut);
  
  // jets
  m_params.setJetSeedThreshold(conf.getParameter<double>("jetSeedThreshold"));
  m_params.setJetNeighbourThreshold(conf.getParameter<double>("jetNeighbourThreshold"));
  m_params.setJetPUSType(conf.getParameter<std::string>("jetPUSType"));
  m_params.setJetCalibrationType(conf.getParameter<std::string>("jetCalibrationType"));
  m_params.setJetCalibrationParams(conf.getParameter<std::vector<double> >("jetCalibrationParams"));
  
  // sums
  m_params.setEtSumEtaMin(0, conf.getParameter<double>("ettEtaMin"));
  m_params.setEtSumEtaMax(0, conf.getParameter<double>("ettEtaMax"));
  m_params.setEtSumEtThreshold(0, conf.getParameter<double>("ettEtThreshold"));
  m_params.setEtSumEtaMin(1, conf.getParameter<double>("httEtaMin"));
  m_params.setEtSumEtaMax(1, conf.getParameter<double>("httEtaMax"));
  m_params.setEtSumEtThreshold(1, conf.getParameter<double>("httEtThreshold"));
  m_params.setEtSumEtaMin(2, conf.getParameter<double>("metEtaMin"));
  m_params.setEtSumEtaMax(2, conf.getParameter<double>("metEtaMax"));
  m_params.setEtSumEtThreshold(2, conf.getParameter<double>("metEtThreshold"));
  m_params.setEtSumEtaMin(3, conf.getParameter<double>("mhtEtaMin"));
  m_params.setEtSumEtaMax(3, conf.getParameter<double>("mhtEtaMax"));
  m_params.setEtSumEtThreshold(3, conf.getParameter<double>("mhtEtThreshold"));
  
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

}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(l1t::L1TCaloParamsESProducer);
