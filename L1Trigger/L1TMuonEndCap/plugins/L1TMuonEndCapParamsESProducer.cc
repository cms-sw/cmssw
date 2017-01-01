#include <iostream>
#include <memory>
#include <iostream>

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducts.h"

#include "CondFormats/L1TObjects/interface/L1TMuonEndCapParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonEndcapParamsRcd.h"
#include "L1Trigger/L1TMuonEndCap/interface/EndCapParamsHelper.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "TXMLEngine.h"

#include "L1Trigger/L1TMuonEndCap/interface/Tree.h"

using namespace std;

// class declaration

class L1TMuonEndCapParamsESProducer : public edm::ESProducer {
public:
  L1TMuonEndCapParamsESProducer(const edm::ParameterSet&);
  ~L1TMuonEndCapParamsESProducer();
  
  typedef std::shared_ptr<L1TMuonEndCapParams> ReturnType;

  ReturnType produce(const L1TMuonEndcapParamsRcd&);
private:
  l1t::EndCapParamsHelper data_;
};

L1TMuonEndCapParamsESProducer::L1TMuonEndCapParamsESProducer(const edm::ParameterSet& iConfig) :
  data_(new L1TMuonEndCapParams())
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced(this);

   data_.SetPtAssignVersion(iConfig.getParameter<int>("PtAssignVersion"));
   data_.SetFirmwareVersion(iConfig.getParameter<int>("firmwareVersion"));
   data_.SetSt1PhiMatchWindow(iConfig.getParameter<int>("St1MatchWindow"));
   data_.SetSt2PhiMatchWindow(iConfig.getParameter<int>("St2MatchWindow"));
   data_.SetSt3PhiMatchWindow(iConfig.getParameter<int>("St3MatchWindow"));
   data_.SetSt4PhiMatchWindow(iConfig.getParameter<int>("St4MatchWindow"));
      
}


L1TMuonEndCapParamsESProducer::~L1TMuonEndCapParamsESProducer()
{
}



//
// member functions
//

// ------------ method called to produce the data  ------------
L1TMuonEndCapParamsESProducer::ReturnType
L1TMuonEndCapParamsESProducer::produce(const L1TMuonEndcapParamsRcd& iRecord)
{
   using namespace edm::es;
   std::shared_ptr<L1TMuonEndCapParams> pEMTFParams(data_.getWriteInstance());
   return pEMTFParams;
   
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TMuonEndCapParamsESProducer);
