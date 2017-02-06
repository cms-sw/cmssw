#include <iostream>
#include <fstream>
#include <stdexcept>

#include "CondTools/L1TriggerExt/interface/L1ConfigOnlineProdBaseExt.h"
#include "CondFormats/L1TObjects/interface/L1TMuonOverlapParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonOverlapParamsRcd.h"
#include "CondFormats/DataRecord/interface/L1TMuonOverlapParamsO2ORcd.h"

class L1TMuonOverlapParamsOnlineProd : public L1ConfigOnlineProdBaseExt<L1TMuonOverlapParamsO2ORcd,L1TMuonOverlapParams> {
private:
public:
    virtual boost::shared_ptr<L1TMuonOverlapParams> newObject(const std::string& objectKey, const L1TMuonOverlapParamsO2ORcd& record) override ;

    L1TMuonOverlapParamsOnlineProd(const edm::ParameterSet&);
    ~L1TMuonOverlapParamsOnlineProd(void){}
};

L1TMuonOverlapParamsOnlineProd::L1TMuonOverlapParamsOnlineProd(const edm::ParameterSet& iConfig) : L1ConfigOnlineProdBaseExt<L1TMuonOverlapParamsO2ORcd,L1TMuonOverlapParams>(iConfig) {}

boost::shared_ptr<L1TMuonOverlapParams> L1TMuonOverlapParamsOnlineProd::newObject(const std::string& objectKey, const L1TMuonOverlapParamsO2ORcd& record) {

    edm::LogError( "L1-O2O" ) << "L1TMuonOverlapParams object with key " << objectKey << " not in ORCON!" ;

    throw std::runtime_error("You are never supposed to get this code running!");

    return boost::shared_ptr< L1TMuonOverlapParams >( new L1TMuonOverlapParams() );

}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TMuonOverlapParamsOnlineProd);
