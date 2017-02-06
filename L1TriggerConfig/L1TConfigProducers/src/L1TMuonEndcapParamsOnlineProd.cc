#include <iostream>
#include <fstream>

#include "CondTools/L1TriggerExt/interface/L1ConfigOnlineProdBaseExt.h"
#include "CondFormats/L1TObjects/interface/L1TMuonEndCapParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonEndcapParamsRcd.h"
#include "CondFormats/DataRecord/interface/L1TMuonEndcapParamsO2ORcd.h"

class L1TMuonEndcapParamsOnlineProd : public L1ConfigOnlineProdBaseExt<L1TMuonEndcapParamsO2ORcd,L1TMuonEndCapParams> {
private:
public:
    virtual boost::shared_ptr<L1TMuonEndCapParams> newObject(const std::string& objectKey, const L1TMuonEndcapParamsO2ORcd& record) override ;

    L1TMuonEndcapParamsOnlineProd(const edm::ParameterSet&);
    ~L1TMuonEndcapParamsOnlineProd(void){}
};

L1TMuonEndcapParamsOnlineProd::L1TMuonEndcapParamsOnlineProd(const edm::ParameterSet& iConfig) : L1ConfigOnlineProdBaseExt<L1TMuonEndcapParamsO2ORcd,L1TMuonEndCapParams>(iConfig) {}

boost::shared_ptr<L1TMuonEndCapParams> L1TMuonEndcapParamsOnlineProd::newObject(const std::string& objectKey, const L1TMuonEndcapParamsO2ORcd& record) {
    edm::LogError( "L1-O2O" ) << "L1TMuonEndcapParams object with key " << objectKey << " not in ORCON!" ;
    return boost::shared_ptr< L1TMuonEndCapParams >( new L1TMuonEndCapParams() );
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TMuonEndcapParamsOnlineProd);
