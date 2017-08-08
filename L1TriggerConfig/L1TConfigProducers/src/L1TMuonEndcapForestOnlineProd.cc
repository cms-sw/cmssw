#include <iostream>
#include <fstream>
#include <stdexcept>

#include "CondTools/L1TriggerExt/interface/L1ConfigOnlineProdBaseExt.h"
#include "CondFormats/L1TObjects/interface/L1TMuonEndCapForest.h"
#include "CondFormats/DataRecord/interface/L1TMuonEndCapForestRcd.h"
#include "CondFormats/DataRecord/interface/L1TMuonEndcapForestO2ORcd.h"

class L1TMuonEndcapForestOnlineProd : public L1ConfigOnlineProdBaseExt<L1TMuonEndcapForestO2ORcd,L1TMuonEndCapForest> {
private:
public:
    virtual std::shared_ptr<L1TMuonEndCapForest> newObject(const std::string& objectKey, const L1TMuonEndcapForestO2ORcd& record) override ;

    L1TMuonEndcapForestOnlineProd(const edm::ParameterSet&);
    ~L1TMuonEndcapForestOnlineProd(void){}
};

L1TMuonEndcapForestOnlineProd::L1TMuonEndcapForestOnlineProd(const edm::ParameterSet& iConfig) : L1ConfigOnlineProdBaseExt<L1TMuonEndcapForestO2ORcd,L1TMuonEndCapForest>(iConfig) {}

std::shared_ptr<L1TMuonEndCapForest> L1TMuonEndcapForestOnlineProd::newObject(const std::string& objectKey, const L1TMuonEndcapForestO2ORcd& record) {

    edm::LogError( "L1-O2O" ) << "L1TMuonEndCapForest object with key " << objectKey << " not in ORCON!" ;

    throw std::runtime_error("You are never supposed to get this code running!");

    std::shared_ptr< L1TMuonEndCapForest > retval = std::make_shared< L1TMuonEndCapForest >();
    return retval;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TMuonEndcapForestOnlineProd);
