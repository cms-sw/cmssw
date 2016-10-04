#include <iomanip>
#include <iostream>

#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DataRecord/interface/L1TMuonGlobalParamsRcd.h"
#include "CondFormats/L1TObjects/interface/L1TMuonGlobalParams.h"
#include "L1Trigger/L1TMuon/interface/L1TMuonGlobalParamsHelper.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondCore/CondDB/interface/Session.h"

#include <iostream>
using namespace std;

class L1TMuonGlobalParamsViewer: public edm::EDAnalyzer {
private:
//    bool printLayerMap;
public:
    virtual void analyze(const edm::Event&, const edm::EventSetup&);
//    string hash(void *buf, size_t len) const ;

    explicit L1TMuonGlobalParamsViewer(const edm::ParameterSet& pset) : edm::EDAnalyzer(){
//       printLayerMap   = pset.getUntrackedParameter<bool>("printLayerMap",  false);
    }
    virtual ~L1TMuonGlobalParamsViewer(void){}
};

/*
#include <openssl/sha.h>
#include <math.h>
#include <iostream>
using namespace std;

string L1TMuonGlobalParamsViewer::hash(void *buf, size_t len) const {
    char tmp[SHA_DIGEST_LENGTH*2+1];
    bzero(tmp,sizeof(tmp));
    SHA_CTX ctx;
    if( !SHA1_Init( &ctx ) )
        throw cms::Exception("L1TCaloParamsReader::hash")<<"SHA1 initialization error";

    if( !SHA1_Update( &ctx, buf, len ) )
        throw cms::Exception("L1TCaloParamsReader::hash")<<"SHA1 processing error";

    unsigned char hash[SHA_DIGEST_LENGTH];
    if( !SHA1_Final(hash, &ctx) )
        throw cms::Exception("L1TCaloParamsReader::hash")<<"SHA1 finalization error";

    // re-write bytes in hex
    for(unsigned int i=0; i<20; i++)
        ::sprintf(&tmp[i*2], "%02x", hash[i]);

    tmp[20*2] = 0;
    return string(tmp);
}
*/
void L1TMuonGlobalParamsViewer::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup){

    // Pull the config from the ES
    edm::ESHandle<L1TMuonGlobalParams> handle1;
    evSetup.get<L1TMuonGlobalParamsRcd>().get( handle1 ) ;
    boost::shared_ptr<L1TMuonGlobalParams> ptr1(new L1TMuonGlobalParams(*(handle1.product ())));

    cout<<"Some fields in L1TMuonGlobalParams: "<<endl;

    ((L1TMuonGlobalParamsHelper*)ptr1.get())->print(cout);

}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

DEFINE_FWK_MODULE(L1TMuonGlobalParamsViewer);

