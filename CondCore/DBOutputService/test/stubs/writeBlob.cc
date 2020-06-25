#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/Calibration/interface/mySiStripNoises.h"

#include <random>

#include "writeBlob.h"

typedef std::minstd_rand base_generator_type;
writeBlob::writeBlob(const edm::ParameterSet& iConfig) : m_StripRecordName("mySiStripNoisesRcd") {}

writeBlob::~writeBlob() { std::cout << "writeBlob::writeBlob" << std::endl; }

void writeBlob::analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) {
  std::cout << "writeBlob::analyze " << std::endl;
  base_generator_type rng(42u);
  std::uniform_real_distribution<> uni_dist(0.0, 1.0);
  auto uni = [&]() { return uni_dist(rng); };

  edm::Service<cond::service::PoolDBOutputService> mydbservice;

  if (!mydbservice.isAvailable()) {
    std::cout << "db service unavailable" << std::endl;
    return;
  }
  try {
    mySiStripNoises* me = new mySiStripNoises;
    unsigned int detidseed = 1234;
    unsigned int bsize = 100;
    unsigned int nAPV = 2;
    for (uint32_t detid = detidseed; detid < (detidseed + bsize); detid++) {
      //Generate Noise for det detid
      std::vector<short> theSiStripVector;
      for (unsigned int strip = 0; strip < 128 * nAPV; ++strip) {
        float noise = uni();
        ;
        me->setData(noise, theSiStripVector);
      }
      me->put(detid, theSiStripVector);
    }

    mydbservice->writeOne(me, mydbservice->currentTime(), m_StripRecordName);
  } catch (const cond::Exception& er) {
    throw cms::Exception("DBOutputServiceUnitTestFailure", "failed writeBlob", er);
    //std::cout<<er.what()<<std::endl;
  } catch (const cms::Exception& er) {
    throw cms::Exception("DBOutputServiceUnitTestFailure", "failed writeBlob", er);
  } /*catch(const std::exception& er){
    std::cout<<"caught std::exception "<<er.what()<<std::endl;
  }catch(...){
    std::cout<<"Funny error"<<std::endl;
    }*/
}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(writeBlob);
