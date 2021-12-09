#include <stdexcept>
#include <string>
#include <iostream>
#include <map>

#include "CondFormats/Calibration/interface/mySiStripNoises.h"
#include "CondFormats/DataRecord/interface/mySiStripNoisesRcd.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

using namespace std;

namespace edmtest {
  class NoisesAnalyzer : public edm::EDAnalyzer {
  public:
    explicit NoisesAnalyzer(edm::ParameterSet const& p) : theNoisesToken_(esConsumes()) {
      std::cout << "NoisesAnalyzer" << std::endl;
    }
    explicit NoisesAnalyzer(int i) { std::cout << "NoisesAnalyzer " << i << std::endl; }
    virtual ~NoisesAnalyzer() { std::cout << "~NoisesAnalyzer " << std::endl; }
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);

  private:
    const edm::ESGetToken<mySiStripNoises, mySiStripNoisesRcd> theNoisesToken_;
  };

  void NoisesAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context) {
    using namespace edm::eventsetup;
    // Context is not used.
    std::cout << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
    std::cout << " ---EVENT NUMBER " << e.id().event() << std::endl;
    edm::eventsetup::EventSetupRecordKey recordKey(
        edm::eventsetup::EventSetupRecordKey::TypeTag::findType("mySiStripNoisesRcd"));
    if (recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
      //record not found
      std::cout << "Record \"mySiStripNoisesRcd"
                << "\" does not exist " << std::endl;
    }
    std::cout << "got context" << std::endl;
    auto const& mynoise = &context.getData(theNoisesToken_);
    std::cout << "Noises* " << mynoise << std::endl;
    unsigned int a = mynoise->v_noises.size();
    std::cout << "size a " << a << std::endl;
    unsigned int b = mynoise->indexes.size();
    std::cout << "size b " << b << std::endl;
    /*for(std::vector<mySiStripNoises::DetRegistry>::const_iterator it=mynoise->indexes.begin(); it!=mynoise->indexes.end(); ++it){
      std::cout << "  detid  " <<it->detid<< std::endl;
      }*/
  }
  DEFINE_FWK_MODULE(NoisesAnalyzer);
}  // namespace edmtest
