#include <stdexcept>
#include <string>
#include <iostream>
#include <map>

#include "CondFormats/Calibration/interface/mySiStripNoises.h"
#include "CondFormats/DataRecord/interface/mySiStripNoisesRcd.h"

#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

using namespace std;

namespace edmtest {
  class NoisesAnalyzer : public edm::global::EDAnalyzer<> {
  public:
    explicit NoisesAnalyzer(edm::ParameterSet const& p) : theNoisesToken_(esConsumes()) {
      edm::LogPrint("NoisesAnalyzer") << "NoisesAnalyzer";
    }
    explicit NoisesAnalyzer(int i) { edm::LogPrint("NoisesAnalyzer") << "NoisesAnalyzer " << i; }
    virtual ~NoisesAnalyzer() { edm::LogPrint("NoisesAnalyzer") << "~NoisesAnalyzer "; }
    void analyze(edm::StreamID, const edm::Event& e, const edm::EventSetup& c) const override;

  private:
    const edm::ESGetToken<mySiStripNoises, mySiStripNoisesRcd> theNoisesToken_;
  };

  void NoisesAnalyzer::analyze(edm::StreamID, const edm::Event& e, const edm::EventSetup& context) const {
    using namespace edm::eventsetup;
    // Context is not used.
    edm::LogPrint("NoisesAnalyzer") << " I AM IN RUN NUMBER " << e.id().run();
    edm::LogPrint("NoisesAnalyzer") << " ---EVENT NUMBER " << e.id().event();
    edm::eventsetup::EventSetupRecordKey recordKey(
        edm::eventsetup::EventSetupRecordKey::TypeTag::findType("mySiStripNoisesRcd"));
    if (recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
      //record not found
      edm::LogPrint("NoisesAnalyzer") << "Record \"mySiStripNoisesRcd\" does not exist";
    }
    edm::LogPrint("NoisesAnalyzer") << "got context";
    auto const& mynoise = &context.getData(theNoisesToken_);
    edm::LogPrint("NoisesAnalyzer") << "Noises* " << mynoise;
    unsigned int a = mynoise->v_noises.size();
    edm::LogPrint("NoisesAnalyzer") << "size a " << a;
    unsigned int b = mynoise->indexes.size();
    edm::LogPrint("NoisesAnalyzer") << "size b " << b;
    /*for(std::vector<mySiStripNoises::DetRegistry>::const_iterator it=mynoise->indexes.begin(); it!=mynoise->indexes.end(); ++it){
      std::cout << "  detid  " <<it->detid<< std::endl;
      }*/
  }
  DEFINE_FWK_MODULE(NoisesAnalyzer);
}  // namespace edmtest
