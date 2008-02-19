/*----------------------------------------------------------------------

Toy EDProducers and EDProducts for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>
#include <iostream>
#include <fstream>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/CSCObjects/interface/CSCDBCrosstalk.h"
#include "CondFormats/DataRecord/interface/CSCDBCrosstalkRcd.h"

using namespace std;

namespace edmtest
{
  class CSCCrossTalkDBReadAnalyzer : public edm::EDAnalyzer
  {
  public:
    explicit  CSCCrossTalkDBReadAnalyzer(edm::ParameterSet const& p) 
    { }
    explicit  CSCCrossTalkDBReadAnalyzer(int i) 
    { }
    virtual ~ CSCCrossTalkDBReadAnalyzer() { }
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  private:
  };
  
  void
   CSCCrossTalkDBReadAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context)
  {
    std::ofstream DBXtalkFile("dbxtalk.dat",std::ios::out);
    int counter=0;
    using namespace edm::eventsetup;
    std::cout <<" I AM IN RUN NUMBER "<<e.id().run() <<std::endl;
    std::cout <<" ---EVENT NUMBER "<<e.id().event() <<std::endl;
    edm::ESHandle<CSCDBCrosstalk> pcrosstalk;
    context.get<CSCDBCrosstalkRcd>().get(pcrosstalk);

    const CSCDBCrosstalk* mycrosstalk=pcrosstalk.product();
    int i;
    for( i=0; i<CSCDBCrosstalk::ArraySize; ++i ){
      counter++;
      DBXtalkFile<<counter<<"  "<<mycrosstalk->crosstalk[i].xtalk_slope_right<<"  "<<mycrosstalk->crosstalk[i].xtalk_intercept_right<<"  "<<mycrosstalk->crosstalk[i].xtalk_slope_left<<"  "<<mycrosstalk->crosstalk[i].xtalk_intercept_left<<std::endl;
    }
  }
  DEFINE_FWK_MODULE(CSCCrossTalkDBReadAnalyzer);
}
