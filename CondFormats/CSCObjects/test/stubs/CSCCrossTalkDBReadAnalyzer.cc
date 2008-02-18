/*----------------------------------------------------------------------

Toy EDProducers and EDProducts for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>
#include <iostream>
#include <fstream>
#include <map>

#include <vector>
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
    using namespace edm::eventsetup;
    // Context is not used.
    std::cout <<" I AM IN RUN NUMBER "<<e.id().run() <<std::endl;
    std::cout <<" ---EVENT NUMBER "<<e.id().event() <<std::endl;
    edm::ESHandle<CSCDBCrosstalk> pcrosstalk;
    context.get<CSCDBCrosstalkRcd>().get(pcrosstalk);

    const CSCDBCrosstalk* mycrosstalk=pcrosstalk.product();
    std::vector<CSCDBCrosstalk::Item>::const_iterator it;
    for( it=mycrosstalk->crosstalk.begin();it!=mycrosstalk->crosstalk.end(); ++it ){
      //no global variables!
      //counter++;
      // DBXtalkFile<<counter<<"  "<<it->xtalk_slope_right<<"  "<<it->xtalk_intercept_right<<"  "<<it->xtalk_chi2_right<<"  "<<it->xtalk_slope_left<<"  "<<it->xtalk_intercept_left<<"  "<<it->xtalk_chi2_left<<std::endl;
    }
  }
  DEFINE_FWK_MODULE(CSCCrossTalkDBReadAnalyzer);
}

