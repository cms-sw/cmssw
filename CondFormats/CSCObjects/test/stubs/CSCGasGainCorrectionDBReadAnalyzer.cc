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

#include "CondFormats/CSCObjects/interface/CSCDBGasGainCorrection.h"
#include "CondFormats/DataRecord/interface/CSCDBGasGainCorrectionRcd.h"

namespace edmtest
{
  class CSCGasGainCorrectionDBReadAnalyzer : public edm::EDAnalyzer
  {
  public:
    explicit  CSCGasGainCorrectionDBReadAnalyzer(edm::ParameterSet const& p) 
    { }
    explicit  CSCGasGainCorrectionDBReadAnalyzer(int i) 
    { }
    virtual ~ CSCGasGainCorrectionDBReadAnalyzer() { }
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  private:
  };
  
  void
  CSCGasGainCorrectionDBReadAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context)
  {
    //const float epsilon = 1.E-09; // some 'small' value to test for non-positive values.
    //*    const float epsilon = 20; // some 'small' value to test 

    using namespace edm::eventsetup;
    std::ofstream DBGasGainCorrectionFile("dbGasGainCorrection.dat",std::ios::out);
    int counter=0;

    std::cout <<" I AM IN RUN NUMBER "<<e.id().run() <<std::endl;
    std::cout <<" ---EVENT NUMBER "<<e.id().event() <<std::endl;

    edm::ESHandle<CSCDBGasGainCorrection> pGasGainCorr;
    context.get<CSCDBGasGainCorrectionRcd>().get(pGasGainCorr);
    
    const CSCDBGasGainCorrection* myGasGainCorr=pGasGainCorr.product();
    CSCDBGasGainCorrection::GasGainContainer::const_iterator it;

    for( it=myGasGainCorr->gasGainCorr.begin();it!=myGasGainCorr->gasGainCorr.end(); ++it ){    
      counter++;
      DBGasGainCorrectionFile<<counter<<"  "<<it->gainCorr<<std::endl;
      //* if ( it->gainCorr <= epsilon ) DBGasGainCorrectionFile << " ERROR? Gain Correction <= " << epsilon << std::endl;
    }
  }
  DEFINE_FWK_MODULE(CSCChipSpeedCorrectionDBReadAnalyzer);
}

