
/*----------------------------------------------------------------------

Sh. Rahatlou, University of Rome & INFN
simple analyzer to dump information about ECAL cond objects

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>
#include <iostream>
#include <map>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalWeightRecAlgoWeights.h"
#include "CondFormats/DataRecord/interface/EcalWeightRecAlgoWeightsRcd.h"
#include "CLHEP/Matrix/Matrix.h"

using namespace std;

  class EcalObjectAnalyzer : public edm::EDAnalyzer
  {
  public:
    explicit  EcalObjectAnalyzer(edm::ParameterSet const& p) 
    { }
    explicit  EcalObjectAnalyzer(int i) 
    { }
    virtual ~ EcalObjectAnalyzer() { }
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  private:
  };

  void
   EcalObjectAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context)
  {
    using namespace edm::eventsetup;
    // Context is not used.
    std::cout <<" I AM IN RUN NUMBER "<<e.id().run() <<std::endl;
    std::cout <<" ---EVENT NUMBER "<<e.id().event() <<std::endl;
    edm::ESHandle<EcalPedestals> pPeds;
    context.get<EcalPedestalsRcd>().get(pPeds);

    //call tracker code
    //
    int channelID=1656;
    //EcalPedestals* myped=const_cast<EcalPedestals*>(pPeds.product());
    const EcalPedestals* myped=pPeds.product();
    std::map<const unsigned int,EcalPedestals::Item>::const_iterator it=myped->m_pedestals.find(channelID);
    if( it!=myped->m_pedestals.end() ){
      std::cout << "Ecal channel: " << channelID
                << "  mean_x1:  " <<it->second.mean_x1 << " rms_x1: " << it->second.rms_x1
                << "  mean_x6:  " <<it->second.mean_x6 << " rms_x6: " << it->second.rms_x6
                << "  mean_x12: " <<it->second.mean_x12 << " rms_x12: " << it->second.rms_x12
                << std::endl;
    }


    edm::ESHandle<EcalWeightRecAlgoWeights> pWgts;
    context.get<EcalWeightRecAlgoWeightsRcd>().get(pWgts);
    const EcalWeightRecAlgoWeights* wgts = pWgts.product();

    typedef std::vector< std::vector<EcalWeight> > EcalWeightMatrix;
    EcalWeightMatrix& mat1 = wgts->getWeightsBeforeGainSwitch();
    EcalWeightMatrix& mat2 = wgts->getWeightsAfterGainSwitch();

    //EcalWeight awgt = (mat1[0])[0];
    std::cout << "before switch [0,0]: " << (mat1[0])[0]() << endl;
    std::cout << "after  switch [0,0]: " << (mat2[0])[0]() << endl;

    HepMatrix clmat1(3,8,0);
    HepMatrix clmat2(3,8,0);
    for(int irow=0; irow<3; irow++) {
     for(int icol=0; icol<8; icol++) {
       clmat1[irow][icol] = (mat1[irow])[icol]();
       clmat2[irow][icol] = (mat2[irow])[icol]();
     }
   }
   std::cout << clmat1 << std::endl;
   std::cout << clmat2 << std::endl;




  } //end of ::Analyze()
  DEFINE_FWK_MODULE(EcalObjectAnalyzer)
