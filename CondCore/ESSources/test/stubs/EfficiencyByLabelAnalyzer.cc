/*----------------------------------------------------------------------

Toy EDProducers and EDProducts for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>
#include <iostream>
#include <map>
#include <typeinfo>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"

#include "CondFormats/Calibration/interface/Efficiency.h"
#include "CondFormats/DataRecord/interface/ExEfficiency.h"

using namespace std;

namespace edmtest
{
  class EfficiencyByLabelAnalyzer : public edm::EDAnalyzer
  {
  public:
    explicit  EfficiencyByLabelAnalyzer(edm::ParameterSet const& p) 
    { 
      std::cout<<"EfficiencyByLabelAnalyzer"<<std::endl;
    }
    explicit  EfficiencyByLabelAnalyzer(int i) 
    { std::cout<<"EfficiencyByLabelAnalyzer "<<i<<std::endl; }
    virtual ~EfficiencyByLabelAnalyzer() {  
      std::cout<<"~EfficiencyByLabelAnalyzer "<<std::endl;
    }
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  private:
  };
  
  void
   EfficiencyByLabelAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context){
    using namespace edm::eventsetup;
    // Context is not used.
    std::cout <<" I AM IN RUN NUMBER "<<e.id().run() <<std::endl;
    std::cout <<" ---EVENT NUMBER "<<e.id().event() <<std::endl;
    edm::eventsetup::EventSetupRecordKey recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType("ExEfficiencyRcd"));
    if( recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
      //record not found
      std::cout <<"Record \"ExEfficiencyRcd"<<"\" does not exist "<<std::endl;
    }
    edm::ESHandle<condex::Efficiency> hEff1;
    edm::ESHandle<condex::Efficiency> hEff2;
    std::cout<<"got eshandle"<<std::endl;
    context.get<ExEfficiencyRcd>().get("vinEff1",hEff1);
    context.get<ExEfficiencyRcd>().get("vinEff2",hEff2);
    std::cout<<"got context"<<std::endl;
    {
      condex::Efficiency const &  eff= *hEff2.product();
      std::cout<<"Efficiency*, type (2) "<< (void*)(&eff) << " " << typeid(eff).name() <<std::endl;
    }
    condex::Efficiency const &  eff= *hEff1.product();
    std::cout<<"Efficiency*, type "<< (void*)(&eff) << " " << typeid(eff).name() <<std::endl;
    for (float pt=0;pt<10; pt+=2) {
     std::cout << "\npt="<<pt<<"    :";
     for (float eta=-3;eta<3; eta+=1)
       std::cout << eff(pt,eta) <<" ";    
   }  
   std::cout << std::endl;
  }

  DEFINE_FWK_MODULE(EfficiencyByLabelAnalyzer);
}


