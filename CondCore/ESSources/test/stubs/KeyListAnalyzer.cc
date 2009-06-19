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

#include "CondFormats/DataRecord/interface/ExEfficiency.h"
#include "CondCore/IOVService/interface/KeyList.h"
#include "CondFormats/Calibration/interface/Conf.h"

using namespace std;

namespace {

  template<typename Conf>
  void print(Conf const & c) {
    std::cout << c.v << " " << c.key << " ; ";
  }

}


namespace edmtest
{
  class KeyListAnalyzer : public edm::EDAnalyzer
  {
  public:
    explicit KeyListAnalyzer(edm::ParameterSet const& p) 
    { 
      std::cout<<"KeyListAnalyzer"<<std::endl;
    }
    explicit KeyListAnalyzer(int i) 
    { std::cout<<"KeyListAnalyzer "<<i<<std::endl; }
    virtual ~KeyListAnalyzer() {  
      std::cout<<"~KeyListAnalyzer "<<std::endl;
    }
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  private:
  };
  
  void
  KeyListAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context){
    using namespace edm::eventsetup;
    // Context is not used.
    std::cout <<" I AM IN RUN NUMBER "<<e.id().run() <<std::endl;
    std::cout <<" ---EVENT NUMBER "<<e.id().event() <<std::endl;
    edm::eventsetup::EventSetupRecordKey 
      recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType("ExDwarfListRcd"));
    if( recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
      //record not found
      std::cout <<"Record \"ExDwarfListRcd "<<"\" does not exist "<<std::endl;
    }
    edm::ESHandle<cond::KeyList> klh;
    std::cout<<"got eshandle"<<std::endl;
    context.get<ExDwarfListRcd>().get(klh);
    std::cout<<"got context"<<std::endl;
    cond::KeyList const &  kl= *klh.product();
    int n=0;
    for (int i=0; i<kl.size(); i++)
      if (kl.elem(i)) { 
	n++;
	if (0==i%2) 
	  print(*kl.get<condex::ConfI>(i));
	else
	  print(*kl.get<condex::ConfF>(i));
      }
    std::cout << "found " << n << " valid keyed confs" << std::endl;
      
    std::cout << std::endl;
  }

  DEFINE_FWK_MODULE(KeyListAnalyzer);
}
