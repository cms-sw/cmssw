#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/DT/interface/DTUserKeyedConfigHandler.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CondCore/IOVService/interface/KeyList.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CondFormats/DTObjects/interface/DTKeyedConfig.h"
#include "CondFormats/DataRecord/interface/DTKeyedConfigListRcd.h"

//typedef popcon::PopConAnalyzer<DTUserKeyedConfigHandler> DTUserKeyedConfigPopConAnalyzer;
class DTUserKeyedConfigPopConAnalyzer: public popcon::PopConAnalyzer<DTUserKeyedConfigHandler> {
 public:
  DTUserKeyedConfigPopConAnalyzer(const edm::ParameterSet& pset):
    popcon::PopConAnalyzer<DTUserKeyedConfigHandler>( pset ) {}
  virtual ~DTUserKeyedConfigPopConAnalyzer(){}
  virtual void analyze(const edm::Event& e, const edm::EventSetup& s){

    edm::ESHandle<cond::KeyList> klh;
    std::cout<<"got eshandle"<<std::endl;
    s.get<DTKeyedConfigListRcd>().get(klh);
    std::cout<<"got context"<<std::endl;
    cond::KeyList const &  kl= *klh.product();
    cond::KeyList* list = const_cast<cond::KeyList*>( &kl );
    for ( int i = 0; i < list->size(); i++ ) {
      if ( list->elem( i ) )
           std::cout << list->get<DTKeyedConfig>( i )->getId() << std::endl;
    }
    DTUserKeyedConfigHandler::setList( list );

  }
 private:
};


DEFINE_FWK_MODULE(DTUserKeyedConfigPopConAnalyzer);

