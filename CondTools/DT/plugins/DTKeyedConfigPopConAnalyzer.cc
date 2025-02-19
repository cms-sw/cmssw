#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/DT/interface/DTKeyedConfigHandler.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CondCore/IOVService/interface/KeyList.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CondFormats/DTObjects/interface/DTKeyedConfig.h"
#include "CondFormats/DataRecord/interface/DTKeyedConfigListRcd.h"

//typedef popcon::PopConAnalyzer<DTKeyedConfigHandler> DTKeyedConfigPopConAnalyzer;
class DTKeyedConfigPopConAnalyzer: public popcon::PopConAnalyzer<DTKeyedConfigHandler> {
 public:
  DTKeyedConfigPopConAnalyzer(const edm::ParameterSet& pset):
    popcon::PopConAnalyzer<DTKeyedConfigHandler>( pset ),
    copyData( pset.getParameter<edm::ParameterSet>("Source").
                      getUntrackedParameter<bool> ( "copyData", true ) ) 
 {}
  virtual ~DTKeyedConfigPopConAnalyzer(){}
  virtual void analyze(const edm::Event& e, const edm::EventSetup& s){

    if ( !copyData ) return;

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
    DTKeyedConfigHandler::setList( list );

  }
 private:
  bool copyData;
};


DEFINE_FWK_MODULE(DTKeyedConfigPopConAnalyzer);

