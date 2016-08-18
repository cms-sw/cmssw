#include <iostream>
#include "CondTools/L1TriggerExt/interface/L1ObjectKeysOnlineProdBaseExt.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class L1TUtmTriggerMenuObjectKeysOnlineProd : public L1ObjectKeysOnlineProdBaseExt {
private:

public:
    virtual void fillObjectKeys( ReturnType pL1TriggerKey ) override ;

    L1TUtmTriggerMenuObjectKeysOnlineProd(const edm::ParameterSet&);
    ~L1TUtmTriggerMenuObjectKeysOnlineProd(void){}
};

L1TUtmTriggerMenuObjectKeysOnlineProd::L1TUtmTriggerMenuObjectKeysOnlineProd(const edm::ParameterSet& iConfig)
  : L1ObjectKeysOnlineProdBaseExt( iConfig ){
}


void L1TUtmTriggerMenuObjectKeysOnlineProd::fillObjectKeys( ReturnType pL1TriggerKey ){

    std::string uGTKey = pL1TriggerKey->subsystemKey( L1TriggerKeyExt::kuGT ) ;

    std::string stage2Schema = "CMS_TRG_L1_CONF" ;

    if( !uGTKey.empty() ) {
       std::string l1_menu_key;
       std::vector< std::string > queryStrings ;
       queryStrings.push_back( "L1_MENU" ) ;

        std::string l1_menu_name, ugt_key;

        // select MP7_PP_CONF_KEY from CMS_S1CALOL2.S1CALOL2_CONF where S1CALOL2_CONF_KEY = objectKey ;
        l1t::OMDSReader::QueryResults queryResult =
            m_omdsReader.basicQuery( queryStrings,
                                     stage2Schema,
                                     "UGT_KEYS",
                                     "UGT_KEYS.ID",
                                     m_omdsReader.singleAttribute(uGTKey)
                                   ) ;

        if( queryResult.queryFailed() || queryResult.numberRows() != 1 ){
            edm::LogError( "L1-O2O" ) << "Cannot get UGT_KEYS.L1_MENU}" ;
            return ;
        }

        if( !queryResult.fillVariable( "L1_MENU", l1_menu_key) ) l1_menu_key = "";
        
        pL1TriggerKey->add( "L1TUtmTriggerMenuRcd",
                            "L1TUtmTriggerMenu",
			    l1_menu_key) ;
    }
}


//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TUtmTriggerMenuObjectKeysOnlineProd);
