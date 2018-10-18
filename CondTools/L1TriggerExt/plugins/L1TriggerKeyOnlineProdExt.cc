#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondTools/L1TriggerExt/plugins/L1TriggerKeyOnlineProdExt.h"

#include "CondTools/L1Trigger/interface/Exception.h"

#include "FWCore/Framework/interface/EventSetup.h"

L1TriggerKeyOnlineProdExt::L1TriggerKeyOnlineProdExt(const edm::ParameterSet& iConfig)
  : m_subsystemLabels( iConfig.getParameter< std::vector< std::string > >(
      "subsystemLabels" ) )
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced(this);

   //now do what ever other initialization is needed
}


L1TriggerKeyOnlineProdExt::~L1TriggerKeyOnlineProdExt()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
L1TriggerKeyOnlineProdExt::ReturnType
L1TriggerKeyOnlineProdExt::produce(const L1TriggerKeyExtRcd& iRecord)
{
   using namespace edm::es;

   // Start with "SubsystemKeysOnly"
   edm::ESHandle< L1TriggerKeyExt > subsystemKeys ;
   try
     {
       iRecord.get( "SubsystemKeysOnly", subsystemKeys ) ;
     }
   catch( l1t::DataAlreadyPresentException& ex )
     {
       throw ex ;
     }

   auto pL1TriggerKey = std::make_unique< L1TriggerKeyExt >(*subsystemKeys) ;

  // Collate object keys
  std::vector< std::string >::const_iterator itr = m_subsystemLabels.begin() ;
  std::vector< std::string >::const_iterator end = m_subsystemLabels.end() ;
  for( ; itr != end ; ++itr )
    {
      edm::ESHandle< L1TriggerKeyExt > objectKeys ;
      try
	{
	  iRecord.get( *itr, objectKeys ) ;
	}
      catch( l1t::DataAlreadyPresentException& ex )
	{
	  throw ex ;
	}

      pL1TriggerKey->add( objectKeys->recordToKeyMap() ) ;
    }

   return pL1TriggerKey ;
}

//define this as a plug-in
//DEFINE_FWK_EVENTSETUP_MODULE(L1TriggerKeyOnlineProdExt);
