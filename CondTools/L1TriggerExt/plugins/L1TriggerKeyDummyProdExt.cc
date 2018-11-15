#include "CondTools/L1TriggerExt/plugins/L1TriggerKeyDummyProdExt.h"

L1TriggerKeyDummyProdExt::L1TriggerKeyDummyProdExt(const edm::ParameterSet& iConfig)
{
   //the following line is needed to tell the framework what
   // data is being produced

  // Label should be empty, "SubsystemKeysOnly" or any subsystem label expected
  // by L1TriggerKeyOnlineProd.
  std::string label = iConfig.getParameter< std::string >( "label" ) ;
  setWhatProduced(this, label);

   //now do what ever other initialization is needed
   m_key.setTSCKey( iConfig.getParameter< std::string >( "tscKey" ) ) ;
   m_key.setSubsystemKey( L1TriggerKeyExt::kuGT,
			  iConfig.getParameter< std::string >( "uGTKey" ) ) ;
   m_key.setSubsystemKey( L1TriggerKeyExt::kuGMT,
			  iConfig.getParameter< std::string >( "uGMTKey" ) ) ;
   m_key.setSubsystemKey( L1TriggerKeyExt::kBMTF,
			  iConfig.getParameter< std::string >( "BMTFKey" ) ) ;
   m_key.setSubsystemKey( L1TriggerKeyExt::kCALO,
			  iConfig.getParameter< std::string >( "CALOKey" ) ) ;

   if( label != "SubsystemKeysOnly" )
     {
       typedef std::vector< edm::ParameterSet > ObjectKeys;
       ObjectKeys keys = iConfig.getParameter< ObjectKeys >( "objectKeys" ) ;

       for( ObjectKeys::const_iterator it = keys.begin ();
	    it != keys.end() ;
	    ++it )
	 {
	   // Replace ?s with spaces.
	   std::string key = it->getParameter< std::string >( "key" ) ;
	   replace( key.begin(), key.end(), '?', ' ' ) ;

	   m_key.add( it->getParameter< std::string >( "record" ),
		      it->getParameter< std::string >( "type" ),
		      key ) ;
	 }
     }
}


L1TriggerKeyDummyProdExt::~L1TriggerKeyDummyProdExt()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
L1TriggerKeyDummyProdExt::ReturnType
L1TriggerKeyDummyProdExt::produce(const L1TriggerKeyExtRcd& iRecord)
{
   using namespace edm::es;
   return std::make_unique< L1TriggerKeyExt >(m_key) ;
}

//define this as a plug-in
//DEFINE_FWK_EVENTSETUP_MODULE(L1TriggerKeyDummyProdExt);
