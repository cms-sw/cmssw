#include "L1TriggerConfig/GctConfigProducers/interface/L1GctHfLutSetupConfigurer.h"

#include <string>

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
L1GctHfLutSetupConfigurer::L1GctHfLutSetupConfigurer(const edm::ParameterSet& iConfig) :
  m_thresholds(iConfig.getParameter< std::vector<unsigned> >("HfLutThresholds"))
{

  // ------------------------------------------------------------------------------------------
  // Read jet counter setup info from config file
}


L1GctHfLutSetupConfigurer::~L1GctHfLutSetupConfigurer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//
    

// ------------ methods called to produce the data  ------------

L1GctHfLutSetupConfigurer::HfLutSetupReturnType 
L1GctHfLutSetupConfigurer::produceHfLutSetup()
{
   boost::shared_ptr<L1GctHfLutSetup> pL1GctHfLutSetup=
     boost::shared_ptr<L1GctHfLutSetup> (new L1GctHfLutSetup());

   unsigned nTypes = (unsigned) L1GctHfLutSetup::numberOfLutTypes;
   for (unsigned t=0; t<nTypes; ++t) {
     pL1GctHfLutSetup->setThresholds( (L1GctHfLutSetup::hfLutType) t, m_thresholds);
   }

   return pL1GctHfLutSetup;
}

