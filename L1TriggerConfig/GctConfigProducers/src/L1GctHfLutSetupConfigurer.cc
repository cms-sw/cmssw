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
  m_etSumThresholds(iConfig.getParameter< std::vector<unsigned> >("HfLutEtSumThresholds")),
  m_countThresholds(iConfig.getParameter< std::vector<unsigned> >("HfLutBitCountThresholds"))
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

   pL1GctHfLutSetup->setThresholds( L1GctHfLutSetup::bitCountPosEtaRing1, m_countThresholds);
   pL1GctHfLutSetup->setThresholds( L1GctHfLutSetup::bitCountPosEtaRing2, m_countThresholds);
   pL1GctHfLutSetup->setThresholds( L1GctHfLutSetup::bitCountNegEtaRing1, m_countThresholds);
   pL1GctHfLutSetup->setThresholds( L1GctHfLutSetup::bitCountNegEtaRing2, m_countThresholds);
   pL1GctHfLutSetup->setThresholds( L1GctHfLutSetup::etSumPosEtaRing1, m_etSumThresholds);
   pL1GctHfLutSetup->setThresholds( L1GctHfLutSetup::etSumPosEtaRing2, m_etSumThresholds);
   pL1GctHfLutSetup->setThresholds( L1GctHfLutSetup::etSumNegEtaRing1, m_etSumThresholds);
   pL1GctHfLutSetup->setThresholds( L1GctHfLutSetup::etSumNegEtaRing2, m_etSumThresholds);

   return pL1GctHfLutSetup;
}

