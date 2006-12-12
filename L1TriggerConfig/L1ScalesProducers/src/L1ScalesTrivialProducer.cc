#include "L1Trigger/L1ScalesProducers/src/L1ScalesTrivialProducer.h"

#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTLookupTables.h"

using std::cout;
using std::endl;
using std::vector;
using std::auto_ptr;


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
L1ScalesTrivialProducer::L1ScalesTrivialProducer(const edm::ParameterSet& ps)
{
 
  cout << "Constructing an L1ScalesTrivialProducer" << endl;

  //the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this, &L1ScalesTrivialProducer::produceEmScale);
  setWhatProduced(this, &L1ScalesTrivialProducer::produceJetScale);

  //now do what ever other initialization is needed
  
  // get numbers from the config file -  all units are GeV
  //  m_emEtScaleInputLsb = ps.getParameter<double>("L1CaloEmEtScaleLSB"); 
  m_emEtScaleInputLsb = L1RCTLookupTables::eGammaLSB();
  m_emEtThresholds = ps.getParameter< vector<double> >("L1CaloEmThresholds");

  //  m_jetEtScaleInputLsb = ps.getParameter<double>("L1CaloRegionEtScaleLSB"); 
  m_jetEtScaleInputLsb = L1RCTLookupTables::jetMETLSB();
  m_jetEtThresholds = ps.getParameter< vector<double> >("L1CaloJetThresholds");

}


L1ScalesTrivialProducer::~L1ScalesTrivialProducer()
{
 
   // do anything here that needs to be done at destruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
std::auto_ptr<L1CaloEtScale> L1ScalesTrivialProducer::produceEmScale(const L1EmEtScaleRcd& iRecord)
{
   using namespace edm::es;

   std::auto_ptr<L1CaloEtScale> emScale = std::auto_ptr<L1CaloEtScale>( new L1CaloEtScale(m_emEtScaleInputLsb, m_emEtThresholds) );

   return emScale ;
}

std::auto_ptr<L1CaloEtScale> L1ScalesTrivialProducer::produceJetScale(const L1JetEtScaleRcd& iRecord)
{
   using namespace edm::es;

   std::auto_ptr<L1CaloEtScale> jetEtScale = std::auto_ptr<L1CaloEtScale>( new L1CaloEtScale(m_jetEtScaleInputLsb, m_jetEtThresholds) );

   return jetEtScale ;
}

