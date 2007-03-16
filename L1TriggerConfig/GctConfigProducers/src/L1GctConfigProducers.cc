#include "L1TriggerConfig/GctConfigProducers/interface/L1GctConfigProducers.h"

#include "CondFormats/L1TObjects/interface/L1GctJetEtCalibrationFunction.h"
#include "CondFormats/DataRecord/interface/L1GctJetCalibFunRcd.h"
#include "CondFormats/DataRecord/interface/L1JetEtScaleRcd.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
L1GctConfigProducers::L1GctConfigProducers(const edm::ParameterSet& iConfig)
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced(this,dependsOn(&L1GctConfigProducers::doWhenChanged));

   //now do what ever other initialization is needed

   std::cout << "Hello from GctConfigProducers" << std::endl;

   // get the CalibrationFunction parameters from the config file
   m_htScaleLSB = iConfig.getParameter<double>("L1CaloHtScaleLsbInGeV");
   m_threshold  = iConfig.getParameter<double>("L1CaloJetZeroSuppressionThresholdInGeV");
   // coefficients for non-tau jet corrections
   for (unsigned i=0; i<L1GctJetEtCalibrationFunction::NUMBER_ETA_VALUES; ++i) {
     std::stringstream ss;
     std::string str;
     ss << "nonTauJetCalib" << i;
     ss >> str;
     m_jetCalibFunc.push_back(iConfig.getParameter< std::vector<double> >(str));
   }
   // coefficients for tau jet corrections
   for (unsigned i=0; i<L1GctJetEtCalibrationFunction::N_CENTRAL_ETA_VALUES; ++i) {
     std::stringstream ss;
     std::string str;
     ss << "tauJetCalib" << i;
     ss >> str;
     m_tauCalibFunc.push_back(iConfig.getParameter< std::vector<double> >(str));
   }
}


L1GctConfigProducers::~L1GctConfigProducers()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
L1GctConfigProducers::ReturnType
L1GctConfigProducers::produce(const L1GctJetCalibFunRcd& iRecord)
{
   using namespace edm::es;
   boost::shared_ptr<L1GctJetEtCalibrationFunction> pL1GctJetEtCalibrationFunction =
     boost::shared_ptr<L1GctJetEtCalibrationFunction> (new L1GctJetEtCalibrationFunction());

   pL1GctJetEtCalibrationFunction->setParams(m_htScaleLSB, m_threshold,
					     m_jetCalibFunc,
					     m_tauCalibFunc);

   pL1GctJetEtCalibrationFunction->setOutputEtScale(m_jetScale);

   std::cout << "Done producing calibFun!!" << std::endl;

   return pL1GctJetEtCalibrationFunction ;
}

/// Add a dependency on the JetEtScale
void L1GctConfigProducers::doWhenChanged(const L1JetEtScaleRcd& jetScaleRcd)
{
  edm::ESHandle<L1CaloEtScale> jsc;
  jetScaleRcd.get(jsc);
  m_jetScale = *jsc.product();
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1GctConfigProducers);
