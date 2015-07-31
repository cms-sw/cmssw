///
/// \class l1t::L1TStage1Layer2Producer
///
/// Description: Emulator for the stage 1 jet algorithms.
///
///
/// \author: R. Alex Barbieri MIT
///


// system include files
#include <boost/shared_ptr.hpp>

// user include files

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"
#include "CondFormats/DataRecord/interface/L1EmEtScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1JetEtScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1HtMissScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1HfRingEtScaleRcd.h"

#include <vector>
#include "DataFormats/L1Trigger/interface/BXVector.h"

//#include "CondFormats/DataRecord/interface/CaloParamsRcd.h"
//#include "CondFormats/L1TCalorimeter/interface/CaloParams.h"
#include "CondFormats/L1TObjects/interface/CaloParams.h"
#include "CondFormats/DataRecord/interface/L1TCaloParamsRcd.h"
//#include "CondFormats/L1TObjects/interface/FirmwareVersion.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloConfigHelper.h"
#include "CondFormats/DataRecord/interface/L1TCaloConfigRcd.h"

#include "DataFormats/L1TCalorimeter/interface/CaloRegion.h"
#include "DataFormats/L1TCalorimeter/interface/CaloEmCand.h"

#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/L1Trigger/interface/CaloSpare.h"

#include "L1Trigger/L1TCalorimeter/interface/Stage1Layer2MainProcessor.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage1Layer2FirmwareFactory.h"

// print statements
//#include <stdio.h>

using namespace std;
using namespace edm;
using namespace l1t;


//
// class declaration
//

  class L1TStage1Layer2Producer : public EDProducer {
  public:
    explicit L1TStage1Layer2Producer(const ParameterSet&);
    ~L1TStage1Layer2Producer();

    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    virtual void produce(Event&, EventSetup const&);
    virtual void beginJob();
    virtual void endJob();
    virtual void beginRun(Run const&iR, EventSetup const&iE);
    virtual void endRun(Run const& iR, EventSetup const& iE);

    // ----------member data ---------------------------
    unsigned long long m_paramsCacheId; // Cache-ID from current parameters, to check if needs to be updated.
    unsigned long long m_configCacheId; // Cache-ID from current parameters, to check if needs to be updated.
    CaloParamsHelper* m_params;
    CaloConfigHelper m_config;


    boost::shared_ptr<Stage1Layer2MainProcessor> m_fw; // Firmware to run per event, depends on database parameters.

    Stage1Layer2FirmwareFactory m_factory; // Factory to produce algorithms based on DB parameters

    std::string m_conditionsLabel;

    // to be extended with other "consumes" stuff
    EDGetToken regionToken;
    EDGetToken candsToken;

  };

  //
  // constructors and destructor
  //
  L1TStage1Layer2Producer::L1TStage1Layer2Producer(const ParameterSet& iConfig)
  {
    // register what you produce
    produces<BXVector<EGamma>>();
    produces<BXVector<Tau>>("rlxTaus");
    produces<BXVector<Tau>>("isoTaus");
    produces<BXVector<Jet>>();
    produces<BXVector<Jet>>("preGtJets");
    produces<BXVector<EtSum>>();
    produces<BXVector<CaloSpare>>("HFRingSums");
    produces<BXVector<CaloSpare>>("HFBitCounts");

    // register what you consume and keep token for later access:
    regionToken = consumes<BXVector<CaloRegion>>(iConfig.getParameter<InputTag>("CaloRegions"));
    candsToken = consumes<BXVector<CaloEmCand>>(iConfig.getParameter<InputTag>("CaloEmCands"));
    //int ifwv=iConfig.getParameter<unsigned>("FirmwareVersion");  // LenA  make configurable for now

    m_conditionsLabel = iConfig.getParameter<std::string>("conditionsLabel");

    m_params = new CaloParamsHelper;

    // set cache id to zero, will be set at first beginRun:
    m_paramsCacheId = 0;
    m_configCacheId = 0;
  }


  L1TStage1Layer2Producer::~L1TStage1Layer2Producer()
  {
  }



//
// member functions
//

// ------------ method called to produce the data ------------
void
L1TStage1Layer2Producer::produce(Event& iEvent, const EventSetup& iSetup)
{

  LogDebug("l1t|stage 1 jets") << "L1TStage1Layer2Producer::produce function called...\n";

  //return;

  //inputs
  Handle<BXVector<CaloRegion>> caloRegions;
  iEvent.getByToken(regionToken,caloRegions);

  Handle<BXVector<CaloEmCand>> caloEmCands;
  iEvent.getByToken(candsToken, caloEmCands);

  int bxFirst = caloRegions->getFirstBX();
  int bxLast = caloRegions->getLastBX();

  //outputs
  std::auto_ptr<EGammaBxCollection> egammas (new EGammaBxCollection);
  std::auto_ptr<TauBxCollection> taus (new TauBxCollection);
  std::auto_ptr<TauBxCollection> isoTaus (new TauBxCollection);
  std::auto_ptr<JetBxCollection> jets (new JetBxCollection);
  std::auto_ptr<JetBxCollection> preGtJets (new JetBxCollection);
  std::auto_ptr<EtSumBxCollection> etsums (new EtSumBxCollection);
  std::auto_ptr<CaloSpareBxCollection> hfSums (new CaloSpareBxCollection);
  std::auto_ptr<CaloSpareBxCollection> hfCounts (new CaloSpareBxCollection);

  egammas->setBXRange(bxFirst, bxLast);
  taus->setBXRange(bxFirst, bxLast);
  isoTaus->setBXRange(bxFirst, bxLast);
  jets->setBXRange(bxFirst, bxLast);
  preGtJets->setBXRange(bxFirst, bxLast);
  etsums->setBXRange(bxFirst, bxLast);
  hfSums->setBXRange(bxFirst, bxLast);
  hfCounts->setBXRange(bxFirst, bxLast);

  //producer is responsible for splitting the BXVector into pieces for
  //the firmware to handle
  for(int i = bxFirst; i <= bxLast; ++i)
  {
    //make local inputs
    std::vector<CaloRegion> *localRegions = new std::vector<CaloRegion>();
    std::vector<CaloEmCand> *localEmCands = new std::vector<CaloEmCand>();

    //make local outputs
    std::vector<EGamma> *localEGammas = new std::vector<EGamma>();
    std::vector<Tau> *localTaus = new std::vector<Tau>();
    std::vector<Tau> *localIsoTaus = new std::vector<Tau>();
    std::vector<Jet> *localJets = new std::vector<Jet>();
    std::vector<Jet> *localPreGtJets = new std::vector<Jet>();
    std::vector<EtSum> *localEtSums = new std::vector<EtSum>();
    CaloSpare *localHfSums = new CaloSpare();
    localHfSums->setType(CaloSpare::HFRingSum);
    CaloSpare *localHfCounts = new CaloSpare();
    localHfCounts->setType(CaloSpare::HFBitCount);

    // copy over the inputs -> there must be a better way to do this
    for(std::vector<CaloRegion>::const_iterator region = caloRegions->begin(i);
	region != caloRegions->end(i); ++region)
      localRegions->push_back(*region);
    for(std::vector<CaloEmCand>::const_iterator emcand = caloEmCands->begin(i);
	emcand != caloEmCands->end(i); ++emcand)
      localEmCands->push_back(*emcand);

    //run the firmware on one event
    m_fw->processEvent(*localEmCands, *localRegions,
		       localEGammas, localTaus, localIsoTaus, localJets, localPreGtJets, localEtSums,
		       localHfSums, localHfCounts);

    // copy the output into the BXVector -> there must be a better way
    for(std::vector<EGamma>::const_iterator eg = localEGammas->begin(); eg != localEGammas->end(); ++eg)
      egammas->push_back(i, *eg);
    for(std::vector<Tau>::const_iterator tau = localTaus->begin(); tau != localTaus->end(); ++tau)
      taus->push_back(i, *tau);
    for(std::vector<Tau>::const_iterator isotau = localIsoTaus->begin(); isotau != localIsoTaus->end(); ++isotau)
      isoTaus->push_back(i, *isotau);
    for(std::vector<Jet>::const_iterator jet = localJets->begin(); jet != localJets->end(); ++jet)
      jets->push_back(i, *jet);
    for(std::vector<Jet>::const_iterator jet = localPreGtJets->begin(); jet != localPreGtJets->end(); ++jet)
      preGtJets->push_back(i, *jet);
    for(std::vector<EtSum>::const_iterator etsum = localEtSums->begin(); etsum != localEtSums->end(); ++etsum)
      etsums->push_back(i, *etsum);
    hfSums->push_back(i, *localHfSums);
    hfCounts->push_back(i, *localHfCounts);

    delete localRegions;
    delete localEmCands;
    delete localEGammas;
    delete localTaus;
    delete localIsoTaus;
    delete localJets;
    delete localPreGtJets;
    delete localEtSums;
    delete localHfSums;
    delete localHfCounts;
  }


  iEvent.put(egammas);
  iEvent.put(taus,"rlxTaus");
  iEvent.put(isoTaus,"isoTaus");
  iEvent.put(jets);
  iEvent.put(preGtJets,"preGtJets");
  iEvent.put(etsums);
  iEvent.put(hfSums,"HFRingSums");
  iEvent.put(hfCounts,"HFBitCounts");
}

// ------------ method called once each job just before starting event loop ------------
void
L1TStage1Layer2Producer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop ------------
void
L1TStage1Layer2Producer::endJob() {
}

// ------------ method called when starting to processes a run ------------

void L1TStage1Layer2Producer::beginRun(Run const&iR, EventSetup const&iE){
  unsigned long long id = 0;

  id = iE.get<L1TCaloParamsRcd>().cacheIdentifier();
  if (id != m_paramsCacheId) {

    m_paramsCacheId = id;

    edm::ESHandle<CaloParams> paramsHandle;

    iE.get<L1TCaloParamsRcd>().get(m_conditionsLabel, paramsHandle);

    // replace our local copy of the parameters with a new one using placement new
    m_params->~CaloParamsHelper();
    m_params = new (m_params) CaloParamsHelper(*paramsHandle.product());

    LogDebug("L1TDebug") << *m_params << std::endl;

    if (! m_params){
      edm::LogError("l1t|caloStage1") << "Could not retrieve params from Event Setup" << std::endl;
    }

  }

  id = iE.get<L1TCaloConfigRcd>().cacheIdentifier();
  if (id != m_configCacheId) {
    m_configCacheId = id;

    edm::ESHandle<CaloConfig> configHandle;

    iE.get<L1TCaloConfigRcd>().get(m_conditionsLabel, configHandle);


    if (! configHandle.product()){
      edm::LogError("l1t|caloStage1") << "Could not retrieve config from Event Setup" << std::endl;
    } else {
      //update our DB payload in CaloConfigHelper:
      m_config.UpdatePayload(configHandle.product());
      //cout << "DEBUG:  L1T Calo Config reports the Layer2 firmware version is " << m_config.fwv() << "\n";
    }      
  }






  LogDebug("l1t|stage 1 jets") << "L1TStage1Layer2Producer::beginRun function called...\n";

  //get the proper scales for conversion to physical et AND gt scales
  edm::ESHandle< L1CaloEtScale > emScale ;
  iE.get< L1EmEtScaleRcd >().get( m_conditionsLabel, emScale ) ;
  m_params->setEmScale(*emScale);

  edm::ESHandle< L1CaloEtScale > jetScale ;
  iE.get< L1JetEtScaleRcd >().get( m_conditionsLabel, jetScale ) ;
  m_params->setJetScale(*jetScale);

  edm::ESHandle< L1CaloEtScale > HtMissScale;
  iE.get< L1HtMissScaleRcd >().get( m_conditionsLabel, HtMissScale ) ;
  m_params->setHtMissScale(*HtMissScale);

  //not sure if I need this one
  edm::ESHandle< L1CaloEtScale > HfRingScale;
  iE.get< L1HfRingEtScaleRcd >().get( m_conditionsLabel, HfRingScale );
  m_params->setHfRingScale(*HfRingScale);


  //unsigned long long id = iE.get<CaloParamsRcd>().cacheIdentifier();

  //if (id != m_paramsCacheId)
  { // Need to update:
    //m_paramsCacheId = id;

    //ESHandle<CaloParams> parameters;
    //iE.get<CaloParamsRcd>().get(parameters);

    // LenA move the setting of the firmware version to the L1TStage1Layer2Producer constructor

    //m_params = boost::shared_ptr<const CaloParams>(parameters.product());
    //m_fwv = boost::shared_ptr<const FirmwareVersion>(new FirmwareVersion());
    //printf("Begin.\n");
    //m_fwv = boost::shared_ptr<FirmwareVersion>(new FirmwareVersion()); //not const during testing
    //printf("Success m_fwv.\n");
    //m_fwv->setFirmwareVersion(1); //hardcode for now, 1=HI, 2=PP
    //printf("Success m_fwv version set.\n");

    // if (! m_params){
    //   LogError("l1t|stage 1 jets") << "L1TStage1Layer2Producer: could not retreive DB params from Event Setup\n";
    // }

    // Set the current algorithm version based on DB pars from database:
    //m_fw = m_factory.create(*m_fwv /*,*m_params*/);
    //printf("Success create.\n");

    //if (! m_fw) {
    //  // we complain here once per run
    //  LogError("l1t|stage 1 jets") << "L1TStage1Layer2Producer: firmware could not be configured.\n";
    //}
  }



    int ifwv=m_config.fwv();
    //cout << "DEBUG:  ifwv is " << ifwv << "\n";
    //m_fwv = boost::shared_ptr<FirmwareVersion>(new FirmwareVersion()); //not const during testing
    if (ifwv == 1){
      LogDebug("l1t|stage1firmware") << "L1TStage1Layer2Producer -- Running HI implementation\n";
      //std::cout << "L1TStage1Layer2Producer -- Running HI implementation\n";
    }else if (ifwv == 2){
      LogDebug("l1t|stage1firmware") << "L1TStage1Layer2Producer -- Running pp implementation\n";
      //std::cout << "L1TStage1Layer2Producer -- Running pp implementation\n";
    } else if (ifwv == 3){
      LogDebug("l1t|stage1firmware") << "L1TStage1Layer2Producer -- Running SimpleHW implementation\n";
      //std::cout << "L1TStage1Layer2Producer -- Running SimpleHW implementation -- for testing only\n";
    }else{
      LogError("l1t|stage1firmware") << "L1TStage1Layer2Producer -- Unknown implementation.\n";
      //std::cout << "L1TStage1Layer2Producer -- Unknown implementation.\n";
    }
    //m_fwv->setFirmwareVersion(ifwv); // =1 HI, =2 PP
    // m_fw = m_factory.create(*m_fwv /*,*m_params*/);
    //m_fwv = ifwv;
    m_fw = m_factory.create(ifwv ,m_params);
    //printf("Success create.\n");
    if (! m_fw) {
      // we complain here once per job
      LogError("l1t|stage1firmware") << "L1TStage1Layer2Producer: firmware could not be configured.\n";
    }




}

// ------------ method called when ending the processing of a run ------------
void L1TStage1Layer2Producer::endRun(Run const& iR, EventSetup const& iE){

}


// ------------ method fills 'descriptions' with the allowed parameters for the module ------------
void
L1TStage1Layer2Producer::fillDescriptions(ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TStage1Layer2Producer);
