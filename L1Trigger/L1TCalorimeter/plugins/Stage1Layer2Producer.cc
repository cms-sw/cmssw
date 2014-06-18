///
/// \class l1t::Stage1Layer2Producer
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
#include "L1Trigger/L1TCalorimeter/interface/CaloParamsStage1.h"

#include "DataFormats/L1TCalorimeter/interface/CaloRegion.h"
#include "DataFormats/L1TCalorimeter/interface/CaloEmCand.h"

#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"

#include "L1Trigger/L1TCalorimeter/interface/Stage1Layer2MainProcessor.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage1Layer2FirmwareFactory.h"

// print statements
//#include <stdio.h>

using namespace std;
using namespace edm;

namespace l1t {

//
// class declaration
//

  class Stage1Layer2Producer : public EDProducer {
  public:
    explicit Stage1Layer2Producer(const ParameterSet&);
    ~Stage1Layer2Producer();

    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    virtual void produce(Event&, EventSetup const&);
    virtual void beginJob();
    virtual void endJob();
    virtual void beginRun(Run const&iR, EventSetup const&iE);
    virtual void endRun(Run const& iR, EventSetup const& iE);

    // ----------member data ---------------------------
    unsigned long long m_paramsCacheId; // Cache-ID from current parameters, to check if needs to be updated.
    CaloParamsStage1* m_params;

    //boost::shared_ptr<const CaloParamsStage1> m_params; // Database parameters for the trigger, to be updated as needed.
    //boost::shared_ptr<const FirmwareVersion> m_fwv;
    //boost::shared_ptr<FirmwareVersion> m_fwv; //not const during testing.
    int m_fwv;

    boost::shared_ptr<Stage1Layer2MainProcessor> m_fw; // Firmware to run per event, depends on database parameters.

    Stage1Layer2FirmwareFactory m_factory; // Factory to produce algorithms based on DB parameters

    // to be extended with other "consumes" stuff
    EDGetToken regionToken;
    EDGetToken candsToken;

    unsigned regionETCutForHT;
    unsigned regionETCutForMET;
    int minGctEtaForSums;
    int maxGctEtaForSums;
    double egRelativeJetIsolationCut;
    double tauRelativeJetIsolationCut;
  };

  //
  // constructors and destructor
  //
  Stage1Layer2Producer::Stage1Layer2Producer(const ParameterSet& iConfig)
  {
    // register what you produce
    produces<BXVector<l1t::EGamma>>();
    produces<BXVector<l1t::Tau>>();
    produces<BXVector<l1t::Jet>>();
    produces<BXVector<l1t::EtSum>>();

    // register what you consume and keep token for later access:
    regionToken = consumes<BXVector<l1t::CaloRegion>>(iConfig.getParameter<InputTag>("CaloRegions"));
    candsToken = consumes<BXVector<l1t::CaloEmCand>>(iConfig.getParameter<InputTag>("CaloEmCands"));
    int ifwv=iConfig.getParameter<unsigned>("FirmwareVersion");  // LenA  make configurable for now


    regionETCutForHT = iConfig.getParameter<unsigned int>("regionETCutForHT");
    regionETCutForMET = iConfig.getParameter<unsigned int>("regionETCutForMET");
    minGctEtaForSums = iConfig.getParameter<int>("minGctEtaForSums");
    maxGctEtaForSums = iConfig.getParameter<int>("maxGctEtaForSums");
    egRelativeJetIsolationCut = iConfig.getParameter<double>("egRelativeJetIsolationCut");
    tauRelativeJetIsolationCut = iConfig.getParameter<double>("tauRelativeJetIsolationCut");


    //m_fwv = boost::shared_ptr<FirmwareVersion>(new FirmwareVersion()); //not const during testing

    if (ifwv == 1){
      LogDebug("l1t|stage 1 jets") << "Stage1Layer2Producer -- Running HI implementation\n";
      std::cout << "Stage1Layer2Producer -- Running HI implementation\n";
    }else if (ifwv == 2){
      LogDebug("l1t|stage 1 jets") << "Stage1Layer2Producer -- Running pp implementation\n";
      std::cout << "Stage1Layer2Producer -- Running pp implementation\n";
    }else{
      LogError("l1t|stage 1 jets") << "Stage1Layer2Producer -- Unknown implementation.\n";
      std::cout << "Stage1Layer2Producer -- Unknown implementation.\n";
    }
    //m_fwv->setFirmwareVersion(ifwv); // =1 HI, =2 PP
    // m_fw = m_factory.create(*m_fwv /*,*m_params*/);
    m_fwv = ifwv;

    m_params = new CaloParamsStage1;

    m_fw = m_factory.create(m_fwv ,m_params);
    //printf("Success create.\n");
    if (! m_fw) {
      // we complain here once per job
      LogError("l1t|stage 1 jets") << "Stage1Layer2Producer: firmware could not be configured.\n";
    }

    // set cache id to zero, will be set at first beginRun:
    m_paramsCacheId = 0;
  }


  Stage1Layer2Producer::~Stage1Layer2Producer()
  {
  }



//
// member functions
//

// ------------ method called to produce the data ------------
void
Stage1Layer2Producer::produce(Event& iEvent, const EventSetup& iSetup)
{

  LogDebug("l1t|stage 1 jets") << "Stage1Layer2Producer::produce function called...\n";

  //return;

  //inputs
  Handle<BXVector<l1t::CaloRegion>> caloRegions;
  iEvent.getByToken(regionToken,caloRegions);

  Handle<BXVector<l1t::CaloEmCand>> caloEmCands;
  iEvent.getByToken(candsToken, caloEmCands);

  int bxFirst = caloRegions->getFirstBX();
  int bxLast = caloRegions->getLastBX();

  //outputs
  std::auto_ptr<l1t::EGammaBxCollection> egammas (new l1t::EGammaBxCollection);
  std::auto_ptr<l1t::TauBxCollection> taus (new l1t::TauBxCollection);
  std::auto_ptr<l1t::JetBxCollection> jets (new l1t::JetBxCollection);
  std::auto_ptr<l1t::EtSumBxCollection> etsums (new l1t::EtSumBxCollection);

  egammas->setBXRange(bxFirst, bxLast);
  taus->setBXRange(bxFirst, bxLast);
  jets->setBXRange(bxFirst, bxLast);
  etsums->setBXRange(bxFirst, bxLast);

  //producer is responsible for splitting the BXVector into pieces for
  //the firmware to handle
  for(int i = bxFirst; i <= bxLast; ++i)
  {
    //make local inputs
    std::vector<l1t::CaloRegion> *localRegions = new std::vector<l1t::CaloRegion>();
    std::vector<l1t::CaloEmCand> *localEmCands = new std::vector<l1t::CaloEmCand>();

    //make local outputs
    std::vector<l1t::EGamma> *localEGammas = new std::vector<l1t::EGamma>();
    std::vector<l1t::Tau> *localTaus = new std::vector<l1t::Tau>();
    std::vector<l1t::Jet> *localJets = new std::vector<l1t::Jet>();
    std::vector<l1t::EtSum> *localEtSums = new std::vector<l1t::EtSum>();

    // copy over the inputs -> there must be a better way to do this
    for(std::vector<l1t::CaloRegion>::const_iterator region = caloRegions->begin(i);
	region != caloRegions->end(i); ++region)
      localRegions->push_back(*region);
    for(std::vector<l1t::CaloEmCand>::const_iterator emcand = caloEmCands->begin(i);
	emcand != caloEmCands->end(i); ++emcand)
      localEmCands->push_back(*emcand);

    //run the firmware on one event
    m_fw->processEvent(*localEmCands, *localRegions,
		       localEGammas, localTaus, localJets, localEtSums);

    // copy the output into the BXVector -> there must be a better way
    for(std::vector<l1t::EGamma>::const_iterator eg = localEGammas->begin(); eg != localEGammas->end(); ++eg)
      egammas->push_back(i, *eg);
    for(std::vector<l1t::Tau>::const_iterator tau = localTaus->begin(); tau != localTaus->end(); ++tau)
      taus->push_back(i, *tau);
    for(std::vector<l1t::Jet>::const_iterator jet = localJets->begin(); jet != localJets->end(); ++jet)
      jets->push_back(i, *jet);
    for(std::vector<l1t::EtSum>::const_iterator etsum = localEtSums->begin(); etsum != localEtSums->end(); ++etsum)
      etsums->push_back(i, *etsum);

    delete localRegions;
    delete localEmCands;
    delete localEGammas;
    delete localTaus;
    delete localJets;
    delete localEtSums;

  }


  iEvent.put(egammas);
  iEvent.put(taus);
  iEvent.put(jets);
  iEvent.put(etsums);

}

// ------------ method called once each job just before starting event loop ------------
void
Stage1Layer2Producer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop ------------
void
Stage1Layer2Producer::endJob() {
}

// ------------ method called when starting to processes a run ------------

void Stage1Layer2Producer::beginRun(Run const&iR, EventSetup const&iE){

  unsigned long long id = iE.get<L1TCaloParamsRcd>().cacheIdentifier();

  if (id != m_paramsCacheId) {

    m_paramsCacheId = id;

    edm::ESHandle<CaloParams> paramsHandle;
    iE.get<L1TCaloParamsRcd>().get(paramsHandle);

    // replace our local copy of the parameters with a new one using placement new
    m_params->~CaloParamsStage1();
    m_params = new (m_params) CaloParamsStage1(*paramsHandle.product());

    LogDebug("L1TDebug") << *m_params << std::endl;

    if (! m_params){
      edm::LogError("l1t|caloStage1") << "Could not retrieve params from Event Setup" << std::endl;
    }

  }


  m_params->setRegionETCutForHT(regionETCutForHT);
  m_params->setRegionETCutForMET(regionETCutForMET);
  m_params->setMinGctEtaForSums(minGctEtaForSums);
  m_params->setMaxGctEtaForSums(maxGctEtaForSums);
  m_params->setEgRelativeJetIsolationCut(egRelativeJetIsolationCut);
  m_params->setTauRelativeJetIsolationCut(tauRelativeJetIsolationCut);

  LogDebug("l1t|stage 1 jets") << "Stage1Layer2Producer::beginRun function called...\n";

  //get the proper scales for conversion to physical et AND gt scales
  edm::ESHandle< L1CaloEtScale > emScale ;
  iE.get< L1EmEtScaleRcd >().get( emScale ) ;
  m_params->setEmScale(*emScale);

  edm::ESHandle< L1CaloEtScale > jetScale ;
  iE.get< L1JetEtScaleRcd >().get( jetScale ) ;
  m_params->setJetScale(*jetScale);

  edm::ESHandle< L1CaloEtScale > HtMissScale;
  iE.get< L1HtMissScaleRcd >().get( HtMissScale ) ;
  m_params->setHtMissScale(*HtMissScale);

  //not sure if I need this one
  edm::ESHandle< L1CaloEtScale > HfRingScale;
  iE.get< L1HfRingEtScaleRcd >().get( HfRingScale );
  m_params->setHfRingScale(*HfRingScale);


  //m_params->setEgLsb(emScale->linearLsb());
  //m_params->setJetLsb(jetScale->linearLsb());

  //unsigned long long id = iE.get<CaloParamsRcd>().cacheIdentifier();

  //if (id != m_paramsCacheId)
  { // Need to update:
    //m_paramsCacheId = id;

    //ESHandle<CaloParams> parameters;
    //iE.get<CaloParamsRcd>().get(parameters);

    // LenA move the setting of the firmware version to the Stage1Layer2Producer constructor

    //m_params = boost::shared_ptr<const CaloParams>(parameters.product());
    //m_fwv = boost::shared_ptr<const FirmwareVersion>(new FirmwareVersion());
    //printf("Begin.\n");
    //m_fwv = boost::shared_ptr<FirmwareVersion>(new FirmwareVersion()); //not const during testing
    //printf("Success m_fwv.\n");
    //m_fwv->setFirmwareVersion(1); //hardcode for now, 1=HI, 2=PP
    //printf("Success m_fwv version set.\n");

    // if (! m_params){
    //   LogError("l1t|stage 1 jets") << "Stage1Layer2Producer: could not retreive DB params from Event Setup\n";
    // }

    // Set the current algorithm version based on DB pars from database:
    //m_fw = m_factory.create(*m_fwv /*,*m_params*/);
    //printf("Success create.\n");

    //if (! m_fw) {
    //  // we complain here once per run
    //  LogError("l1t|stage 1 jets") << "Stage1Layer2Producer: firmware could not be configured.\n";
    //}
  }


}

// ------------ method called when ending the processing of a run ------------
void Stage1Layer2Producer::endRun(Run const& iR, EventSetup const& iE){

}


// ------------ method fills 'descriptions' with the allowed parameters for the module ------------
void
Stage1Layer2Producer::fillDescriptions(ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

} // namespace

//define this as a plug-in
DEFINE_FWK_MODULE(l1t::Stage1Layer2Producer);
