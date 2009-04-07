#include <CLHEP/Random/RandomEngine.h>

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "GeneratorInterface/CosmicMuonGenerator/interface/CosMuoGenProducer.h"

edm::CosMuoGenProducer::CosMuoGenProducer( const ParameterSet & pset ) :
  //RanS(pset.getParameter<int>("RanSeed", 123456)), //get seed now from Framework
  MinP(pset.getParameter<double>("MinP")),
  MinP_CMS(pset.getParameter<double>("MinP_CMS")),
  MaxP(pset.getParameter<double>("MaxP")),
  MinT(pset.getParameter<double>("MinTheta")),
  MaxT(pset.getParameter<double>("MaxTheta")),
  MinPh(pset.getParameter<double>("MinPhi")),
  MaxPh(pset.getParameter<double>("MaxPhi")),
  MinS(pset.getParameter<double>("MinT0")),
  MaxS(pset.getParameter<double>("MaxT0")),
  ELSF(pset.getParameter<double>("ElossScaleFactor")),
  RTarget(pset.getParameter<double>("RadiusOfTarget")),
  ZTarget(pset.getParameter<double>("ZDistOfTarget")),
  ZCTarget(pset.getParameter<double>("ZCentrOfTarget")),
  TrackerOnly(pset.getParameter<bool>("TrackerOnly")),
  TIFOnly_constant(pset.getParameter<bool>("TIFOnly_constant")),
  TIFOnly_linear(pset.getParameter<bool>("TIFOnly_linear")),
  MTCCHalf(pset.getParameter<bool>("MTCCHalf")),
  PlugVtx(pset.getParameter<double>("PlugVx")),
  PlugVtz(pset.getParameter<double>("PlugVz")),
  MinEn(pset.getParameter<double>("MinEnu")),
  MaxEn(pset.getParameter<double>("MaxEnu")),
  AllMu(pset.getParameter<bool>("AcptAllMu")),
  extCrossSect(pset.getUntrackedParameter<double>("crossSection", -1.)),
  extFilterEff(pset.getUntrackedParameter<double>("filterEfficiency", -1.)),
  cmVerbosity_(pset.getParameter<bool>("Verbosity"))
  {
    //if not specified (i.e. negative) then use MinP also for MinP_CMS
    if(MinP_CMS < 0) MinP_CMS = MinP;

    edm::Service<RandomNumberGenerator> rng;
    if (!rng.isAvailable())
      throw cms::Exception("Configuration")
         << "The RandomNumberProducer module requires the RandomNumberGeneratorService\n"
            "which appears to be absent.  Please add that service to your configuration\n"
            "or remove the modules that require it." << std::endl;

    // set up the generator
    CosMuoGen = new CosmicMuonGenerator();
// Begin JMM change
//  CosMuoGen->setNumberOfEvents(numberEventsInRun());
    CosMuoGen->setNumberOfEvents(999999999);
// End of JMM change
    CosMuoGen->setRanSeed(RanS);
    CosMuoGen->setMinP(MinP);
    CosMuoGen->setMinP_CMS(MinP_CMS);
    CosMuoGen->setMaxP(MaxP);
    CosMuoGen->setMinTheta(MinT);
    CosMuoGen->setMaxTheta(MaxT);
    CosMuoGen->setMinPhi(MinPh);
    CosMuoGen->setMaxPhi(MaxPh);
    CosMuoGen->setMinT0(MinS);
    CosMuoGen->setMaxT0(MaxS);
    CosMuoGen->setElossScaleFactor(ELSF);
    CosMuoGen->setRadiusOfTarget(RTarget);
    CosMuoGen->setZDistOfTarget(ZTarget);
    CosMuoGen->setZCentrOfTarget(ZCTarget);
    CosMuoGen->setTrackerOnly(TrackerOnly);
    CosMuoGen->setTIFOnly_constant(TIFOnly_constant);
    CosMuoGen->setTIFOnly_linear(TIFOnly_linear);
    CosMuoGen->setMTCCHalf(MTCCHalf);
    CosMuoGen->setPlugVx(PlugVtx);
    CosMuoGen->setPlugVz(PlugVtz);    
    CosMuoGen->setMinEnu(MinEn);
    CosMuoGen->setMaxEnu(MaxEn);    
    CosMuoGen->setAcptAllMu(AllMu);
    CosMuoGen->initialize(&rng->getEngine());
    produces<HepMCProduct>();
    produces<GenEventInfoProduct>();
    produces<GenRunInfoProduct, edm::InRun>();
  }

edm::CosMuoGenProducer::~CosMuoGenProducer(){
  //CosMuoGen->terminate();
  delete CosMuoGen;
  //  delete fEvt;
  clear();
}

void edm::CosMuoGenProducer::endRun( Run &run, const EventSetup& es )
{
  std::auto_ptr<GenRunInfoProduct> genRunInfo(new GenRunInfoProduct());

  double cs = CosMuoGen->getRate(); // flux in Hz, not s^-1m^-2
  genRunInfo->setInternalXSec(cs);
  genRunInfo->setExternalXSecLO(extCrossSect);
  genRunInfo->setFilterEfficiency(extFilterEff);

  run.put(genRunInfo);

  CosMuoGen->terminate();
}

void edm::CosMuoGenProducer::clear(){}

void edm::CosMuoGenProducer::produce(Event &e, const edm::EventSetup &es)
{  
  // generate event
  CosMuoGen->nextEvent();

  // delete and re-create fEvt (memory)
  // delete fEvt;
  fEvt = new HepMC::GenEvent();
  HepMC::GenVertex* Vtx = new  HepMC::GenVertex(HepMC::FourVector(CosMuoGen->OneMuoEvt.vx(),
								  CosMuoGen->OneMuoEvt.vy(),
								  CosMuoGen->OneMuoEvt.vz(),
								  CosMuoGen->OneMuoEvt.t0()));
  HepMC::FourVector p_in(CosMuoGen->OneMuoEvt.px_in(),CosMuoGen->OneMuoEvt.py_in(),CosMuoGen->OneMuoEvt.pz_in(),CosMuoGen->OneMuoEvt.e_in());
  HepMC::GenParticle* Part_in = 
    new HepMC::GenParticle(p_in,CosMuoGen->OneMuoEvt.id_in(),3);//Comment mother particle
  Vtx->add_particle_in(Part_in); 

  HepMC::FourVector p(CosMuoGen->OneMuoEvt.px(),CosMuoGen->OneMuoEvt.py(),CosMuoGen->OneMuoEvt.pz(),CosMuoGen->OneMuoEvt.e());
  HepMC::GenParticle* Part = 
    new HepMC::GenParticle(p,CosMuoGen->OneMuoEvt.id(),1); //Final state daughter particle
  Vtx->add_particle_out(Part); 

  fEvt->add_vertex(Vtx);
  fEvt->set_event_number(e.id().event());
  fEvt->set_signal_process_id(13);

  if (cmVerbosity_) fEvt->print();

  std::auto_ptr<HepMCProduct> CMProduct(new HepMCProduct());
  CMProduct->addHepMCData( fEvt );
  e.put(CMProduct);

  std::auto_ptr<GenEventInfoProduct> genEventInfo(new GenEventInfoProduct( fEvt ));
  e.put(genEventInfo);
}
