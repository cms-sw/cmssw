
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ServiceRegistry/interface/RandomEngineSentry.h"

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
  MultiMuon(pset.getParameter<bool>("MultiMuon")),
  MultiMuonFileName(pset.getParameter<std::string>("MultiMuonFileName")),
  MultiMuonFileFirstEvent(pset.getParameter<int>("MultiMuonFileFirstEvent")),
  MultiMuonNmin(pset.getParameter<int>("MultiMuonNmin")),
  TIFOnly_constant(pset.getParameter<bool>("TIFOnly_constant")),
  TIFOnly_linear(pset.getParameter<bool>("TIFOnly_linear")),
  MTCCHalf(pset.getParameter<bool>("MTCCHalf")),
  PlugVtx(pset.getParameter<double>("PlugVx")),
  PlugVtz(pset.getParameter<double>("PlugVz")),
  VarRhoAir(pset.getParameter<double>("RhoAir")),
  VarRhoWall(pset.getParameter<double>("RhoWall")),
  VarRhoRock(pset.getParameter<double>("RhoRock")),
  VarRhoClay(pset.getParameter<double>("RhoClay")),
  VarRhoPlug(pset.getParameter<double>("RhoPlug")),
  ClayLayerWidth(pset.getParameter<double>("ClayWidth")),
  MinEn(pset.getParameter<double>("MinEnu")),
  MaxEn(pset.getParameter<double>("MaxEnu")),
  NuPrdAlt(pset.getParameter<double>("NuProdAlt")),
  AllMu(pset.getParameter<bool>("AcptAllMu")),
  extCrossSect(pset.getUntrackedParameter<double>("crossSection", -1.)),
  extFilterEff(pset.getUntrackedParameter<double>("filterEfficiency", -1.)),
  cmVerbosity_(pset.getParameter<bool>("Verbosity")),
  isInitialized_(false)
  {
    //if not specified (i.e. negative) then use MinP also for MinP_CMS
    if(MinP_CMS < 0) MinP_CMS = MinP;

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
    CosMuoGen->setMultiMuon(MultiMuon);
    CosMuoGen->setMultiMuonFileName(MultiMuonFileName);
    CosMuoGen->setMultiMuonFileFirstEvent(MultiMuonFileFirstEvent);
    CosMuoGen->setMultiMuonNmin(MultiMuonNmin);
    CosMuoGen->setTIFOnly_constant(TIFOnly_constant);
    CosMuoGen->setTIFOnly_linear(TIFOnly_linear);
    CosMuoGen->setMTCCHalf(MTCCHalf);
    CosMuoGen->setPlugVx(PlugVtx);
    CosMuoGen->setPlugVz(PlugVtz);    
    CosMuoGen->setRhoAir(VarRhoAir);
    CosMuoGen->setRhoWall(VarRhoWall);
    CosMuoGen->setRhoRock(VarRhoRock);
    CosMuoGen->setRhoClay(VarRhoClay);
    CosMuoGen->setRhoPlug(VarRhoPlug);
    CosMuoGen->setClayWidth(ClayLayerWidth);
    CosMuoGen->setMinEnu(MinEn);
    CosMuoGen->setMaxEnu(MaxEn);    
    CosMuoGen->setNuProdAlt(NuPrdAlt);
    CosMuoGen->setAcptAllMu(AllMu);
    produces<HepMCProduct>("unsmeared");
    produces<GenEventInfoProduct>();
    produces<GenRunInfoProduct, edm::InRun>();
  }

edm::CosMuoGenProducer::~CosMuoGenProducer(){
  //CosMuoGen->terminate();
  delete CosMuoGen;
  //  delete fEvt;
  clear();
}

void edm::CosMuoGenProducer::beginLuminosityBlock(LuminosityBlock const& lumi, EventSetup const&)
{
  if(!isInitialized_) {
    isInitialized_ = true;
    RandomEngineSentry<CosmicMuonGenerator> randomEngineSentry(CosMuoGen, lumi.index());
    CosMuoGen->initialize(randomEngineSentry.randomEngine());
  }
}

void edm::CosMuoGenProducer::endRunProduce( Run &run, const EventSetup& es )
{
  std::auto_ptr<GenRunInfoProduct> genRunInfo(new GenRunInfoProduct());

  double cs = CosMuoGen->getRate(); // flux in Hz, not s^-1m^-2
  if (MultiMuon) genRunInfo->setInternalXSec(0.);
  else genRunInfo->setInternalXSec(cs);
  genRunInfo->setExternalXSecLO(extCrossSect);
  genRunInfo->setFilterEfficiency(extFilterEff);

  run.put(genRunInfo);

  CosMuoGen->terminate();
}

void edm::CosMuoGenProducer::clear(){}

void edm::CosMuoGenProducer::produce(Event &e, const edm::EventSetup &es)
{  
  RandomEngineSentry<CosmicMuonGenerator> randomEngineSentry(CosMuoGen, e.streamID());

  // generate event
  if (!MultiMuon) {
    CosMuoGen->nextEvent();
  }
  else {
    bool success = CosMuoGen->nextMultiEvent();
    if (!success) std::cout << "CosMuoGenProducer.cc: CosMuoGen->nextMultiEvent() failed!" 
			    << std::endl;
  }

  if (Debug) {
    std::cout << "CosMuoGenProducer.cc: CosMuoGen->EventWeight=" << CosMuoGen->EventWeight 
	      << "  CosMuoGen: Nmuons=" << CosMuoGen->Id_sf.size() << std::endl; 
    std::cout << "CosMuoGen->Id_at=" << CosMuoGen->Id_at
	      << "  CosMuoGen->Vx_at=" << CosMuoGen->Vx_at 
	      << "  CosMuoGen->Vy_at=" << CosMuoGen->Vy_at
	      << "  CosMuoGen->Vz_at=" << CosMuoGen->Vz_at 
	      << "  CosMuoGen->T0_at=" << CosMuoGen->T0_at << std::endl;
    std::cout << "  Px=" << CosMuoGen->Px_at
	      << "  Py=" << CosMuoGen->Py_at
	      << "  Pz=" << CosMuoGen->Pz_at << std::endl;
    for (unsigned int i=0; i<CosMuoGen->Id_sf.size(); ++i) {
      std::cout << "Id_sf[" << i << "]=" << CosMuoGen->Id_sf[i]
		<< "  Vx_sf[" << i << "]=" << CosMuoGen->Vx_sf[i]
		<< "  Vy_sf=" << CosMuoGen->Vy_sf[i]
		<< "  Vz_sf=" << CosMuoGen->Vz_sf[i]
		<< "  T0_sf=" << CosMuoGen->T0_sf[i]
		<< "  Px_sf=" << CosMuoGen->Px_sf[i]
		<< "  Py_sf=" << CosMuoGen->Py_sf[i]
		<< "  Pz_sf=" << CosMuoGen->Pz_sf[i] << std::endl;
      std::cout << "phi_sf=" << atan2(CosMuoGen->Px_sf[i],CosMuoGen->Pz_sf[i]) << std::endl;
      std::cout << "Id_ug[" << i << "]=" << CosMuoGen->Id_ug[i] 
		<< "  Vx_ug[" << i << "]=" << CosMuoGen->Vx_ug[i] 
		<< "  Vy_ug=" << CosMuoGen->Vy_ug[i]
		<< "  Vz_ug=" << CosMuoGen->Vz_ug[i]
		<< "  T0_ug=" << CosMuoGen->T0_ug[i]
		<< "  Px_ug=" << CosMuoGen->Px_ug[i]
		<< "  Py_ug=" << CosMuoGen->Py_ug[i]
		<< "  Pz_ug=" << CosMuoGen->Pz_ug[i] << std::endl;
      std::cout << "phi_ug=" << atan2(CosMuoGen->Px_ug[i],CosMuoGen->Pz_ug[i]) << std::endl;;
    }
  }


  fEvt = new HepMC::GenEvent();
  
  HepMC::GenVertex* Vtx_at = new  HepMC::GenVertex(HepMC::FourVector(CosMuoGen->Vx_at, //[mm]
  							     CosMuoGen->Vy_at, //[mm]
  							     CosMuoGen->Vz_at, //[mm]
  							     CosMuoGen->T0_at)); //[mm]
  //cout << "CosMuoGenProducer.cc: Vy_at=" << CosMuoGen->Vy_at << endl;
  HepMC::FourVector p_at(CosMuoGen->Px_at,CosMuoGen->Py_at,CosMuoGen->Pz_at,CosMuoGen->E_at);
  HepMC::GenParticle* Part_at =
    new HepMC::GenParticle(p_at,CosMuoGen->Id_at, 3);//Comment mother particle in
  Vtx_at->add_particle_in(Part_at);


  //loop here in case of multi muon events (else just one iteration)
  for (unsigned int i=0; i<CosMuoGen->Id_sf.size(); ++i) {

    HepMC::FourVector p_sf(CosMuoGen->Px_sf[i],CosMuoGen->Py_sf[i],CosMuoGen->Pz_sf[i],CosMuoGen->E_sf[i]);
    HepMC::GenParticle* Part_sf_in =
      new HepMC::GenParticle(p_sf,CosMuoGen->Id_sf[i], 3); //Comment daughter particle
    Vtx_at->add_particle_out(Part_sf_in);
    
    HepMC::GenVertex* Vtx_sf = new HepMC::GenVertex(HepMC::FourVector(CosMuoGen->Vx_sf[i],                             CosMuoGen->Vy_sf[i], CosMuoGen->Vz_sf[i], CosMuoGen->T0_sf[i])); //[mm]
    HepMC::GenParticle* Part_sf_out =
      new HepMC::GenParticle(p_sf,CosMuoGen->Id_sf[i], 3); //Comment daughter particle
    
    Vtx_sf->add_particle_in(Part_sf_in);
    Vtx_sf->add_particle_out(Part_sf_out);
    
    fEvt->add_vertex(Vtx_sf); //one per muon

    HepMC::GenVertex* Vtx_ug = new HepMC::GenVertex(HepMC::FourVector(CosMuoGen->Vx_ug[i],                             CosMuoGen->Vy_ug[i], CosMuoGen->Vz_ug[i], CosMuoGen->T0_ug[i])); //[mm]
    
    HepMC::FourVector p_ug(CosMuoGen->Px_ug[i],CosMuoGen->Py_ug[i],CosMuoGen->Pz_ug[i],CosMuoGen->E_ug[i]);
    HepMC::GenParticle* Part_ug =
      new HepMC::GenParticle(p_ug,CosMuoGen->Id_ug[i], 1);//Final state daughter particle

    Vtx_ug->add_particle_in(Part_sf_out);
    Vtx_ug->add_particle_out(Part_ug);

    fEvt->add_vertex(Vtx_ug); //one per muon

  }

  fEvt->add_vertex(Vtx_at);
  fEvt->set_signal_process_vertex(Vtx_at);

  fEvt->set_event_number(e.id().event());
  fEvt->set_signal_process_id(13);

  fEvt->weights().push_back( CosMuoGen->EventWeight ); // just one event weight 
  fEvt->weights().push_back( CosMuoGen->Trials ); // int Trials number (unweighted) 


  if (cmVerbosity_) fEvt->print();

  std::auto_ptr<HepMCProduct> CMProduct(new HepMCProduct());
  CMProduct->addHepMCData( fEvt );
  e.put(CMProduct, "unsmeared");

  std::auto_ptr<GenEventInfoProduct> genEventInfo(new GenEventInfoProduct( fEvt ));
  e.put(genEventInfo);

}
