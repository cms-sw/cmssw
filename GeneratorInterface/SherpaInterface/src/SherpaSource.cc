/*
 *  $Revision: 1.5 $
 *  
 *  Martin Niegel 
 *  niegel@cern.ch
 *  20/03/2007
 * 
 */



#include "SHERPA-MC/Message.H"
#include "SHERPA-MC/prof.hh"
#include "SHERPA-MC/Random.H"
#include "SHERPA-MC/Exception.H"
#include "SHERPA-MC/Run_Parameter.H"
#include "SHERPA-MC/My_Root.H"
#include "SHERPA-MC/Input_Output_Handler.H"

#include "GeneratorInterface/SherpaInterface/interface/SherpaSource.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "CLHEP/Random/RandomEngine.h"
//#include "CLHEP/Random/RandFlat.h"


#include <iostream>
#include "time.h"

using namespace edm;
using namespace std;
using namespace SHERPA;
using namespace HepMC;


// Generator modifications
// ***********************


//HepMC::ConvertHEPEVT conv
// ***********************
//#include "HepMC/IO_Ascii.h"
#include "SHERPA-MC/HepMC2_Interface.H"



//used for defaults
//  static const unsigned long kNanoSecPerSec = 1000000000;
//  static const unsigned long kAveEventPerSec = 200;

SherpaSource::SherpaSource( const ParameterSet & pset, 
			    InputSourceDescription const& desc ) :
  GeneratedInputSource(pset, desc), evt(0),
  libDir_(pset.getUntrackedParameter<string>("libDir","Sherpa_Process")),
  resultDir_(pset.getUntrackedParameter<string>("resultDir","Result"))
  
{

  libDir_    =  pset.getUntrackedParameter<string>("libDir","Sherpa_Process");
  resultDir_ =  pset.getUntrackedParameter<string>("resultDir","Result");

   string shRun  = "./Sherpa";
   string shPath = "PATH=" + libDir_;
   string shRes  = "RESULT_DIRECTORY=" + libDir_ + "/" + resultDir_;
   string shRng  = "EXTERNAL_RNG=CMS_RNG";
   char* argv[4];
   argv[0]=(char*)shRun.c_str();
   argv[1]=(char*)shPath.c_str();
   argv[2]=(char*)shRes.c_str();
   argv[3]=(char*)shRng.c_str();

  set_prof();	
  cout << "SherpaSource: initializing Sherpa. " << endl;
  Generator.InitializeTheRun(4,argv);
  cout << "SherpaSource: InitializeTheRun(argc,argv)" << endl;
  Generator.InitializeTheEventHandler();
  cout << "SherpaSource: InitializeTheEventHandler() " << endl;
  produces<HepMCProduct>();
  cout << "SherpaSource: starting event generation ... " << endl;


}


SherpaSource::~SherpaSource(){
  cout << "SherpaSource: summarizing the run " << endl;
  Generator.SummarizeRun();
  cout << "SherpaSource: event generation done. " << endl;
  clear(); 
}

void SherpaSource::clear() {
 
}


bool SherpaSource::produce(Event & e) {

 auto_ptr<HepMCProduct> bare_product(new HepMCProduct);   

 if (Generator.GenerateOneEvent()) { 
  HepMC::GenEvent* evt = Generator.GetIOHandler()->GetHepMC2Interface()->GenEvent();
  HepMC::GenEvent *copyEvt = new HepMC::GenEvent (*evt);      
	   
  if (evt) bare_product->addHepMCData(copyEvt);   
 
  e.put(bare_product);
  return true;
 }
 else {
   return false;
 } 
}


using namespace ATOOLS;
DECLARE_GETTER(CMS_RNG_Getter,"CMS_RNG",External_RNG,RNG_Key);
External_RNG *CMS_RNG_Getter::operator()(const RNG_Key &) const
{ return new CMS_RNG(); }
void CMS_RNG_Getter::PrintInfo(std::ostream &str,const size_t) const
{ str<<"CMS RNG interface"; }

double CMS_RNG::Get(){
 edm::Service<edm::RandomNumberGenerator> rng;
   if ( ! rng.isAvailable()) {
     throw cms::Exception("Configuration")
       << "SherpaInterface requires the RandomNumberGeneratorService\n"
          "which is not present in the configuration file.  You must add the service\n"
          "in the configuration file or remove the modules that require it.";
   }
//   double rngNumber = RandFlat::shoot(rng->getEngine());
   CLHEP::HepRandomEngine& engine = rng->getEngine();
   double rngNumber = engine.flat();
//   std::cout << "rno: " << rngNumber << std::endl;
   return rngNumber;
}
