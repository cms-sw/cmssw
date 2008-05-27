/*
 *  $Revision: 0.01 $
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
#include "CLHEP/Random/JamesRandom.h"
#include "CLHEP/Random/RandFlat.h"

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
#include "HepMC/IO_Ascii.h"
#include "SHERPA-MC/HepMC2_Interface.H"



//used for defaults
  static const unsigned long kNanoSecPerSec = 1000000000;
  static const unsigned long kAveEventPerSec = 200;

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
   char* argv[3];
   argv[0]=(char*)shRun.c_str();
   argv[1]=(char*)shPath.c_str();
   argv[2]=(char*)shRes.c_str();
  


  // Set SHERPA parameters in a single ParameterSet
  ParameterSet sherpa_params = 
    pset.getParameter<ParameterSet>("SherpaParameters") ;
  
  // The parameter sets to be read (default, min bias, user ...) in the
  // proper order.
  vector<string> setNames = 
    sherpa_params.getParameter<vector<string> >("parameterSets");



  cout << "----------------------------------------------" << endl;
  cout << "Set Sherpa random number seed " << endl;

  Service<RandomNumberGenerator> rng;
  long seed = (long)(rng->mySeed());
  cout << " seed= " << seed << endl ;
  fRandomEngine = new HepJamesRandom(seed) ;
  fRandomGenerator = new RandFlat(fRandomEngine) ;
  cout << "Internal BaseFlatGunSource is initialzed" << endl ;
  int seed1  = fRandomGenerator->fireInt(0,31328);//allowed random number range for Sherpa 1.0.11
  int seed2  = fRandomGenerator->fireInt(0,30081);//allowed random number range for Sherpa 1.0.11

   cout << " seed1= " << seed1 << endl ;
   cout << " seed2= " << seed2 << endl ;

  cout << "----------------------------------------------" << endl;
  // Loop over the sets
  for ( unsigned i=0; i<setNames.size(); ++i ) {
    
    string mySet = setNames[i];
    
    // Read the SHERPA parameters for each set of parameters
    vector<string> pars = 
      sherpa_params.getParameter<vector<string> >(mySet);
    
   
    cout << "Write Sherpa parameter set " << mySet <<" to "<<mySet<<".dat "<<endl;
    
    string datfile =  libDir_ + "/" + mySet+".dat";
   
        std::ofstream os(datfile.c_str());
    
    // Loop over all parameters and stop in case of mistake
    for( vector<string>::const_iterator  
	   itPar = pars.begin(); itPar != pars.end(); ++itPar ) {
           os<<(*itPar)<<endl;
    }
       if(mySet=="Run") os<<"RANDOM_SEED = "<<seed1<<" "<<seed2<<endl;  
       //causes warnings if only few events are generated !!
  }

    cout << "----------------------------------------------" << endl;
   set_prof();	
 

  cout << "SherpaSource: initializing Sherpa. " << endl;
  Generator.InitializeTheRun(3,argv);
 cout << "SherpaSource: InitializeTheRun(argc,argv)" << endl;
  Generator.InitializeTheEventHandler();
  cout << "SherpaSource: InitializeTheEventHandler() " << endl;
 produces<HepMCProduct>();
  cout << "SherpaSource: starting event generation ... " << endl;

      msg_Out()<<"=========================================================================="<<std::endl
                       <<"Sherpa will start event generation now : "<<std::endl               
                       <<"=========================================================================="<<std::endl;
 

}


SherpaSource::~SherpaSource(){
  cout << "SherpaSource: event generation done. " << endl;
  clear(); 
}

void SherpaSource::clear() {
 
}


bool SherpaSource::produce(Event & e) {

 auto_ptr<HepMCProduct> bare_product(new HepMCProduct);   

   int i = numberEventsInRun() - remainingEvents() ;
   if (i%100==0 && i!=0) {

  cout<<"numberEventsInRun() :"<<numberEventsInRun() <<endl;
  cout<<"remainingEvents()   :"<< remainingEvents()<<endl;
  cout<<"Sherpa : Passed "<<i<<" events."<<std::endl;
 std::cout <<" ================================== " << i <<std::endl; 
 
   }


 if (Generator.GenerateOneEvent()){ 

  HepMC::GenEvent* evt = Generator.GetIOHandler()->GetHepMC2Interface()->GenEvent();
  HepMC::GenEvent *copyEvt = new HepMC::GenEvent (*evt);      
	   
  if (evt)  bare_product->addHepMCData(copyEvt);   
 
  e.put(bare_product);
  return true;
 }
 else {

   return false;
 } 
}
