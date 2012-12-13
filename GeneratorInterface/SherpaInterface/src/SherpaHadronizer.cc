#include <iostream>
#include <sstream>
#include <string>
#include <memory>
#include <stdint.h>


#include "HepMC/GenEvent.h"

#include "SHERPA/Main/Sherpa.H"
#include "ATOOLS/Org/Message.H"
#include "ATOOLS/Math/Random.H"
#include "ATOOLS/Org/Exception.H"
#include "ATOOLS/Org/Run_Parameter.H"
#include "ATOOLS/Org/MyStrStream.H"
//#include "SHERPA/Tools/Input_Output_Handler.H"
//#include "SHERPA/Tools/HepMC2_Interface.H"
#include "ATOOLS/Org/Library_Loader.H"
#include "SHERPA/Single_Events/Event_Handler.H"
#include "AddOns/HepMC/HepMC2_Interface.H"

#include "GeneratorInterface/Core/interface/ParameterCollector.h"
#include "GeneratorInterface/Core/interface/BaseHadronizer.h"
#include "GeneratorInterface/Core/interface/GeneratorFilter.h"
#include "GeneratorInterface/Core/interface/HadronizerFilter.h"
#include "GeneratorInterface/Core/interface/RNDMEngineAccess.h"
#include "GeneratorInterface/SherpaInterface/interface/SherpackFetcher.h"

class SherpaHadronizer : public gen::BaseHadronizer {
public:
  SherpaHadronizer(const edm::ParameterSet &params);
  ~SherpaHadronizer();
  
  bool readSettings( int ) { return true; }
  bool initializeForInternalPartons();
  bool declareStableParticles(const std::vector<int> &pdgIds);
  bool declareSpecialSettings( const std::vector<std::string> ) { return true; }
  void statistics();
  bool generatePartonsAndHadronize();
  bool decay();
  bool residualDecay();
  void finalizeEvent();
  const char *classname() const { return "SherpaHadronizer"; }
  
private:
  
  std::string SherpaProcess;
  std::string SherpaChecksum;
  std::string SherpaPath;
  std::string SherpaPathPiece;
  std::string SherpaResultDir;
  double SherpaDefaultWeight;
  edm::ParameterSet	SherpaParameterSet;
  unsigned int	maxEventsToPrint;
  
  SHERPA::Sherpa Generator;
  CLHEP::HepRandomEngine* randomEngine;
  
};

class CMS_SHERPA_RNG: public ATOOLS::External_RNG {
public:
  CMS_SHERPA_RNG(){randomEngine = &gen::getEngineReference();};
private: 
  CLHEP::HepRandomEngine* randomEngine;
  double Get();
};



SherpaHadronizer::SherpaHadronizer(const edm::ParameterSet &params) :
  BaseHadronizer(params),
  SherpaParameterSet(params.getParameter<edm::ParameterSet>("SherpaParameters"))
{
  if (!params.exists("SherpaProcess")) SherpaProcess="";
	else SherpaProcess=params.getParameter<std::string>("SherpaProcess");
  if (!params.exists("SherpaPath")) SherpaPath="";
    else SherpaPath=params.getParameter<std::string>("SherpaPath");
  if (!params.exists("SherpaPathPiece")) SherpaPathPiece="";
    else SherpaPathPiece=params.getParameter<std::string>("SherpaPathPiece");
  if (!params.exists("SherpaResultDir")) SherpaResultDir="Result";
    else SherpaResultDir=params.getParameter<std::string>("SherpaResultDir");
  if (!params.exists("SherpaDefaultWeight")) SherpaDefaultWeight=1.;
    else SherpaDefaultWeight=params.getParameter<double>("SherpaDefaultWeight");
  if (!params.exists("maxEventsToPrint")) maxEventsToPrint=0;
    else maxEventsToPrint=params.getParameter<int>("maxEventsToPrint");

	
  spf::SherpackFetcher Fetcher(params);
  int retval=Fetcher.Fetch();	
  if (retval != 0) {
	std::cout << "SherpaHadronizer: Preparation of Sherpack failed ... " << std::endl;
	std::cout << "SherpaHadronizer: Error code: " << retval << std::endl;
	std::terminate();  
	  
  }	  
  // The ids (names) of parameter sets to be read (Analysis,Run) to create Analysis.dat, Run.dat
  //They are given as a vstring.  
  std::vector<std::string> setNames = SherpaParameterSet.getParameter<std::vector<std::string> >("parameterSets");
  //Loop all set names...
  for ( unsigned i=0; i<setNames.size(); ++i ) {
    // ...and read the parameters for each set given in vstrings
    std::vector<std::string> pars = SherpaParameterSet.getParameter<std::vector<std::string> >(setNames[i]);
    std::cout << "Write Sherpa parameter set " << setNames[i] <<" to "<<setNames[i]<<".dat "<<std::endl;
    std::string datfile =  SherpaPath + "/" + setNames[i] +".dat";
    std::ofstream os(datfile.c_str());  
    // Loop over all strings and write the according *.dat
    for(std::vector<std::string>::const_iterator itPar = pars.begin(); itPar != pars.end(); ++itPar ) {
      os<<(*itPar)<<std::endl;
    } 
  }

  //To be conform to the default Sherpa usage create a command line:
  //name of executable  (only for demonstration, could also be empty)
  std::string shRun  = "./Sherpa";
  //Path where the Sherpa libraries are stored
  std::string shPath = "PATH=" + SherpaPath;
  // new for Sherpa 1.3.0, suggested by authors
  std::string shPathPiece = "PATH_PIECE=" + SherpaPathPiece; 
  //Path where results are stored 
  std::string shRes  = "RESULT_DIRECTORY=" + SherpaResultDir; // from Sherpa 1.2.0 on
  //Name of the external random number class
  std::string shRng  = "EXTERNAL_RNG=CMS_SHERPA_RNG";
  
  //create the command line
  char* argv[5];
  argv[0]=(char*)shRun.c_str();
  argv[1]=(char*)shPath.c_str();
  argv[2]=(char*)shPathPiece.c_str();
  argv[3]=(char*)shRes.c_str();
  argv[4]=(char*)shRng.c_str();
  
  //initialize Sherpa with the command line
  Generator.InitializeTheRun(5,argv);
}

SherpaHadronizer::~SherpaHadronizer()
{
}

bool SherpaHadronizer::initializeForInternalPartons()
{
  ATOOLS::s_loader->LoadLibrary("SherpaHepMCOutput");
  //initialize Sherpa
  Generator.InitializeTheEventHandler();
  
  return true;
}

#if 0
// naive Sherpa HepMC status fixup //FIXME 
static int getStatus(const HepMC::GenParticle *p)
{
  return status;
}
#endif

//FIXME
bool SherpaHadronizer::declareStableParticles(const std::vector<int> &pdgIds)
{
#if 0
  for(std::vector<int>::const_iterator iter = pdgIds.begin();
      iter != pdgIds.end(); ++iter)
    if (!markStable(*iter))
      return false;
  
  return true;
#else
  return false;
#endif
}


void SherpaHadronizer::statistics()
{
  //calculate statistics
  Generator.SummarizeRun(); 

  //get the xsec from EventHandler
  SHERPA::Event_Handler* theEventHandler = Generator.GetEventHandler();
  double xsec_val = theEventHandler->TotalXS();
  double xsec_err = theEventHandler->TotalErr();
  
  //set the internal cross section in pb in GenRunInfoProduct 
  runInfo().setInternalXSec(GenRunInfoProduct::XSec(xsec_val,xsec_err));

}


bool SherpaHadronizer::generatePartonsAndHadronize()
{
  //get the next event and check if it produced
  if (Generator.GenerateOneEvent()) { 
    //convert it to HepMC2
    //SHERPA::Input_Output_Handler* ioh = Generator.GetIOHandler();
    //get the event weight from blobs
    ATOOLS::Blob_List* blobs = Generator.GetEventHandler()-> GetBlobs();
    ATOOLS::Blob* sp(blobs->FindFirst(ATOOLS::btp::Signal_Process));
    double weight((*sp)["Weight"]->Get<double>());
    double ef((*sp)["Enhance"]->Get<double>());
    double weight_norm((*sp)["Weight_Norm"]->Get<double>());
    // in case of unweighted events sherpa puts the max weight as event weight. 
    // this is not optimal, we want 1 for unweighted events, so we check 
    // whether we are producing unweighted events ("EVENT_GENERATION_MODE" == "1")
    if ( ATOOLS::ToType<int>( ATOOLS::rpa->gen.Variable("EVENT_GENERATION_MODE") ) == 1 ) {
      if (ef > 0.) {
        weight = SherpaDefaultWeight/ef/weight_norm;
      } else {
        weight = -1234.;
      }
    }
    //create and empty event and then hand it to SherpaIOHandler to fill it
    SHERPA::HepMC2_Interface hm2i;
    HepMC::GenEvent* evt = new HepMC::GenEvent();
    hm2i.Sherpa2HepMC(blobs, *evt, weight);
    resetEvent(evt);         
    return true;
  }
  else {
    return false;
  }  
}

bool SherpaHadronizer::decay()
{
	return true;
}

bool SherpaHadronizer::residualDecay()
{
	return true;
}

void SherpaHadronizer::finalizeEvent()
{
#if 0
	for(HepMC::GenEvent::particle_iterator iter = event->particles_begin();
	    iter != event->particles_end(); iter++)
		(*iter)->set_status(getStatus(*iter));
#endif
	//******** Verbosity *******
	if (maxEventsToPrint > 0) {
		maxEventsToPrint--;
		event()->print();		
	}
}

//GETTER for the external random numbers
DECLARE_GETTER(CMS_SHERPA_RNG_Getter,"CMS_SHERPA_RNG",ATOOLS::External_RNG,ATOOLS::RNG_Key);

ATOOLS::External_RNG *CMS_SHERPA_RNG_Getter::operator()(const ATOOLS::RNG_Key &) const
{ return new CMS_SHERPA_RNG(); }

void CMS_SHERPA_RNG_Getter::PrintInfo(std::ostream &str,const size_t) const
{ str<<"CMS_SHERPA_RNG interface"; }

double CMS_SHERPA_RNG::Get(){
   return randomEngine->flat();
   }
   
#include "GeneratorInterface/ExternalDecays/interface/ExternalDecayDriver.h"

typedef edm::GeneratorFilter<SherpaHadronizer, gen::ExternalDecayDriver> SherpaGeneratorFilter;
DEFINE_FWK_MODULE(SherpaGeneratorFilter);
