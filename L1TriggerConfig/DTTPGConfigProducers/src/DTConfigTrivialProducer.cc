#include "L1TriggerConfig/DTTPGConfigProducers/src/DTConfigTrivialProducer.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include <ostream>

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
DTConfigTrivialProducer::DTConfigTrivialProducer(const edm::ParameterSet& ps)
{
 
  //the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this);

  //now do what ever other initialization is needed
  
  //get and store parameter set 
  m_ps = ps;
  m_manager = new DTConfigManager();
}


DTConfigTrivialProducer::~DTConfigTrivialProducer()
{
 
   // do anything here that needs to be done at destruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//
// ------------ method called to produce the data  ------------
std::auto_ptr<DTConfigManager> DTConfigTrivialProducer::produce (const DTConfigManagerRcd& iRecord)
{
   using namespace edm::es;

   buildManager();

   std::auto_ptr<DTConfigManager> dtConfig = std::auto_ptr<DTConfigManager>( m_manager );

   return dtConfig ;
}

void DTConfigTrivialProducer::buildManager()
{

  //create config classes&C.
  edm::ParameterSet conf_ps = m_ps.getParameter<edm::ParameterSet>("DTTPGParameters");
  edm::ParameterSet conf_map = m_ps.getUntrackedParameter<edm::ParameterSet>("DTTPGMap");
  bool dttpgdebug = conf_ps.getUntrackedParameter<bool>("Debug");
  DTConfigSectColl sectcollconf(conf_ps.getParameter<edm::ParameterSet>("SectCollParameters"));
  edm::ParameterSet tups = conf_ps.getParameter<edm::ParameterSet>("TUParameters");
  DTConfigBti bticonf(tups.getParameter<edm::ParameterSet>("BtiParameters"));
  DTConfigTraco tracoconf(tups.getParameter<edm::ParameterSet>("TracoParameters"));
  DTConfigTSTheta tsthetaconf(tups.getParameter<edm::ParameterSet>("TSThetaParameters"));
  DTConfigTSPhi tsphiconf(tups.getParameter<edm::ParameterSet>("TSPhiParameters"));
  DTConfigTrigUnit trigunitconf(tups);
  
  m_manager->setDTTPGDebug(dttpgdebug);

  for (int iwh=-2;iwh<=2;++iwh){
    for (int ist=1;ist<=4;++ist){
      for (int ise=1;ise<=12;++ise){
	DTChamberId chambid(iwh,ist,ise);
	std::ostringstream os;
	os << "wh" << chambid.wheel() << "st" << chambid.station() << "se" << chambid.sector();
	vector<int> nmap = conf_map.getUntrackedParameter<vector<int> >(os.str().c_str());
//      std::cout << "  untracked vint32 wh" << chambid.wheel()
// 		  << "st" << chambid.station()
// 		  << "se" << chambid.sector() << " = { ";
	if(dttpgdebug)
	  {
	    std::cout << " Filling configuration for chamber : wh " << chambid.wheel() << 
	      ", st " << chambid.station() << 
	      ", se " << chambid.sector() << endl;
	  }
	
	//fill the bti map
	for (int isl=1;isl<=3;isl++){
	  int ncell = nmap[isl-1];
	  //	  std::cout << ncell <<" , ";
	  for (int ibti=0;ibti<ncell;ibti++)
	    {
	      m_manager->setDTConfigBti(DTBtiId(chambid,isl,ibti+1),bticonf);
	      if(dttpgdebug)
		std::cout << "Filling BTI config for chamber : wh " << chambid.wheel() << 
		  ", st " << chambid.station() << 
		  ", se " << chambid.sector() << 
		  "... sl " << isl << 
		  ", bti " << ibti+1 << endl;
	    }     
	}
	
	// fill the traco map
	int ntraco = nmap[3];
	//std::cout << ntraco << " }" << std::endl;
	for (int itraco=0;itraco<ntraco;itraco++)
	  { 
	    m_manager->setDTConfigTraco(DTTracoId(chambid,itraco+1),tracoconf);
	    if(dttpgdebug)
	      std::cout << "Filling TRACO config for chamber : wh " << chambid.wheel() << 
		", st " << chambid.station() << 
		", se " << chambid.sector() << 
		", traco " << itraco+1 << endl;
	  }     
	
	// fill TS & TrigUnit
	m_manager->setDTConfigTSTheta(chambid,tsthetaconf);
	m_manager->setDTConfigTSPhi(chambid,tsphiconf);
	m_manager->setDTConfigTrigUnit(chambid,trigunitconf);
	
      }
    }
  }

  for (int iwh=-2;iwh<=2;++iwh){
    for (int ise=13;ise<=14;++ise){
      int ist =4;
      DTChamberId chambid(iwh,ist,ise);
      std::ostringstream os;
      os << "wh" << chambid.wheel() << "st" << chambid.station() << "se" << chambid.sector();
      vector<int> nmap = conf_map.getUntrackedParameter<vector<int> >(os.str().c_str());
//       std::cout << "  untracked vint32 wh" << chambid.wheel()
// 		<< "st" << chambid.station()
// 		<< "se" << chambid.sector() << " = { ";
      if(dttpgdebug)
	{
	  std::cout << " Filling configuration for chamber : wh " << chambid.wheel() << 
	    ", st " << chambid.station() << 
	    ", se " << chambid.sector() << endl;
	}
      
      //fill the bti map
      for (int isl=1;isl<=3;isl++){
	int ncell = nmap[isl-1];
// 	std::cout << ncell <<" , ";
	for (int ibti=0;ibti<ncell;ibti++)
	  {
	    m_manager->setDTConfigBti(DTBtiId(chambid,isl,ibti+1),bticonf);
	    if(dttpgdebug)
	      std::cout << "Filling BTI config for chamber : wh " << chambid.wheel() << 
		", st " << chambid.station() << 
		", se " << chambid.sector() << 
		"... sl " << isl << 
		", bti " << ibti+1 << endl;
	  }     
      }
      
      // fill the traco map
      int ntraco = nmap[3];
//       std::cout << ntraco << " }" << std::endl;
      for (int itraco=0;itraco<ntraco;itraco++)
	{ 
	  m_manager->setDTConfigTraco(DTTracoId(chambid,itraco+1),tracoconf);
	  if(dttpgdebug)
	    std::cout << "Filling TRACO config for chamber : wh " << chambid.wheel() << 
	      ", st " << chambid.station() << 
	      ", se " << chambid.sector() << 
	      ", traco " << itraco+1 << endl;
	}     
      
      // fill TS & TrigUnit
      m_manager->setDTConfigTSTheta(chambid,tsthetaconf);
      m_manager->setDTConfigTSPhi(chambid,tsphiconf);
      m_manager->setDTConfigTrigUnit(chambid,trigunitconf);
      
    }
  }
  
  //loop on Sector Collectors
  for (int wh=-2;wh<=2;wh++)
    for (int se=1;se<=12;se++)
      m_manager->setDTConfigSectColl(DTSectCollId(wh,se),sectcollconf);

}
