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
// constructors and destructor
//
DTConfigTrivialProducer::DTConfigTrivialProducer(const edm::ParameterSet& ps)
{

  setWhatProduced(this);
  
  //get and store parameter set 
  m_ps = ps;
  m_manager = new DTConfigManager();
  m_tpgParams = new DTTPGParameters();

  // set debug
  edm::ParameterSet conf_ps = m_ps.getParameter<edm::ParameterSet>("DTTPGParameters");  
  m_debug = conf_ps.getUntrackedParameter<bool>("Debug");

  if (m_debug) 
    cout << "DTConfigTrivialProducer::DTConfigTrivialProducer()" << endl;

  m_manager->setDTTPGDebug(m_debug);

  // DB specific requests
  bool tracoLutsFromDB = m_ps.getParameter<bool>("TracoLutsFromDB");
  bool useBtiAcceptParam = m_ps.getParameter<bool>("UseBtiAcceptParam");

  // set specific DB requests
  m_manager->setLutFromDB(tracoLutsFromDB);
  m_manager->setUseAcceptParam(useBtiAcceptParam);  // CB Are these needed here???
}


DTConfigTrivialProducer::~DTConfigTrivialProducer()
{

  if (m_debug) 
    cout << "DTConfigTrivialProducer::~DTConfigTrivialProducer()" << endl;
 

}


//
// member functions
//

std::auto_ptr<DTConfigManager> DTConfigTrivialProducer::produce (const DTConfigManagerRcd& iRecord)
{

  if (m_debug) 
    cout << "DTConfigTrivialProducer::produce()" << endl;

   using namespace edm::es;
   buildManager();

   //m_manager->getDTConfigPedestals()->print();

   std::auto_ptr<DTConfigManager> dtConfig = std::auto_ptr<DTConfigManager>( m_manager );

   return dtConfig ;

}

void DTConfigTrivialProducer::buildManager()
{

  if (m_debug) 
    cout << "DTConfigTrivialProducer::buildManager()" << endl;

  //create config classes&C.
  edm::ParameterSet conf_ps = m_ps.getParameter<edm::ParameterSet>("DTTPGParameters");
  edm::ParameterSet conf_map = m_ps.getUntrackedParameter<edm::ParameterSet>("DTTPGMap");
  DTConfigSectColl sectcollconf(conf_ps.getParameter<edm::ParameterSet>("SectCollParameters"));
  edm::ParameterSet tups = conf_ps.getParameter<edm::ParameterSet>("TUParameters");
  DTConfigBti bticonf(tups.getParameter<edm::ParameterSet>("BtiParameters"));
  DTConfigTraco tracoconf(tups.getParameter<edm::ParameterSet>("TracoParameters"));
  DTConfigLUTs lutconf(tups.getParameter<edm::ParameterSet>("LutParameters"));
  DTConfigTSTheta tsthetaconf(tups.getParameter<edm::ParameterSet>("TSThetaParameters"));
  DTConfigTSPhi tsphiconf(tups.getParameter<edm::ParameterSet>("TSPhiParameters"));
  DTConfigTrigUnit trigunitconf(tups);
   
  for (int iwh=-2;iwh<=2;++iwh){
    for (int ist=1;ist<=4;++ist){
      for (int ise=1;ise<=12;++ise){
	DTChamberId chambid(iwh,ist,ise);
	vector<int> nmap = conf_map.getUntrackedParameter<vector<int> >(mapEntryName(chambid).c_str());

	if(m_debug)
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
	      if(m_debug)
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
	    if(m_debug)
	      std::cout << "Filling TRACO config for chamber : wh " << chambid.wheel() << 
		", st " << chambid.station() << 
		", se " << chambid.sector() << 
		", traco " << itraco+1 << endl;
	  }     
	
	// fill TS & TrigUnit
	m_manager->setDTConfigTSTheta(chambid,tsthetaconf);
	m_manager->setDTConfigTSPhi(chambid,tsphiconf);
	m_manager->setDTConfigTrigUnit(chambid,trigunitconf);
	
	// fill LUTs
	m_manager->setDTConfigLUTs(chambid,lutconf);
        m_manager->setLutFromDB(false);    // 110204 SV to be sure to compute luts from geometry         
      }
    }
  }

  for (int iwh=-2;iwh<=2;++iwh){
    for (int ise=13;ise<=14;++ise){
      int ist =4;
      DTChamberId chambid(iwh,ist,ise);
      vector<int> nmap = conf_map.getUntrackedParameter<vector<int> >(mapEntryName(chambid).c_str());

      if(m_debug)
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
	    if(m_debug)
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
	  if(m_debug)
	    std::cout << "Filling TRACO config for chamber : wh " << chambid.wheel() << 
	      ", st " << chambid.station() << 
	      ", se " << chambid.sector() << 
	      ", traco " << itraco+1 << endl;
	}     
      
      // fill TS & TrigUnit
      m_manager->setDTConfigTSTheta(chambid,tsthetaconf);
      m_manager->setDTConfigTSPhi(chambid,tsphiconf);
      m_manager->setDTConfigTrigUnit(chambid,trigunitconf);

      // fill LUTs
      m_manager->setDTConfigLUTs(chambid,lutconf);
      m_manager->setLutFromDB(false);    // 110204 SV to be sure to compute luts from geometry         
    }
  }
  
  //loop on Sector Collectors
  for (int wh=-2;wh<=2;wh++)
    for (int se=1;se<=12;se++)
      m_manager->setDTConfigSectColl(DTSectCollId(wh,se),sectcollconf);

  //fake collection of pedestals
  m_manager->setDTConfigPedestals(buildTrivialPedestals());

}

DTConfigPedestals DTConfigTrivialProducer::buildTrivialPedestals()
{

  int counts = m_ps.getParameter<int>("bxOffset");
  float fine = m_ps.getParameter<double>("finePhase");
   
  if (m_debug) 
    cout << "DTConfigTrivialProducer::buildPedestals()" << endl;

  //DTTPGParameters tpgParams;
  for (int iwh=-2;iwh<=2;++iwh){
    for (int ist=1;ist<=4;++ist){
      for (int ise=1;ise<=14;++ise){
	if (ise>12 && ist!=4) continue;

	DTChamberId chId(iwh,ist,ise);
	m_tpgParams->set(chId,counts,fine,DTTimeUnits::ns);
      }
    }
  }

  DTConfigPedestals tpgPedestals;
  tpgPedestals.setUseT0(false);
  tpgPedestals.setES(m_tpgParams);
 
  return tpgPedestals;

}


std::string DTConfigTrivialProducer::mapEntryName(const DTChamberId & chambid) const
{
  int iwh = chambid.wheel();
  std::ostringstream os;
  os << "wh";
  if (iwh < 0) {
     os << 'm' << -iwh;
   } else {
     os << iwh;
  }
  os << "st" << chambid.station() << "se" << chambid.sector();
  return os.str();
}
