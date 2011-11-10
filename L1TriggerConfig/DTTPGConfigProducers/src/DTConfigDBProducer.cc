#include "L1TriggerConfig/DTTPGConfigProducers/src/DTConfigDBProducer.h"

#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"

#include "CondFormats/DataRecord/interface/DTT0Rcd.h"
#include "CondFormats/DTObjects/interface/DTCCBConfig.h"
#include "CondFormats/DataRecord/interface/DTCCBConfigRcd.h"
#include "CondFormats/DataRecord/interface/DTTPGParametersRcd.h"
#include "CondFormats/DTObjects/interface/DTKeyedConfig.h"
#include "CondFormats/DataRecord/interface/DTKeyedConfigListRcd.h"
#include "CondFormats/DTObjects/interface/DTConfigAbstractHandler.h"

#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigManagerRcd.h"
#include "L1TriggerConfig/DTTPGConfigProducers/src/DTPosNegType.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <iomanip>

using std::cout;
using std::endl;
using std::vector;
using std::auto_ptr;

//
// constructors and destructor
//

DTConfigDBProducer::DTConfigDBProducer(const edm::ParameterSet& p)
{
  // tell the framework what record is being produced
  setWhatProduced(this,&DTConfigDBProducer::produce);

  cfgConfig = p.getParameter<bool>("cfgConfig");
    
  // get and store parameter set and config manager pointer
  m_ps = p;
  m_manager = new DTConfigManager();
  
  // debug flags
  m_debugDB    = p.getParameter<bool>("debugDB"); 
  m_debugBti   = p.getParameter<int>("debugBti");
  m_debugTraco = p.getParameter<int>("debugTraco");
  m_debugTSP   = p.getParameter<bool>("debugTSP");
  m_debugTST   = p.getParameter<bool>("debugTST");
  m_debugTU    = p.getParameter<bool>("debugTU");
  m_debugSC    = p.getParameter<bool>("debugSC");
  m_debugLUTs  = p.getParameter<bool>("debugLUTs");
  m_debugPed   = p.getParameter<bool>("debugPed");

  m_UseT0 = p.getParameter<bool>("UseT0");  // CB check for a better way to do it

  // DB specific requests
  bool tracoLutsFromDB   = p.getParameter<bool>("TracoLutsFromDB");
  bool useBtiAcceptParam = p.getParameter<bool>("UseBtiAcceptParam");

  // set specific DB requests
  m_manager->setLutFromDB(tracoLutsFromDB);
  m_manager->setUseAcceptParam(useBtiAcceptParam);

  // initialize flags to check if data are present in OMDS 
  flagDBBti 	= false;
  flagDBTraco 	= false;
  flagDBTSS 	= false;
  flagDBTSM 	= false;
  flagDBLUTS    = false;

  // set debug
  edm::ParameterSet conf_ps = m_ps.getParameter<edm::ParameterSet>("DTTPGParameters");  
  bool dttpgdebug = conf_ps.getUntrackedParameter<bool>("Debug");
  m_manager->setDTTPGDebug(dttpgdebug);
}


DTConfigDBProducer::~DTConfigDBProducer()
{

}


//
// member functions
//

std::auto_ptr<DTConfigManager> DTConfigDBProducer::produce(const DTConfigManagerRcd& iRecord)
{
   using namespace edm;

   int code; 
   if(cfgConfig){
     m_manager->setLutFromDB(false);
     configFromCfg();
     buildTrivialPedestals();
     code = 2;
   }  else{
     code = readDTCCBConfig(iRecord);
     readDBPedestalsConfig(iRecord); // no return code if fails exception is raised by ESHandle getter
   }
   if(code==-1) {
     //throw cms::Exception("DTTPG") << "DTConfigDBProducer::produce : " << endl
     //				   << "generic error pharsing DT CCB config strings." << endl
     //                              << "Run module with debug flags enable to get more info" << endl;
     m_manager->setCCBConfigValidity(false);
   } else if(code==2) {
     LogVerbatim ("DTTPG") << "DTConfigDBProducer::produce : Trivial : " << endl
                           << "configurations has been read from cfg" << endl; 
   } else if(code==0) {
     LogVerbatim ("DTTPG") << "DTConfigDBProducer::produce : " << endl 
                           << "Configurations successfully read from OMDS" << endl; 
   } else {
     LogProblem ("DTTPG") << "DTConfigDBProducer::produce : " << endl
			  << "Wrong congiguration rertun CODE" << endl;
   }

   std::auto_ptr<DTConfigManager> dtConfig = std::auto_ptr<DTConfigManager>( m_manager );

   return dtConfig ;
}

void DTConfigDBProducer::readDBPedestalsConfig(const DTConfigManagerRcd& iRecord){

  edm::ESHandle<DTTPGParameters> dttpgParams;
  iRecord.getRecord<DTTPGParametersRcd>().get(dttpgParams);

  DTConfigPedestals pedestals;
  pedestals.setDebug(m_debugPed);

  if (m_UseT0) {

    edm::ESHandle<DTT0> t0i;
    iRecord.getRecord<DTT0Rcd>().get(t0i);
 
    pedestals.setUseT0(true);
    pedestals.setES(dttpgParams.product(),t0i.product());

  } else {

    pedestals.setUseT0(false);
    pedestals.setES(dttpgParams.product());

  }

  m_manager->setDTConfigPedestals(pedestals);

}

int DTConfigDBProducer::readDTCCBConfig(const DTConfigManagerRcd& iRecord)
{
  using namespace edm::eventsetup;

  // initialize CCB validity flag
  m_manager->setCCBConfigValidity(true);

  // get DTCCBConfigRcd from DTConfigManagerRcd (they are dependent records)
  edm::ESHandle<DTCCBConfig> ccb_conf;
  iRecord.getRecord<DTCCBConfigRcd>().get(ccb_conf);
  int ndata = std::distance( ccb_conf->begin(), ccb_conf->end() );

  DTConfigAbstractHandler* cfgCache = DTConfigAbstractHandler::getInstance();
  const DTKeyedConfigListRcd& keyRecord = iRecord.getRecord<DTKeyedConfigListRcd>();

  if(m_debugDB)
  {
  	cout << ccb_conf->version() << endl;
  	cout << ndata << " data in the container" << endl;
  }

  edm::ValidityInterval iov(iRecord.getRecord<DTCCBConfigRcd>().validityInterval() );
  unsigned int currValidityStart = iov.first().eventID().run();
  unsigned int currValidityEnd   = iov.last( ).eventID().run();

  if(m_debugDB)
  	cout 	<< "valid since run " << currValidityStart
            	<< " to run "         << currValidityEnd << endl;  
	    
  // if there are no data in the container, configuration from cfg files...	    
  if( ndata==0 ){
    //throw cms::Exception("DTTPG") << "DTConfigDBProducer::readDTCCBConfig : " << endl 
    //				  << "DTCCBConfigRcd is empty!" << endl;
    m_manager->setCCBConfigValidity(false);
  }

  // get DTTPGMap for retrieving bti number and traco number
  edm::ParameterSet conf_map = m_ps.getUntrackedParameter<edm::ParameterSet>("DTTPGMap");

  // loop over chambers
  DTCCBConfig::ccb_config_map configKeys( ccb_conf->configKeyMap() );
  DTCCBConfig::ccb_config_iterator iter = configKeys.begin();
  DTCCBConfig::ccb_config_iterator iend = configKeys.end();

  // read data from CCBConfig
  while ( iter != iend ) {
    // get chamber id
      	const DTCCBId& ccbId = iter->first;
	if(m_debugDB)
      		cout << " Filling configuration for chamber : wh " << ccbId.wheelId   << " st "
        	        << ccbId.stationId << " se "
                	<< ccbId.sectorId  << " -> " << endl;

	// get chamber type and id from ccbId
      	int mbtype = DTPosNegType::getCT( ccbId.wheelId, ccbId.sectorId, ccbId.stationId );
      	int posneg = DTPosNegType::getPN( ccbId.wheelId, ccbId.sectorId, ccbId.stationId );
	if(m_debugDB)
      		cout << "Chamber type : " <<  mbtype
      			<< " posneg : " << posneg << endl; 
	DTChamberId chambid(ccbId.wheelId, ccbId.stationId, ccbId.sectorId);
			
	// get brick identifiers list
      	const std::vector<int>& ccbConf = iter->second;
      	std::vector<int>::const_iterator cfgIter = ccbConf.begin();
      	std::vector<int>::const_iterator cfgIend = ccbConf.end();

        //TSS-TSM buffers
        unsigned short int tss_buffer[7][31];
        unsigned short int tsm_buffer[9];
        int ntss=0;

	// loop over configuration bricks
      	while ( cfgIter != cfgIend ) {
		// get brick identifier
        	int id = *cfgIter++;
		if(m_debugDB)
			cout << " BRICK " << id << endl;  

		// create strings list
        	std::vector<std::string> list;
                cfgCache->getData( keyRecord, id, list );

		// loop over strings
        	std::vector<std::string>::const_iterator s_iter = list.begin();
        	std::vector<std::string>::const_iterator s_iend = list.end();
        	while ( s_iter != s_iend ) {
			if(m_debugDB)
				cout << "        ----> " << *s_iter << endl;
				
			// copy string in unsigned int buffer
			std::string str = *s_iter++;
			unsigned short int buffer[100];		//2 bytes
			int c = 0;
			const char* cstr = str.c_str();
  			const char* ptr = cstr + 2;
  			const char* end = cstr + str.length();
			while ( ptr < end ) {
    				char c1 = *ptr++;
    				int i1 = 0;
    				if ( ( c1 >= '0' ) && ( c1 <= '9' ) ) i1 =      c1 - '0';
    				if ( ( c1 >= 'a' ) && ( c1 <= 'f' ) ) i1 = 10 + c1 - 'a';
    				if ( ( c1 >= 'A' ) && ( c1 <= 'F' ) ) i1 = 10 + c1 - 'A';
    				char c2 = *ptr++;
    				int i2 = 0;
    				if ( ( c2 >= '0' ) && ( c2 <= '9' ) ) i2 =      c2 - '0';
    				if ( ( c2 >= 'a' ) && ( c2 <= 'f' ) ) i2 = 10 + c2 - 'a';
    				if ( ( c2 >= 'A' ) && ( c2 <= 'F' ) ) i2 = 10 + c2 - 'A';
    				buffer[c] = ( i1 * 16 ) + i2;
				c++;
			}// end loop over string

			// BTI configuration string	
			if (buffer[2]==0x54){
				if(m_debugDB)
					cout << "BTI STRING found in DB" << endl;

				// BTI configuration read for BTI
				flagDBBti = true;
				
				// compute sl and bti number from board and chip
				int brd=buffer[3]; // Board Nr.
  				int chip=buffer[4]; // Chip Nr.

  				if (brd>7) {
					cout << "Not existing board ... " << brd << endl;
					return -1; // Non-existing board
				}	
  				if (chip>31) {
					cout << "Not existing chip... " << chip << endl;
					return -1; // Non existing chip 
				}
				
				// Is it Phi or Theta board?
  				bool ThetaSL, PhiSL;
				PhiSL=false;
				ThetaSL=false;
  				switch (mbtype) {
  					case 1: // mb1
    						if (brd==6 || brd==7) {
							ThetaSL=true; 
							brd-=6;
						}
    						else if ((brd<3 && chip<32) || (brd==3 && chip<8)) 
							PhiSL=true;
    						break;
  					case 2: // mb2
    						if (brd==6 || brd==7) {
							ThetaSL=true; 
							brd-=6;
						}
    						else if (brd<4 && chip<32) 
							PhiSL=true;
    						break;
  					case 3: // mb3
    						if (brd==6 || brd==7) {
							ThetaSL=true; 
							brd-=6;
						}
    						else if (brd<5 && chip<32) 
							PhiSL=true;
    						break;
  					case 4: // mb4-s, mb4_8
    						if (brd<6 && chip<32) 
							PhiSL=true;
    						break;
  					case 5: // mb4-9
    						if (brd<3 && chip<32) 
							PhiSL=true;
    						break;
  					case 6: // mb4-4
    						if (brd<5 && chip<32) 
							PhiSL=true;
    						break;
  					case 7: // mb4-10
    						if (brd<4 && chip<32) 
							PhiSL=true;
    						break;
  				}
 				if (!PhiSL && !ThetaSL) {
					cout << "MB type " << mbtype << endl; 
					cout << "Board " << brd << " chip " <<chip << endl;
					cout << "Not phi SL nor Theta SL" << endl; 
					return -1; // Not PhiSL nor ThetaSL 
				}	 
					
				// compute SL number and bti number
				int isl;
				int ibti;
				if (PhiSL) {
    					if ((chip%8)<4) 
						isl=1; 	// Phi1 
    					else 
						isl=3; 	// Phi2 
					ibti=brd*16+(int)(chip/8)*4+(chip%4); 
				}
			        else if (ThetaSL){
    					isl=2; 		// Theta 
					if ((chip%8)<4)
      						ibti=brd*32+ chip-4*(int)(chip/8);
    					else
      						ibti=brd*32+ chip+12-4*(int)(chip/8);
				}		 
			
				// BTI config constructor from strings	
			        DTConfigBti bticonf(m_debugBti,buffer);
                                 
				m_manager->setDTConfigBti(DTBtiId(chambid,isl,ibti+1),bticonf);
			    	
	      			if(m_debugDB)
					cout << 	"Filling BTI config for chamber : wh " << chambid.wheel() << 
		  					", st " << chambid.station() << 
		  					", se " << chambid.sector() << 
		  					"... sl " << isl << 
		  					", bti " << ibti+1 << endl;				
			}
				
			// TRACO configuration string 			
			if (buffer[2]==0x15){
				if(m_debugDB)
					cout << "TRACO STRING found in DB" << endl;
				// TRACO configuration read from OMDS
				flagDBTraco = true;
				
				// TRACO config constructor from strings	
			        int traco_brd = buffer[3]; 	// Board Nr.;
  				int traco_chip = buffer[4]; 	// Chip Nr.;
				int itraco = traco_brd * 4 + traco_chip + 1;
				DTConfigTraco tracoconf(m_debugTraco,buffer);
          			m_manager->setDTConfigTraco(DTTracoId(chambid,itraco),tracoconf);
				
				if(m_debugDB)
                			cout << 	"Filling TRACO config for chamber : wh " << chambid.wheel() <<
                					", st " << chambid.station() <<
                					", se " << chambid.sector() <<
					                ", board " << traco_brd <<
					                ", chip " << traco_chip <<
                					", traco " << itraco << endl;	
			}
			
			
			// TSS configuration string	
			if (buffer[2]==0x16){
				if(m_debugDB)
					cout << "TSS STRING found in DB" << endl;
				// TSS configuration read from OMDS
				flagDBTSS = true;
				
				unsigned short int itss=buffer[3];
                  		for (int i=0;i<31;i++) 
					tss_buffer[itss][i]=buffer[i];
                  		ntss++;
                	}
			
			// TSM configuration string
                	if (buffer[2]==0x17){
				if(m_debugDB)
					cout << "TSM STRING found in DB" << endl;
			 
				// TSM configuration read from OMDS
				flagDBTSM = true;			 
			 
                        	for (int i=0; i<9; i++) 
					tsm_buffer[i]=buffer[i];
                	} 
			
			// LUT configuration string
			if (buffer[2]==0xA8){
				if(m_debugDB)
					cout << "LUT STRING found in DB" << endl;

				// LUT parameters read from OMDS
                                flagDBLUTS = true;
				DTConfigLUTs lutconf(m_debugLUTs,buffer);
				//lutconf.setDebug(m_debugLUTs);
				m_manager->setDTConfigLUTs(chambid,lutconf);
			}
			
		}//end string iteration					
      	}//end brick iteration
	
 	//TSS + TSM configurations are set in DTConfigTSPhi constructor
	if(flagDBTSM && flagDBTSS) {
	  DTConfigTSPhi tsphiconf(m_debugTSP,tss_buffer,ntss,tsm_buffer);
	  m_manager->setDTConfigTSPhi(chambid,tsphiconf);
	}

 	// get configuration for TSTheta, SC and TU from .cfg
        edm::ParameterSet conf_ps = m_ps.getParameter<edm::ParameterSet>("DTTPGParameters");  
	edm::ParameterSet tups = conf_ps.getParameter<edm::ParameterSet>("TUParameters");

	// TSTheta configuration from .cfg
        DTConfigTSTheta tsthetaconf(tups.getParameter<edm::ParameterSet>("TSThetaParameters"));
	tsthetaconf.setDebug(m_debugTST);
	m_manager->setDTConfigTSTheta(chambid,tsthetaconf);

	// SC configuration from .cfg
        DTConfigSectColl sectcollconf(conf_ps.getParameter<edm::ParameterSet>("SectCollParameters"));
	sectcollconf.setDebug(m_debugSC);
        m_manager->setDTConfigSectColl(DTSectCollId(chambid.wheel(),chambid.sector()),sectcollconf);

	// TU configuration from .cfg
  	DTConfigTrigUnit trigunitconf(tups);
	trigunitconf.setDebug(m_debugTU);
        m_manager->setDTConfigTrigUnit(chambid,trigunitconf);
       
      	++iter;
  }

  // moved to exception handling no attempt to configure from cfg is DB is missing
  // SV comment exception handling and activate flag in DTConfigManager
  if(!flagDBBti || !flagDBTraco || !flagDBTSS || !flagDBTSM ){
    //throw cms::Exception("DTTPG") << "DTConfigDBProducer::readDTCCBConfig :"  << endl
    //				  << "(at least) part of the CCB strings needed to configure"  << endl
    //				  << "DTTPG emulator were not found in DTCCBConfigRcd" << endl;
    m_manager->setCCBConfigValidity(false);
    
  }
  if(!flagDBLUTS && m_manager->lutFromDB()==true){
    //throw cms::Exception("DTTPG") << "DTConfigDBProducer::readDTCCBConfig : " << endl
    //				  << "Asked to configure the emulator using Lut seeds from DB "
    //				  << "but no configuration parameters found in DTCCBConfigRcd." << endl;
    m_manager->setCCBConfigValidity(false);
  } 
  
  return 0;
}

std::string DTConfigDBProducer::mapEntryName(const DTChamberId & chambid) const
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


void DTConfigDBProducer::configFromCfg(){

  // ... but still set CCB validity flag to let the emulator run
  m_manager->setCCBConfigValidity(true);

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
  DTConfigLUTs lutconf(tups.getParameter<edm::ParameterSet>("LutParameters"));
 
 
  for (int iwh=-2;iwh<=2;++iwh){
    for (int ist=1;ist<=4;++ist){
      for (int ise=1;ise<=12;++ise){
	DTChamberId chambid(iwh,ist,ise);
	vector<int> nmap = conf_map.getUntrackedParameter<vector<int> >(mapEntryName(chambid).c_str());

	if(dttpgdebug)
	  {
	    cout << " Filling configuration for chamber : wh " << chambid.wheel() << 
	      ", st " << chambid.station() << 
	      ", se " << chambid.sector() << endl;
	  }
	
	//fill the bti map
	if(!flagDBBti){
		for (int isl=1;isl<=3;isl++){
	  		int ncell = nmap[isl-1];
	  		//	  cout << ncell <<" , ";
	  		for (int ibti=0;ibti<ncell;ibti++){
	      			m_manager->setDTConfigBti(DTBtiId(chambid,isl,ibti+1),bticonf);
	      			if(dttpgdebug)
					cout << "Filling BTI config for chamber : wh " << chambid.wheel() << 
		  				", st " << chambid.station() << 
		  				", se " << chambid.sector() << 
		  				"... sl " << isl << 
		  				", bti " << ibti+1 << endl;
	    		}     
		}
	}		
	
	// fill the traco map
	if(!flagDBTraco){
		int ntraco = nmap[3];
		//cout << ntraco << " }" << endl;
		for (int itraco=0;itraco<ntraco;itraco++){ 
	    		m_manager->setDTConfigTraco(DTTracoId(chambid,itraco+1),tracoconf);
	    		if(dttpgdebug)
	      			cout << "Filling TRACO config for chamber : wh " << chambid.wheel() << 
					", st " << chambid.station() << 
					", se " << chambid.sector() << 
					", traco " << itraco+1 << endl;
	  	}     
	}
	
	// fill TS & TrigUnit
	if(!flagDBTSS || !flagDBTSM)
	{	
		m_manager->setDTConfigTSTheta(chambid,tsthetaconf);
		m_manager->setDTConfigTSPhi(chambid,tsphiconf);
		m_manager->setDTConfigTrigUnit(chambid,trigunitconf);
	}

      }
    }
  }

  for (int iwh=-2;iwh<=2;++iwh){
    for (int ise=13;ise<=14;++ise){
      int ist =4;
      DTChamberId chambid(iwh,ist,ise);
      vector<int> nmap = conf_map.getUntrackedParameter<vector<int> >(mapEntryName(chambid).c_str());

      if(dttpgdebug)
	{
	  cout << " Filling configuration for chamber : wh " << chambid.wheel() << 
	    ", st " << chambid.station() << 
	    ", se " << chambid.sector() << endl;
	}
      
      //fill the bti map
      if(!flagDBBti){
      		for (int isl=1;isl<=3;isl++){
			int ncell = nmap[isl-1];
			// 	cout << ncell <<" , ";
			for (int ibti=0;ibti<ncell;ibti++){
	    			m_manager->setDTConfigBti(DTBtiId(chambid,isl,ibti+1),bticonf);
	    			if(dttpgdebug)
	      				cout << "Filling BTI config for chamber : wh " << chambid.wheel() << 
						", st " << chambid.station() << 
						", se " << chambid.sector() << 
						"... sl " << isl << 
						", bti " << ibti+1 << endl;
	  		}     
      		}
      }	
      
      // fill the traco map
      if(!flagDBTraco){
      		int ntraco = nmap[3];	
		//       cout << ntraco << " }" << endl;
      		for (int itraco=0;itraco<ntraco;itraco++){ 
	  		m_manager->setDTConfigTraco(DTTracoId(chambid,itraco+1),tracoconf);
	  		if(dttpgdebug)
	    			cout << "Filling TRACO config for chamber : wh " << chambid.wheel() << 
	      			", st " << chambid.station() << 
	      			", se " << chambid.sector() << 
	      			", traco " << itraco+1 << endl;
		}     
      }
      
      // fill TS & TrigUnit
      if(!flagDBTSS || !flagDBTSM)
      {
      		m_manager->setDTConfigTSTheta(chambid,tsthetaconf);
      		m_manager->setDTConfigTSPhi(chambid,tsphiconf);
      		m_manager->setDTConfigTrigUnit(chambid,trigunitconf);
      }
    }
  }
  
  //loop on Sector Collectors
  for (int wh=-2;wh<=2;wh++)
    for (int se=1;se<=12;se++)
      m_manager->setDTConfigSectColl(DTSectCollId(wh,se),sectcollconf);
      
  //fake collection of pedestals
  m_manager->setDTConfigPedestals(buildTrivialPedestals());

  return;          

}

DTConfigPedestals DTConfigDBProducer::buildTrivialPedestals()
{
  DTTPGParameters* m_tpgParams = new DTTPGParameters();

  int counts = m_ps.getParameter<int>("bxOffset");
  float fine = m_ps.getParameter<double>("finePhase");
   
  if (m_debugPed) 
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


