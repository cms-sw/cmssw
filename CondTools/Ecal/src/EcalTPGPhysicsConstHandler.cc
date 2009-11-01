#include "CondTools/Ecal/interface/EcalTPGPhysicsConstHandler.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/RunTPGConfigDat.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigMainInfo.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigParamDat.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigLUTInfo.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include<iostream>

popcon::EcalTPGPhysicsConstHandler::EcalTPGPhysicsConstHandler(const edm::ParameterSet & ps)
  :    m_name(ps.getUntrackedParameter<std::string>("name","EcalTPGPhysicsConstHandler")) {

        edm::LogInfo("EcalTPGPhysicsConstHandler") << "EcalTPGPhysicsConst Source handler constructor.";
        m_firstRun=(unsigned long)atoi( ps.getParameter<std::string>("firstRun").c_str());
        m_lastRun=(unsigned long)atoi( ps.getParameter<std::string>("lastRun").c_str());
        m_sid= ps.getParameter<std::string>("OnlineDBSID");
        m_user= ps.getParameter<std::string>("OnlineDBUser");
        m_pass= ps.getParameter<std::string>("OnlineDBPassword");
        m_locationsource= ps.getParameter<std::string>("LocationSource");
        m_location=ps.getParameter<std::string>("Location");
        m_gentag=ps.getParameter<std::string>("GenTag");
	m_runtype=ps.getParameter<std::string>("RunType");

        edm::LogInfo("EcalTPGPhysicsConstHandler")<< m_sid<<"/"<<m_user<<"/"<<m_pass<<"/"<<m_location<<"/"<<m_gentag;

}

popcon::EcalTPGPhysicsConstHandler::~EcalTPGPhysicsConstHandler()
{
}


void popcon::EcalTPGPhysicsConstHandler::getNewObjects()
{

	edm::LogInfo("EcalTPGPhysicsConstHandler") << "Started GetNewObjects!!!";

	//check whats already inside of database
	if (tagInfo().size){
  	//check whats already inside of database
    	std::cout << "got offlineInfo = " << std::endl;
	std::cout << "tag name = " << tagInfo().name << std::endl;
	std::cout << "size = " << tagInfo().size <<  std::endl;
    	} else {
    	std::cout << " First object for this tag " << std::endl;
    	}

	int max_since=0;
	max_since=(int)tagInfo().lastInterval.first;
	edm::LogInfo("EcalTPGPhysicsConstHandler") << "max_since : "  << max_since;
	Ref physC_db = lastPayload();
	
	edm::LogInfo("EcalTPGPhysicsConstHandler") << "retrieved last payload ";

	// here we retrieve all the runs after the last from online DB 
	edm::LogInfo("EcalTPGPhysicsConstHandler") << "Retrieving run list from ONLINE DB ... ";

	edm::LogInfo("EcalTPGPhysicsConstHandler") << "Making connection...";
	econn = new EcalCondDBInterface( m_sid, m_user, m_pass );
	edm::LogInfo("EcalTPGPhysicsConstHandler") << "Done.";
	
	if (!econn)
	  {
	    cout << " connection parameters " <<m_sid <<"/"<<m_user<<"/"<<m_pass<<endl;
	    //	    cerr << e.what() << endl;
	    throw cms::Exception("OMDS not available");
	  } 

	
	LocationDef my_locdef;
	my_locdef.setLocation(m_location); 

	RunTypeDef my_rundef;
	my_rundef.setRunType(m_runtype); 

	RunTag  my_runtag;
	my_runtag.setLocationDef( my_locdef );
	my_runtag.setRunTypeDef(  my_rundef );
	my_runtag.setGeneralTag(m_gentag); 


        readFromFile("last_tpg_physC_settings.txt");

 	int min_run=m_i_run_number+1;

	if(m_firstRun<(unsigned int)m_i_run_number) {
	  min_run=(int) m_i_run_number+1;
	} else {
	  min_run=(int)m_firstRun;
	}
	
	std::cout<<"m_i_run_number"<< m_i_run_number <<"m_firstRun "<<m_firstRun<< "max_since " <<max_since<< endl;

	if(min_run<(unsigned int)max_since) {
	  min_run=  (int)max_since+1; // we have to add 1 to the last transferred one
	} 

	int max_run=(int)m_lastRun;
	edm::LogInfo("EcalTPGPhysicsConstHandler") << "min_run= " << min_run << "max_run= " << max_run;
	
        RunList my_list;
	my_list=econn->fetchRunListByLocation(my_runtag,min_run,max_run,my_locdef);

      
	std::vector<RunIOV> run_vec=  my_list.getRuns();
	int num_runs=run_vec.size();

	std::cout <<"number of runs is : "<< num_runs<< endl;

	unsigned long irun;
	if(num_runs>0){
	
	  for(int kr=0; kr<run_vec.size(); kr++){

	    irun=(unsigned long) run_vec[kr].getRunNumber();

	    std::cout<<" **************** "<<std::endl;
	    std::cout<<" **************** "<<std::endl;
	    std::cout<<" run= "<<irun<<std::endl;
            
	    // retrieve the data :
	    map<EcalLogicID, RunTPGConfigDat> dataset;
	    econn->fetchDataSet(&dataset, &run_vec[kr]);
	    
	    std::string the_config_tag="";
	    int the_config_version=0;
	    
	    map< EcalLogicID,  RunTPGConfigDat>::const_iterator it;
	    
	    int nr=0;
	    for( it=dataset.begin(); it!=dataset.end(); it++ )
	    {
	      ++nr;
	      //EcalLogicID ecalid  = it->first;
	      RunTPGConfigDat  dat = it->second;
	      the_config_tag=dat.getConfigTag();
	      the_config_version=dat.getVersion();
	    } 
	      
	      
	    // it is all the same for all SM... get the last one 


	    std::cout<<" run= "<<irun<<" tag "<<the_config_tag<<" version="<<the_config_version <<std::endl;

	    // here we should check if it is the same as previous run.


	    if((the_config_tag != m_i_tag || the_config_version != m_i_version ) && nr>0 ) {
	      std::cout<<"the tag is different from last transferred run ... retrieving last config set from DB"<<endl;

	      FEConfigMainInfo fe_main_info;
	      fe_main_info.setConfigTag(the_config_tag);
	      fe_main_info.setVersion(the_config_version);

	      try{ 
		std::cout << " before fetch config set" << std::endl;	    
		econn-> fetchConfigSet(&fe_main_info);
		std::cout << " after fetch config set" << std::endl;	   
	   
	   
            	// now get TPGPhysicsConst
            	int linId=fe_main_info.getLinId();
	    	int fgrId=fe_main_info.getFgrId();
	    	int lutId=fe_main_info.getLUTId();
		
		if ((linId != m_i_physClin) || (fgrId != m_i_physCfgr) || (lutId != m_i_physClut) ) {
	          
		  std::cout<<"one of the parameters: linId, LutId or fgrId is different from" <<endl;
		  std::cout<<"last transferred run ..."<<endl;
 
	          FEConfigLinInfo fe_physLin_info;
	    	  FEConfigFgrInfo fe_physFgr_info;
		  FEConfigLUTInfo fe_physLut_info;
		  fe_physLin_info.setId(linId);
	          fe_physFgr_info.setId(fgrId);
		  fe_physLut_info.setId(lutId);
		  
		  econn-> fetchConfigSet(&fe_physLin_info);
	          econn-> fetchConfigSet(&fe_physFgr_info);
		  econn-> fetchConfigSet(&fe_physLut_info);
	          map<EcalLogicID, FEConfigLinParamDat> dataset_TpgPhysicsLin;
	          map<EcalLogicID, FEConfigLUTParamDat> dataset_TpgPhysicsLut;
		  map<EcalLogicID, FEConfigFgrParamDat> dataset_TpgPhysicsFgr;
		  
		  econn->fetchDataSet(&dataset_TpgPhysicsLin, &fe_physLin_info);
		  econn->fetchDataSet(&dataset_TpgPhysicsLut, &fe_physLut_info);
		  econn->fetchDataSet(&dataset_TpgPhysicsFgr, &fe_physFgr_info);

	          EcalTPGPhysicsConst* physC = new EcalTPGPhysicsConst;
                  typedef map<EcalLogicID, FEConfigLinParamDat>::const_iterator CIfeLin;
	          typedef map<EcalLogicID, FEConfigLUTParamDat>::const_iterator CIfeLUT;
	          typedef map<EcalLogicID, FEConfigFgrParamDat>::const_iterator CIfeFgr;

		  EcalLogicID ecidLin_xt;
		  EcalLogicID ecidLut_xt;
		  EcalLogicID ecidFgr_xt;
	          FEConfigLinParamDat rd_physLin;
		  FEConfigLUTParamDat rd_physLut;
		  FEConfigFgrParamDat rd_physFgr;
	    	  
		  map<int,float> EtSatLinEB;
		  map<int,float> EtSatLinEE;
		  typedef map<int,float>::const_iterator itEtSat;
		  
		  map<int,EcalTPGPhysicsConst::Item> temporaryMapEB;
		  map<int,EcalTPGPhysicsConst::Item> temporaryMapEE;
		  typedef map<int,EcalTPGPhysicsConst::Item>::iterator iterEB;
		  typedef map<int,EcalTPGPhysicsConst::Item>::iterator iterEE;
		  
                 
		  for (CIfeLin p0 = dataset_TpgPhysicsLin.begin(); p0 != dataset_TpgPhysicsLin.end(); p0++) 
	          { 
	      	    ecidLin_xt = p0->first;
	            rd_physLin = p0->second;
	  
	            std::string ecid_nameLin=ecidLin_xt.getName();

		    if(ecid_nameLin=="EB") {
		      DetId eb(DetId::Ecal, EcalBarrel);	      
		      EtSatLinEB.insert(make_pair(eb.rawId(),rd_physLin.getETSat()));  
		    }
		    else if (ecid_nameLin=="EE"){
		    DetId ee(DetId::Ecal, EcalEndcap);		      
		      EtSatLinEE.insert(make_pair(ee.rawId(),rd_physLin.getETSat()));
		    }
		  }   

				  
		  int icells=0;
	    	  for (CIfeLUT p1 = dataset_TpgPhysicsLut.begin(); p1 != dataset_TpgPhysicsLut.end(); p1++) 
	          { 
	      	    ecidLut_xt = p1->first;
	            rd_physLut = p1->second;
	  
	            std::string ecid_nameLut=ecidLut_xt.getName(); 

	      	    // Ecal barrel detector
	            if(ecid_nameLut=="EB") {
	            
	      	      DetId eb(DetId::Ecal, EcalBarrel);
		      
		      for (itEtSat it1 = EtSatLinEB.begin() ; it1 != EtSatLinEB.end(); it1++){

			if (it1->first == eb.rawId()){ 
		          float ETSatLin = it1->second;
		          
	                  if (rd_physLut.getETSat() == ETSatLin) {
			    EcalTPGPhysicsConst::Item item;
	                    item.EtSat=rd_physLut.getETSat();
	                    item.ttf_threshold_Low=rd_physLut.getTTThreshlow();
	                    item.ttf_threshold_High=rd_physLut.getTTThreshhigh(); 
		            temporaryMapEB.insert(make_pair(eb.rawId(),item));
			  }
			  else throw cms::Exception("The values of the ETSatLin and ETSatLut are different.");
		        }
			
		      }
		      	      	      	              
		       	
	              ++icells;
	            }
	    	    else if (ecid_nameLut=="EE") {
	      	      // Ecal endcap detector	  
	              
	      	      DetId ee(DetId::Ecal, EcalEndcap);
	  
	              for (itEtSat it2 = EtSatLinEE.begin(); it2 != EtSatLinEE.end(); it2++){

			if (it2->first == ee.rawId()){ 
		          float ETSatLin = it2->second;
		       
	                  if (rd_physLut.getETSat() == ETSatLin) {
			    EcalTPGPhysicsConst::Item item;
	      	      	    item.EtSat=rd_physLut.getETSat();
	      	      	    item.ttf_threshold_Low=rd_physLut.getTTThreshlow();
	      	      	    item.ttf_threshold_High=rd_physLut.getTTThreshhigh(); 
	                    temporaryMapEE.insert( make_pair(ee.rawId(),item) );
			  }
			  else throw cms::Exception("The values of the ETSatLin and ETSatLut are different.");
		        }
			
		      }
		      		       	   
	      	      ++icells;	  
	  	    }
	          }
		  
		  int icellsEB=0;
		  int icellsEE=0;
	          for (CIfeFgr p2 = dataset_TpgPhysicsFgr.begin(); p2 != dataset_TpgPhysicsFgr.end(); p2++) 
	          { 
		    ecidFgr_xt = p2->first;
	            rd_physFgr  = p2->second;
	  
	      	    std::string ecid_nameFgr=ecidFgr_xt.getName();

	      	    // Ecal barrel detector
	            if(ecid_nameFgr=="EB") {
		      
	      	      DetId eb(DetId::Ecal, EcalBarrel);
		      
	  	      int count;
		      for ( iterEB itt=temporaryMapEB.begin() ; itt != temporaryMapEB.end() ; itt++ ){
                       			
		        if (itt->first == eb.rawId()){ 

			  (itt->second).FG_lowThreshold=rd_physFgr.getFGlowthresh();
	                  (itt->second).FG_highThreshold=rd_physFgr.getFGhighthresh();
	                  (itt->second).FG_lowRatio=rd_physFgr.getFGlowratio();
	                  (itt->second).FG_highRatio= rd_physFgr.getFGhighratio();
						   
			   physC->setValue(eb.rawId(),itt->second);
			
			}
			
			count++; 
		      }	  
	  
	              ++icellsEB;
		      
	            }
	    	    else if (ecid_nameFgr=="EE") {
	      	      // Ecal endcap detector

	      	      DetId ee(DetId::Ecal, EcalEndcap);
		
	              int countEE = 0;
	  	      for ( iterEE itEE=temporaryMapEE.begin() ; itEE != temporaryMapEE.end() ; itEE++ ){                       

		        if (itEE->first == ee.rawId()){ 
			  
			  (itEE->second).FG_lowThreshold=rd_physFgr.getFGlowthresh();
	                  (itEE->second).FG_highThreshold=rd_physFgr.getFGhighthresh();
	                  // the last two is empty for the EE
			  (itEE->second).FG_lowRatio=rd_physFgr.getFGlowratio();
	                  (itEE->second).FG_highRatio= rd_physFgr.getFGhighratio();
			   
			   physC->setValue(ee.rawId(),itEE->second); 
			}
			
			countEE++; 
		      }
	   
	      	      ++icellsEE;	  
	  	    }
	          }
		  

 	        Time_t snc= (Time_t) irun ;
	      	      
 	        m_to_transfer.push_back(std::make_pair((EcalTPGPhysicsConst*)physC,snc));
	      
	      
	          m_i_run_number=irun;
		  m_i_tag=the_config_tag;
		  m_i_version=the_config_version;
		  m_i_physClin=linId;
		  m_i_physClut=lutId;
		  m_i_physCfgr=fgrId;
		  
		  writeFile("last_tpg_physC_settings.txt");

		} else {

		  m_i_run_number=irun;
		  m_i_tag=the_config_tag;
		  m_i_version=the_config_version;

		  writeFile("last_tpg_physC_settings.txt");

		  std::cout<< " even if the tag/version is not the same, the physics constants id is the same -> no transfer needed "<< std::endl; 

		}

	      }       
	      
	      
	      
	      catch (std::exception &e) { 
		std::cout << "ERROR: THIS CONFIG DOES NOT EXIST: tag=" <<the_config_tag
			  <<" version="<<the_config_version<< std::endl;
		cout << e.what() << endl;
		m_i_run_number=irun;

	      }
	      std::cout<<" **************** "<<std::endl;
	      
	    } else if(nr==0) {
	      m_i_run_number=irun;
	      std::cout<< " no tag saved to RUN_TPGCONFIG_DAT by EcalSupervisor -> no transfer needed "<< std::endl; 
	      std::cout<<" **************** "<<std::endl;
	    } else {
	      m_i_run_number=irun;
	      m_i_tag=the_config_tag;
	      m_i_version=the_config_version;
	      std::cout<< " the tag/version is the same -> no transfer needed "<< std::endl; 
	      std::cout<<" **************** "<<std::endl;
	      writeFile("last_tpg_physC_settings.txt");
	    }
    }
  }
	  
	delete econn;
	edm::LogInfo("EcalTPGPhysicsConstHandler") << "Ecal - > end of getNewObjects -----------";
	
}


void  popcon::EcalTPGPhysicsConstHandler::readFromFile(const char* inputFile) {
  //-------------------------------------------------------------
  
  m_i_tag="";
  m_i_version=0;
  m_i_run_number=0;
  m_i_physClin=0;
  m_i_physClut=0;
  m_i_physCfgr=0;
		  
  FILE *inpFile; // input file
  inpFile = fopen(inputFile,"r");
  if(!inpFile) {
    edm::LogError("EcalTPGPhysicsConstHandler")<<"*** Can not open file: "<<inputFile;
  }

  char line[256];
    
  std::ostringstream str;

  fgets(line,255,inpFile);
  m_i_tag=to_string(line);
  str << "gen tag " << m_i_tag << endl ;  // should I use this? 

  fgets(line,255,inpFile);
  m_i_version=atoi(line);
  str << "version= " << m_i_version << endl ;  

  fgets(line,255,inpFile);
  m_i_run_number=atoi(line);
  str << "run_number= " << m_i_run_number << endl ;  

  fgets(line,255,inpFile);
  m_i_physClin=atoi(line);
  str << "physClin_config= " << m_i_physClin << endl ;  

  fgets(line,255,inpFile);
  m_i_physClut=atoi(line);
  str << "physClut_config= " << m_i_physClut << endl ;  
  
  fgets(line,255,inpFile);
  m_i_physCfgr=atoi(line);
  str << "physCfgr_config= " << m_i_physCfgr << endl ;  
    
  fclose(inpFile);           // close inp. file

}

void  popcon::EcalTPGPhysicsConstHandler::writeFile(const char* inputFile) {
  //-------------------------------------------------------------
  
  
  ofstream myfile;
  myfile.open (inputFile);
  myfile << m_i_tag <<endl;
  myfile << m_i_version <<endl;
  myfile << m_i_run_number <<endl;
  myfile << m_i_physClin <<endl;
  myfile << m_i_physClut <<endl;
  myfile << m_i_physCfgr <<endl;

  myfile.close();

}

