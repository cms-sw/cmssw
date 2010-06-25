#include "L1TriggerConfig/CSCTFConfigProducers/interface/CSCTFConfigOnlineProd.h"
#include <cstdio>

boost::shared_ptr< L1MuCSCTFConfiguration >
CSCTFConfigOnlineProd::newObject( const std::string& objectKey )
{
  
  edm::LogInfo( "L1-O2O: CSCTFConfigOnlineProd" ) << "Producing "
					     << " L1MuCSCTFConfiguration"
					     << "with key CSCTF_KEY="
					     << objectKey;

  std::string csctfreg[12];

  // loop over the 12 SPs forming the CSCTF crate
  for (int iSP=1;iSP<13; iSP++) {

    char spName[2];
    if (iSP<10) sprintf(spName,"0%d",iSP);
    else        sprintf(spName, "%d",iSP);

    std::string spkey = objectKey + "00" + spName;

    edm::LogInfo( "L1-O2O: CSCTFConfigOnlineProd" ) << "spkey: " 
						    << spkey;
  
  
     //  SELECT Multiple columns  FROM TABLE with correct key: 
     std::vector< std::string > columns ;
     columns.push_back( "STATIC_CONFIG" ) ;
     columns.push_back( "ETA_CONFIG" ) ;
     
     //SELECT * FROM CMS_CSC_TF.CSCTF_SP_CONF WHERE CSCTF_SP_CONF.SP_KEY = spkey
     l1t::OMDSReader::QueryResults results = m_omdsReader.basicQuery(
								     columns,
								     "CMS_CSC_TF",
								     "CSCTF_SP_CONF",
								     "CSCTF_SP_CONF.SP_KEY",
								     m_omdsReader.singleAttribute( spkey )
								     ) ;
  
     if( results.queryFailed() ) // check if query was successful
       {
	edm::LogError( "L1-O2O" ) << "Problem with L1CSCTFParameters key." ;
	// return empty configuration
	return boost::shared_ptr< L1MuCSCTFConfiguration >( new L1MuCSCTFConfiguration() ) ;
       }
  

     std::string conf_stat, conf_eta;
     results.fillVariable( "STATIC_CONFIG", conf_stat );
     results.fillVariable( "ETA_CONFIG",    conf_eta  );
     //   std::cout<<conf_stat<<std::endl;

     edm::LogInfo( "conf_stat queried" ) << conf_stat;
     edm::LogInfo( "conf_eta queried" )  << conf_eta;
    
     for(size_t pos=conf_stat.find("\\n"); pos!=std::string::npos; pos=conf_stat.find("\\n",pos)) 
       { 
	 conf_stat[pos]=' '; 
	 conf_stat[pos+1]='\n'; 
       }
    
     for(size_t pos=conf_eta.find("\\n"); pos!=std::string::npos; pos=conf_eta.find("\\n",pos)) 
       { 
	 conf_eta[pos]=' '; 
	 conf_eta[pos+1]='\n'; 
       }
  
     std::string conf_read=conf_eta+conf_stat;
     // write all registers for a given SP
     csctfreg[iSP-1]=conf_read;
  }  
  
  // return the final object with the configuration for all CSCTF
  return boost::shared_ptr< L1MuCSCTFConfiguration >( new L1MuCSCTFConfiguration(csctfreg) ) ;    


//   // Execute SQL queries to get data from OMDS (using key) and make C++ object
//   // Example: SELECT A_PARAMETER FROM CMS_XXX.XXX_CONF WHERE XXX_CONF.XXX_KEY = objectKey
  
//  //  SELECT Multiple columns  FROM TABLE with correct key: 
//  std::vector< std::string > columns ;
//  columns.push_back( "STATIC_CONFIG" ) ;
//  columns.push_back( "ETA_CONFIG" ) ;
//  l1t::OMDSReader::QueryResults results = m_omdsReader.basicQuery(
// 								 columns,
// 								 "CMS_CSC_TF",
// 								 "CSCTF_SP_CONF",
// 								 "CSCTF_SP_CONF.SP_KEY",
// 								 m_omdsReader.singleAttribute( objectKey )
// 								 ) ;
  
//  if( results.queryFailed() ) // check if query was successful
//    {
//      edm::LogError( "L1-O2O" ) << "Problem with L1CSCTFParameters key." ;
//      return boost::shared_ptr< L1MuCSCTFConfiguration >( new L1MuCSCTFConfiguration( ) ) ;
//    }
  
//  //    double datum ;
//  std::string conf_stat, conf_eta;
//  results.fillVariable( "STATIC_CONFIG", conf_stat ) ;
//  results.fillVariable( "ETA_CONFIG", conf_eta ) ;
//  //   std::cout<<conf_stat<<std::endl;
//  edm::LogInfo( "conf_stat queried" ) << conf_stat ;
//  edm::LogInfo( "conf_eta queried" ) << conf_eta ;
//  std::string csctfreg[12];
//  std::vector<std::string> spreg;
//  std::string registers;

//  for(size_t pos=conf_stat.find("\\n"); pos!=std::string::npos; pos=conf_stat.find("\\n",pos)) 
//    { 
//      conf_stat[pos]=' '; 
//      conf_stat[pos+1]='\n'; 
//    }

//  //std::cout<<" ******** \n"<<conf_stat<<std::endl;


//  for(size_t pos=conf_eta.find("\\n"); pos!=std::string::npos; pos=conf_eta.find("\\n",pos)) 
//    { 
//      conf_eta[pos]=' '; 
//      conf_eta[pos+1]='\n'; 

//    }

//  //std::cout<<" ******** \n"<<conf_eta<<std::endl;

//  std::string conf_fixed= "CSR_LQE F1 M1 0xFFFF \nCSR_LQE F1 M2 0xFFFF \nCSR_LQE F1 M3 0xFFFF \nCSR_LQE F2 M1 0xFFFF \nCSR_LQE F2 M2 0xFFFF \nCSR_LQE F2 M3 0xFFFF \nCSR_LQE F3 M1 0xFFFF \nCSR_LQE F3 M2 0xFFFF \nCSR_LQE F3 M3 0xFFFF \nCSR_LQE F4 M1 0xFFFF \nCSR_LQE F4 M2 0xFFFF \nCSR_LQE F4 M3 0xFFFF \nCSR_LQE F5 M1 0xFFFF \nCSR_LQE F5 M2 0xFFFF \nCSR_LQE F5 M3 0xFFFF \nCSR_KFL SP MA 0x0000 \nDAT_FTR SP MA 0xFF   \nCSR_SFC SP MA 0x1000 \n";


//  std::string conf_read=conf_fixed+conf_eta+conf_stat;
//  // std::cout<<" ******** \n"<<conf_read<<std::endl;


//  // spreg='"test1","test2","test3"';
//  // std::cout<<spreg<<std::endl;
//  //    for(std::vector<std::string>::const_iterator line=spreg.begin(); line!=spreg.end(); line++)
//  // 	registers += *line + "\n";

//  //
//  for (int jsp=0;jsp<12; jsp++) {
//    csctfreg[jsp]=conf_read;
//  }

//  L1MuCSCTFConfiguration myconfig(csctfreg);

//  // print out
//  //const std::string* outcfg=myconfig.configAsText();
//  //   for (int jsp=0;jsp<12; jsp++) {
//  //     std::cout<<"outcfg... "<<jsp<<std::endl;
//  //     std::cout<<outcfg[jsp]<<std::endl;
//  //  }
// return boost::shared_ptr< L1MuCSCTFConfiguration >( new L1MuCSCTFConfiguration(csctfreg) ) ;
}


