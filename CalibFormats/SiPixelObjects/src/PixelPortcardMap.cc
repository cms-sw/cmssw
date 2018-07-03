//
// This class provides the mapping between
// portcards and the modules controlled by
// the card
//
//
//
 
#include "CalibFormats/SiPixelObjects/interface/PixelPortcardMap.h"
#include "CalibFormats/SiPixelObjects/interface/PixelTimeFormatter.h"

#include <sstream>
#include <cassert>
#include <stdexcept>

using namespace pos;
using namespace std;

PixelPortcardMap::PixelPortcardMap(std::vector< std::vector < std::string> > &tableMat):PixelConfigBase(" "," "," "){

  std::string mthn = "[PixelPortcardMap::PixelPortcardMap()]\t\t\t    " ;
  std::vector< std::string > ins = tableMat[0];
  std::map<std::string , int > colM;
  std::vector<std::string > colNames;
/*
  EXTENSION_TABLE_NAME: PIXEL_PORTCARD_MAP (VIEW: CONF_KEY_PORTCARD_MAP_V)

  CONFIG_KEY				    NOT NULL VARCHAR2(80)
  KEY_TYPE				    NOT NULL VARCHAR2(80)
  KEY_ALIAS				    NOT NULL VARCHAR2(80)
  VERSION					     VARCHAR2(40)
  KIND_OF_COND  			    NOT NULL VARCHAR2(40)
  PORT_CARD				    NOT NULL VARCHAR2(200)
  PANEL_NAME				    NOT NULL VARCHAR2(200)
  TBM_MODE					     VARCHAR2(200)
  AOH_CHAN				    NOT NULL NUMBER(38)
*/
  colNames.push_back("CONFIG_KEY"  );
  colNames.push_back("KEY_TYPE"    );
  colNames.push_back("KEY_ALIAS"   );
  colNames.push_back("VERSION"     );
  colNames.push_back("KIND_OF_COND");
  colNames.push_back("PORT_CARD"   );
  colNames.push_back("PANEL_NAME"  );
  colNames.push_back("TBM_MODE"    );
  colNames.push_back("AOH_CHAN"    );
/*
  colNames.push_back("CONFIG_KEY_ID" );
  colNames.push_back("CONFG_KEY"     );
  colNames.push_back("VERSION"       );
  colNames.push_back("KIND_OF_COND"  );
  colNames.push_back("SERIAL_NUMBER" );
  colNames.push_back("PORT_CARD"     );
  colNames.push_back("PANEL_NAME"    );
  colNames.push_back("TBM_MODE"      );
  colNames.push_back("AOH_CHAN"      );
 */ 
  for(unsigned int c = 0 ; c < ins.size() ; c++)
    {
      for(unsigned int n=0; n<colNames.size(); n++)
	{
	  if(tableMat[0][c] == colNames[n])
	    {
	      colM[colNames[n]] = c;
	      break;
	    }
	}
    }//end for
  /*
  for(unsigned int n=0; n<colNames.size(); n++)
    {
      if(colM.find(colNames[n]) == colM.end())
	{
	  std::cerr << __LINE__ << "]\t" << mthn 
	            << "Couldn't find in the database the column with name " << colNames[n] << std::endl;
	  assert(0);
	}
    }
  */
	
	
  std::string portcardname;
  std::string modulename;
  unsigned int aoh;
  std::string aohstring;
  std::string tbmChannel;
  
  for(unsigned int r = 1 ; r < tableMat.size() ; r++){    //Goes to every row of the Matrix
    
    portcardname = tableMat[r][colM["PORT_CARD"]];
    modulename   = tableMat[r][colM["PANEL_NAME"]];
    aohstring    = tableMat[r][colM["AOH_CHAN"]];
    tbmChannel   = tableMat[r][colM["TBM_MODE"]] ;
//     cout << "[PixelPortcardMap::PixelPortcardMap()]\t\t\t    "
// 	 << "Portcardname: " << portcardname
// 	 << "\tmodulename: "   << modulename
// 	 << "\taohstring: "    << aohstring
// 	 << "\ttbmChannel:"   << tbmChannel
// 	 << endl ;
    //aohname.erase(0,20);  // Is going to be change when Umesh put a AOH Channel column in the view.
    aoh = (((unsigned int)atoi(aohstring.c_str())));
    //std::cout<<aoh<<std::endl;
    PixelModuleName module(modulename);
    if (module.modulename()!=modulename)
      {
	std::cout << __LINE__ << "]\t" << mthn << "Modulename: " << modulename          << std::endl;
	std::cout << __LINE__ << "]\t" << mthn << "Parsed to : " << module.modulename() << std::endl;
	assert(0);
      }
    if(tbmChannel == "")
      {
	tbmChannel = "A";// assert(0); // add TBMChannel to the input, then remove assert
      }
    PixelChannel channel(module, tbmChannel);
    std::pair<std::string, int> portcardAndAOH(portcardname, aoh);
    map_[channel] = portcardAndAOH;
  }//end for r


}//end constructor
//*****************************************************************************

PixelPortcardMap::PixelPortcardMap(std::string filename):
  PixelConfigBase(" "," "," "){

  std::string mthn = "[PixelPortcardMap::PixelPortcardMap()]\t\t\t    " ;
  std::ifstream in(filename.c_str());

  if (!in.good()){
    std::cout << __LINE__ << "]\t" << mthn << "Could not open: " << filename <<std::endl;
    throw std::runtime_error("Failed to open file "+filename);
  }
  else {
    std::cout << __LINE__ << "]\t" << mthn << "Reading from: "   << filename <<std::endl;
  }
  
  std::string dummy;

  in >> dummy;
  in >> dummy;
  in >> dummy;
  in >> dummy;
  in >> dummy;

  do {
    
    std::string portcardname;
    std::string modulename;
    std::string TBMChannel = "A";
    std::string aoh_string;
    unsigned int aoh;

    in >> portcardname >> modulename >> aoh_string ;
    if (aoh_string == "A" || aoh_string == "B") // Optionally, the TBM channel may be specified after the module name.  Check for this.
      {
	TBMChannel = aoh_string;
	in >> aoh_string;
      }
    aoh = atoi(aoh_string.c_str());
    
    if (!in.eof() ){
      PixelModuleName module(modulename);
      if (module.modulename()!=modulename){
	std::cout << __LINE__ << "]\t" << mthn << "Modulename: " << modulename          << std::endl;
	std::cout << __LINE__ << "]\t" << mthn << "Parsed to : " << module.modulename() << std::endl;
	assert(0);
      }

      PixelChannel channel(module, TBMChannel);
      std::pair<std::string, int> portcardAndAOH(portcardname, aoh);
      map_[channel] = portcardAndAOH;
    }
	    

  }while (!in.eof());
}

    
/*const std::map<PixelModuleName, int>& PixelPortcardMap::modules(std::string portcard) const{

  std::map<std::string,std::map<PixelModuleName, int> >::const_iterator theportcard=map_.find(portcard);

  if (theportcard==map_.end()) {
    std::cout << "Could not find portcard with name:"<<portcard<<std::endl;
  }

  return theportcard->second;

}*/

PixelPortcardMap::~PixelPortcardMap(){}


void PixelPortcardMap::writeASCII(std::string dir) const {

  
  std::string mthn = "[PixelPortcardMap::writeASCII()]\t\t\t\t    " ;
  if (dir!="") dir+="/";
  string filename=dir+"portcardmap.dat";
  
  ofstream out(filename.c_str());
  if(!out.good()){
    cout << __LINE__ << "]\t" << mthn << "Could not open file: " << filename << endl;
    assert(0);
  }

  out <<"# Portcard          Module                     AOH channel" <<endl;
  std::map< PixelChannel, std::pair<std::string, int> >::const_iterator i=map_.begin();
 for(;i!=map_.end();++i){
    out << i->second.first<<"   "
	<< i->first.module()<<"       "
	<< i->first.TBMChannel()<<"       "
	<< i->second.second<<endl;
  }
  out.close();


}



const std::set< std::pair< std::string, int > > PixelPortcardMap::PortCardAndAOHs(const PixelModuleName& aModule) const
{
	std::set< std::pair< std::string, int > > returnThis;
	
	// Loop over the entire map, searching for elements matching PixelModuleName.  Add matching elements to returnThis.
	for( std::map< PixelChannel, std::pair<std::string, int> >::const_iterator map_itr = map_.begin(); map_itr != map_.end(); ++map_itr )
	{
		if ( map_itr->first.modulename() == aModule.modulename() )
		{
			returnThis.insert(map_itr->second);
		}
	}
	
	return returnThis;
}

// Added by Dario for Debbie (the PixelPortcardMap::portcards is way to slow for the interactive tool)
bool PixelPortcardMap::getName(std::string moduleName, std::string &portcardName)
{
	for( std::map< PixelChannel, std::pair<std::string, int> >::const_iterator map_itr = map_.begin(); map_itr != map_.end(); ++map_itr )
	{
		if ( map_itr->first.modulename() == moduleName )
		{
			portcardName = map_itr->second.first;
			return true ;
		}
	}
        return false ;
}

const std::set< std::string > PixelPortcardMap::portcards(const PixelModuleName& aModule) const
{
	std::set< std::string > returnThis;
	const std::set< std::pair< std::string, int > > portCardAndAOHs = PortCardAndAOHs(aModule);
	for ( std::set< std::pair< std::string, int > >::const_iterator portCardAndAOHs_itr = portCardAndAOHs.begin(); portCardAndAOHs_itr != portCardAndAOHs.end(); ++portCardAndAOHs_itr)
	{
		returnThis.insert( (*portCardAndAOHs_itr).first );
	}
	return returnThis;
}

const std::pair< std::string, int > PixelPortcardMap::PortCardAndAOH(const PixelModuleName& aModule, const PixelTBMChannel& TBMChannel) const
{
	return PortCardAndAOH(PixelChannel(aModule, TBMChannel));
}

const std::pair< std::string, int > PixelPortcardMap::PortCardAndAOH(const PixelModuleName& aModule, const std::string& TBMChannel) const
{
	return PortCardAndAOH(PixelChannel(aModule, TBMChannel));
}

const std::pair< std::string, int > PixelPortcardMap::PortCardAndAOH(const PixelChannel& aChannel) const
{
	std::map< PixelChannel, std::pair<std::string, int> >::const_iterator found = map_.find(aChannel);
	if ( found == map_.end() )
	{
		std::pair< std::string, int > returnThis("none", 0);
		return returnThis;
	}
	else
	{
		return found->second;
	}
}

std::set< PixelModuleName > PixelPortcardMap::modules(std::string portCardName) const
{
	std::set< PixelModuleName > returnThis;
	
	// Loop over the entire map, searching for elements matching portCardName.  Add matching elements to returnThis.
	for( std::map< PixelChannel, std::pair<std::string, int> >::const_iterator map_itr = map_.begin(); map_itr != map_.end(); ++map_itr )
	{
		if ( map_itr->second.first == portCardName )
		{
			returnThis.insert(map_itr->first.module());
		}
	}
	
	return returnThis;
}

std::set< std::string > PixelPortcardMap::portcards(const PixelDetectorConfig* detconfig)
{
	std::set< std::string > returnThis;

	if(detconfig != nullptr){
	
	  //still done done in an awkward way, but this avoids an
          //double nested loop that we had in the first implementation
	  const std::vector <PixelModuleName>& moduleList=detconfig->getModuleList();
	  std::set< std::string > moduleNames;
	  for(std::vector <PixelModuleName>::const_iterator it=moduleList.begin(), it_end=moduleList.end(); it!=it_end; ++it){
	    moduleNames.insert(it->modulename());
	  }

	  for( std::map< PixelChannel, std::pair<std::string, int> >::const_iterator map_itr = map_.begin(); map_itr != map_.end(); ++map_itr )
	    {
	      if ( moduleNames.find(map_itr->first.modulename()) != moduleNames.end() ){
		  returnThis.insert(map_itr->second.first);
		}
	    }
	
	 
	  
	 
	}
	else{
	
	  for( std::map< PixelChannel, std::pair<std::string, int> >::const_iterator map_itr = map_.begin(); map_itr != map_.end(); ++map_itr )
	    {
	      
	      returnThis.insert(map_itr->second.first);
	    }
	  
	}
	
	return returnThis;
}
//=============================================================================================
void PixelPortcardMap::writeXMLHeader(pos::PixelConfigKey key, 
                                      int version, 
                                      std::string path, 
                                      std::ofstream *outstream,
                                      std::ofstream *out1stream,
                                      std::ofstream *out2stream) const
{
  std::string mthn = "[PixelPortcardMap::writeXMLHeader()]\t\t\t    " ;
  std::stringstream fullPath ;
  fullPath << path << "/Pixel_PortCardMap_" << PixelTimeFormatter::getmSecTime() << ".xml" ;
  std::cout << __LINE__ << "]\t" << mthn << "Writing to: " << fullPath.str() << std::endl ;
  
  outstream->open(fullPath.str().c_str()) ;
  
  *outstream << "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"			 	     << std::endl ;
  *outstream << "<ROOT xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance'>" 		 	             << std::endl ;
  *outstream << " <HEADER>"								         	     << std::endl ;
  *outstream << "  <TYPE>"								         	     << std::endl ;
  *outstream << "   <EXTENSION_TABLE_NAME>PIXEL_PORTCARD_MAP</EXTENSION_TABLE_NAME>"          	             << std::endl ;
  *outstream << "   <NAME>Pixel Port Card Map</NAME>"				         	             << std::endl ;
  *outstream << "  </TYPE>"								         	     << std::endl ;
  *outstream << "  <RUN>"								         	     << std::endl ;
  *outstream << "   <RUN_TYPE>Pixel Port Card Map</RUN_TYPE>" 		                                     << std::endl ;
  *outstream << "   <RUN_NUMBER>1</RUN_NUMBER>"					         	             << std::endl ;
  *outstream << "   <RUN_BEGIN_TIMESTAMP>" << pos::PixelTimeFormatter::getTime() << "</RUN_BEGIN_TIMESTAMP>" << std::endl ;
  *outstream << "   <LOCATION>CERN P5</LOCATION>"                                                            << std::endl ; 
  *outstream << "  </RUN>"								         	     << std::endl ;
  *outstream << " </HEADER>"								         	     << std::endl ;
  *outstream << ""										 	     << std::endl ;
  *outstream << " <DATA_SET>"								         	     << std::endl ;
  *outstream << "  <PART>"                                                                                   << std::endl ;
  *outstream << "   <NAME_LABEL>CMS-PIXEL-ROOT</NAME_LABEL>"                                                 << std::endl ;
  *outstream << "   <KIND_OF_PART>Detector ROOT</KIND_OF_PART>"                                              << std::endl ;
  *outstream << "  </PART>"                                                                                  << std::endl ;
  *outstream << "  <VERSION>"             << version      << "</VERSION>"				     << std::endl ;
  *outstream << "  <COMMENT_DESCRIPTION>" << getComment() << "</COMMENT_DESCRIPTION>"			     << std::endl ;
  *outstream << "  <CREATED_BY_USER>"     << getAuthor()  << "</CREATED_BY_USER>"  			     << std::endl ;
}

//=============================================================================================
void PixelPortcardMap::writeXML(std::ofstream *outstream,
                                std::ofstream *out1stream,
                                std::ofstream *out2stream) const 
{
  std::string mthn = "[PixelPortcardMap::writeXML()]\t\t\t    " ;


  std::map< PixelChannel, std::pair<std::string, int> >::const_iterator i=map_.begin();
  for(;i!=map_.end();++i){
     *outstream << "  <DATA>"                                                    			     << std::endl ;
     *outstream << "   <PORT_CARD>"  << i->second.first       << "</PORT_CARD>"  			     << std::endl ;
     *outstream << "   <PANEL_NAME>" << i->first.module()     << "</PANEL_NAME>" 			     << std::endl ;
     *outstream << "   <TBM_MODE>"   << i->first.TBMChannel() << "</TBM_MODE>"   			     << std::endl ;
     *outstream << "   <AOH_CHAN>"   << i->second.second      << "</AOH_CHAN>"  			     << std::endl ;
     *outstream << "  </DATA>"                                                   			     << std::endl ;
   }  
}

//=============================================================================================
void PixelPortcardMap::writeXMLTrailer(std::ofstream *outstream,
                                       std::ofstream *out1stream,
                                       std::ofstream *out2stream) const
{
  std::string mthn = "[PixelPortcardMap::writeXMLTrailer()]\t\t\t    " ;
  
  *outstream << " </DATA_SET>" 						    	 	              	     << std::endl ;
  *outstream << "</ROOT> "								              	     << std::endl ;

  outstream->close() ;
}

