//
// This class provides the mapping between
// portcards and the modules controlled by
// the card
//
//
//
 
#include "CalibFormats/SiPixelObjects/interface/PixelPortcardMap.h"

#include <cassert>

using namespace pos;
using namespace std;

PixelPortcardMap::PixelPortcardMap(std::vector< std::vector < std::string> > &tableMat):PixelConfigBase(" "," "," "){

  std::vector< std::string > ins = tableMat[0];
  std::map<std::string , int > colM;
  std::vector<std::string > colNames;
  colNames.push_back("CONFIG_KEY_ID");//0
  colNames.push_back("CONFG_KEY");//1
  colNames.push_back("VERSION");//2
  colNames.push_back("KIND_OF_COND");
  colNames.push_back("SERIAL_NUMBER");
  colNames.push_back("PORT_CARD");
  colNames.push_back("PANEL_NAME");
  colNames.push_back("AOH_CHAN");

  for(unsigned int c = 0 ; c < ins.size() ; c++){
    for(unsigned int n=0; n<colNames.size(); n++){
      if(tableMat[0][c] == colNames[n]){
	colM[colNames[n]] = c;
	break;
      }
    }
  }//end for
  for(unsigned int n=0; n<colNames.size(); n++){
    if(colM.find(colNames[n]) == colM.end()){
      std::cerr << "[PixelPortcardMap::PixelPortcardMap()]\tCouldn't find in the database the column with name " << colNames[n] << std::endl;
      assert(0);
    }
  }
	
	
	
  std::string portcardname;
  std::string modulename;
  unsigned int aoh;
  std::string aohstring;
  
  for(unsigned int r = 1 ; r < tableMat.size() ; r++){    //Goes to every row of the Matrix

    portcardname = tableMat[r][colM["PORT_CARD"]];
    modulename =   tableMat[r][colM["PANEL_NAME"]];
    aohstring = tableMat[r][colM["AOH_NAME"]];
    //aohname.erase(0,20);  // Is going to be change when Umesh put a AOH Channel column in the view.
    aoh = (((unsigned int)atoi(aohstring.c_str()))+1);
    //std::cout<<aoh<<std::endl;
    PixelModuleName module(modulename);
    if (module.modulename()!=modulename){
      std::cout << "Modulename:"<<modulename<<std::endl;
      std::cout << "Parsed to:"<<module.modulename()<<std::endl;
      assert(0);
    }

    std::string tbmChannel = "A";// assert(0); // add TBMChannel to the input, then remove assert
    PixelChannel channel(module, tbmChannel);
    std::pair<std::string, int> portcardAndAOH(portcardname, aoh);
    map_[channel] = portcardAndAOH;
  }//end for r


}//end constructor
//*****************************************************************************

PixelPortcardMap::PixelPortcardMap(std::string filename):
  PixelConfigBase(" "," "," "){

  std::ifstream in(filename.c_str());

  if (!in.good()){
    std::cout << "[PixelPortcardMap::PixelPortcardMap()]\t\t\t    Could not open: " << filename <<std::endl;
    assert(0);
  }
  else {
    std::cout << "[PixelPortcardMap::PixelPortcardMap()]\t\t\t    Reading from: "   << filename <<std::endl;
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
	std::cout << "Modulename:"<<modulename<<std::endl;
	std::cout << "Parsed to:"<<module.modulename()<<std::endl;
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

  
  if (dir!="") dir+="/";
  string filename=dir+"portcardmap.dat";
  
  ofstream out(filename.c_str());
  if(!out.good()){
    cout << "Could not open file:"<<filename<<endl;
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

std::set< std::string > PixelPortcardMap::portcards()
{
	std::set< std::string > returnThis;
	
	// Loop over the entire map, and add all port cards to returnThis.
	for( std::map< PixelChannel, std::pair<std::string, int> >::const_iterator map_itr = map_.begin(); map_itr != map_.end(); ++map_itr )
	{
		returnThis.insert(map_itr->second.first);
	}
	
	return returnThis;
}
