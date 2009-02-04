//
// This class specifies the settings on the TKPCIFEC
// and the settings on the portcard
//
//
//
//

#include "CalibFormats/SiPixelObjects/interface/PixelPortCardConfig.h"
#include "CalibFormats/SiPixelObjects/interface/PixelPortCardSettingNames.h"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <assert.h>

using namespace std;
using namespace pos::PortCardSettingNames;
using namespace pos;

//added by Umesh
PixelPortCardConfig::PixelPortCardConfig(vector < vector< string> >  &tableMat):PixelConfigBase(" "," "," ")
{
  string mthn = "[PixelPortCardConfig::PixelPortCardConfig()]\t\t    " ;
  map<string , int > colM;
  vector<string> colNames;

  /** 
      View's name : CONF_KEY_PORTCARD_SETTINGS_MV

      CONFIG_KEY_ID				NOT NULL NUMBER(38)
      CONFIG_KEY				NOT NULL VARCHAR2(80)
      VERSION                                            VARCHAR2(40)
      CONDITION_DATA_SET_ID                     NOT NULL NUMBER(38)
      KIND_OF_CONDITION_ID                      NOT NULL NUMBER(38)
      KIND_OF_COND                              NOT NULL VARCHAR2(40)
      SERIAL_NUMBER                                      VARCHAR2(40)
      PORT_CARD_ID                                       NUMBER(38)
      PORT_CARD                                 NOT NULL VARCHAR2(200)
      TRKFEC_NAME                               NOT NULL VARCHAR2(200)
      RINGADDRESS                                        VARCHAR2(40)
      CHANNELADDRESS                                     NUMBER
      CCUADDRESS                                         VARCHAR2(40)
      I2C_CNTRL                                 NOT NULL NUMBER(38)
      I2CSPEED                                           NUMBER(38)
      AOH_BIAS1                                          NUMBER(38)
      AOH_BIAS2                                          NUMBER(38)
      AOH_BIAS3                                          NUMBER(38)
      AOH_BIAS4                                          NUMBER(38)
      AOH_BIAS5                                          NUMBER(38)
      AOH_BIAS6                                          NUMBER(38)
      AOH_GAIN1                                          NUMBER(38)
      AOH_GAIN2                                          NUMBER(38)
      AOH_GAIN3                                          NUMBER(38)
      AOH_GAIN4                                          NUMBER(38)
      AOH_GAIN5                                          NUMBER(38)
      AOH_GAIN6                                          NUMBER(38)
      AOH1_BIAS1                                         NUMBER(38)
      AOH1_BIAS2                                         NUMBER(38)
      AOH1_BIAS3                                         NUMBER(38)
      AOH1_BIAS4                                         NUMBER(38)
      AOH1_BIAS5                                         NUMBER(38)
      AOH1_BIAS6                                         NUMBER(38)
      AOH1_GAIN1                                         NUMBER(38)
      AOH1_GAIN2                                         NUMBER(38)
      AOH1_GAIN3                                         NUMBER(38)
      AOH1_GAIN4                                         NUMBER(38)
      AOH1_GAIN5                                         NUMBER(38)
      AOH1_GAIN6                                         NUMBER(38)
      AOH2_BIAS1                                         NUMBER(38)
      AOH2_BIAS2                                         NUMBER(38)
      AOH2_BIAS3                                         NUMBER(38)
      AOH2_BIAS4                                         NUMBER(38)
      AOH2_BIAS5                                         NUMBER(38)
      AOH2_BIAS6                                         NUMBER(38)
      AOH2_GAIN1                                         NUMBER(38)
      AOH2_GAIN2                                         NUMBER(38)
      AOH2_GAIN3                                         NUMBER(38)
      AOH2_GAIN4                                         NUMBER(38)
      AOH2_GAIN5                                         NUMBER(38)
      AOH2_GAIN6                                         NUMBER(38)
      AOH3_BIAS1                                         NUMBER(38)
      AOH3_BIAS2                                         NUMBER(38)
      AOH3_BIAS3                                         NUMBER(38)
      AOH3_BIAS4                                         NUMBER(38)
      AOH3_BIAS5                                         NUMBER(38)
      AOH3_BIAS6                                         NUMBER(38)
      AOH3_GAIN1                                         NUMBER(38)
      AOH3_GAIN2                                         NUMBER(38)
      AOH3_GAIN3                                         NUMBER(38)
      AOH3_GAIN4                                         NUMBER(38)
      AOH3_GAIN5                                         NUMBER(38)
      AOH3_GAIN6                                         NUMBER(38)
      AOH4_BIAS1                                         NUMBER(38)
      AOH4_BIAS2                                         NUMBER(38)
      AOH4_BIAS3                                         NUMBER(38)
      AOH4_BIAS4                                         NUMBER(38)
      AOH4_BIAS5                                         NUMBER(38)
      AOH4_BIAS6                                         NUMBER(38)
      AOH4_GAIN1                                         NUMBER(38)
      AOH4_GAIN2                                         NUMBER(38)
      AOH4_GAIN3                                         NUMBER(38)
      AOH4_GAIN4                                         NUMBER(38)
      AOH4_GAIN5                                         NUMBER(38)
      AOH4_GAIN6                                         NUMBER(38)
      AOH5_BIAS1                                         NUMBER(38)
      AOH5_BIAS2                                         NUMBER(38)
      AOH5_BIAS3                                         NUMBER(38)
      AOH5_BIAS4                                         NUMBER(38)
      AOH5_BIAS5                                         NUMBER(38)
      AOH5_BIAS6                                         NUMBER(38)
      AOH5_GAIN1                                         NUMBER(38)
      AOH5_GAIN2                                         NUMBER(38)
      AOH5_GAIN3                                         NUMBER(38)
      AOH5_GAIN4                                         NUMBER(38)
      AOH5_GAIN5                                         NUMBER(38)
      AOH5_GAIN6                                         NUMBER(38)
      AOH6_BIAS1                                         NUMBER(38)
      AOH6_BIAS2                                         NUMBER(38)
      AOH6_BIAS3                                         NUMBER(38)
      AOH6_BIAS4                                         NUMBER(38)
      AOH6_BIAS5                                         NUMBER(38)
      AOH6_BIAS6                                         NUMBER(38)
      AOH6_GAIN1                                         NUMBER(38)
      AOH6_GAIN2                                         NUMBER(38)
      AOH6_GAIN3                                         NUMBER(38)
      AOH6_GAIN4                                         NUMBER(38)
      AOH6_GAIN5                                         NUMBER(38)
      AOH6_GAIN6                                         NUMBER(38)
      DELAY25_GCR                               NOT NULL NUMBER(38)
      DELAY25_SCL                               NOT NULL NUMBER(38)
      DELAY25_TRG                               NOT NULL NUMBER(38)
      DELAY25_SDA                               NOT NULL NUMBER(38)
      DELAY25_RCL                               NOT NULL NUMBER(38)
      DELAY25_RDA                               NOT NULL NUMBER(38)
      DOH_BIAS0                                 NOT NULL NUMBER(38)
      DOH_BIAS1                                 NOT NULL NUMBER(38)
      DOH_SEU_GAIN                              NOT NULL NUMBER(38)
      PLL_CTR1                                  NOT NULL NUMBER(38)
      PLL_CTR2                                  NOT NULL NUMBER(38)
      PLL_CTR3                                  NOT NULL NUMBER(38)
      PLL_CTR4_5                                NOT NULL NUMBER(38)  
  */

  colNames.push_back("CONFIG_KEY_ID"           );
  colNames.push_back("CONFIG_KEY"              );
  colNames.push_back("VERSION"                 );
  colNames.push_back("CONDITION_DATA_SET_ID"   );
  colNames.push_back("KIND_OF_CONDITION_ID"    );
  colNames.push_back("KIND_OF_COND"            );
  colNames.push_back("SERIAL_NUMBER"           );
  colNames.push_back("PORT_CARD_ID"            );
  colNames.push_back("PORT_CARD"               );
  colNames.push_back("TRKFEC_NAME"             );
  colNames.push_back("RINGADDRESS"             );
  colNames.push_back("CHANNELADDRESS"          );
  colNames.push_back("CCUADDRESS"              );
  colNames.push_back("I2C_CNTRL"               );
  colNames.push_back("I2CSPEED"                );
  colNames.push_back("AOH_BIAS1"               );
  colNames.push_back("AOH_BIAS2"               );
  colNames.push_back("AOH_BIAS3"               );
  colNames.push_back("AOH_BIAS4"               );
  colNames.push_back("AOH_BIAS5"               );
  colNames.push_back("AOH_BIAS6"               );
  colNames.push_back("AOH_GAIN1"               );
  colNames.push_back("AOH_GAIN2"               );
  colNames.push_back("AOH_GAIN3"               );
  colNames.push_back("AOH_GAIN4"               );
  colNames.push_back("AOH_GAIN5"               );
  colNames.push_back("AOH_GAIN6"               );
  colNames.push_back("AOH1_BIAS1"              );
  colNames.push_back("AOH1_BIAS2"              );
  colNames.push_back("AOH1_BIAS3"              );
  colNames.push_back("AOH1_BIAS4"              );
  colNames.push_back("AOH1_BIAS5"              );
  colNames.push_back("AOH1_BIAS6"              );
  colNames.push_back("AOH1_GAIN1"              );
  colNames.push_back("AOH1_GAIN2"              );
  colNames.push_back("AOH1_GAIN3"              );
  colNames.push_back("AOH1_GAIN4"              );
  colNames.push_back("AOH1_GAIN5"              );
  colNames.push_back("AOH1_GAIN6"              );
  colNames.push_back("AOH2_BIAS1"              );
  colNames.push_back("AOH2_BIAS2"              );
  colNames.push_back("AOH2_BIAS3"              );
  colNames.push_back("AOH2_BIAS4"              );
  colNames.push_back("AOH2_BIAS5"              );
  colNames.push_back("AOH2_BIAS6"              );
  colNames.push_back("AOH2_GAIN1"              );
  colNames.push_back("AOH2_GAIN2"              );
  colNames.push_back("AOH2_GAIN3"              );
  colNames.push_back("AOH2_GAIN4"              );
  colNames.push_back("AOH2_GAIN5"              );
  colNames.push_back("AOH2_GAIN6"              );
  colNames.push_back("AOH3_BIAS1"              );
  colNames.push_back("AOH3_BIAS2"              );
  colNames.push_back("AOH3_BIAS3"              );
  colNames.push_back("AOH3_BIAS4"              );
  colNames.push_back("AOH3_BIAS5"              );
  colNames.push_back("AOH3_BIAS6"              );
  colNames.push_back("AOH3_GAIN1"              );
  colNames.push_back("AOH3_GAIN2"              );
  colNames.push_back("AOH3_GAIN3"              );
  colNames.push_back("AOH3_GAIN4"              );
  colNames.push_back("AOH3_GAIN5"              );
  colNames.push_back("AOH3_GAIN6"              );
  colNames.push_back("AOH4_BIAS1"              );
  colNames.push_back("AOH4_BIAS2"              );
  colNames.push_back("AOH4_BIAS3"              );
  colNames.push_back("AOH4_BIAS4"              );
  colNames.push_back("AOH4_BIAS5"              );
  colNames.push_back("AOH4_BIAS6"              );
  colNames.push_back("AOH4_GAIN1"              );
  colNames.push_back("AOH4_GAIN2"              );
  colNames.push_back("AOH4_GAIN3"              );
  colNames.push_back("AOH4_GAIN4"              );
  colNames.push_back("AOH4_GAIN5"              );
  colNames.push_back("AOH4_GAIN6"              );
  colNames.push_back("AOH5_BIAS1"              );
  colNames.push_back("AOH5_BIAS2"              );
  colNames.push_back("AOH5_BIAS3"              );
  colNames.push_back("AOH5_BIAS4"              );
  colNames.push_back("AOH5_BIAS5"              );
  colNames.push_back("AOH5_BIAS6"              );
  colNames.push_back("AOH5_GAIN1"              );
  colNames.push_back("AOH5_GAIN2"              );
  colNames.push_back("AOH5_GAIN3"              );
  colNames.push_back("AOH5_GAIN4"              );
  colNames.push_back("AOH5_GAIN5"              );
  colNames.push_back("AOH5_GAIN6"              );
  colNames.push_back("AOH6_BIAS1"              );
  colNames.push_back("AOH6_BIAS2"              );
  colNames.push_back("AOH6_BIAS3"              );
  colNames.push_back("AOH6_BIAS4"              );
  colNames.push_back("AOH6_BIAS5"              );
  colNames.push_back("AOH6_BIAS6"              );
  colNames.push_back("AOH6_GAIN1"              );
  colNames.push_back("AOH6_GAIN2"              );
  colNames.push_back("AOH6_GAIN3"              );
  colNames.push_back("AOH6_GAIN4"              );
  colNames.push_back("AOH6_GAIN5"              );
  colNames.push_back("AOH6_GAIN6"              );
  colNames.push_back("DELAY25_GCR"             );
  colNames.push_back("DELAY25_SCL"             );
  colNames.push_back("DELAY25_TRG"             );
  colNames.push_back("DELAY25_SDA"             );
  colNames.push_back("DELAY25_RCL"             );
  colNames.push_back("DELAY25_RDA"             );
  colNames.push_back("DOH_BIAS0"               );
  colNames.push_back("DOH_BIAS1"               );
  colNames.push_back("DOH_SEU_GAIN"            );
  colNames.push_back("PLL_CTR1"                );
  colNames.push_back("PLL_CTR2"                );
  colNames.push_back("PLL_CTR3"                );
  colNames.push_back("PLL_CTR4_5"              );

         
  for(unsigned int c = 0 ; c < tableMat[0].size() ; c++)
    {
      for(unsigned int n=0; n<colNames.size(); n++)
	{
	  if(tableMat[0][c] == colNames[n]){
	    colM[colNames[n]] = c;
	    break;
	  }
	}
    }//end for
  for(unsigned int n=0; n<colNames.size(); n++)
    {
      if(colM.find(colNames[n]) == colM.end())
	{
	  std::cerr << mthn << "\tCouldn't find in the database the column with name " << colNames[n] << std::endl;
	  assert(0);
	}
    }

  portcardname_ = tableMat[1][colM["PORT_CARD"]] ;
  cout << mthn << "Loading PortCard " << portcardname_ << endl ;
  if(portcardname_.find("FPix") != std::string::npos)
    {
      type_ = "fpix" ;
    }
  else if(portcardname_.find("FPix") != std::string::npos)
    {
      type_ = "bpix" ;
    }
  fillNameToAddress();
  fillDBToFileAddress() ;
  
  TKFECID_ = tableMat[1][colM["TRKFEC_NAME"]];
//   01234567890123
//   FPix_Trk_FEC_1
  TKFECID_.replace(0,13, "tkfec") ;
  ringAddress_    = atoi(tableMat[1][colM["RINGADDRESS"]].c_str()    ) ;
  ccuAddress_	  = atoi(tableMat[1][colM["CCUADDRESS"]].c_str()     ) ;
  channelAddress_ = atoi(tableMat[1][colM["CHANNELADDRESS"]].c_str() ) ;
  i2cSpeed_       = atoi(tableMat[1][colM["I2CSPEED"]].c_str()       ) ;

  cout << mthn << 
    "ringAddress_\t"    << ringAddress_	    << endl <<
    "ccuAddress_\t"     << ccuAddress_	    << endl <<
    "channelAddress_\t" << channelAddress_  << endl <<
    "i2cSpeed_\t"	<< i2cSpeed_        << endl ;
  

  for(unsigned int col = 0 ; col < tableMat[1].size() ; col++)    //Goes to every column of the Matrix
    {
      std::string settingName;
      unsigned int i2c_address;
      unsigned int i2c_values;
      
      settingName = tableMat[0][col] ;
      i2c_values  = atoi(tableMat[1][col].c_str()) ;
      
      // Special handling for AOHX_GainY
      if( type_ == "fpix" && settingName.find("AOH_") != string::npos && settingName.find("GAIN") != string::npos // contains both "AOH_" and "Gain"
	       && settingName.find("123") == string::npos && settingName.find("456") == string::npos ) // does not contain "123" or "456"
	{
	  setDataBaseAOHGain(settingName, i2c_values);
	  cout << mthn << "Setting " << settingName << "\tto value " << std::hex << i2c_values << std::dec << std::endl ;
	}
      else if(type_ == "bpix" && settingName.find("AOH") != string::npos && settingName.find("GAIN") != string::npos // contains both "AOH" and "Gain"
	      && settingName.find("AOH_") == string::npos                                                            // must not contain AOH_ 'cause this is for forward
	      && settingName.find("123")  == string::npos && settingName.find("456") == string::npos )               // does not contain "123" or "456"
	{
	  setDataBaseAOHGain(settingName, i2c_values);
	  cout << mthn << "Setting " << settingName << "\tto value " << std::hex << i2c_values << std::dec << std::endl ;
	}
      else // no special handling for this name
	{
	  map<string,string>::iterator iter = nameDBtoFileConversion_.find(settingName);
	  if(iter == nameDBtoFileConversion_.end()) continue ;
	  map<string, unsigned int>::iterator foundName_itr = nameToAddress_.find(nameDBtoFileConversion_[settingName]);
	  
	  if ( foundName_itr != nameToAddress_.end() )
	    {
	      i2c_address = foundName_itr->second;
	    }
	  else
	    {
	      i2c_address = strtoul(settingName.c_str(), 0, 16); // convert string to integer using base 16
	    }
	  pair<unsigned int, unsigned int> p(i2c_address, i2c_values);
	  cout << mthn << "Setting\t" << "|"<<settingName<<"|->"<<nameDBtoFileConversion_[settingName] 
	       << "\twith pair:\t(" 
	       << i2c_address
	       << ","
	       << i2c_values
	       << ")"
	       << endl ;
	  device_.push_back(p);
	}
    } // End of table columns
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
PixelPortCardConfig::PixelPortCardConfig(std::string filename):
  PixelConfigBase(" "," "," "){

  //std::cout << "filename:"<<filename<<std::endl;

  unsigned int portcardpos=filename.find(std::string("portcard_"));
  //std::cout << "portcardpos:"<<portcardpos<<std::endl;
  assert(portcardpos!=std::string::npos);
  unsigned int datpos=filename.find(std::string(".dat"));
  //std::cout << "datpos:"<<datpos<<std::endl;
  assert(datpos!=std::string::npos);
  assert(datpos>portcardpos);
  
  portcardname_=filename.substr(portcardpos+9,datpos-portcardpos-9);

  //std::cout << "Portcard name extracted from file name:"<<portcardname_<<std::endl;

  std::ifstream in(filename.c_str());
  
  if(!in.good()){
    std::cout<<"[PixelPortCardConfig::PixelPortCardConfig()]\t\t    Could not open: "<< filename << std::endl;
    assert(0);
  }
  else {
    std::cout<<"[PixelPortCardConfig::PixelPortCardConfig()]\t\t    Opened: "        << filename << std::endl;
  }
  
  string dummy;

  in >> dummy;
  if ( dummy == "Name:" ) // check that port card name matches the file name
  {
    in >> dummy; assert( dummy==portcardname_ );
    in >> dummy;
  }
  if ( dummy == "Type:" ) // read in the type, defaulting to "fpix" if not specified
  {
    in >> type_;
    assert( type_ == "fpix" || type_ == "bpix" );
    in >> dummy;
  }
  else
  {
    type_ = "fpix";
  }
  fillNameToAddress();
  assert(dummy=="TKFECID:");        in >> TKFECID_;
  in >> dummy; assert(dummy=="ringAddress:");    in >> std::hex >> ringAddress_;
  in >> dummy; assert(dummy=="ccuAddress:");     in >> std::hex >> ccuAddress_;
  in >> dummy; assert(dummy=="channelAddress:"); in >> std::hex >> channelAddress_;
  in >> dummy; assert(dummy=="i2cSpeed:");       in >> std::hex >> i2cSpeed_;
    
  //std::cout<<TKFECAddress_<<", "<<ringAddress_<<", "<<ccuAddress_<<", "<<channelAddress_<<", "<<i2cSpeed_<<std::endl;
  
  assert( nameToAddress_.size() != 0 );
  do {
      
    std::string settingName;
    unsigned int i2c_address;
    unsigned int i2c_values;
    
    in >> settingName >> std::hex >> i2c_values >> std::dec;
    if (in.eof()) break;
    
    if ( settingName[settingName.size()-1] == ':' ) settingName.resize( settingName.size()-1 ); // remove ':' from end of string, if it's there
    
    // Special handling for AOHX_GainY
    if ( settingName.find("AOH") != string::npos && settingName.find("Gain") != string::npos // contains both "AOH" and "Gain"
      && settingName.find("123") == string::npos && settingName.find("456") == string::npos ) // does not contain "123" or "456"
    {
    	setAOHGain(settingName, i2c_values);
    }
    else if ( settingName == k_PLL_CTR4 || settingName == k_PLL_CTR5 ) // special handling
    {
    	unsigned int last_CTR2 = 0x0;
    	if ( containsSetting(k_PLL_CTR2) ) last_CTR2 = getdeviceValuesForSetting( k_PLL_CTR2 );
    	
    	device_.push_back( make_pair(getdeviceAddressForSetting(k_PLL_CTR2), new_PLL_CTR2_value(settingName, last_CTR2)) );
    	device_.push_back( make_pair(getdeviceAddressForSetting(k_PLL_CTR4or5), i2c_values) );
    }
    else // no special handling for this name
    {
    	std::map<std::string, unsigned int>::iterator foundName_itr = nameToAddress_.find(settingName);
    
    	if ( foundName_itr != nameToAddress_.end() )
    	{
    		i2c_address = foundName_itr->second;
    	}
    	else
    	{
    		i2c_address = strtoul(settingName.c_str(), 0, 16); // convert string to integer using base 16
    	}
    	pair<unsigned int, unsigned int> p(i2c_address, i2c_values);
	device_.push_back(p);
    }
  }
  while (!in.eof());
  
  in.close();

}

unsigned int PixelPortCardConfig::new_PLL_CTR2_value(std::string CTR4or5, unsigned int last_CTR2) const
{
	if      ( CTR4or5 == k_PLL_CTR4 ) return 0xdf & last_CTR2;
	else if ( CTR4or5 == k_PLL_CTR5 ) return 0x20 | last_CTR2;
	else assert(0);
}

void PixelPortCardConfig::setAOHGain(std::string settingName, unsigned int value)
{
	assert( settingName.find("AOH") != string::npos && settingName.find("Gain") != string::npos // contains both "AOH" and "Gain"
        && settingName.find("123") == string::npos && settingName.find("456") == string::npos ); // does not contain "123" or "456"
	
	unsigned int i2c_address;
	
	// Get the i2c address of this AOH, and the channel on the AOH.
	string::size_type GainPosition = settingName.find("Gain");
	unsigned int whichAOH;
	if ( settingName[GainPosition-2] == 'H' ) whichAOH = 0; // fpix
	else  // bpix
	{
		char whichAOHDigit[2]={0,0};
		whichAOHDigit[0]=settingName[GainPosition-2];
		whichAOH = atoi( whichAOHDigit );
	}
	char digit[2]={0,0};
	digit[0]=settingName[GainPosition+4];
	unsigned int channelOnAOH = atoi( digit );
	assert( (type_=="fpix" && whichAOH==0)||(type_=="bpix" && 1 <= whichAOH&&whichAOH <= 4) );
	assert( 1 <= channelOnAOH && channelOnAOH <= 6 );
	
	if      ( whichAOH == 0 && channelOnAOH <= 3 ) i2c_address = k_fpix_AOH_Gain123_address;
	else if ( whichAOH == 0 && channelOnAOH >= 4 ) i2c_address = k_fpix_AOH_Gain456_address;
	else if ( whichAOH == 1 && channelOnAOH <= 3 ) i2c_address = k_bpix_AOH1_Gain123_address;
	else if ( whichAOH == 1 && channelOnAOH >= 4 ) i2c_address = k_bpix_AOH1_Gain456_address;
	else if ( whichAOH == 2 && channelOnAOH <= 3 ) i2c_address = k_bpix_AOH2_Gain123_address;
	else if ( whichAOH == 2 && channelOnAOH >= 4 ) i2c_address = k_bpix_AOH2_Gain456_address;
	else if ( whichAOH == 3 && channelOnAOH <= 3 ) i2c_address = k_bpix_AOH3_Gain123_address;
	else if ( whichAOH == 3 && channelOnAOH >= 4 ) i2c_address = k_bpix_AOH3_Gain456_address;
	else if ( whichAOH == 4 && channelOnAOH <= 3 ) i2c_address = k_bpix_AOH4_Gain123_address;
	else if ( whichAOH == 4 && channelOnAOH >= 4 ) i2c_address = k_bpix_AOH4_Gain456_address;
	else assert(0);
	
	// Search for this address in the previously-defined settings.
	bool foundOne = false;
	for (unsigned int i=0;i<device_.size();i++)
	{
		if ( device_[i].first == i2c_address ) // Change this setting in all previous instances
		{
			foundOne = true;
			unsigned int oldValue = device_[i].second;
			if      ( channelOnAOH%3 == 1 )
				device_[i].second = (0x3c & oldValue) + ((value & 0x3)<<0); // replace bits 0 and 1 with value
			else if ( channelOnAOH%3 == 2 )
				device_[i].second = (0x33 & oldValue) + ((value & 0x3)<<2); // replace bits 2 and 3 with value
			else if ( channelOnAOH%3 == 0 )
				device_[i].second = (0x0f & oldValue) + ((value & 0x3)<<4); // replace bits 4 and 5 with value
			else assert(0);
			//std::cout << "Changed setting "<< k_fpix_AOH_Gain123 <<"(address 0x"<<std::hex<<k_fpix_AOH_Gain123_address<<") from 0x"<<oldValue<<" to 0x"<< device_[i].second << std::dec <<"\n";
		}
	}
	if ( foundOne ) return;
	else // If this was not set previously, add this setting with the other two gains set to zero.
	{
		unsigned int i2c_value;
		if      ( channelOnAOH%3 == 1 ) i2c_value  = ((value & 0x3)<<0);
		else if ( channelOnAOH%3 == 2 ) i2c_value  = ((value & 0x3)<<2);
		else if ( channelOnAOH%3 == 0 ) i2c_value  = ((value & 0x3)<<4);
		else assert(0);
		
		pair<unsigned int, unsigned int> p(i2c_address, i2c_value);
		device_.push_back(p);
		return;
	}
}

void PixelPortCardConfig::setDataBaseAOHGain(std::string settingName, unsigned int value)
{
	unsigned int i2c_address;
	
	// Get the i2c address of this AOH, and the channel on the AOH.
	string::size_type GainPosition = settingName.find("GAIN");
	unsigned int whichAOH;
	if(type_ == "fpix")
	  {
	    whichAOH = 0 ; // fpix
	  }
	else  // bpix
	{
		char whichAOHDigit[2]={0,0};
		whichAOHDigit[0]=settingName[GainPosition-2];
		whichAOH = atoi( whichAOHDigit );
	}
	char digit[2]={0,0};
	digit[0]=settingName[GainPosition+4];
	unsigned int channelOnAOH = atoi( digit );
	assert( (type_=="fpix" && whichAOH==0)||(type_=="bpix" && 1 <= whichAOH&&whichAOH <= 4) );
	assert( 1 <= channelOnAOH && channelOnAOH <= 6 );
	
	if      ( whichAOH == 0 && channelOnAOH <= 3 ) i2c_address = k_fpix_AOH_Gain123_address;
	else if ( whichAOH == 0 && channelOnAOH >= 4 ) i2c_address = k_fpix_AOH_Gain456_address;
	else if ( whichAOH == 1 && channelOnAOH <= 3 ) i2c_address = k_bpix_AOH1_Gain123_address;
	else if ( whichAOH == 1 && channelOnAOH >= 4 ) i2c_address = k_bpix_AOH1_Gain456_address;
	else if ( whichAOH == 2 && channelOnAOH <= 3 ) i2c_address = k_bpix_AOH2_Gain123_address;
	else if ( whichAOH == 2 && channelOnAOH >= 4 ) i2c_address = k_bpix_AOH2_Gain456_address;
	else if ( whichAOH == 3 && channelOnAOH <= 3 ) i2c_address = k_bpix_AOH3_Gain123_address;
	else if ( whichAOH == 3 && channelOnAOH >= 4 ) i2c_address = k_bpix_AOH3_Gain456_address;
	else if ( whichAOH == 4 && channelOnAOH <= 3 ) i2c_address = k_bpix_AOH4_Gain123_address;
	else if ( whichAOH == 4 && channelOnAOH >= 4 ) i2c_address = k_bpix_AOH4_Gain456_address;
	else assert(0);
	
	// Search for this address in the previously-defined settings.
	bool foundOne = false;
	for (unsigned int i=0;i<device_.size();i++)
	{
		if ( device_[i].first == i2c_address ) // Change this setting in all previous instances
		{
			foundOne = true;
			unsigned int oldValue = device_[i].second;
			if      ( channelOnAOH%3 == 1 )
				device_[i].second = (0x3c & oldValue) + ((value & 0x3)<<0); // replace bits 0 and 1 with value
			else if ( channelOnAOH%3 == 2 )
				device_[i].second = (0x33 & oldValue) + ((value & 0x3)<<2); // replace bits 2 and 3 with value
			else if ( channelOnAOH%3 == 0 )
				device_[i].second = (0x0f & oldValue) + ((value & 0x3)<<4); // replace bits 4 and 5 with value
			else assert(0);
			//std::cout << "Changed setting "<< k_fpix_AOH_Gain123 <<"(address 0x"<<std::hex<<k_fpix_AOH_Gain123_address<<") from 0x"<<oldValue<<" to 0x"<< device_[i].second << std::dec <<"\n";
		}
	}
	if ( foundOne ) return;
	else // If this was not set previously, add this setting with the other two gains set to zero.
	{
		unsigned int i2c_value;
		if      ( channelOnAOH%3 == 1 ) i2c_value  = ((value & 0x3)<<0);
		else if ( channelOnAOH%3 == 2 ) i2c_value  = ((value & 0x3)<<2);
		else if ( channelOnAOH%3 == 0 ) i2c_value  = ((value & 0x3)<<4);
		else assert(0);
		
		pair<unsigned int, unsigned int> p(i2c_address, i2c_value);
		device_.push_back(p);
		return;
	}
}

void PixelPortCardConfig::fillNameToAddress()
{
	if ( nameToAddress_.size() != 0 ) return;
	
	if ( type_ == "fpix" )
	{
		nameToAddress_[PortCardSettingNames::k_AOH_Bias1] = PortCardSettingNames::k_fpix_AOH_Bias1_address;
		nameToAddress_[PortCardSettingNames::k_AOH_Bias2] = PortCardSettingNames::k_fpix_AOH_Bias2_address;
		nameToAddress_[PortCardSettingNames::k_AOH_Bias3] = PortCardSettingNames::k_fpix_AOH_Bias3_address;
		nameToAddress_[PortCardSettingNames::k_AOH_Bias4] = PortCardSettingNames::k_fpix_AOH_Bias4_address;
		nameToAddress_[PortCardSettingNames::k_AOH_Bias5] = PortCardSettingNames::k_fpix_AOH_Bias5_address;
		nameToAddress_[PortCardSettingNames::k_AOH_Bias6] = PortCardSettingNames::k_fpix_AOH_Bias6_address;
		nameToAddress_[PortCardSettingNames::k_AOH_Gain123] = PortCardSettingNames::k_fpix_AOH_Gain123_address;
		nameToAddress_[PortCardSettingNames::k_AOH_Gain456] = PortCardSettingNames::k_fpix_AOH_Gain456_address;
		
		nameToAddress_[PortCardSettingNames::k_PLL_CTR1] = PortCardSettingNames::k_fpix_PLL_CTR1_address;
		nameToAddress_[PortCardSettingNames::k_PLL_CTR2] = PortCardSettingNames::k_fpix_PLL_CTR2_address;
		nameToAddress_[PortCardSettingNames::k_PLL_CTR3] = PortCardSettingNames::k_fpix_PLL_CTR3_address;
		nameToAddress_[PortCardSettingNames::k_PLL_CTR4or5] = PortCardSettingNames::k_fpix_PLL_CTR4or5_address;
		
		nameToAddress_[PortCardSettingNames::k_Delay25_RDA] = PortCardSettingNames::k_fpix_Delay25_RDA_address;
		nameToAddress_[PortCardSettingNames::k_Delay25_RCL] = PortCardSettingNames::k_fpix_Delay25_RCL_address;
		nameToAddress_[PortCardSettingNames::k_Delay25_SDA] = PortCardSettingNames::k_fpix_Delay25_SDA_address;
		nameToAddress_[PortCardSettingNames::k_Delay25_TRG] = PortCardSettingNames::k_fpix_Delay25_TRG_address;
		nameToAddress_[PortCardSettingNames::k_Delay25_SCL] = PortCardSettingNames::k_fpix_Delay25_SCL_address;
		nameToAddress_[PortCardSettingNames::k_Delay25_GCR] = PortCardSettingNames::k_fpix_Delay25_GCR_address;
		
		nameToAddress_[PortCardSettingNames::k_DOH_Ch0Bias_CLK]  = PortCardSettingNames::k_fpix_DOH_Ch0Bias_CLK_address;
		nameToAddress_[PortCardSettingNames::k_DOH_Dummy]        = PortCardSettingNames::k_fpix_DOH_Dummy_address;
		nameToAddress_[PortCardSettingNames::k_DOH_Ch1Bias_Data] = PortCardSettingNames::k_fpix_DOH_Ch1Bias_Data_address;
		nameToAddress_[PortCardSettingNames::k_DOH_Gain_SEU]     = PortCardSettingNames::k_fpix_DOH_Gain_SEU_address;
	}
	else if ( type_ == "bpix" )
	{
		nameToAddress_[PortCardSettingNames::k_AOH1_Bias1] = PortCardSettingNames::k_bpix_AOH1_Bias1_address;
		nameToAddress_[PortCardSettingNames::k_AOH1_Bias2] = PortCardSettingNames::k_bpix_AOH1_Bias2_address;
		nameToAddress_[PortCardSettingNames::k_AOH1_Bias3] = PortCardSettingNames::k_bpix_AOH1_Bias3_address;
		nameToAddress_[PortCardSettingNames::k_AOH1_Bias4] = PortCardSettingNames::k_bpix_AOH1_Bias4_address;
		nameToAddress_[PortCardSettingNames::k_AOH1_Bias5] = PortCardSettingNames::k_bpix_AOH1_Bias5_address;
		nameToAddress_[PortCardSettingNames::k_AOH1_Bias6] = PortCardSettingNames::k_bpix_AOH1_Bias6_address;
		nameToAddress_[PortCardSettingNames::k_AOH1_Gain123] = PortCardSettingNames::k_bpix_AOH1_Gain123_address;
		nameToAddress_[PortCardSettingNames::k_AOH1_Gain456] = PortCardSettingNames::k_bpix_AOH1_Gain456_address;
		
		nameToAddress_[PortCardSettingNames::k_AOH2_Bias1] = PortCardSettingNames::k_bpix_AOH2_Bias1_address;
		nameToAddress_[PortCardSettingNames::k_AOH2_Bias2] = PortCardSettingNames::k_bpix_AOH2_Bias2_address;
		nameToAddress_[PortCardSettingNames::k_AOH2_Bias3] = PortCardSettingNames::k_bpix_AOH2_Bias3_address;
		nameToAddress_[PortCardSettingNames::k_AOH2_Bias4] = PortCardSettingNames::k_bpix_AOH2_Bias4_address;
		nameToAddress_[PortCardSettingNames::k_AOH2_Bias5] = PortCardSettingNames::k_bpix_AOH2_Bias5_address;
		nameToAddress_[PortCardSettingNames::k_AOH2_Bias6] = PortCardSettingNames::k_bpix_AOH2_Bias6_address;
		nameToAddress_[PortCardSettingNames::k_AOH2_Gain123] = PortCardSettingNames::k_bpix_AOH2_Gain123_address;
		nameToAddress_[PortCardSettingNames::k_AOH2_Gain456] = PortCardSettingNames::k_bpix_AOH2_Gain456_address;
		
		nameToAddress_[PortCardSettingNames::k_AOH3_Bias1] = PortCardSettingNames::k_bpix_AOH3_Bias1_address;
		nameToAddress_[PortCardSettingNames::k_AOH3_Bias2] = PortCardSettingNames::k_bpix_AOH3_Bias2_address;
		nameToAddress_[PortCardSettingNames::k_AOH3_Bias3] = PortCardSettingNames::k_bpix_AOH3_Bias3_address;
		nameToAddress_[PortCardSettingNames::k_AOH3_Bias4] = PortCardSettingNames::k_bpix_AOH3_Bias4_address;
		nameToAddress_[PortCardSettingNames::k_AOH3_Bias5] = PortCardSettingNames::k_bpix_AOH3_Bias5_address;
		nameToAddress_[PortCardSettingNames::k_AOH3_Bias6] = PortCardSettingNames::k_bpix_AOH3_Bias6_address;
		nameToAddress_[PortCardSettingNames::k_AOH3_Gain123] = PortCardSettingNames::k_bpix_AOH3_Gain123_address;
		nameToAddress_[PortCardSettingNames::k_AOH3_Gain456] = PortCardSettingNames::k_bpix_AOH3_Gain456_address;
		
		nameToAddress_[PortCardSettingNames::k_AOH4_Bias1] = PortCardSettingNames::k_bpix_AOH4_Bias1_address;
		nameToAddress_[PortCardSettingNames::k_AOH4_Bias2] = PortCardSettingNames::k_bpix_AOH4_Bias2_address;
		nameToAddress_[PortCardSettingNames::k_AOH4_Bias3] = PortCardSettingNames::k_bpix_AOH4_Bias3_address;
		nameToAddress_[PortCardSettingNames::k_AOH4_Bias4] = PortCardSettingNames::k_bpix_AOH4_Bias4_address;
		nameToAddress_[PortCardSettingNames::k_AOH4_Bias5] = PortCardSettingNames::k_bpix_AOH4_Bias5_address;
		nameToAddress_[PortCardSettingNames::k_AOH4_Bias6] = PortCardSettingNames::k_bpix_AOH4_Bias6_address;
		nameToAddress_[PortCardSettingNames::k_AOH4_Gain123] = PortCardSettingNames::k_bpix_AOH4_Gain123_address;
		nameToAddress_[PortCardSettingNames::k_AOH4_Gain456] = PortCardSettingNames::k_bpix_AOH4_Gain456_address;
		
		nameToAddress_[PortCardSettingNames::k_PLL_CTR1] = PortCardSettingNames::k_bpix_PLL_CTR1_address;
		nameToAddress_[PortCardSettingNames::k_PLL_CTR2] = PortCardSettingNames::k_bpix_PLL_CTR2_address;
		nameToAddress_[PortCardSettingNames::k_PLL_CTR3] = PortCardSettingNames::k_bpix_PLL_CTR3_address;
		nameToAddress_[PortCardSettingNames::k_PLL_CTR4or5] = PortCardSettingNames::k_bpix_PLL_CTR4or5_address;
		
		nameToAddress_[PortCardSettingNames::k_Delay25_RDA] = PortCardSettingNames::k_bpix_Delay25_RDA_address;
		nameToAddress_[PortCardSettingNames::k_Delay25_RCL] = PortCardSettingNames::k_bpix_Delay25_RCL_address;
		nameToAddress_[PortCardSettingNames::k_Delay25_SDA] = PortCardSettingNames::k_bpix_Delay25_SDA_address;
		nameToAddress_[PortCardSettingNames::k_Delay25_TRG] = PortCardSettingNames::k_bpix_Delay25_TRG_address;
		nameToAddress_[PortCardSettingNames::k_Delay25_SCL] = PortCardSettingNames::k_bpix_Delay25_SCL_address;
		nameToAddress_[PortCardSettingNames::k_Delay25_GCR] = PortCardSettingNames::k_bpix_Delay25_GCR_address;
		
		nameToAddress_[PortCardSettingNames::k_DOH_Ch0Bias_CLK]  = PortCardSettingNames::k_bpix_DOH_Ch0Bias_CLK_address;
		nameToAddress_[PortCardSettingNames::k_DOH_Dummy]        = PortCardSettingNames::k_bpix_DOH_Dummy_address;
		nameToAddress_[PortCardSettingNames::k_DOH_Ch1Bias_Data] = PortCardSettingNames::k_bpix_DOH_Ch1Bias_Data_address;
		nameToAddress_[PortCardSettingNames::k_DOH_Gain_SEU]     = PortCardSettingNames::k_bpix_DOH_Gain_SEU_address;
	}
	else assert(0);
	
	return;
}

void PixelPortCardConfig::fillDBToFileAddress()
{
  if(type_ == "fpix")
    {
      //   nameDBtoFileConversion_["CONFIG_KEY_ID"         ] = ;
      //   nameDBtoFileConversion_["CONFIG_KEY"            ] = ;
      //   nameDBtoFileConversion_["VERSION"               ] = ;
      //   nameDBtoFileConversion_["CONDITION_DATA_SET_ID" ] = ;
      //   nameDBtoFileConversion_["KIND_OF_CONDITION_ID"  ] = ;
      //   nameDBtoFileConversion_["KIND_OF_COND"          ] = ;
      //   nameDBtoFileConversion_["SERIAL_NUMBER"         ] = ;
      //   nameDBtoFileConversion_["PORT_CARD_ID"          ] = ;
      //   nameDBtoFileConversion_["PORT_CARD"             ] = ;
      //   nameDBtoFileConversion_["TRKFEC_NAME"           ] = ;
      //   nameDBtoFileConversion_["RINGADDRESS"           ] = ;
      //   nameDBtoFileConversion_["CHANNELADDRESS"        ] = ;
      //   nameDBtoFileConversion_["CCUADDRESS"            ] = ;
      //   nameDBtoFileConversion_["I2C_CNTRL"             ] = ;
      //   nameDBtoFileConversion_["I2CSPEED"              ] = ;
      nameDBtoFileConversion_["AOH_BIAS1"		  ] = k_AOH_Bias1  ;
      nameDBtoFileConversion_["AOH_BIAS2"		  ] = k_AOH_Bias2  ;
      nameDBtoFileConversion_["AOH_BIAS3"		  ] = k_AOH_Bias3  ;
      nameDBtoFileConversion_["AOH_BIAS4"		  ] = k_AOH_Bias4  ;
      nameDBtoFileConversion_["AOH_BIAS5"		  ] = k_AOH_Bias5  ;
      nameDBtoFileConversion_["AOH_BIAS6"		  ] = k_AOH_Bias6  ;
      //       nameDBtoFileConversion_["AOH1_BIAS1"            ] = k_AOH1_Bias1 ;
      //       nameDBtoFileConversion_["AOH1_BIAS2"		  ] = k_AOH1_Bias2 ;
      //       nameDBtoFileConversion_["AOH1_BIAS3"		  ] = k_AOH1_Bias3 ;
      //       nameDBtoFileConversion_["AOH1_BIAS4"		  ] = k_AOH1_Bias4 ;
      //       nameDBtoFileConversion_["AOH1_BIAS5"		  ] = k_AOH1_Bias5 ;
      //       nameDBtoFileConversion_["AOH1_BIAS6"		  ] = k_AOH1_Bias6 ;
      //   nameDBtoFileConversion_["AOH1_GAIN1"            ] = ;
      //   nameDBtoFileConversion_["AOH1_GAIN2"            ] = ;
      //   nameDBtoFileConversion_["AOH1_GAIN3"            ] = ;
      //   nameDBtoFileConversion_["AOH1_GAIN4"            ] = ;
      //   nameDBtoFileConversion_["AOH1_GAIN5"            ] = ;
      //   nameDBtoFileConversion_["AOH1_GAIN6"            ] = ;
      //       nameDBtoFileConversion_["AOH2_BIAS1"            ] = k_AOH2_Bias1 ;
      //       nameDBtoFileConversion_["AOH2_BIAS2"		  ] = k_AOH2_Bias2 ;
      //       nameDBtoFileConversion_["AOH2_BIAS3"		  ] = k_AOH2_Bias3 ;
      //       nameDBtoFileConversion_["AOH2_BIAS4"		  ] = k_AOH2_Bias4 ;
      //       nameDBtoFileConversion_["AOH2_BIAS5"		  ] = k_AOH2_Bias5 ;
      //       nameDBtoFileConversion_["AOH2_BIAS6"		  ] = k_AOH2_Bias6 ;
      //   nameDBtoFileConversion_["AOH2_GAIN1"            ] = ;
      //   nameDBtoFileConversion_["AOH2_GAIN2"            ] = ;
      //   nameDBtoFileConversion_["AOH2_GAIN3"            ] = ;
      //   nameDBtoFileConversion_["AOH2_GAIN4"            ] = ;
      //   nameDBtoFileConversion_["AOH2_GAIN5"            ] = ;
      //   nameDBtoFileConversion_["AOH2_GAIN6"            ] = ;
      //       nameDBtoFileConversion_["AOH3_BIAS1"            ] = k_AOH3_Bias1 ;
      //       nameDBtoFileConversion_["AOH3_BIAS2"		  ] = k_AOH3_Bias2 ;
      //       nameDBtoFileConversion_["AOH3_BIAS3"		  ] = k_AOH3_Bias3 ;
      //       nameDBtoFileConversion_["AOH3_BIAS4"		  ] = k_AOH3_Bias4 ;
      //       nameDBtoFileConversion_["AOH3_BIAS5"		  ] = k_AOH3_Bias5 ;
      //       nameDBtoFileConversion_["AOH3_BIAS6"		  ] = k_AOH3_Bias6 ;
      //   nameDBtoFileConversion_["AOH3_GAIN1"            ] = ;
      //   nameDBtoFileConversion_["AOH3_GAIN2"            ] = ;
      //   nameDBtoFileConversion_["AOH3_GAIN3"            ] = ;
      //   nameDBtoFileConversion_["AOH3_GAIN4"            ] = ;
      //   nameDBtoFileConversion_["AOH3_GAIN5"            ] = ;
      //   nameDBtoFileConversion_["AOH3_GAIN6"            ] = ;
      //       nameDBtoFileConversion_["AOH4_BIAS1"            ] = k_AOH4_Bias1 ;
      //       nameDBtoFileConversion_["AOH4_BIAS2"		  ] = k_AOH4_Bias2 ;
      //       nameDBtoFileConversion_["AOH4_BIAS3"		  ] = k_AOH4_Bias3 ;
      //       nameDBtoFileConversion_["AOH4_BIAS4"		  ] = k_AOH4_Bias4 ;
      //       nameDBtoFileConversion_["AOH4_BIAS5"		  ] = k_AOH4_Bias5 ;
      //       nameDBtoFileConversion_["AOH4_BIAS6"		  ] = k_AOH4_Bias6 ;
      //   nameDBtoFileConversion_["AOH4_GAIN1"            ] = ;
      //   nameDBtoFileConversion_["AOH4_GAIN2"            ] = ;
      //   nameDBtoFileConversion_["AOH4_GAIN3"            ] = ;
      //   nameDBtoFileConversion_["AOH4_GAIN4"            ] = ;
      //   nameDBtoFileConversion_["AOH4_GAIN5"            ] = ;
      //   nameDBtoFileConversion_["AOH4_GAIN6"            ] = ;
      //   nameDBtoFileConversion_["AOH5_BIAS1"            ] = ;
      //   nameDBtoFileConversion_["AOH5_BIAS2"            ] = ;
      //   nameDBtoFileConversion_["AOH5_BIAS3"            ] = ;
      //   nameDBtoFileConversion_["AOH5_BIAS4"            ] = ;
      //   nameDBtoFileConversion_["AOH5_BIAS5"            ] = ;
      //   nameDBtoFileConversion_["AOH5_BIAS6"            ] = ;
      //   nameDBtoFileConversion_["AOH5_GAIN1"            ] = ;
      //   nameDBtoFileConversion_["AOH5_GAIN2"            ] = ;
      //   nameDBtoFileConversion_["AOH5_GAIN3"            ] = ;
      //   nameDBtoFileConversion_["AOH5_GAIN4"            ] = ;
      //   nameDBtoFileConversion_["AOH5_GAIN5"            ] = ;
      //   nameDBtoFileConversion_["AOH5_GAIN6"            ] = ;
      //   nameDBtoFileConversion_["AOH6_BIAS1"            ] = ;
      //   nameDBtoFileConversion_["AOH6_BIAS2"            ] = ;
      //   nameDBtoFileConversion_["AOH6_BIAS3"            ] = ;
      //   nameDBtoFileConversion_["AOH6_BIAS4"            ] = ;
      //   nameDBtoFileConversion_["AOH6_BIAS5"            ] = ;
      //   nameDBtoFileConversion_["AOH6_BIAS6"            ] = ;
      //   nameDBtoFileConversion_["AOH6_GAIN1"            ] = ;
      //   nameDBtoFileConversion_["AOH6_GAIN2"            ] = ;
      //   nameDBtoFileConversion_["AOH6_GAIN3"            ] = ;
      //   nameDBtoFileConversion_["AOH6_GAIN4"            ] = ;
      //   nameDBtoFileConversion_["AOH6_GAIN5"            ] = ;
      //   nameDBtoFileConversion_["AOH6_GAIN6"            ] = ;
      nameDBtoFileConversion_["DELAY25_GCR"              ] = k_Delay25_RDA ;
      nameDBtoFileConversion_["DELAY25_SCL"              ] = k_Delay25_RCL ;
      nameDBtoFileConversion_["DELAY25_TRG"              ] = k_Delay25_SDA ;
      nameDBtoFileConversion_["DELAY25_SDA"              ] = k_Delay25_TRG ;
      nameDBtoFileConversion_["DELAY25_RCL"              ] = k_Delay25_SCL ;
      nameDBtoFileConversion_["DELAY25_RDA"              ] = k_Delay25_GCR ;
      //   nameDBtoFileConversion_["DEL3_GCR"              ] = ;
      //   nameDBtoFileConversion_["DEL3_SCL"              ] = ;
      //   nameDBtoFileConversion_["DEL3_TRG"              ] = ;
      //   nameDBtoFileConversion_["DEL3_SDA"              ] = ;
      //   nameDBtoFileConversion_["DEL3_RCL"              ] = ;
      //   nameDBtoFileConversion_["DEL3_RDA"              ] = ;
      nameDBtoFileConversion_["DOH_BIAS0"                ] = k_DOH_Ch0Bias_CLK  ;
      nameDBtoFileConversion_["DOH_BIAS1"                ] = k_DOH_Ch1Bias_Data ;
      nameDBtoFileConversion_["DOH_SEU_GAIN"             ] = k_DOH_Gain_SEU     ;
      //   nameDBtoFileConversion_["DOH3_BIAS0"            ] =  ;
      //   nameDBtoFileConversion_["DOH3_BIAS1"            ] =  ;
      //   nameDBtoFileConversion_["DOH3_SEU_GAIN"         ] =  ;
      nameDBtoFileConversion_["PLL_CTR1"                 ] = k_PLL_CTR1 ;
      nameDBtoFileConversion_["PLL_CTR2"                 ] = k_PLL_CTR2 ;
      nameDBtoFileConversion_["PLL_CTR3"                 ] = k_PLL_CTR3 ;
      nameDBtoFileConversion_["PLL_CTR4_5"               ] = k_PLL_CTR4 ;
      //   nameDBtoFileConversion_["PLL3_CTR1"             ] = ;
      //   nameDBtoFileConversion_["PLL3_CTR2"             ] = ;
      //   nameDBtoFileConversion_["PLL3_CTR3"             ] = ;
      //   nameDBtoFileConversion_["PLL3_CTR4_5"           ] = ;
    }
  else if(type_ == "bpix")
    {
      //   nameDBtoFileConversion_["CONFIG_KEY_ID"         ] = ;
      //   nameDBtoFileConversion_["CONFIG_KEY"            ] = ;
      //   nameDBtoFileConversion_["VERSION"               ] = ;
      //   nameDBtoFileConversion_["CONDITION_DATA_SET_ID" ] = ;
      //   nameDBtoFileConversion_["KIND_OF_CONDITION_ID"  ] = ;
      //   nameDBtoFileConversion_["KIND_OF_COND"          ] = ;
      //   nameDBtoFileConversion_["SERIAL_NUMBER"         ] = ;
      //   nameDBtoFileConversion_["PORT_CARD_ID"          ] = ;
      //   nameDBtoFileConversion_["PORT_CARD"             ] = ;
      //   nameDBtoFileConversion_["TRKFEC_NAME"           ] = ;
      //   nameDBtoFileConversion_["RINGADDRESS"           ] = ;
      //   nameDBtoFileConversion_["CHANNELADDRESS"        ] = ;
      //   nameDBtoFileConversion_["CCUADDRESS"            ] = ;
      //   nameDBtoFileConversion_["I2C_CNTRL"             ] = ;
      //   nameDBtoFileConversion_["I2CSPEED"              ] = ;
      nameDBtoFileConversion_["AOH1_BIAS1"            ] = k_AOH1_Bias1 ;
      nameDBtoFileConversion_["AOH1_BIAS2"		  ] = k_AOH1_Bias2 ;
      nameDBtoFileConversion_["AOH1_BIAS3"		  ] = k_AOH1_Bias3 ;
      nameDBtoFileConversion_["AOH1_BIAS4"		  ] = k_AOH1_Bias4 ;
      nameDBtoFileConversion_["AOH1_BIAS5"		  ] = k_AOH1_Bias5 ;
      nameDBtoFileConversion_["AOH1_BIAS6"		  ] = k_AOH1_Bias6 ;
      //   nameDBtoFileConversion_["AOH1_GAIN1"            ] = ;
      //   nameDBtoFileConversion_["AOH1_GAIN2"            ] = ;
      //   nameDBtoFileConversion_["AOH1_GAIN3"            ] = ;
      //   nameDBtoFileConversion_["AOH1_GAIN4"            ] = ;
      //   nameDBtoFileConversion_["AOH1_GAIN5"            ] = ;
      //   nameDBtoFileConversion_["AOH1_GAIN6"            ] = ;
      nameDBtoFileConversion_["AOH2_BIAS1"            ] = k_AOH2_Bias1 ;
      nameDBtoFileConversion_["AOH2_BIAS2"		  ] = k_AOH2_Bias2 ;
      nameDBtoFileConversion_["AOH2_BIAS3"		  ] = k_AOH2_Bias3 ;
      nameDBtoFileConversion_["AOH2_BIAS4"		  ] = k_AOH2_Bias4 ;
      nameDBtoFileConversion_["AOH2_BIAS5"		  ] = k_AOH2_Bias5 ;
      nameDBtoFileConversion_["AOH2_BIAS6"		  ] = k_AOH2_Bias6 ;
      //   nameDBtoFileConversion_["AOH2_GAIN1"            ] = ;
      //   nameDBtoFileConversion_["AOH2_GAIN2"            ] = ;
      //   nameDBtoFileConversion_["AOH2_GAIN3"            ] = ;
      //   nameDBtoFileConversion_["AOH2_GAIN4"            ] = ;
      //   nameDBtoFileConversion_["AOH2_GAIN5"            ] = ;
      //   nameDBtoFileConversion_["AOH2_GAIN6"            ] = ;
      nameDBtoFileConversion_["AOH3_BIAS1"            ] = k_AOH3_Bias1 ;
      nameDBtoFileConversion_["AOH3_BIAS2"		  ] = k_AOH3_Bias2 ;
      nameDBtoFileConversion_["AOH3_BIAS3"		  ] = k_AOH3_Bias3 ;
      nameDBtoFileConversion_["AOH3_BIAS4"		  ] = k_AOH3_Bias4 ;
      nameDBtoFileConversion_["AOH3_BIAS5"		  ] = k_AOH3_Bias5 ;
      nameDBtoFileConversion_["AOH3_BIAS6"		  ] = k_AOH3_Bias6 ;
      //   nameDBtoFileConversion_["AOH3_GAIN1"            ] = ;
      //   nameDBtoFileConversion_["AOH3_GAIN2"            ] = ;
      //   nameDBtoFileConversion_["AOH3_GAIN3"            ] = ;
      //   nameDBtoFileConversion_["AOH3_GAIN4"            ] = ;
      //   nameDBtoFileConversion_["AOH3_GAIN5"            ] = ;
      //   nameDBtoFileConversion_["AOH3_GAIN6"            ] = ;
      nameDBtoFileConversion_["AOH4_BIAS1"            ] = k_AOH4_Bias1 ;
      nameDBtoFileConversion_["AOH4_BIAS2"		  ] = k_AOH4_Bias2 ;
      nameDBtoFileConversion_["AOH4_BIAS3"		  ] = k_AOH4_Bias3 ;
      nameDBtoFileConversion_["AOH4_BIAS4"		  ] = k_AOH4_Bias4 ;
      nameDBtoFileConversion_["AOH4_BIAS5"		  ] = k_AOH4_Bias5 ;
      nameDBtoFileConversion_["AOH4_BIAS6"		  ] = k_AOH4_Bias6 ;
      //   nameDBtoFileConversion_["AOH4_GAIN1"            ] = ;
      //   nameDBtoFileConversion_["AOH4_GAIN2"            ] = ;
      //   nameDBtoFileConversion_["AOH4_GAIN3"            ] = ;
      //   nameDBtoFileConversion_["AOH4_GAIN4"            ] = ;
      //   nameDBtoFileConversion_["AOH4_GAIN5"            ] = ;
      //   nameDBtoFileConversion_["AOH4_GAIN6"            ] = ;
      //   nameDBtoFileConversion_["AOH5_BIAS1"            ] = ;
      //   nameDBtoFileConversion_["AOH5_BIAS2"            ] = ;
      //   nameDBtoFileConversion_["AOH5_BIAS3"            ] = ;
      //   nameDBtoFileConversion_["AOH5_BIAS4"            ] = ;
      //   nameDBtoFileConversion_["AOH5_BIAS5"            ] = ;
      //   nameDBtoFileConversion_["AOH5_BIAS6"            ] = ;
      //   nameDBtoFileConversion_["AOH5_GAIN1"            ] = ;
      //   nameDBtoFileConversion_["AOH5_GAIN2"            ] = ;
      //   nameDBtoFileConversion_["AOH5_GAIN3"            ] = ;
      //   nameDBtoFileConversion_["AOH5_GAIN4"            ] = ;
      //   nameDBtoFileConversion_["AOH5_GAIN5"            ] = ;
      //   nameDBtoFileConversion_["AOH5_GAIN6"            ] = ;
      //   nameDBtoFileConversion_["AOH6_BIAS1"            ] = ;
      //   nameDBtoFileConversion_["AOH6_BIAS2"            ] = ;
      //   nameDBtoFileConversion_["AOH6_BIAS3"            ] = ;
      //   nameDBtoFileConversion_["AOH6_BIAS4"            ] = ;
      //   nameDBtoFileConversion_["AOH6_BIAS5"            ] = ;
      //   nameDBtoFileConversion_["AOH6_BIAS6"            ] = ;
      //   nameDBtoFileConversion_["AOH6_GAIN1"            ] = ;
      //   nameDBtoFileConversion_["AOH6_GAIN2"            ] = ;
      //   nameDBtoFileConversion_["AOH6_GAIN3"            ] = ;
      //   nameDBtoFileConversion_["AOH6_GAIN4"            ] = ;
      //   nameDBtoFileConversion_["AOH6_GAIN5"            ] = ;
      //   nameDBtoFileConversion_["AOH6_GAIN6"            ] = ;
      nameDBtoFileConversion_["DELAY25_GCR"              ] = k_Delay25_RDA ;
      nameDBtoFileConversion_["DELAY25_SCL"              ] = k_Delay25_RCL ;
      nameDBtoFileConversion_["DELAY25_TRG"              ] = k_Delay25_SDA ;
      nameDBtoFileConversion_["DELAY25_SDA"              ] = k_Delay25_TRG ;
      nameDBtoFileConversion_["DELAY25_RCL"              ] = k_Delay25_SCL ;
      nameDBtoFileConversion_["DELAY25_RDA"              ] = k_Delay25_GCR ;
      //   nameDBtoFileConversion_["DEL3_GCR"              ] = ;
      //   nameDBtoFileConversion_["DEL3_SCL"              ] = ;
      //   nameDBtoFileConversion_["DEL3_TRG"              ] = ;
      //   nameDBtoFileConversion_["DEL3_SDA"              ] = ;
      //   nameDBtoFileConversion_["DEL3_RCL"              ] = ;
      //   nameDBtoFileConversion_["DEL3_RDA"              ] = ;
      nameDBtoFileConversion_["DOH_BIAS0"            ] = k_DOH_Ch0Bias_CLK  ;
      nameDBtoFileConversion_["DOH_BIAS1"            ] = k_DOH_Ch1Bias_Data ;
      nameDBtoFileConversion_["DOH_SEU_GAIN"         ] = k_DOH_Gain_SEU     ;
      //   nameDBtoFileConversion_["DOH3_BIAS0"            ] =  ;
      //   nameDBtoFileConversion_["DOH3_BIAS1"            ] =  ;
      //   nameDBtoFileConversion_["DOH3_SEU_GAIN"         ] =  ;
      nameDBtoFileConversion_["PLL_CTR1"             ] = k_PLL_CTR1 ;
      nameDBtoFileConversion_["PLL_CTR2"             ] = k_PLL_CTR2 ;
      nameDBtoFileConversion_["PLL_CTR3"             ] = k_PLL_CTR3 ;
      nameDBtoFileConversion_["PLL_CTR4_5"           ] = k_PLL_CTR4 ;
      //   nameDBtoFileConversion_["PLL3_CTR1"             ] = ;
      //   nameDBtoFileConversion_["PLL3_CTR2"             ] = ;
      //   nameDBtoFileConversion_["PLL3_CTR3"             ] = ;
      //   nameDBtoFileConversion_["PLL3_CTR4_5"           ] = ;
    }
    
    
}


void PixelPortCardConfig::writeASCII(std::string dir) const {

  if (dir!="") dir+="/";
  std::string filename=dir+"portcard_"+portcardname_+".dat";

  std::ofstream out(filename.c_str());
  if (!out.good()){
    std::cout << "Could not open file:"<<filename.c_str()<<std::endl;
    assert(0);
  }

  out << "Name: " << portcardname_ << std::endl;
  out << "Type: " << type_ << std::endl;
  out << "TKFECID: " << TKFECID_ << std::endl;
  out << "ringAddress: 0x" <<std::hex<< ringAddress_ <<std::dec<< std::endl;
  out << "ccuAddress: 0x" <<std::hex<< ccuAddress_ <<std::dec<< std::endl;
  
  out << "channelAddress: 0x" <<std::hex<< channelAddress_ <<std::dec<< std::endl;

  out << "i2cSpeed: 0x" <<std::hex<< i2cSpeed_ <<std::dec<< std::endl;

  bool found_PLL_CTR2 = false;
  unsigned int last_PLL_CTR2_value = 0x0;
  for (unsigned int i=0;i<device_.size();i++)
  {
    unsigned int deviceAddress = device_.at(i).first;
    
    // Special handling for AOH gains
    if (    ( type_=="fpix" && deviceAddress == k_fpix_AOH_Gain123_address )
         || ( type_=="fpix" && deviceAddress == k_fpix_AOH_Gain456_address )
         || ( type_=="bpix" && deviceAddress == k_bpix_AOH1_Gain123_address )
         || ( type_=="bpix" && deviceAddress == k_bpix_AOH1_Gain456_address )
         || ( type_=="bpix" && deviceAddress == k_bpix_AOH2_Gain123_address )
         || ( type_=="bpix" && deviceAddress == k_bpix_AOH2_Gain456_address )
         || ( type_=="bpix" && deviceAddress == k_bpix_AOH3_Gain123_address )
         || ( type_=="bpix" && deviceAddress == k_bpix_AOH3_Gain456_address )
         || ( type_=="bpix" && deviceAddress == k_bpix_AOH4_Gain123_address )
         || ( type_=="bpix" && deviceAddress == k_bpix_AOH4_Gain456_address )
       )
    {
		std::string whichAOHString;
		unsigned int zeroOrThree;
		if      ( type_=="fpix" && deviceAddress == k_fpix_AOH_Gain123_address )  { whichAOHString = "";  zeroOrThree = 0; }
		else if ( type_=="fpix" && deviceAddress == k_fpix_AOH_Gain456_address )  { whichAOHString = "";  zeroOrThree = 3; }
		else if ( type_=="bpix" && deviceAddress == k_bpix_AOH1_Gain123_address ) { whichAOHString = "1"; zeroOrThree = 0; }
		else if ( type_=="bpix" && deviceAddress == k_bpix_AOH1_Gain456_address ) { whichAOHString = "1"; zeroOrThree = 3; }
		else if ( type_=="bpix" && deviceAddress == k_bpix_AOH2_Gain123_address ) { whichAOHString = "2"; zeroOrThree = 0; }
		else if ( type_=="bpix" && deviceAddress == k_bpix_AOH2_Gain456_address ) { whichAOHString = "2"; zeroOrThree = 3; }
		else if ( type_=="bpix" && deviceAddress == k_bpix_AOH3_Gain123_address ) { whichAOHString = "3"; zeroOrThree = 0; }
		else if ( type_=="bpix" && deviceAddress == k_bpix_AOH3_Gain456_address ) { whichAOHString = "3"; zeroOrThree = 3; }
		else if ( type_=="bpix" && deviceAddress == k_bpix_AOH4_Gain123_address ) { whichAOHString = "4"; zeroOrThree = 0; }
		else if ( type_=="bpix" && deviceAddress == k_bpix_AOH4_Gain456_address ) { whichAOHString = "4"; zeroOrThree = 3; }
		else assert(0);
		
		out << "AOH"<<whichAOHString<<"_Gain"<<zeroOrThree+1<<": 0x"<< (((device_[i].second) & 0x03)>>0) << std::endl; // output bits 0 & 1
		out << "AOH"<<whichAOHString<<"_Gain"<<zeroOrThree+2<<": 0x"<< (((device_[i].second) & 0x0c)>>2) << std::endl; // output bits 2 & 3
		out << "AOH"<<whichAOHString<<"_Gain"<<zeroOrThree+3<<": 0x"<< (((device_[i].second) & 0x30)>>4) << std::endl; // output bits 4 & 5
		continue;
    }
    // End of special handling
    
    // Check to see if there's a name corresponding to this address.
    std::string settingName = "";
    for ( std::map<std::string, unsigned int>::const_iterator nameToAddress_itr = nameToAddress_.begin(); nameToAddress_itr != nameToAddress_.end(); ++nameToAddress_itr )
    {
      if ( nameToAddress_itr->second == deviceAddress ) {settingName = nameToAddress_itr->first; break;}
    }
    
    // Special handling for PLL addresses.
    if ( settingName == k_PLL_CTR2 )
    {
    	if ( found_PLL_CTR2 && last_PLL_CTR2_value == device_.at(i).second ) continue; // don't save duplicate CTR2 settings
    	found_PLL_CTR2 = true;
    	last_PLL_CTR2_value = device_.at(i).second;
    }
    if ( found_PLL_CTR2 && settingName == k_PLL_CTR4or5 ) // change name to PLL_CTR4 or PLL_CTR5
    {
    	if ( (last_PLL_CTR2_value & 0x20) == 0x0 ) settingName = k_PLL_CTR4;
    	else                                       settingName = k_PLL_CTR5;
    }
    // end of special handling
    
    if ( settingName=="" ) out << "0x" <<std::hex<< device_.at(i).first <<std::dec;
    else                   out << settingName << ":";
    
    out << " 0x" <<std::hex<< device_.at(i).second <<std::dec<< std::endl;
  }

  out.close();
}

unsigned int PixelPortCardConfig::getdevicesize() const{
  return device_.size();
}

std::string  PixelPortCardConfig::getTKFECID() const{
  return TKFECID_;
}

unsigned int PixelPortCardConfig::getringAddress() const{
  return ringAddress_;
}

unsigned int PixelPortCardConfig::getccuAddress() const{
  return ccuAddress_;
}

unsigned int PixelPortCardConfig::getchannelAddress() const{
  return channelAddress_;
}

unsigned int PixelPortCardConfig::geti2cSpeed() const{
  return i2cSpeed_;
}

std::string  PixelPortCardConfig::gettype() const{
  return type_;
}

unsigned int PixelPortCardConfig::getdeviceAddress(unsigned int i) const{
  assert(i<device_.size());
  return device_[i].first;
}

unsigned int PixelPortCardConfig::getdeviceValues(unsigned int i) const{
  assert(i<device_.size());
  return device_[i].second;
}

void PixelPortCardConfig::setdeviceValues(unsigned int address, unsigned int value) {
  for (int i=device_.size()-1; i>=0; i--) // Change only the last occurance of address, if there are more than one.
    {
      if( device_.at(i).first==address )
        {
          device_.at(i).second=value;
          return;
        }
    }
  
  // This address wasn't found in the list, so add it and its value.
  pair<unsigned int, unsigned int> p(address, value);
  device_.push_back(p);
  
  return;
}

void PixelPortCardConfig::setdeviceValues(std::string settingName, unsigned int value)
{
	setdeviceValues( getdeviceAddressForSetting(settingName), value );
	return;
}

unsigned int PixelPortCardConfig::getdeviceAddressForSetting(std::string settingName) const
{
  //std::cout << "[PixelPortCardConfig::getdeviceAddressForSetting()]\t    settingName: " << settingName<< std::endl ;
  std::map<std::string, unsigned int>::const_iterator foundName_itr = nameToAddress_.find(settingName);
  assert( foundName_itr != nameToAddress_.end() );
  return foundName_itr->second;
}

unsigned int PixelPortCardConfig::getdeviceValuesForSetting(std::string settingName) const
{
	return getdeviceValuesForAddress( getdeviceAddressForSetting(settingName) );
}

unsigned int PixelPortCardConfig::getdeviceValuesForAddress(unsigned int address) const
{
	for (int i=device_.size()-1; i>=0; i--) // Get the last occurance of address, if there are more than one.
    {
      if( device_.at(i).first==address )
        {
          return device_.at(i).second;
        }
    }
  assert(0); // didn't find this device
  return 0;
}

bool PixelPortCardConfig::containsDeviceAddress(unsigned int deviceAddress) const
{
	for ( std::vector<std::pair<unsigned int, unsigned int> >::const_iterator device_itr = device_.begin(); device_itr != device_.end(); device_itr++ )
	{
		if ( device_itr->first == deviceAddress ) return true;
	}
	return false;
}

unsigned int PixelPortCardConfig::AOHBiasAddressFromAOHNumber(unsigned int AOHNumber) const
{
	if ( type_ == "fpix" )
	{
		if      (AOHNumber == 1) return PortCardSettingNames::k_fpix_AOH_Bias1_address;
		else if (AOHNumber == 2) return PortCardSettingNames::k_fpix_AOH_Bias2_address;
		else if (AOHNumber == 3) return PortCardSettingNames::k_fpix_AOH_Bias3_address;
		else if (AOHNumber == 4) return PortCardSettingNames::k_fpix_AOH_Bias4_address;
		else if (AOHNumber == 5) return PortCardSettingNames::k_fpix_AOH_Bias5_address;
		else if (AOHNumber == 6) return PortCardSettingNames::k_fpix_AOH_Bias6_address;
		else {std::cout << "ERROR: For fpix, AOH number must be in the range 1-6, but the given AOH number was "<<AOHNumber<<"."<<std::endl; assert(0);}
	}
	else if ( type_ == "bpix" )
	{
		if      (AOHNumber == 1) return PortCardSettingNames::k_bpix_AOH1_Bias1_address;
		else if (AOHNumber == 2) return PortCardSettingNames::k_bpix_AOH1_Bias2_address;
		else if (AOHNumber == 3) return PortCardSettingNames::k_bpix_AOH1_Bias3_address;
		else if (AOHNumber == 4) return PortCardSettingNames::k_bpix_AOH1_Bias4_address;
		else if (AOHNumber == 5) return PortCardSettingNames::k_bpix_AOH1_Bias5_address;
		else if (AOHNumber == 6) return PortCardSettingNames::k_bpix_AOH1_Bias6_address;
		else if (AOHNumber == 7) return PortCardSettingNames::k_bpix_AOH2_Bias1_address;
		else if (AOHNumber == 8) return PortCardSettingNames::k_bpix_AOH2_Bias2_address;
		else if (AOHNumber == 9) return PortCardSettingNames::k_bpix_AOH2_Bias3_address;
		else if (AOHNumber ==10) return PortCardSettingNames::k_bpix_AOH2_Bias4_address;
		else if (AOHNumber ==11) return PortCardSettingNames::k_bpix_AOH2_Bias5_address;
		else if (AOHNumber ==12) return PortCardSettingNames::k_bpix_AOH2_Bias6_address;
		else if (AOHNumber ==13) return PortCardSettingNames::k_bpix_AOH3_Bias1_address;
		else if (AOHNumber ==14) return PortCardSettingNames::k_bpix_AOH3_Bias2_address;
		else if (AOHNumber ==15) return PortCardSettingNames::k_bpix_AOH3_Bias3_address;
		else if (AOHNumber ==16) return PortCardSettingNames::k_bpix_AOH3_Bias4_address;
		else if (AOHNumber ==17) return PortCardSettingNames::k_bpix_AOH3_Bias5_address;
		else if (AOHNumber ==18) return PortCardSettingNames::k_bpix_AOH3_Bias6_address;
		else if (AOHNumber ==19) return PortCardSettingNames::k_bpix_AOH4_Bias1_address;
		else if (AOHNumber ==20) return PortCardSettingNames::k_bpix_AOH4_Bias2_address;
		else if (AOHNumber ==21) return PortCardSettingNames::k_bpix_AOH4_Bias3_address;
		else if (AOHNumber ==22) return PortCardSettingNames::k_bpix_AOH4_Bias4_address;
		else if (AOHNumber ==23) return PortCardSettingNames::k_bpix_AOH4_Bias5_address;
		else if (AOHNumber ==24) return PortCardSettingNames::k_bpix_AOH4_Bias6_address;
		else {std::cout << "ERROR: For bpix, AOH number must be in the range 1-24, but the given AOH number was "<<AOHNumber<<"."<<std::endl; assert(0);}
	}
	else assert(0);
}

std::string PixelPortCardConfig::AOHGainStringFromAOHNumber(unsigned int AOHNumber) const
{
	if ( type_ == "fpix" )
	{
		if      (AOHNumber == 1) return "AOH_Gain1";
		else if (AOHNumber == 2) return "AOH_Gain2";
		else if (AOHNumber == 3) return "AOH_Gain3";
		else if (AOHNumber == 4) return "AOH_Gain4";
		else if (AOHNumber == 5) return "AOH_Gain5";
		else if (AOHNumber == 6) return "AOH_Gain6";
		else {std::cout << "ERROR: For fpix, AOH number must be in the range 1-6, but the given AOH number was "<<AOHNumber<<"."<<std::endl; assert(0);}
	}
	else if ( type_ == "bpix" )
	{
		if      (AOHNumber == 1) return "AOH1_Gain1";
		else if (AOHNumber == 2) return "AOH1_Gain2";
		else if (AOHNumber == 3) return "AOH1_Gain3";
		else if (AOHNumber == 4) return "AOH1_Gain4";
		else if (AOHNumber == 5) return "AOH1_Gain5";
		else if (AOHNumber == 6) return "AOH1_Gain6";
		else if (AOHNumber == 7) return "AOH2_Gain1";
		else if (AOHNumber == 8) return "AOH2_Gain2";
		else if (AOHNumber == 9) return "AOH2_Gain3";
		else if (AOHNumber ==10) return "AOH2_Gain4";
		else if (AOHNumber ==11) return "AOH2_Gain5";
		else if (AOHNumber ==12) return "AOH2_Gain6";
		else if (AOHNumber ==13) return "AOH3_Gain1";
		else if (AOHNumber ==14) return "AOH3_Gain2";
		else if (AOHNumber ==15) return "AOH3_Gain3";
		else if (AOHNumber ==16) return "AOH3_Gain4";
		else if (AOHNumber ==17) return "AOH3_Gain5";
		else if (AOHNumber ==18) return "AOH3_Gain6";
		else if (AOHNumber ==19) return "AOH4_Gain1";
		else if (AOHNumber ==20) return "AOH4_Gain2";
		else if (AOHNumber ==21) return "AOH4_Gain3";
		else if (AOHNumber ==22) return "AOH4_Gain4";
		else if (AOHNumber ==23) return "AOH4_Gain5";
		else if (AOHNumber ==24) return "AOH4_Gain6";
		else {std::cout << "ERROR: For bpix, AOH number must be in the range 1-24, but the given AOH number was "<<AOHNumber<<"."<<std::endl; assert(0);}
	}
	else assert(0);
}

unsigned int PixelPortCardConfig::AOHGainAddressFromAOHNumber(unsigned int AOHNumber) const
{
	unsigned int address;
	if ( type_ == "fpix" )
	{
		if      (AOHNumber == 1) address =  PortCardSettingNames::k_fpix_AOH_Gain123_address;
		else if (AOHNumber == 2) address =  PortCardSettingNames::k_fpix_AOH_Gain123_address;
		else if (AOHNumber == 3) address =  PortCardSettingNames::k_fpix_AOH_Gain123_address;
		else if (AOHNumber == 4) address =  PortCardSettingNames::k_fpix_AOH_Gain456_address;
		else if (AOHNumber == 5) address =  PortCardSettingNames::k_fpix_AOH_Gain456_address;
		else if (AOHNumber == 6) address =  PortCardSettingNames::k_fpix_AOH_Gain456_address;
		else {std::cout << "ERROR: For fpix, AOH number must be in the range 1-6, but the given AOH number was "<<AOHNumber<<"."<<std::endl; assert(0);}
	}
	else if ( type_ == "bpix" )
	{
		if      (AOHNumber == 1) address =  PortCardSettingNames::k_bpix_AOH1_Gain123_address;
		else if (AOHNumber == 2) address =  PortCardSettingNames::k_bpix_AOH1_Gain123_address;
		else if (AOHNumber == 3) address =  PortCardSettingNames::k_bpix_AOH1_Gain123_address;
		else if (AOHNumber == 4) address =  PortCardSettingNames::k_bpix_AOH1_Gain456_address;
		else if (AOHNumber == 5) address =  PortCardSettingNames::k_bpix_AOH1_Gain456_address;
		else if (AOHNumber == 6) address =  PortCardSettingNames::k_bpix_AOH1_Gain456_address;
		else if (AOHNumber == 7) address =  PortCardSettingNames::k_bpix_AOH2_Gain123_address;
		else if (AOHNumber == 8) address =  PortCardSettingNames::k_bpix_AOH2_Gain123_address;
		else if (AOHNumber == 9) address =  PortCardSettingNames::k_bpix_AOH2_Gain123_address;
		else if (AOHNumber ==10) address =  PortCardSettingNames::k_bpix_AOH2_Gain456_address;
		else if (AOHNumber ==11) address =  PortCardSettingNames::k_bpix_AOH2_Gain456_address;
		else if (AOHNumber ==12) address =  PortCardSettingNames::k_bpix_AOH2_Gain456_address;
		else if (AOHNumber ==13) address =  PortCardSettingNames::k_bpix_AOH3_Gain123_address;
		else if (AOHNumber ==14) address =  PortCardSettingNames::k_bpix_AOH3_Gain123_address;
		else if (AOHNumber ==15) address =  PortCardSettingNames::k_bpix_AOH3_Gain123_address;
		else if (AOHNumber ==16) address =  PortCardSettingNames::k_bpix_AOH3_Gain456_address;
		else if (AOHNumber ==17) address =  PortCardSettingNames::k_bpix_AOH3_Gain456_address;
		else if (AOHNumber ==18) address =  PortCardSettingNames::k_bpix_AOH3_Gain456_address;
		else if (AOHNumber ==19) address =  PortCardSettingNames::k_bpix_AOH4_Gain123_address;
		else if (AOHNumber ==20) address =  PortCardSettingNames::k_bpix_AOH4_Gain123_address;
		else if (AOHNumber ==21) address =  PortCardSettingNames::k_bpix_AOH4_Gain123_address;
		else if (AOHNumber ==22) address =  PortCardSettingNames::k_bpix_AOH4_Gain456_address;
		else if (AOHNumber ==23) address =  PortCardSettingNames::k_bpix_AOH4_Gain456_address;
		else if (AOHNumber ==24) address =  PortCardSettingNames::k_bpix_AOH4_Gain456_address;
		else {std::cout << "ERROR: For bpix, AOH number must be in the range 1-24, but the given AOH number was "<<AOHNumber<<"."<<std::endl; assert(0);}
	}
	else assert(0);
	
	return address;
}

unsigned int PixelPortCardConfig::getAOHGain(unsigned int AOHNumber) const
{
	const unsigned int address = AOHGainAddressFromAOHNumber(AOHNumber);
	const unsigned int threeGainsValue = getdeviceValuesForAddress(address);
	
	if ( AOHNumber%3 == 1 ) return (((threeGainsValue) & 0x03)>>0); // return bits 0 & 1
	if ( AOHNumber%3 == 2 ) return (((threeGainsValue) & 0x0c)>>2); // return bits 2 & 3
	if ( AOHNumber%3 == 0 ) return (((threeGainsValue) & 0x30)>>4); // return bits 4 & 5
	
	assert(0);
}
