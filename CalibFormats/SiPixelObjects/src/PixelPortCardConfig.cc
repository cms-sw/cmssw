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
   fillNameToAddress();
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
    std::cout<<"Could not open:"<<filename<<std::endl;
    assert(0);
  }
  else {
    std::cout<<"Opened:"<<filename<<std::endl;
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
    
    // Special handling for AOH_GainX
    if ( (type_=="fpix")&&( settingName == k_fpix_AOH_Gain1 || settingName == k_fpix_AOH_Gain2 || settingName == k_fpix_AOH_Gain3 ) )
    {
    	// Search for AOH_Gain123 in the previously-defined settings.
    	bool foundOne = false;
    	for (unsigned int i=0;i<device_.size();i++)
    	{
    		if ( device_[i].first == k_fpix_AOH_Gain123_address ) // Change AOH_GainX in all previous instances
    		{
    			foundOne = true;
    			unsigned int oldValue = device_[i].second;
    			if      ( settingName == k_fpix_AOH_Gain1 )
    				device_[i].second = (0x3c & oldValue) + ((i2c_values & 0x3)<<0); // replace bits 0 and 1 with i2c_values
    			else if ( settingName == k_fpix_AOH_Gain2 )
    				device_[i].second = (0x33 & oldValue) + ((i2c_values & 0x3)<<2); // replace bits 2 and 3 with i2c_values
    			else if ( settingName == k_fpix_AOH_Gain3 )
    				device_[i].second = (0x0f & oldValue) + ((i2c_values & 0x3)<<4); // replace bits 4 and 5 with i2c_values
    			else assert(0);
    			//std::cout << "Changed setting "<< k_fpix_AOH_Gain123 <<"(address 0x"<<std::hex<<k_fpix_AOH_Gain123_address<<") from 0x"<<oldValue<<" to 0x"<< device_[i].second << std::dec <<"\n";
    		}
    	}
    	if ( foundOne ) continue;
    	else // If this was not set previously, add AOH_Gain123 with the other two gains set to zero.
    	{
    		i2c_address = k_fpix_AOH_Gain123_address;
    		if      ( settingName == k_fpix_AOH_Gain1 ) i2c_values  = ((i2c_values & 0x3)<<0);
    		else if ( settingName == k_fpix_AOH_Gain2 ) i2c_values  = ((i2c_values & 0x3)<<2);
    		else if ( settingName == k_fpix_AOH_Gain3 ) i2c_values  = ((i2c_values & 0x3)<<4);
    		else assert(0);
    		settingName = k_fpix_AOH_Gain123;
    	}
    }
    else if ( (type_=="fpix")&&( settingName == k_fpix_AOH_Gain4 || settingName == k_fpix_AOH_Gain5 || settingName == k_fpix_AOH_Gain6 ) )
    {
    	// Search for AOH_Gain456 in the previously-defined settings.
    	bool foundOne = false;
    	for (unsigned int i=0;i<device_.size();i++)
    	{
    		if ( device_[i].first == k_fpix_AOH_Gain456_address ) // Change AOH_GainX in all previous instances
    		{
    			foundOne = true;
    			unsigned int oldValue = device_[i].second;
    			if      ( settingName == k_fpix_AOH_Gain4 )
    				device_[i].second = (0x3c & oldValue) + ((i2c_values & 0x3)<<0); // replace bits 0 and 1 with i2c_values
    			else if ( settingName == k_fpix_AOH_Gain5 )
    				device_[i].second = (0x33 & oldValue) + ((i2c_values & 0x3)<<2); // replace bits 2 and 3 with i2c_values
    			else if ( settingName == k_fpix_AOH_Gain6 )
    				device_[i].second = (0x0f & oldValue) + ((i2c_values & 0x3)<<4); // replace bits 4 and 5 with i2c_values
    			else assert(0);
    			//std::cout << "Changed setting "<< k_fpix_AOH_Gain456 <<"(address 0x"<<std::hex<<k_fpix_AOH_Gain456_address<<") from 0x"<<oldValue<<" to 0x"<< device_[i].second << std::dec <<"\n";
    		}
    	}
    	if ( foundOne ) continue;
    	else // If this was not set previously, add AOH_Gain456 with the other two gains set to zero.
    	{
    		i2c_address = k_fpix_AOH_Gain456_address;
    		if      ( settingName == k_fpix_AOH_Gain4 ) i2c_values  = ((i2c_values & 0x3)<<0);
    		else if ( settingName == k_fpix_AOH_Gain5 ) i2c_values  = ((i2c_values & 0x3)<<2);
    		else if ( settingName == k_fpix_AOH_Gain6 ) i2c_values  = ((i2c_values & 0x3)<<4);
    		else assert(0);
    		settingName = k_fpix_AOH_Gain456;
    	}
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
    }
    
    //std::cout << "Setting name = " << settingName << ", i2c address = 0x" << std::hex << i2c_address <<", value = 0x"<< i2c_values << std::dec << std::endl;
      
    if (!in.eof()){
      pair<unsigned int, unsigned int> p(i2c_address, i2c_values);
      device_.push_back(p);
    }

  }
  while (!in.eof());
  
  in.close();

}

void PixelPortCardConfig::fillNameToAddress()
{
	if ( nameToAddress_.size() != 0 ) return;
	
	if ( type_ == "fpix" )
	{
		nameToAddress_[PortCardSettingNames::k_fpix_Delay25_GCR] = PortCardSettingNames::k_fpix_Delay25_GCR_address;
		nameToAddress_[PortCardSettingNames::k_fpix_Delay25_SCL] = PortCardSettingNames::k_fpix_Delay25_SCL_address;
		nameToAddress_[PortCardSettingNames::k_fpix_Delay25_TRG] = PortCardSettingNames::k_fpix_Delay25_TRG_address;
		nameToAddress_[PortCardSettingNames::k_fpix_Delay25_SDA] = PortCardSettingNames::k_fpix_Delay25_SDA_address;
		nameToAddress_[PortCardSettingNames::k_fpix_Delay25_RCL] = PortCardSettingNames::k_fpix_Delay25_RCL_address;
		nameToAddress_[PortCardSettingNames::k_fpix_Delay25_RDA] = PortCardSettingNames::k_fpix_Delay25_RDA_address;
		nameToAddress_[PortCardSettingNames::k_fpix_AOH_Bias1] = PortCardSettingNames::k_fpix_AOH_Bias1_address;
		nameToAddress_[PortCardSettingNames::k_fpix_AOH_Bias2] = PortCardSettingNames::k_fpix_AOH_Bias2_address;
		nameToAddress_[PortCardSettingNames::k_fpix_AOH_Bias3] = PortCardSettingNames::k_fpix_AOH_Bias3_address;
		nameToAddress_[PortCardSettingNames::k_fpix_AOH_Bias4] = PortCardSettingNames::k_fpix_AOH_Bias4_address;
		nameToAddress_[PortCardSettingNames::k_fpix_AOH_Bias5] = PortCardSettingNames::k_fpix_AOH_Bias5_address;
		nameToAddress_[PortCardSettingNames::k_fpix_AOH_Bias6] = PortCardSettingNames::k_fpix_AOH_Bias6_address;
		nameToAddress_[PortCardSettingNames::k_fpix_AOH_Gain123] = PortCardSettingNames::k_fpix_AOH_Gain123_address;
		nameToAddress_[PortCardSettingNames::k_fpix_AOH_Gain456] = PortCardSettingNames::k_fpix_AOH_Gain456_address;
	}
	else if ( type_ == "bpix" )
	{
		nameToAddress_[PortCardSettingNames::k_bpix_dummy] = PortCardSettingNames::k_bpix_dummy_address;
	}
	else assert(0);
	
	return;
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

  for (unsigned int i=0;i<device_.size();i++)
  {
    unsigned int deviceAddress = device_.at(i).first;
    
    // Special handling for AOH_Gain123
    if ( (type_=="fpix")&&( deviceAddress == k_fpix_AOH_Gain123_address ) )
    {
    	out << k_fpix_AOH_Gain1<<": 0x"<< (((device_[i].second) & 0x03)>>0) << std::endl; // output bits 0 & 1
    	out << k_fpix_AOH_Gain2<<": 0x"<< (((device_[i].second) & 0x0c)>>2) << std::endl; // output bits 2 & 3
    	out << k_fpix_AOH_Gain3<<": 0x"<< (((device_[i].second) & 0x30)>>4) << std::endl; // output bits 4 & 5
    	continue;
    }
     // Special handling for AOH_Gain456
    if ( (type_=="fpix")&&( deviceAddress == k_fpix_AOH_Gain456_address ) )
    {
    	out << k_fpix_AOH_Gain4<<": 0x"<< (((device_[i].second) & 0x03)>>0) << std::endl; // output bits 0 & 1
    	out << k_fpix_AOH_Gain5<<": 0x"<< (((device_[i].second) & 0x0c)>>2) << std::endl; // output bits 2 & 3
    	out << k_fpix_AOH_Gain6<<": 0x"<< (((device_[i].second) & 0x30)>>4) << std::endl; // output bits 4 & 5
    	continue;
    }
    // End of special handling
    
    // Check to see if there's a name corresponding to this address.
    std::string settingName = "";
    for ( std::map<std::string, unsigned int>::const_iterator nameToAddress_itr = nameToAddress_.begin(); nameToAddress_itr != nameToAddress_.end(); ++nameToAddress_itr )
    {
      if ( nameToAddress_itr->second == deviceAddress ) {settingName = nameToAddress_itr->first; break;}
    }
    
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
  for (unsigned int i=0; i<device_.size(); i++)
    {
      if( device_.at(i).first==address )
        {
          device_.at(i).second=value;
          break;
        }
    }
  return;
}

void PixelPortCardConfig::setdeviceValues(std::string settingName, unsigned int value)
{
	setdeviceValues( getdeviceAddressForSetting(settingName), value );
	return;
}

unsigned int PixelPortCardConfig::getdeviceAddressForSetting(std::string settingName) const
{
	std::map<std::string, unsigned int>::const_iterator foundName_itr = nameToAddress_.find(settingName);
	assert( foundName_itr != nameToAddress_.end() );
	return foundName_itr->second;
}

unsigned int PixelPortCardConfig::getdeviceValuesForSetting(std::string settingName) const
{
	unsigned int address = getdeviceAddressForSetting(settingName);
	for (unsigned int i=0; i<device_.size(); i++)
    {
      if( device_.at(i).first==address )
        {
          return device_.at(i).second;
        }
    }
  assert(0); // didn't find this device
  return 0;
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
		std::cout << "ERROR: For bpix, AOH number must be in the range ?-?, but the given AOH number was "<<AOHNumber<<"."<<std::endl; assert(0);
	}
	else assert(0);
	
	assert(0); // should never get here
}
