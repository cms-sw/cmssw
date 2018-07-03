#include "CondFormats/L1TObjects/interface/L1MuCSCPtLut.h"
#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include <sstream>
#include <iostream>
#include <cerrno>

void L1MuCSCPtLut::readFromDBS( std::string& ptLUT )
{

  //edm::LogInfo( "L1-O2O: L1MuCSCPtLut" ) <<" Reading from DBS";

  // the ptLUT returned by the OMDS query will have a structure like number"\n"number"\n"
  // e.g., 32896\n32896\n32896\n32896\n and so on
  // remove the \n separator to write the pt_lut[1<<21] line-by-line
  for(size_t pos=ptLUT.find("\\n"); pos!=std::string::npos; pos=ptLUT.find("\\n",pos)){ 
    ptLUT[pos]=' '; 
    ptLUT[pos+1]='\n'; 
  }
  
  unsigned long length = 1<<21; //length of the ptLUT file

  std::stringstream file(ptLUT);
  if( file.fail() )
    throw cms::Exception("Cannot open the ptLUT")<<"L1MuCSCPtLut cannot open "
						 <<"ptLUT from DBS (errno="
						 <<errno<<")"<<std::endl;
  
  // filling the ptLUT
  unsigned int address=0;
  for(address=0; !file.eof() && address<length; address++) {
    char buff[1024];
    file.getline(buff,1024);
    int ptOutput = atoi(buff);

    // uncomment if you want to see line-by-line
    //edm::LogInfo( "L1-O2O: L1MuCSCPtLut" ) << "writing line " 
    //				       << ptOutput;

    // Warning: this may throw non-cms like exception
    pt_lut[address] = ptOutput; 
  }
  
  if( address!=length ) 
    throw cms::Exception("Incorrect LUT size")<<"L1MuCSCPtLut read "<<address
					      <<" words from DBS instead of expected "
					      <<length <<std::endl;
}
