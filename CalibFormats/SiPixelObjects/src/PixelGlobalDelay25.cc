//
// This class stores the information about the global delay25 delay settings
// For the time being, the affected delay25 channels are SDA, SCL, and TRG
// (as implemented in PixelTKFECSupervisor)
//

#include "CalibFormats/SiPixelObjects/interface/PixelGlobalDelay25.h"
#include "CalibFormats/SiPixelObjects/interface/PixelTimeFormatter.h"

#include <sstream>
#include <fstream>
#include <map>
#include <assert.h>
#include <math.h>


using namespace pos;
using namespace std;

PixelGlobalDelay25::PixelGlobalDelay25(vector<vector<string> > &tableMat):PixelConfigBase(" "," "," ")
{
  std::string mthn = "[PixelGlobalDelay25::PixelGlobalDelay25()]\t\t    " ;
  vector<string> ins = tableMat[0];
  map<string , int > colM;
  vector<string> colNames;

  /**
    VIEW_NAME: CONF_KEY_GLOBAL_DELAY25_V
    
    Name                                        Null?    Type               POS variable
    ----------------------------------------- -------- ---------------------------------------------------------------
    CONFIG_KEY 
    KEY_TYPE
    KEY_ALIAS_ID
    KEY_ALIAS
    VERSION
    KIND_OF_COND
    GLOBALDELAY25                                       VARCHAR

  */
  colNames.push_back("CONFIG_KEY"   ) ;
  colNames.push_back("KEY_TYPE"     ) ;
  colNames.push_back("KEY_ALIAS_ID" ) ;
  colNames.push_back("KEY_ALIAS"    ) ;
  colNames.push_back("VERSION"      ) ;
  colNames.push_back("KIND_OF_COND" ) ;
  colNames.push_back("GLOBALDELAY25") ;
  for(unsigned int c = 0 ; c < ins.size() ; c++)
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
          std::cerr << "[PixelGlobalDelay25::PixelGlobalDelay25()]\tCouldn't find in the database the column with name " << colNames[n] << std::endl;
          assert(0);
        }
    }
  sscanf(tableMat[1][colM["GLOBALDELAY25"]].c_str(),"%x",&delay_);
  std::cout << __LINE__ << "]\t" << mthn << "[DB] read global delay 0x" << std::hex << delay_ << std::dec << endl;  

  if (delay_>=50) {
    std::cout << __LINE__ << "]\t" << mthn << "global delay is out of range (>= 1 Tclk)."  << std::endl;
    std::cout << __LINE__ << "]\t" << mthn << "will not apply any global delays."          << std::endl;
    std::cout << __LINE__ << "]\t" << mthn << "increase the delays in the TPLL if needed." << std::endl;
    delay_=0;
  }
}


PixelGlobalDelay25::PixelGlobalDelay25(std::string filename):
    PixelConfigBase(" "," "," "){

    std::string mthn = "[PixelGlobalDelay25::PixelGlobalDelay25()]\t\t\t    " ;
    std::ifstream in(filename.c_str());

    if (!in.good()){
	std::cout << __LINE__ << "]\t" << mthn << "Could not open: " << filename << std::endl;
	assert(0);
    }
    else {
	std::cout << __LINE__ << "]\t" << mthn << "Opened: " << filename << std::endl;
    }

    in >> std::hex >> delay_ >> std::dec;
    std::cout << __LINE__ << "]\t" << mthn << "read global delay 0x" << std::hex << delay_ << std::dec << endl;  

    in.close();

    if (delay_>=50) {
      std::cout << __LINE__ << "]\t" << mthn << "global delay is out of range (>= 1 Tclk)."  << std::endl;
      std::cout << __LINE__ << "]\t" << mthn << "will not apply any global delays."          << std::endl;
      std::cout << __LINE__ << "]\t" << mthn << "increase the delays in the TPLL if needed." << std::endl;
      delay_=0;
    }
}

 
PixelGlobalDelay25::~PixelGlobalDelay25() {}


unsigned int PixelGlobalDelay25::getDelay(unsigned int offset) const{
  std::string mthn = "[PixelGlobalDelay25::getDelay()]\t\t\t    " ;
  unsigned int ret=offset+delay_;
  if (ret > 127) {
    std::cout << __LINE__ << "]\t" << mthn <<"the required total delay "<<ret<<" is out of range."    << endl;
    std::cout << __LINE__ << "]\t" << mthn <<"original setting: "<<offset<<", global delay: "<<delay_ << endl;
    std::cout << __LINE__ << "]\t" << mthn <<"we will keep the default delay setting..."              << endl;

    ret=offset;
  }

  std::cout << __LINE__ << "]\t" << mthn << "getDelay("<<offset<<") returns "<<ret<<endl;
  return ret;
}


unsigned int PixelGlobalDelay25::getCyclicDelay(unsigned int offset) const{
  std::string mthn = "[PixelGlobalDelay25::getCyclicDelay()]\t\t\t    " ;
  unsigned int ret=offset+delay_;
  if (ret > 120) ret-=50;
  std::cout<< __LINE__ << "]\t" << mthn << "getCyclicDelay("<<offset<<") returns "<<ret<<endl;
  return ret;
}


unsigned int PixelGlobalDelay25::getTTCrxDelay(unsigned int offset) const{
  // Computes the TTCrx delay settting required to compensate for the global Delay25 shift.
  //
  // 'offset' is the current register setting in the TTCrx register
  //
  // The unit of delay_ is 0.499 ns (Delay25 granularity) that needs to be converted
  // to the units of the TTCrx delay generator 103.96 ps

  std::string mthn = "[PixelGlobalDelay25::getTTCrxDelay()]\t\t\t    " ;
  unsigned int K=(offset/16*16+offset%16*15+30)%240;
  K+=(unsigned int)floor((delay_*0.499)/0.1039583 + 0.5); // add max 235


  unsigned int ret;
  if (K>239) {
    std::cout << __LINE__ << "]\t" << mthn << "the required TTCrx fine delay "<<K<<" is out of range."<<endl;
    std::cout << __LINE__ << "]\t" << mthn << "this can happen if the register was initialized to 0"<<endl;
    std::cout << __LINE__ << "]\t" << mthn << "(i.e. delay of 3.1 ns) and the required delay is >21.7 ns."<<endl;    
    std::cout << __LINE__ << "]\t" << mthn << "we will keep the current delay setting..."<<endl;
    ret=offset;
  }else{
    unsigned int n=K%15;
    unsigned int m=((K/15)-n+14)%16;
    ret=16*n+m;
  }
  
  std::cout << __LINE__ << "]\t" << mthn << "getTTCrxDelay("<<offset<<") returns "<<ret<<endl;
  return ret;
  //return offset;
}


unsigned int PixelGlobalDelay25::getTTCrxDelay() const{
  // Computes the TTCrx delay settting required to compensate for the global Delay25 shift.
  //
  // Assumes that the current register setting in the TTCrx is 0 ns (14)
  //
  // The unit of delay_ is 0.499 ns (Delay25 granularity) that needs to be converted
  // to the units of the TTCrx delay generator 103.96 ps
  
  return getTTCrxDelay(14);
}



void PixelGlobalDelay25::writeASCII(std::string dir) const {

  std::string mthn = "[PixelGlobalDelay25::writeASCII()]\t\t\t    " ;
  if (dir!="") dir+="/";
  string filename=dir+"globaldelay25.dat";

  ofstream out(filename.c_str());
  if(!out.good()){
    cout << __LINE__ << "]\t" << mthn << "Could not open file:"<<filename<<endl;
    assert(0);
  }

  out << "0x" << hex << delay_ << dec << endl;

  out.close();
}

//=============================================================================================
void PixelGlobalDelay25::writeXMLHeader(pos::PixelConfigKey key, 
					int version, 
					std::string path, 
					std::ofstream *outstream,
					std::ofstream *out1stream,
					std::ofstream *out2stream)  const
{
  std::stringstream s ; s << __LINE__ << "]\t[[PixelGlobalDelay25::writeASCII()]\t\t\t    " ;
  std::string mthn = s.str() ;
  std::stringstream fullPath ;
  fullPath << path << "/Pixel_GlobalDelay25_" << PixelTimeFormatter::getmSecTime() << ".xml" ;
  cout << mthn << "Writing to: " << fullPath.str() << endl ;
  
  outstream->open(fullPath.str().c_str()) ;
  
  *outstream << "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"                                    << std::endl ;
  *outstream << "<ROOT xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance'>"                               << std::endl ;
  *outstream << ""                                                                                           << std::endl ; 
  *outstream << " <!-- " << mthn << "-->"                                                                    << std::endl ; 
  *outstream << ""                                                                                           << std::endl ; 
  *outstream << " <HEADER>"                                                                                  << std::endl ;
  *outstream << "  <TYPE>"                                                                                   << std::endl ;
  *outstream << "   <EXTENSION_TABLE_NAME>PIXEL_GLOBAL_DELAY25</EXTENSION_TABLE_NAME>"                       << std::endl ;
  *outstream << "   <NAME>Pixel Global Delay25</NAME>"                                                       << std::endl ;
  *outstream << "  </TYPE>"                                                                                  << std::endl ;
  *outstream << "  <RUN>"                                                                                    << std::endl ;
  *outstream << "   <RUN_TYPE>Pixel Global Delay25</RUN_TYPE>"                                               << std::endl ;
  *outstream << "   <RUN_NUMBER>1</RUN_NUMBER>"                                                              << std::endl ;
  *outstream << "   <RUN_BEGIN_TIMESTAMP>" << pos::PixelTimeFormatter::getTime() << "</RUN_BEGIN_TIMESTAMP>" << std::endl ;
  *outstream << "   <LOCATION>CERN P5</LOCATION>"                                                            << std::endl ; 
  *outstream << "  </RUN>"                                                                                   << std::endl ;
  *outstream << " </HEADER>"                                                                                 << std::endl ;
  *outstream << ""                                                                                           << std::endl ;
  *outstream << "  <DATA_SET>"                                                                               << std::endl ;
  *outstream << " "                                                                                          << std::endl ;
  *outstream << "   <VERSION>"             << version      << "</VERSION>"                                   << std::endl ;
  *outstream << "   <COMMENT_DESCRIPTION>" << getComment() << "</COMMENT_DESCRIPTION>"                       << std::endl ;
  *outstream << "   <INITIATED_BY_USER>"   << getAuthor()  << "</INITIATED_BY_USER>"                         << std::endl ; 
  *outstream << " "                                                                                          << std::endl ;
  *outstream << "   <PART>"                                                                                  << std::endl ;
  *outstream << "    <NAME_LABEL>CMS-PIXEL-ROOT</NAME_LABEL>"                                                << std::endl ;
  *outstream << "    <KIND_OF_PART>Detector ROOT</KIND_OF_PART>"                                             << std::endl ;
  *outstream << "   </PART>"                                                                                 << std::endl ;

}

//=============================================================================================
void PixelGlobalDelay25::writeXML( std::ofstream *outstream,
				   std::ofstream *out1stream,
				   std::ofstream *out2stream)  const
{
  std::stringstream s ; s << __LINE__ << "]\t[PixelGlobalDelay25::writeASCII()]\t\t\t    " ;
  std::string mthn = s.str() ;
  *outstream << "  <DATA>"                                                                           << std::endl ;
  *outstream << "   <GLOBALDELAY25>0x" << hex << delay_ << dec << "</GLOBALDELAY25>"                 << std::endl ;
  *outstream << "  </DATA>"                                                                          << std::endl ;
  *outstream << " "                                                                                  << std::endl ;
}

//=============================================================================================
void PixelGlobalDelay25::writeXMLTrailer(std::ofstream *outstream,
					 std::ofstream *out1stream,
					 std::ofstream *out2stream) const
{
  std::stringstream s ; s << __LINE__ << "]\t[PixelGlobalDelay25::writeASCII()]\t\t\t    " ;
  std::string mthn = s.str() ;
  
  *outstream << " "                                                                                          << std::endl ;
  *outstream << " </DATA_SET>"                                                                               << std::endl ;
  *outstream << "</ROOT> "                                                                                   << std::endl ;

  outstream->close() ;
}

