#ifndef PixelGlobalDelay25_h
#define PixelGlobalDelay25_h
/**
* \file CalibFormats/SiPixelObjects/interface/PixelGlobalDelay25.h
* \brief This class specifies which delay25 channels are delayed over the entire pixel detector and by how much
*
*   A longer explanation will be placed here later
*
*/
#include <iostream>
#include <vector>
#include <string>
#include "CalibFormats/SiPixelObjects/interface/PixelConfigBase.h"

namespace pos{
/*!  \ingroup ConfigurationObjects "Configuration Objects"
*    
*  @{
*
*  \class PixelGlobalDelay25 PixelGlobalDelay25.h
*  \brief This class specifies which delay25 channels are delayed over the entire pixel detector and by how much
*
*   A longer explanation will be placed here later
*
*/
  class PixelGlobalDelay25: public PixelConfigBase {

  public:

    PixelGlobalDelay25(std::string filename);                         // create from file
    PixelGlobalDelay25(std::vector<std::vector<std::string> > & tab); // create from DB
    virtual ~PixelGlobalDelay25(); 

    unsigned int getDelay(      unsigned int offset=0) const; // delays in steps of 0.499 ns (Delay25 step)
    unsigned int getCyclicDelay(unsigned int offset=0) const; // delays in steps of 0.499 ns (Delay25 step)
    unsigned int getTTCrxDelay( unsigned int offset  ) const; // delays in steps of 0.10396 ns (TTCrx step)
    unsigned int getTTCrxDelay(                      ) const; // delays in steps of 0.10396 ns (TTCrx step)
                                        		      // but assumes that the default TTCrx delay is 0 ns
    virtual void setDelay(      unsigned int delay) {delay_ = delay ;}

    virtual void writeASCII(std::string dir) const;
    //    void 	 writeXML(      pos::PixelConfigKey key, int version, std::string path)                     const ;
    virtual void writeXMLHeader(  pos::PixelConfigKey key, 
				  int version, 
				  std::string path, 
				  std::ofstream *out,
				  std::ofstream *out1 = NULL,
				  std::ofstream *out2 = NULL
				  ) const ;
    virtual void writeXML(        std::ofstream *out,					   	 	    
			   	  std::ofstream *out1 = NULL ,
			   	  std::ofstream *out2 = NULL )  const ;
    virtual void writeXMLTrailer( std::ofstream *out, 
				  std::ofstream *out1 = NULL,
				  std::ofstream *out2 = NULL
				  ) const ;

  private:
    unsigned int delay_;
    
  };
}
/* @} */
#endif
