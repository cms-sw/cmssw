#ifndef PixelConfigBase_h
#define PixelConfigBase_h
/*! \file CalibFormats/SiPixelObjects/interface/PixelConfigBase.h
*   \brief This file contains the base class for "pixel configuration data" 
*          management
*
*   A longer explanation will be placed here later
*/
//
// Base class for pixel configuration data
// provide a place to implement common interfaces
// for these objects. Any configuration data
// object that is to be accessed from the database
// should derive from this class.
//

#include <string>
#include "CalibFormats/SiPixelObjects/interface/PixelConfigKey.h"
#include "CalibFormats/SiPixelObjects/interface/PixelBase64.h"


namespace pos{
/*!  \ingroup ConfigurationObjects "Configuration Objects"
*    
*  @{
*
*   \class PixelConfigBase PixelConfigBase.h "interface/PixelConfigBase.h"
*   \brief This file contains the base class for "pixel configuration data" 
*          management
*
*   A longer explanation will be placed here later
*/
  class PixelConfigBase {

  public:

    //A few things that you should provide
    //description : purpose of this object
    //creator : who created this configuration object
    //date : time/date of creation (should probably not be
    //       a string, but I have no idea what CMS uses.
    PixelConfigBase(std::string description,
		    std::string creator,
                    std::string date);

    virtual ~PixelConfigBase(){}

    std::string description();
    std::string creator();
    std::string date();

    //Interface to write out data to ascii file
    virtual void writeASCII(std::string dir="") const = 0;
    //Interface to write out data to XML file for DB population
    virtual void writeXMLHeader(  pos::PixelConfigKey key, 
				  int version, 
				  std::string path, 
				  std::ofstream *out,
				  std::ofstream *out1,
				  std::ofstream *out2)  const {;}
    virtual void writeXML( 	  std::ofstream *out,
			   	  std::ofstream *out1,
			   	  std::ofstream *out2 ) const {;}
    virtual void writeXMLTrailer( std::ofstream *out,
				  std::ofstream *ou1, 
				  std::ofstream *out2)  const {;}
    virtual void writeXML(      pos::PixelConfigKey key, int version, std::string path)                     const {;}
    virtual void writeXMLHeader(pos::PixelConfigKey key, int version, std::string path, std::ofstream *out) const {;}
    virtual void writeXML(                                                              std::ofstream *out) const {std::cout << __LINE__ << " " << __PRETTY_FUNCTION__ << "\tUnimplemented method" << std::endl ;;}
    virtual void writeXMLTrailer(                                                       std::ofstream *out) const {;}

    void setAuthor (std::string author)  {creator_ = author ;} 
    void setComment(std::string comment) {comment_ = comment;} 
    std::string getAuthor()const         {return creator_   ;}
    std::string getComment()const        {return base64_encode((unsigned char *)comment_.c_str(), comment_.length()) ;}
    
  private:

    std::string description_;
    std::string creator_ ;
    std::string date_    ;
    std::string comment_ ;

  };

/* @} */
}

#endif
