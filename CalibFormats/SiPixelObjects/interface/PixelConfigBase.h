#ifndef PixelConfigBase_h
#define PixelConfigBase_h
//
// Base class for pixel configuration data
// provide a place to implement common interfaces
// for these objects. Any configuration data
// object that is to be accessed from the database
// should derive from this class
//

#include <string>

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

    std::string description();
    std::string creator();
    std::string date();

 private:

    std::string description_;
    std::string creator_;
    std::string date_;
     

};


#endif
