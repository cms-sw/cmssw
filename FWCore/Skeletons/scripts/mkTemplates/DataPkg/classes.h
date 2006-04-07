#include "DataFormats/Common/interface/Wrapper.h"

//Add includes for your classes here
#include "skelsubsys/datapkgname/YOUR_CLASS_GOES_HERE.h"

namespace {
   namespace {
      //add 'dummy' Wrapper variable for each class type you put into the Event
      edm::Wrapper<YOUR_CLASS_GOES_HERE> dummy1;
   }
}
