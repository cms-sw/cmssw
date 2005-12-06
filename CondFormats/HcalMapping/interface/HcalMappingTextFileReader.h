/* -*- C++ -*- */
#ifndef HcalMappingTextFileReader_h_included
#define HcalMappingTextFileReader_h_included 1

#include "CondFormats/HcalMapping/interface/HcalMapping.h"
#include <memory>

/** \class HcalMappingTextFileReader

   Format of the text file is columns of data separated by whitespace.

   Column 1 : line number
   Column 2 : crate
   Column 3 : HTR slot
   Column 4 : HTR top/bottom [t/b]
   Column 5 : DCC number (local/minus 579)
   Column 6 : DCC spigot
   Column 7 : Fiber index
   Column 8 : Fiber channel id
   Column 9 : Subdet (HB,HE,HF,HO)
   Column 10: IETA
   Column 11: IPHI
   Column 12: Depth
    
   $Date: 2005/10/04 14:32:40 $
   $Revision: 1.2 $
   \author J. Mans - Minnesota
*/
class HcalMappingTextFileReader {
public:
  static std::auto_ptr<HcalMapping> readFromFile(const char* filename, bool maintainL2E=false);
};


#endif // HcalMappingTextFileReader_h_included
