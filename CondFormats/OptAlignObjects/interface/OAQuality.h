#ifndef OAQuality_H
#define OAQuality_H

enum OAQuality { oa_fixed, oa_calibrated, oa_unknown };

struct OAQualityTranslator {
    
  static const char * name(OAQuality oaq) 
  {
    static const char* c[] = { 
      "fixed",
      "calibrated",
      "unknown"   
    };
    return c[oaq];   			  
  }
  
  static const OAQuality index( const int& ind ) {
    switch (ind) {
    case 0:
      return oa_fixed;
      break;
    case 1:
      return oa_calibrated;
      break;
    case 2:
      return oa_unknown;
      break;
    default:
      return oa_unknown;
      break;
    }
  }

};
#endif
