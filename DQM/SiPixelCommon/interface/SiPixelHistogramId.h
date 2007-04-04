#ifndef SiPixelCommon_SiPixelHistogramId_h
#define SiPixelCommon_SiPixelHistogramId_h
// -*- C++ -*-
//
// Package:     SiPixelCommon
// Class  :     SiPixelHistogramId
// 
/**\class SiPixelHistogramId SiPixelHistogramId.h DQM/SiPixelCommon/interface/SiPixelHistogramId.h

 Description: Returns histogram Id

 Usage:
    <usage>

*/
//
// Original Author:  chiochia
//         Created:  Wed Feb 22 16:07:51 CET 2006
// $Id: SiPixelHistogramId.h,v 1.1 2006/03/08 12:55:11 chiochia Exp $
//

#include <string>
#include <boost/cstdint.hpp>

class SiPixelHistogramId
{
 public:

  /// Constructor
  SiPixelHistogramId();
  /// Constructor
  SiPixelHistogramId(std::string dataCollection);
  /// Destructor
  virtual ~SiPixelHistogramId();
  /// Create Histogram Id
  std::string createHistoId( std::string variable,  uint32_t& rawId );

  // extract the component_id and the id_type from a histogram id
  //uint32_t    getComponentId(std::string histoid);
  //std::string getComponentType(std::string histoid);
 private:

  std::string dataCollection_;
  std::string separator_;

  //SiPixelHistoId(const SiPixelHistoId&); // stop default
  //const SiPixelHistoId& operator=(const SiPixelHistoId&); // stop default
  //std::string returnIdPart(std::string histoid, uint32_t whichpart);
  // ---------- member data --------------------------------
  //std::string separator1;
  //std::string separator2;
};

#endif
