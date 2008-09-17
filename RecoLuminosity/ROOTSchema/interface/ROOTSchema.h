#ifndef ROOTSCHEMA_H
#define ROOTSCHEMA_H

/*

Author: Adam Hunt - Princeton University
Date: 2007-10-05

*/

#include "RecoLuminosity/ROOTSchema/interface/ROOTFileWriter.h"

namespace HCAL_HLX{

  class ROOTSchema: public ROOTFileWriter{
    
  public:
    
    ROOTSchema();
    ~ROOTSchema();
    
    bool ProcessSection(const HCAL_HLX::LUMI_SECTION & lumiSection);

  };
}

#endif
