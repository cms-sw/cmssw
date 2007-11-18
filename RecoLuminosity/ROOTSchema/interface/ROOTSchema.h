#ifndef ROOTSCHEMA_H
#define ROOTSCHEMA_H

/*

Author: Adam Hunt - Princeton University
Date: 2007-10-05

*/

#include "RecoLuminosity/ROOTSchema/interface/ROOTFileBase.h"

using std::cout;
using std::endl;

class ROOTSchema: public ROOTFileBase{
        
public:
      
    ROOTSchema();
    ~ROOTSchema();

    void ProcessSection(const HCAL_HLX::LUMI_SECTION & lumiSection);
};

#endif
