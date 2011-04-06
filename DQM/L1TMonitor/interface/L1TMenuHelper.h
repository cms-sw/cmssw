#ifndef L1TMenuHelper_H
#define L1TMenuHelper_H

/*
 * \file L1TMenuHelper.h
 *
 * $Date: 2011/03/10 15:12:25 $
 * $Revision: 1.1 $
 * \author J. Pela
 *
*/

// system include files
#include <memory>
#include <unistd.h>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"

#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtUtils.h"

#include <iostream>
#include <fstream>
#include <vector>


// Simplified structure for single object conditions information
struct SingleObjectCondition{

  std::string           name;
  L1GtConditionCategory conditionCategory;
  L1GtConditionType     conditionType;
  L1GtObject            object;
  unsigned int          threshold;

};

// Simplified structure for single object conditions information
struct SingleObjectTrigger{

  std::string           alias;
  unsigned int          bit;
  int                   prescale;
  unsigned int          threshold;

};

class L1TMenuHelper {

  public:

    L1TMenuHelper(const edm::EventSetup& iSetup); // Constructor
    ~L1TMenuHelper();                             // Destructor

    // Get Lowest Unprescaled Single Object Triggers
    std::map<std::string,std::string> getLUSOTrigger(std::map<std::string,bool> iCategories, int IndexRefPrescaleFactors);

    // To convert enum to strings
    std::string enumToStringL1GtObject           (L1GtObject            iObject);
    std::string enumToStringL1GtConditionType    (L1GtConditionType     iConditionType);
    std::string enumToStringL1GtConditionCategory(L1GtConditionCategory iConditionCategory);

  private:

    L1GtUtils myUtils;

    const L1GtTriggerMenu*                m_l1GtMenu;
    const std::vector<std::vector<int> >* m_prescaleFactorsAlgoTrig;

};



#endif
