#ifndef DQM_L1TMONITOR_L1TMENUHELPER_H
#define DQM_L1TMONITOR_L1TMENUHELPER_H

/*
 * \file L1TMenuHelper.h
 *
 * $Date: 2011/04/14 13:03:11 $
 * $Revision: 1.2 $
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
  unsigned int          quality;
  unsigned int          etaRange;
  unsigned int          threshold;

};

// Simplified structure for single object conditions information
struct SingleObjectTrigger{

  L1GtObject   object;
  std::string  alias;
  unsigned int bit;
  int          prescale;
  unsigned int threshold; //
  unsigned int quality;   // Only aplicable to Muons
  unsigned int etaRange;  // Only aplicable to Muons

  bool operator< (const SingleObjectTrigger &iSOT){

    // First we order by lowest prescale
    if     (this->prescale <  iSOT.prescale){return true;}
    else if(this->prescale == iSOT.prescale){

      // If L1GtObject == Mu we compare also with respect to conditions
      if(this->object == Mu){

        if     (this->quality >  iSOT.quality){return true;}
        else if(this->quality <  iSOT.quality){return false;}
        else if(this->quality == iSOT.quality){     

          if     (this->etaRange >  iSOT.etaRange){return true;}
          else if(this->etaRange <  iSOT.etaRange){return false;}
          else if(this->etaRange == iSOT.etaRange){return this->threshold < iSOT.threshold;}

        }
      }else{return this->threshold < iSOT.threshold;}
    }

    return false;

  };

  friend bool operator< (const SingleObjectTrigger &iSOT1,const SingleObjectTrigger &iSOT2){

    // First we order by lowest prescale
    if     (iSOT1.prescale <  iSOT2.prescale){return true;}
    else if(iSOT1.prescale == iSOT2.prescale){

      // If L1GtObject == Mu we compare also with respect to conditions
      if(iSOT1.object == Mu){

        if     (iSOT1.quality >  iSOT2.quality){return true;}
        else if(iSOT1.quality <  iSOT2.quality){return false;}
        else if(iSOT1.quality == iSOT2.quality){     

          if     (iSOT1.etaRange >  iSOT2.etaRange){return true;}
          else if(iSOT1.etaRange <  iSOT2.etaRange){return false;}
          else if(iSOT1.etaRange == iSOT2.etaRange){return iSOT1.threshold < iSOT2.threshold;}

        }
      }else{return iSOT1.threshold < iSOT2.threshold;}
    }

    return false;

  };

};

class L1TMenuHelper {

  public:

    L1TMenuHelper(const edm::EventSetup& iSetup); // Constructor
    ~L1TMenuHelper();                             // Destructor

    // Get Lowest Unprescaled Single Object Triggers
    std::map<std::string,std::string> getLUSOTrigger(std::map<std::string,bool> iCategories, int IndexRefPrescaleFactors);
    std::map<std::string,std::string> testAlgos     (std::map<std::string,std::string>);
    

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
