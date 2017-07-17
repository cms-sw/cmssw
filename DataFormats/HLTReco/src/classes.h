#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/HLTReco/interface/ModuleTiming.h"
#include "DataFormats/HLTReco/interface/HLTPerformanceInfo.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/HLTReco/interface/HLTPrescaleTable.h"
#include "DataFormats/Common/interface/Wrapper.h"

#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "DataFormats/Candidate/interface/CompositeCandidateFwd.h"

#include "DataFormats/Common/interface/Ref.h"


namespace DataFormats_HLTReco {
  struct dictionary {

    edm::EventTime                                et0;

    edm::Wrapper<edm::EventTime>                wet10;

    // Performance Information
    HLTPerformanceInfo pw0;
    edm::Wrapper<HLTPerformanceInfo> pw1;
    HLTPerformanceInfoCollection pw2; 
    edm::Wrapper<HLTPerformanceInfoCollection> pw3; 

    HLTPerformanceInfo::Module pw4;
    HLTPerformanceInfo::Path pw6;
    std::vector<HLTPerformanceInfo::Module> pw8;
    std::vector<HLTPerformanceInfo::Module>::const_iterator pw9;
    std::vector<HLTPerformanceInfo::Path> pw10;
    std::vector<HLTPerformanceInfo::Path>::const_iterator pw11;
    //HLTPerformanceInfo::Path::const_iterator pw13;

    std::vector<trigger::TriggerObjectType> v_t_tot;
    std::vector<trigger::TriggerObjectType>::const_iterator v_t_tot_ci;
    edm::Wrapper<std::vector<trigger::TriggerObjectType> > w_v_t_tot;

    trigger::TriggerObjectCollection toc;
    trigger::TriggerRefsCollections trc;
    trigger::TriggerFilterObjectWithRefs tfowr;
    trigger::TriggerEvent te;
    trigger::TriggerEventWithRefs tewr;

    edm::Wrapper<trigger::TriggerObjectCollection> wtoc;
    edm::Wrapper<trigger::TriggerFilterObjectWithRefs> wtfowr;
    edm::Wrapper<trigger::TriggerEvent> wte;
    edm::Wrapper<trigger::TriggerEventWithRefs> wtewr;

    std::map<std::string,std::vector<unsigned int> > msu;
    trigger::HLTPrescaleTable hpt;
    edm::Wrapper<trigger::HLTPrescaleTable> whpt;

  };
}
