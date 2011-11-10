/* HLTTau Path  Analyzer
 Michail Bachtis
 University of Wisconsin - Madison
 bachtis@hep.wisc.edu
 */

#ifndef HLTTauDQMPathPlotter_h
#define HLTTauDQMPathPlotter_h

#include "DQM/HLTEvF/interface/HLTTauDQMPlotter.h"

#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerRefsCollections.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"

class HLTTauDQMPathPlotter : public HLTTauDQMPlotter {
public:
    HLTTauDQMPathPlotter( const edm::ParameterSet&, bool, std::string );
    ~HLTTauDQMPathPlotter();
    const std::string name() { return name_; }
    void analyze( const edm::Event&, const edm::EventSetup&, const std::map<int,LVColl>& );
    
private:
    void endJob() ;
    LVColl getFilterCollection( size_t, int, const trigger::TriggerEventWithRefs& );
    
    //InputTag of TriggerEventWithRefs to analyze
    edm::InputTag triggerEventObject_;
    
    //The filters
    std::vector<edm::ParameterSet> filters_;
    std::vector<HLTTauDQMPlotter::FilterObject> filterObjs_;
    
    //Reference parameters
    edm::ParameterSet reference_;
    bool doRefAnalysis_;
    unsigned int refNTriggeredTaus_;
    unsigned int refNTriggeredLeptons_;
    double refTauPt_;
    double refLeptonPt_;
        
    //MonitorElements
    MonitorElement *accepted_events;
    MonitorElement *accepted_events_matched;
};
#endif
