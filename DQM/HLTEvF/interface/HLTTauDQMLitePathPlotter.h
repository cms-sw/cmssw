/* HLTTau Path  Analyzer
 Michail Bachtis
 University of Wisconsin - Madison
 bachtis@hep.wisc.edu
 */

#ifndef HLTTauDQMLitePathPlotter_h
#define HLTTauDQMLitePathPlotter_h

#include "DQM/HLTEvF/interface/HLTTauDQMPlotter.h"

#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerRefsCollections.h"

class HLTTauDQMLitePathPlotter : public HLTTauDQMPlotter {
public:
    
    HLTTauDQMLitePathPlotter( const edm::ParameterSet&, int, int, int, double, bool, double, std::string );
    ~HLTTauDQMLitePathPlotter();
    const std::string name() { return name_; }
    void analyze( const edm::Event&, const edm::EventSetup&, const std::map<int,LVColl>& );
    
private:
    void endJob();
    LVColl getFilterCollection( size_t, int, const trigger::TriggerEvent& );
    LVColl getObjectCollection( int, const trigger::TriggerEvent& );
    
    /// InputTag of TriggerEventWithRefs to analyze
    edm::InputTag triggerEvent_;
    
    //The filters
    std::vector<edm::ParameterSet> filters_;
    std::vector<HLTTauDQMPlotter::FilterObject> filterObjs_;
    
    bool doRefAnalysis_;
    double matchDeltaR_;
    double minEt_;
    double maxEt_;
    int binsEt_;
    int binsEta_;
    int binsPhi_;
    
    double refTauPt_;
    double refLeptonPt_;
    
    //MonitorElements for paths
    MonitorElement *accepted_events;
    MonitorElement *accepted_events_matched;
    MonitorElement *ref_events;
    
    std::vector<MonitorElement*> mass_distribution;
    
    //MonitorElements for objects
    MonitorElement *tauEt;
    MonitorElement *tauEta;
    MonitorElement *tauPhi;
    
    MonitorElement *tauEtEffNum;
    MonitorElement *tauEtaEffNum;
    MonitorElement *tauPhiEffNum;
    
    MonitorElement *tauEtEffDenom;
    MonitorElement *tauEtaEffDenom;
    MonitorElement *tauPhiEffDenom;
      
    class LVSorter {
    public:
        LVSorter() {}
        ~LVSorter() {}
        bool operator()(LV p1, LV p2) {
            return p1.Et() < p2.Et();
        }
    };
};
#endif
