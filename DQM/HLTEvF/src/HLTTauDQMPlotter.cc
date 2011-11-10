#include "DQM/HLTEvF/interface/HLTTauDQMPlotter.h"

HLTTauDQMPlotter::HLTTauDQMPlotter() {
    //Declare DQM Store
    store_ = edm::Service<DQMStore>().operator->();
    validity_ = false;
}

HLTTauDQMPlotter::~HLTTauDQMPlotter() {
}

std::pair<bool,LV> HLTTauDQMPlotter::match( const LV& jet, const LVColl& McInfo, double dr ) {
    bool matched = false;
    LV out;
    for ( std::vector<LV>::const_iterator it = McInfo.begin(); it != McInfo.end(); ++it ) {
        double delta = ROOT::Math::VectorUtil::DeltaR(jet,*it);
        if ( delta < dr ) {
            matched = true;
            out = *it;
            break;
        }
    }
    return std::pair<bool,LV>(matched,out);
}

std::string HLTTauDQMPlotter::triggerTag() {
    if ( triggerTagAlias_ != "" ) {
        return dqmBaseFolder_+triggerTagAlias_;
    }
    return dqmBaseFolder_+triggerTag_;
}



HLTTauDQMPlotter::FilterObject::FilterObject( const edm::ParameterSet& ps ) {
    validity_ = false;
    try {
        FilterName_ = ps.getUntrackedParameter<edm::InputTag>("FilterName");
        MatchDeltaR_ = ps.getUntrackedParameter<double>("MatchDeltaR",0.5);
        NTriggeredTaus_ = ps.getUntrackedParameter<unsigned int>("NTriggeredTaus");
        TauType_ = ps.getUntrackedParameter<int>("TauType");
        NTriggeredLeptons_ = ps.getUntrackedParameter<unsigned int>("NTriggeredLeptons");
        LeptonType_ = ps.getUntrackedParameter<int>("LeptonType");
        Alias_ = ps.getUntrackedParameter<std::string>("Alias",FilterName_.label());
        validity_ = true;
    } catch ( cms::Exception &e ) {
        edm::LogWarning("HLTTauDQMPlotter::FilterObject") << e.what() << std::endl;
    }
}

HLTTauDQMPlotter::FilterObject::~FilterObject() {
}

int HLTTauDQMPlotter::FilterObject::leptonId() {
    if ( LeptonType_ == trigger::TriggerL1IsoEG || LeptonType_ == trigger::TriggerL1NoIsoEG || LeptonType_ == trigger::TriggerElectron || std::abs(LeptonType_) == 11 ) {
        return 11;
    } else if ( LeptonType_ == trigger::TriggerL1Mu || LeptonType_ == trigger::TriggerMuon || std::abs(LeptonType_) == 13 ) {
        return 13;
    } else if ( LeptonType_ == trigger::TriggerL1TauJet || LeptonType_ == trigger::TriggerL1CenJet || LeptonType_ == trigger::TriggerTau || std::abs(LeptonType_) == 15 ) {
        return 15;
    }
    return 0;
}
