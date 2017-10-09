#ifndef RecoJets_JetProducers_AnomalousTower_h
#define RecoJets_JetProducers_AnomalousTower_h

#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

class AnomalousTower
{
public:
    explicit AnomalousTower(const edm::ParameterSet&);
    virtual ~AnomalousTower() {}

    // operator() returns "true" if the tower is anomalous
    virtual bool operator()(const reco::Candidate& input) const;

private:
    AnomalousTower();

    const unsigned maxBadEcalCells;          // maximum number of bad ECAL cells
    const unsigned maxRecoveredEcalCells;    // maximum number of recovered ECAL cells
    const unsigned maxProblematicEcalCells;  // maximum number of problematic ECAL cells
    const unsigned maxBadHcalCells;          // maximum number of bad HCAL cells
    const unsigned maxRecoveredHcalCells;    // maximum number of recovered HCAL cells
    const unsigned maxProblematicHcalCells;  // maximum number of problematic HCAL cells
};

#endif // RecoJets_JetProducers_AnomalousTower_h
