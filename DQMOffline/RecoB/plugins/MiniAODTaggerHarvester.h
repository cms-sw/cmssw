#ifndef MiniAODTaggerHarvester_H
#define MiniAODTaggerHarvester_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMOffline/RecoB/interface/JetTagPlotter.h"

/** \class MiniAODTaggerHarvester
 *
 *  Tagger harvester to run on MiniAOD
 *
 */

class MiniAODTaggerHarvester : public DQMEDHarvester {
public:
    explicit MiniAODTaggerHarvester(const edm::ParameterSet& pSet);
    ~MiniAODTaggerHarvester() override;

private:
    void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;

    std::unique_ptr<JetTagPlotter> jetTagPlotter_;


    //add consumes
    std::string folder_;
    edm::ParameterSet disrParameters_;

    bool doCTagPlots_;
    bool dodifferentialPlots_;
    double discrCut_;

    bool etaActive_;
    double etaMin_;
    double etaMax_;
    bool ptActive_;
    double ptMin_;
    double ptMax_;
};

#endif
