#include "DQMOffline/RecoB/plugins/MiniAODTaggerHarvester.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"



MiniAODTaggerHarvester::MiniAODTaggerHarvester(const edm::ParameterSet& pSet)
: folder_(pSet.getParameter<std::string>("folder")),
disrParameters_(pSet.getParameter<edm::ParameterSet>("parameters")),

isMC_(pSet.getParameter<bool>("MCplots")),
doCTagPlots_(pSet.getParameter<bool>("CTagPlots")),
dodifferentialPlots_(pSet.getParameter<bool>("differentialPlots")),
discrCut_(pSet.getParameter<double>("discrCut")),

etaActive_(pSet.getParameter<bool>("etaActive")),
etaMin_(pSet.getParameter<double>("etaMin")),
etaMax_(pSet.getParameter<double>("etaMax")),
ptActive_(pSet.getParameter<bool>("ptActive")),
ptMin_(pSet.getParameter<double>("ptMin")),
ptMax_(pSet.getParameter<double>("ptMax"))

{

}


MiniAODTaggerHarvester::~MiniAODTaggerHarvester() { }


void MiniAODTaggerHarvester::dqmEndJob(DQMStore::IBooker& ibook, DQMStore::IGetter& iget)
{
    int MClevel = 0;
    if(isMC_)
    {
        MClevel = 4;
    }

    jetTagPlotter_ = std::make_unique<JetTagPlotter>(folder_,
                                                     EtaPtBin(etaActive_, etaMin_, etaMax_, ptActive_, ptMin_, ptMax_),
                                                     disrParameters_,
                                                     MClevel,
                                                     true,
                                                     ibook,
                                                     doCTagPlots_,
                                                     dodifferentialPlots_,
                                                     discrCut_
                                                    );

    jetTagPlotter_->finalize(ibook, iget);
}


//define this as a plug-in
DEFINE_FWK_MODULE(MiniAODTaggerHarvester);
