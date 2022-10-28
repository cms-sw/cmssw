#include "DQMOffline/RecoB/plugins/MiniAODTaggerHarvester.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"

MiniAODTaggerHarvester::MiniAODTaggerHarvester(const edm::ParameterSet& pSet)
    : folder_(pSet.getParameter<std::string>("folder")),
      discrParameters_(pSet.getParameter<edm::ParameterSet>("parameters")),

      mclevel_(pSet.getParameter<int>("MClevel")),
      doCTagPlots_(pSet.getParameter<bool>("CTagPlots")),
      dodifferentialPlots_(pSet.getParameter<bool>("differentialPlots")),
      discrCut_(pSet.getParameter<double>("discrCut")),

      etaActive_(pSet.getParameter<bool>("etaActive")),
      etaMin_(pSet.getParameter<double>("etaMin")),
      etaMax_(pSet.getParameter<double>("etaMax")),
      ptActive_(pSet.getParameter<bool>("ptActive")),
      ptMin_(pSet.getParameter<double>("ptMin")),
      ptMax_(pSet.getParameter<double>("ptMax"))

{}

MiniAODTaggerHarvester::~MiniAODTaggerHarvester() {}

void MiniAODTaggerHarvester::dqmEndJob(DQMStore::IBooker& ibook, DQMStore::IGetter& iget) {
  jetTagPlotter_ = std::make_unique<JetTagPlotter>(folder_,
                                                   EtaPtBin(etaActive_, etaMin_, etaMax_, ptActive_, ptMin_, ptMax_),
                                                   discrParameters_,
                                                   mclevel_,
                                                   true,
                                                   ibook,
                                                   doCTagPlots_,
                                                   dodifferentialPlots_,
                                                   discrCut_);

  jetTagPlotter_->finalize(ibook, iget);
}

//define this as a plug-in
DEFINE_FWK_MODULE(MiniAODTaggerHarvester);
