#include "DQM/HLTEvF/plugins/PSMonitor.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQM/TrackingMonitor/interface/GetLumi.h"


// -----------------------------
//  constructors and destructor
// -----------------------------

PSMonitor::PSMonitor( const edm::ParameterSet& iConfig ) : 
  folderName_   ( iConfig.getParameter<std::string>("FolderName") )
  , ugtBXToken_ ( consumes<GlobalAlgBlkBxCollection>(iConfig.getParameter<edm::InputTag>("ugtBXInputTag") ) )
{

  /// Prescale service
  if ( edm::Service<edm::service::PrescaleService>().isAvailable() )
      psService_ = edm::Service<edm::service::PrescaleService>().operator->();

  psColumnIndexVsLS_ = nullptr;

  edm::ParameterSet histoPSet    = iConfig.getParameter<edm::ParameterSet>("histoPSet");
  edm::ParameterSet psColumnPSet = histoPSet.getParameter<edm::ParameterSet>("psColumnPSet");
  edm::ParameterSet lsPSet       = histoPSet.getParameter<edm::ParameterSet>("lsPSet");

  getHistoPSet(psColumnPSet, ps_binning_);
  getHistoPSet(lsPSet,       ls_binning_);

}

void PSMonitor::getHistoPSet(edm::ParameterSet& pset, MEbinning& mebinning)
{
  mebinning.nbins = pset.getParameter<int32_t>("nbins");
  mebinning.xmin  = 0.;
  mebinning.xmax  = double(pset.getParameter<int32_t>("nbins"));
}


void PSMonitor::bookHistograms(DQMStore::IBooker     & ibooker,
			       edm::Run const        & iRun,
			       edm::EventSetup const & iSetup) 
{  
  
  std::string histname, histtitle;

  std::string currentFolder = folderName_ ;
  ibooker.setCurrentFolder(currentFolder);

  std::vector<std::string> psLabels = psService_->getLvl1Labels();
  int nbins   = ( !psLabels.empty() ? psLabels.size()         : ps_binning_.nbins );
  double xmin = ( !psLabels.empty() ? 0.                      : ps_binning_.xmin  );
  double xmax = ( !psLabels.empty() ? double(psLabels.size()) : ps_binning_.xmax  );

  histname = "psColumnIndexVsLS"; histtitle = "PS column index vs LS";
  psColumnIndexVsLS_ = ibooker.book2D(histname, histtitle, 
				      ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax,
				      nbins, xmin, xmax);
  psColumnIndexVsLS_->setAxisTitle("LS",1);
  psColumnIndexVsLS_->setAxisTitle("PS column index",2);
  
  int ibin = 1;
  for ( auto l : psLabels ) {
    psColumnIndexVsLS_->setBinLabel(ibin,l,2);
    ibin++;
  }

}

void PSMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup)  {

  int ls = iEvent.id().luminosityBlock();

  int psColumn = -1;
  
  edm::Handle<GlobalAlgBlkBxCollection> ugtBXhandle;
  iEvent.getByToken(ugtBXToken_, ugtBXhandle);
  if (ugtBXhandle.isValid() and not ugtBXhandle->isEmpty(0))
    psColumn = ugtBXhandle->at(0, 0).getPreScColumn();

  psColumnIndexVsLS_->Fill(ls, psColumn);

}

void PSMonitor::fillHistoPSetDescription(edm::ParameterSetDescription & pset, int value)
{
  pset.add<int>( "nbins", value);
}

void PSMonitor::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>( "ugtBXInputTag", edm::InputTag("hltGtStage2Digis") );
  desc.add<std::string>  ( "FolderName",    "HLT/PSMonitoring" );

  edm::ParameterSetDescription histoPSet;

  edm::ParameterSetDescription psColumnPSet;
  fillHistoPSetDescription(psColumnPSet,8);
  histoPSet.add<edm::ParameterSetDescription>("psColumnPSet", psColumnPSet);

  edm::ParameterSetDescription lsPSet;
  fillHistoPSetDescription(lsPSet,2500);
  histoPSet.add<edm::ParameterSetDescription>("lsPSet", lsPSet);

  desc.add<edm::ParameterSetDescription>("histoPSet",histoPSet);

  descriptions.add("psMonitoring", desc);
}

// Define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PSMonitor);
