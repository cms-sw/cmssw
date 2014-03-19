
/*
 *  See header file for a description of this class.
 *
 *  \author:  Mia Tosi,40 3-B32,+41227671609 
 */

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/TrackingMonitor/interface/VertexMonitor.h"

#include "DQM/TrackingMonitor/interface/GetLumi.h"

#include "TMath.h"


VertexMonitor::VertexMonitor(const edm::ParameterSet& iConfig, const edm::InputTag& primaryVertexInputTag, const edm::InputTag& selectedPrimaryVertexInputTag, std::string pvLabel)
    : conf_( iConfig )
    , primaryVertexInputTag_         ( primaryVertexInputTag )
    , selectedPrimaryVertexInputTag_ ( selectedPrimaryVertexInputTag )
    , label_                         ( pvLabel )
    , NumberOfPVtx(NULL)
    , NumberOfPVtxVsBXlumi(NULL)
    , NumberOfPVtxVsGoodPVtx(NULL)
    , NumberOfGoodPVtx(NULL)
    , NumberOfGoodPVtxVsBXlumi(NULL)
    , FractionOfGoodPVtx(NULL)
    , FractionOfGoodPVtxVsBXlumi(NULL)
    , FractionOfGoodPVtxVsGoodPVtx(NULL)
    , FractionOfGoodPVtxVsPVtx(NULL)
    , NumberOfBADndofPVtx(NULL)
    , NumberOfBADndofPVtxVsBXlumi(NULL)
    , NumberOfBADndofPVtxVsGoodPVtx(NULL)
    , GoodPVtxSumPt(NULL)
    , GoodPVtxSumPtVsBXlumi(NULL)
    , GoodPVtxSumPtVsGoodPVtx(NULL)
    , GoodPVtxNumberOfTracks(NULL)
    , GoodPVtxNumberOfTracksVsBXlumi(NULL)
    , GoodPVtxNumberOfTracksVsGoodPVtx(NULL)
    , GoodPVtxNumberOfTracksVsGoodPVtxNdof(NULL)
    , GoodPVtxChi2oNDFVsGoodPVtx(NULL)
    , GoodPVtxChi2oNDFVsBXlumi(NULL)
    , GoodPVtxChi2ProbVsGoodPVtx(NULL)
    , GoodPVtxChi2ProbVsBXlumi(NULL)
    , doAllPlots_       ( conf_.getParameter<bool>("doAllPlots")        )
    , doPlotsVsBXlumi_  ( conf_.getParameter<bool>("doPlotsVsBXlumi")   )
    , doPlotsVsGoodPVtx_( conf_.getParameter<bool>("doPlotsVsGoodPVtx") )

{
   //now do what ever initialization is needed
  if ( doPlotsVsBXlumi_ )
    lumiDetails_ = new GetLumi( iConfig.getParameter<edm::ParameterSet>("BXlumiSetup") );

}

VertexMonitor::VertexMonitor(const edm::ParameterSet& iConfig, const edm::InputTag& primaryVertexInputTag, const edm::InputTag& selectedPrimaryVertexInputTag, std::string pvLabel, edm::ConsumesCollector& iC) : VertexMonitor(iConfig,primaryVertexInputTag,selectedPrimaryVertexInputTag,pvLabel)
{

  if ( doPlotsVsBXlumi_ )
    lumiDetails_ = new GetLumi( iConfig.getParameter<edm::ParameterSet>("BXlumiSetup"), iC );

  pvToken_    = iC.consumes<reco::VertexCollection>(primaryVertexInputTag_);
  selpvToken_ = iC.consumes<reco::VertexCollection>(selectedPrimaryVertexInputTag_); 
}

VertexMonitor::~VertexMonitor()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  //  if (lumiDetails_) delete lumiDetails_;
}


//
// member functions
//

// -- Analyse
// ------------ method called for each event  ------------
// ------------------------------------------------------- //
void
VertexMonitor::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  double bxlumi = 0.;
  if ( doPlotsVsBXlumi_ )
    bxlumi = lumiDetails_->getValue(iEvent);
  std::cout << "bxlumi : " << bxlumi << std::endl;

  size_t totalNumPV = 0;
  size_t totalNumBADndofPV = 0;
  edm::Handle< reco::VertexCollection > pvHandle;
  iEvent.getByToken(pvToken_, pvHandle );
  if ( pvHandle.isValid() )
    {
      totalNumPV = pvHandle->size();
      std::cout << "totalNumPV : " << totalNumPV << std::endl;
      for (reco::VertexCollection::const_iterator pv = pvHandle->begin();
	   pv != pvHandle->end(); ++pv) {
	//--- count pv w/ ndof < 4 
	if (pv->ndof() <  4.) totalNumBADndofPV++;
      }
    } else return;
  NumberOfPVtx        -> Fill( totalNumPV        );
  NumberOfBADndofPVtx -> Fill( totalNumBADndofPV );
  if ( doPlotsVsBXlumi_ ) {
    NumberOfPVtxVsBXlumi        -> Fill( bxlumi, totalNumPV        );
    NumberOfBADndofPVtxVsBXlumi -> Fill( bxlumi, totalNumBADndofPV );
  }
  
  size_t totalNumGoodPV = 0;
  edm::Handle< reco::VertexCollection > selpvHandle;
  iEvent.getByToken(selpvToken_, selpvHandle );
  if ( selpvHandle.isValid() )
    totalNumGoodPV = selpvHandle->size();
  else return;
  std::cout << "totalNumGoodPV: " << totalNumGoodPV << std::endl;
  if ( doPlotsVsGoodPVtx_ ) {
    NumberOfPVtxVsGoodPVtx        -> Fill( totalNumGoodPV, totalNumPV        );
    NumberOfBADndofPVtxVsGoodPVtx -> Fill( totalNumGoodPV, totalNumBADndofPV );
  }

  double fracGoodPV = double(totalNumGoodPV)/double(totalNumPV);
  std::cout << "fracGoodPV: " << fracGoodPV << std::endl;

  NumberOfGoodPVtx    -> Fill( totalNumGoodPV    );
  FractionOfGoodPVtx  -> Fill( fracGoodPV        );
  if ( doPlotsVsBXlumi_ ) {
    NumberOfGoodPVtxVsBXlumi    -> Fill( bxlumi, totalNumGoodPV    );
    FractionOfGoodPVtxVsBXlumi  -> Fill( bxlumi, fracGoodPV        );
  }
  if ( doPlotsVsGoodPVtx_ ) {
    FractionOfGoodPVtxVsGoodPVtx  -> Fill( totalNumGoodPV, fracGoodPV        );
    FractionOfGoodPVtxVsPVtx      -> Fill( totalNumPV,     fracGoodPV        );
  }

  if ( selpvHandle->size() ) {
    double sumpt    = 0;
    size_t ntracks  = 0;
    double chi2ndf  = 0.; 
    double chi2prob = 0.;

    if (!selpvHandle->at(0).isFake()) {
      
      reco::Vertex pv = selpvHandle->at(0);
      
      ntracks  = pv.tracksSize();
      chi2ndf  = pv.normalizedChi2();
      chi2prob = TMath::Prob(pv.chi2(),(int)pv.ndof());
      
      for (reco::Vertex::trackRef_iterator itrk = pv.tracks_begin();
	   itrk != pv.tracks_end(); ++itrk) {
	double pt = (**itrk).pt();
	sumpt += pt*pt;
      }
      GoodPVtxSumPt           -> Fill( sumpt   );
      GoodPVtxNumberOfTracks  -> Fill( ntracks );

      if ( doPlotsVsBXlumi_ ) {
	GoodPVtxSumPtVsBXlumi           -> Fill( bxlumi, sumpt    );
	GoodPVtxNumberOfTracksVsBXlumi  -> Fill( bxlumi, ntracks  );
	GoodPVtxChi2oNDFVsBXlumi        -> Fill( bxlumi, chi2ndf  );
	GoodPVtxChi2ProbVsBXlumi        -> Fill( bxlumi, chi2prob );
      }
      if ( doPlotsVsGoodPVtx_ ) {
	GoodPVtxSumPtVsGoodPVtx          -> Fill( totalNumGoodPV, sumpt    );
	GoodPVtxNumberOfTracksVsGoodPVtx -> Fill( totalNumGoodPV, ntracks  );
	GoodPVtxChi2oNDFVsGoodPVtx       -> Fill( totalNumGoodPV, chi2ndf  );
	GoodPVtxChi2ProbVsGoodPVtx       -> Fill( totalNumGoodPV, chi2prob );
      }
    }
  }
}


// ------------ method called once each job just before starting event loop  ------------
void 
VertexMonitor::initHisto(DQMStore::IBooker & ibooker)
{
    // parameters from the configuration
    std::string MEFolderName   = conf_.getParameter<std::string>("PVFolderName"); 

    // get binning from the configuration
    int    GoodPVtxBin   = conf_.getParameter<int>("GoodPVtxBin");
    double GoodPVtxMin   = conf_.getParameter<double>("GoodPVtxMin");
    double GoodPVtxMax   = conf_.getParameter<double>("GoodPVtxMax");

    // book histo
    // ----------------------//
    ibooker.setCurrentFolder(MEFolderName+"/"+label_);

    histname = "NumberOfPVtx_" + label_;
    NumberOfPVtx = ibooker.book1D(histname,histname, GoodPVtxBin,GoodPVtxMin,GoodPVtxMax);
    NumberOfPVtx->setAxisTitle("Number of PV",1);
    NumberOfPVtx->setAxisTitle("Number of Events",2);
    
    histname = "NumberOfGoodPVtx_" + label_;
    NumberOfGoodPVtx = ibooker.book1D(histname,histname, GoodPVtxBin,GoodPVtxMin,GoodPVtxMax);
    NumberOfGoodPVtx->setAxisTitle("Number of Good PV",1);
    NumberOfGoodPVtx->setAxisTitle("Number of Events",2);
    
    histname = "FractionOfGoodPVtx_" + label_;
    FractionOfGoodPVtx = ibooker.book1D(histname,histname, 100,0.,1.);
    FractionOfGoodPVtx->setAxisTitle("fraction of Good PV",1);
    FractionOfGoodPVtx->setAxisTitle("Number of Events",2);
    
    histname = "NumberOfBADndofPVtx_" + label_;
    NumberOfBADndofPVtx = ibooker.book1D(histname,histname,GoodPVtxBin,GoodPVtxMin,GoodPVtxMax);
    NumberOfBADndofPVtx->setAxisTitle("Number of BADndof #PV",1);
    NumberOfBADndofPVtx->setAxisTitle("Number of Events",2);
    
    histname = "GoodPVtxSumPt_" + label_;
    GoodPVtxSumPt = ibooker.book1D(histname,histname,100,0.,500.);
    GoodPVtxSumPt->setAxisTitle("primary vertex #Sum p_{T}^{2} [GeV^{2}/c^{2}]",1);
    GoodPVtxSumPt->setAxisTitle("Number of events",2);

    histname = "GoodPVtxNumberOfTracks_" + label_;
    GoodPVtxNumberOfTracks = ibooker.book1D(histname,histname,100,0.,100.);
    GoodPVtxNumberOfTracks->setAxisTitle("primary vertex number of tracks",1);
    GoodPVtxNumberOfTracks->setAxisTitle("Number of events",2);

    if ( doPlotsVsBXlumi_ ) {
      // get binning from the configuration
      edm::ParameterSet BXlumiParameters = conf_.getParameter<edm::ParameterSet>("BXlumiSetup");
      int    BXlumiBin   = BXlumiParameters.getParameter<int>("BXlumiBin");
      double BXlumiMin   = BXlumiParameters.getParameter<double>("BXlumiMin");
      double BXlumiMax   = BXlumiParameters.getParameter<double>("BXlumiMax");

      ibooker.setCurrentFolder(MEFolderName+"/"+label_+"/PUmonitoring/");

      histname = "NumberOfPVtxVsBXlumi_" + label_;
      NumberOfPVtxVsBXlumi = ibooker.bookProfile(histname,histname, BXlumiBin,BXlumiMin,BXlumiMax,0.,60.,"");
      NumberOfPVtxVsBXlumi->getTH1()->SetBit(TH1::kCanRebin);
      NumberOfPVtxVsBXlumi->setAxisTitle("lumi BX [10^{30}Hzcm^{-2}]",1);
      NumberOfPVtxVsBXlumi->setAxisTitle("Mean number of PV",2);

      histname = "NumberOfGoodPVtxVsBXlumi_" + label_;
      NumberOfGoodPVtxVsBXlumi = ibooker.bookProfile(histname,histname, BXlumiBin,BXlumiMin,BXlumiMax,0.,60.,"");
      NumberOfGoodPVtxVsBXlumi->getTH1()->SetBit(TH1::kCanRebin);
      NumberOfGoodPVtxVsBXlumi->setAxisTitle("lumi BX [10^{30}Hzcm^{-2}]",1);
      NumberOfGoodPVtxVsBXlumi->setAxisTitle("Mean number of PV",2);

      histname = "FractionOfGoodPVtxVsBXlumi_" + label_;
      FractionOfGoodPVtxVsBXlumi = ibooker.bookProfile(histname,histname, BXlumiBin,BXlumiMin,BXlumiMax,0.,1.5,"");
      FractionOfGoodPVtxVsBXlumi->getTH1()->SetBit(TH1::kCanRebin);
      FractionOfGoodPVtxVsBXlumi->setAxisTitle("lumi BX [10^{30}Hzcm^{-2}]",1);
      FractionOfGoodPVtxVsBXlumi->setAxisTitle("Mean number of PV",2);

      histname = "NumberOfBADndofPVtxVsBXlumi_" + label_;
      NumberOfBADndofPVtxVsBXlumi = ibooker.bookProfile(histname,histname, BXlumiBin,BXlumiMin,BXlumiMax,0.,60.,"");
      NumberOfBADndofPVtxVsBXlumi->getTH1()->SetBit(TH1::kCanRebin);
      NumberOfBADndofPVtxVsBXlumi->setAxisTitle("BADndof #PV",1);
      NumberOfBADndofPVtxVsBXlumi->setAxisTitle("Number of Events",2);
    
      histname = "GoodPVtxSumPtVsBXlumi_" + label_;
      GoodPVtxSumPtVsBXlumi = ibooker.bookProfile(histname,histname, BXlumiBin,BXlumiMin,BXlumiMax,0.,500.,"");
      GoodPVtxSumPtVsBXlumi->getTH1()->SetBit(TH1::kCanRebin);
      GoodPVtxSumPtVsBXlumi->setAxisTitle("lumi BX [10^{30}Hzcm^{-2}]",1);
      GoodPVtxSumPtVsBXlumi->setAxisTitle("Mean pv #Sum p_{T}^{2} [GeV^{2}/c]^{2}",2);
      
      histname = "GoodPVtxNumberOfTracksVsBXlumi_" + label_;
      GoodPVtxNumberOfTracksVsBXlumi = ibooker.bookProfile(histname,histname, BXlumiBin,BXlumiMin,BXlumiMax,0.,100.,"");
      GoodPVtxNumberOfTracksVsBXlumi->getTH1()->SetBit(TH1::kCanRebin);
      GoodPVtxNumberOfTracksVsBXlumi->setAxisTitle("lumi BX [10^{30}Hzcm^{-2}]",1);
      GoodPVtxNumberOfTracksVsBXlumi->setAxisTitle("Mean pv number of tracks",2);

      // get binning from the configuration
      double Chi2NDFMin = conf_.getParameter<double>("Chi2NDFMin");
      double Chi2NDFMax = conf_.getParameter<double>("Chi2NDFMax");

      double Chi2ProbMin  = conf_.getParameter<double>("Chi2ProbMin");
      double Chi2ProbMax  = conf_.getParameter<double>("Chi2ProbMax");

      histname = "Chi2oNDFVsBXlumi_" + label_;
      Chi2oNDFVsBXlumi  = ibooker.bookProfile(histname,histname,BXlumiBin, BXlumiMin,BXlumiMax,Chi2NDFMin,Chi2NDFMax,"");
      Chi2oNDFVsBXlumi -> getTH1()->SetBit(TH1::kCanRebin);
      Chi2oNDFVsBXlumi -> setAxisTitle("lumi BX [10^{30}Hzcm^{-2}]",1);
      Chi2oNDFVsBXlumi -> setAxisTitle("Mean #chi^{2}/ndof",2);

      histname = "Chi2ProbVsBXlumi_" + label_;
      Chi2ProbVsBXlumi  = ibooker.bookProfile(histname,histname,BXlumiBin, BXlumiMin,BXlumiMax,Chi2ProbMin,Chi2ProbMax,"");
      Chi2ProbVsBXlumi -> getTH1()->SetBit(TH1::kCanRebin);
      Chi2ProbVsBXlumi -> setAxisTitle("lumi BX [10^{30}Hzcm^{-2}]",1);
      Chi2ProbVsBXlumi -> setAxisTitle("Mean #chi^{2}/prob",2);

      histname = "GoodPVtxChi2oNDFVsBXlumi_" + label_;
      GoodPVtxChi2oNDFVsBXlumi  = ibooker.bookProfile(histname,histname,BXlumiBin, BXlumiMin,BXlumiMax,Chi2NDFMin,Chi2NDFMax,"");
      GoodPVtxChi2oNDFVsBXlumi -> getTH1()->SetBit(TH1::kCanRebin);
      GoodPVtxChi2oNDFVsBXlumi -> setAxisTitle("lumi BX [10^{30}Hzcm^{-2}]",1);
      GoodPVtxChi2oNDFVsBXlumi -> setAxisTitle("Mean PV #chi^{2}/ndof",2);

      histname = "GoodPVtxChi2ProbVsBXlumi_" + label_;
      GoodPVtxChi2ProbVsBXlumi  = ibooker.bookProfile(histname,histname,BXlumiBin, BXlumiMin,BXlumiMax,Chi2ProbMin,Chi2ProbMax,"");
      GoodPVtxChi2ProbVsBXlumi -> getTH1()->SetBit(TH1::kCanRebin);
      GoodPVtxChi2ProbVsBXlumi -> setAxisTitle("lumi BX [10^{30}Hzcm^{-2}]",1);
      GoodPVtxChi2ProbVsBXlumi -> setAxisTitle("Mean PV #chi^{2}/prob",2);
    }

    if ( doPlotsVsGoodPVtx_ ) {

      ibooker.setCurrentFolder(MEFolderName+"/"+label_+"/PUmonitoring/VsGoodPVtx");

      histname = "NumberOfPVtxVsGoodPVtx_" + label_;
      NumberOfPVtxVsGoodPVtx = ibooker.bookProfile(histname,histname, GoodPVtxBin,GoodPVtxMin,GoodPVtxMax,0.,60.,"");
      NumberOfPVtxVsGoodPVtx->getTH1()->SetBit(TH1::kCanRebin);
      NumberOfPVtxVsGoodPVtx->setAxisTitle("Number of Good PV",1);
      NumberOfPVtxVsGoodPVtx->setAxisTitle("Mean number of PV",2);

      histname = "FractionOfGoodPVtxVsGoodPVtx_" + label_;
      FractionOfGoodPVtxVsGoodPVtx = ibooker.bookProfile(histname,histname, GoodPVtxBin,GoodPVtxMin,GoodPVtxMax,0.,60.,"");
      FractionOfGoodPVtxVsGoodPVtx->getTH1()->SetBit(TH1::kCanRebin);
      FractionOfGoodPVtxVsGoodPVtx->setAxisTitle("Number of Good PV",1);
      FractionOfGoodPVtxVsGoodPVtx->setAxisTitle("Mean fraction of Good PV",2);

      histname = "FractionOfGoodPVtxVsPVtx_" + label_;
      FractionOfGoodPVtxVsPVtx = ibooker.bookProfile(histname,histname, GoodPVtxBin,GoodPVtxMin,GoodPVtxMax,0.,60.,"");
      FractionOfGoodPVtxVsPVtx->getTH1()->SetBit(TH1::kCanRebin);
      FractionOfGoodPVtxVsPVtx->setAxisTitle("Number of Good PV",1);
      FractionOfGoodPVtxVsPVtx->setAxisTitle("Mean number of Good PV",2);

      histname = "NumberOfBADndofPVtxVsGoodPVtx_" + label_;
      NumberOfBADndofPVtxVsGoodPVtx = ibooker.bookProfile(histname,histname, GoodPVtxBin,GoodPVtxMin,GoodPVtxMax,0.,60.,"");
      NumberOfBADndofPVtxVsGoodPVtx->getTH1()->SetBit(TH1::kCanRebin);
      NumberOfBADndofPVtxVsGoodPVtx->setAxisTitle("Number of Good PV",1);
      NumberOfBADndofPVtxVsGoodPVtx->setAxisTitle("Mean Number of BAD PV",2);
    
      histname = "GoodPVtxSumPtVsGoodPVtx_" + label_;
      GoodPVtxSumPtVsGoodPVtx = ibooker.bookProfile(histname,histname, GoodPVtxBin,GoodPVtxMin,GoodPVtxMax,0.,500.,"");
      GoodPVtxSumPtVsGoodPVtx->getTH1()->SetBit(TH1::kCanRebin);
      GoodPVtxSumPtVsGoodPVtx->setAxisTitle("Number of Good PV",1);
      GoodPVtxSumPtVsGoodPVtx->setAxisTitle("Mean pv #Sum p_{T}^{2} [GeV^{2}/c]^{2}",2);
      
      histname = "GoodPVtxNumberOfTracksVsGoodPVtx_" + label_;
      GoodPVtxNumberOfTracksVsGoodPVtx = ibooker.bookProfile(histname,histname, GoodPVtxBin,GoodPVtxMin,GoodPVtxMax,0.,100.,"");
      GoodPVtxNumberOfTracksVsGoodPVtx->getTH1()->SetBit(TH1::kCanRebin);
      GoodPVtxNumberOfTracksVsGoodPVtx->setAxisTitle("Number of Good PV",1);
      GoodPVtxNumberOfTracksVsGoodPVtx->setAxisTitle("Mean pv number of tracks",2);
      
      // get binning from the configuration
      double Chi2NDFMin = conf_.getParameter<double>("Chi2NDFMin");
      double Chi2NDFMax = conf_.getParameter<double>("Chi2NDFMax");

      double Chi2ProbMin  = conf_.getParameter<double>("Chi2ProbMin");
      double Chi2ProbMax  = conf_.getParameter<double>("Chi2ProbMax");

      histname = "GoodPVtxChi2oNDFVsGoodPVtx_" + label_;
      GoodPVtxChi2oNDFVsGoodPVtx  = ibooker.bookProfile(histname,histname,GoodPVtxBin, GoodPVtxMin,GoodPVtxMax,Chi2NDFMin,Chi2NDFMax,"");
      GoodPVtxChi2oNDFVsGoodPVtx -> getTH1()->SetBit(TH1::kCanRebin);
      GoodPVtxChi2oNDFVsGoodPVtx -> setAxisTitle("Number of Good PV",1);
      GoodPVtxChi2oNDFVsGoodPVtx -> setAxisTitle("Mean PV #chi^{2}/ndof",2);

      histname = "GoodPVtxChi2ProbVsGoodPVtx_" + label_;
      GoodPVtxChi2ProbVsGoodPVtx  = ibooker.bookProfile(histname,histname,GoodPVtxBin, GoodPVtxMin,GoodPVtxMax,Chi2ProbMin,Chi2ProbMax,"");
      GoodPVtxChi2ProbVsGoodPVtx -> getTH1()->SetBit(TH1::kCanRebin);
      GoodPVtxChi2ProbVsGoodPVtx -> setAxisTitle("Number of Good PV",1);
      GoodPVtxChi2ProbVsGoodPVtx -> setAxisTitle("Mean PV #chi^{2}/prob",2);

    }
}
 

void 
VertexMonitor::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void 
VertexMonitor::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
VertexMonitor::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}
