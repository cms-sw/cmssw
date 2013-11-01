#include "DQMOffline/RecoB/plugins/PrimaryVertexMonitor.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/isFinite.h"


#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"     

#include "TMath.h"

using namespace reco;
using namespace edm;

PrimaryVertexMonitor::PrimaryVertexMonitor(const edm::ParameterSet& pSet)
  : conf_          ( pSet )
  , dqmStore_      ( edm::Service<DQMStore>().operator->() )
  , TopFolderName_ ( pSet.getParameter<std::string>("TopFolderName") )
  , AlignmentLabel_( pSet.getParameter<std::string>("AlignmentLabel"))
  , nbvtx(NULL)
  , bsX(NULL)
  , bsY(NULL)
  , bsZ(NULL)
  , bsSigmaZ(NULL)
  , bsDxdz(NULL)
  , bsDydz(NULL)
  , bsBeamWidthX(NULL)
  , bsBeamWidthY(NULL)
  , bsType(NULL)
  , sumpt(NULL)
  , ntracks(NULL)
  , weight(NULL)
  , chi2ndf(NULL)
  , chi2prob(NULL)
  , dxy(NULL)
  , dz(NULL)
  , dxyErr(NULL)
  , dzErr(NULL)
  , dxyVsPhi(NULL)
  , dzVsPhi(NULL)
  , dxyVsEta(NULL)
  , dzVsEta(NULL)
{
  //  dqmStore_ = edm::Service<DQMStore>().operator->();


  vertexInputTag_   = pSet.getParameter<InputTag>("vertexLabel");
  beamSpotInputTag_ = pSet.getParameter<InputTag>("beamSpotLabel");
  vertexToken_   = consumes<reco::VertexCollection>(vertexInputTag_);
  beamspotToken_ = consumes<reco::BeamSpot>        (beamSpotInputTag_);

}

// -- BeginRun
//---------------------------------------------------------------------------------//
void 
PrimaryVertexMonitor::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{

  std::string dqmLabel = "";

  //
  // Book all histograms.
  //

  //  get the store
  dqmLabel = TopFolderName_+"/"+vertexInputTag_.label();
  dqmStore_->setCurrentFolder(dqmLabel);

//   xPos = dqmStore_->book1D ("xPos","x Coordinate" ,100, -0.1, 0.1);

  nbvtx      = dqmStore_->book1D("vtxNbr","Reconstructed Vertices in Event",50,-0.5,49.5);

  nbtksinvtx[0] = dqmStore_->book1D("otherVtxTrksNbr","Reconstructed Tracks in Vertex (other Vtx)",40,-0.5,99.5); 
  trksWeight[0] = dqmStore_->book1D("otherVtxTrksWeight","Total weight of Tracks in Vertex (other Vtx)",40,0,100.); 
  vtxchi2[0]    = dqmStore_->book1D("otherVtxChi2","#chi^{2} (other Vtx)",100,0.,200.);
  vtxndf[0]     = dqmStore_->book1D("otherVtxNdf","ndof (other Vtx)",100,0.,200.);
  vtxprob[0]    = dqmStore_->book1D("otherVtxProb","#chi^{2} probability (other Vtx)",100,0.,1.);
  nans[0]       = dqmStore_->book1D("otherVtxNans","Illegal values for x,y,z,xx,xy,xz,yy,yz,zz (other Vtx)",9,0.5,9.5);

  nbtksinvtx[1] = dqmStore_->book1D("tagVtxTrksNbr","Reconstructed Tracks in Vertex (tagged Vtx)",100,-0.5,99.5); 
  trksWeight[1] = dqmStore_->book1D("tagVtxTrksWeight","Total weight of Tracks in Vertex (tagged Vtx)",100,0,100.); 
  vtxchi2[1]    = dqmStore_->book1D("tagVtxChi2","#chi^{2} (tagged Vtx)",100,0.,200.);
  vtxndf[1]     = dqmStore_->book1D("tagVtxNdf","ndof (tagged Vtx)",100,0.,200.);
  vtxprob[1]    = dqmStore_->book1D("tagVtxProb","#chi^{2} probability (tagged Vtx)",100,0.,1.);
  nans[1]       = dqmStore_->book1D("tagVtxNans","Illegal values for x,y,z,xx,xy,xz,yy,yz,zz (tagged Vtx)",9,0.5,9.5);

  xrec[0]	 = dqmStore_->book1D("otherPosX","Position x Coordinate (other Vtx)",100,-0.1,0.1);
  yrec[0]	 = dqmStore_->book1D("otherPosY","Position y Coordinate (other Vtx)",100,-0.1,0.1);
  zrec[0]        = dqmStore_->book1D("otherPosZ","Position z Coordinate (other Vtx)",100,-20.,20.);
  xDiff[0]	 = dqmStore_->book1D("otherDiffX","X distance from BeamSpot (other Vtx)",100,-500,500);
  yDiff[0]	 = dqmStore_->book1D("otherDiffY","Y distance from BeamSpot (other Vtx)",100,-500,500);
  xerr[0]	 = dqmStore_->book1D("otherErrX","Uncertainty x Coordinate (other Vtx)",100,-0.1,0.1);
  yerr[0]	 = dqmStore_->book1D("otherErrY","Uncertainty y Coordinate (other Vtx)",100,-0.1,0.1);
  zerr[0]        = dqmStore_->book1D("otherErrZ","Uncertainty z Coordinate (other Vtx)",100,-20.,20.);
  xerrVsTrks[0]	 = dqmStore_->book2D("otherErrVsWeightX","Uncertainty x Coordinate vs. track weight (other Vtx)",100,0,100.,100,-0.1,0.1);
  yerrVsTrks[0]	 = dqmStore_->book2D("otherErrVsWeightY","Uncertainty y Coordinate vs. track weight (other Vtx)",100,0,100.,100,-0.1,0.1);
  zerrVsTrks[0]	 = dqmStore_->book2D("otherErrVsWeightZ","Uncertainty z Coordinate vs. track weight (other Vtx)",100,0,100.,100,-0.1,0.1);


  xrec[1]     = dqmStore_->book1D("tagPosX","Position x Coordinate (tagged Vtx)",100,-0.1,0.1);
  yrec[1]     = dqmStore_->book1D("tagPosY","Position y Coordinate (tagged Vtx)",100,-0.1,0.1);
  zrec[1]     = dqmStore_->book1D("tagPosZ","Position z Coordinate (tagged Vtx)",100,-20.,20.);
  xDiff[1]    = dqmStore_->book1D("tagDiffX","X distance from BeamSpot (tagged Vtx)",100,-500, 500);
  yDiff[1]    = dqmStore_->book1D("tagDiffY","Y distance from BeamSpot (tagged Vtx)",100,-500, 500);
  xerr[1]     = dqmStore_->book1D("tagErrX","Uncertainty x Coordinate (tagged Vtx)",100,0.,100);
  yerr[1]     = dqmStore_->book1D("tagErrY","Uncertainty y Coordinate (tagged Vtx)",100,0.,100);
  zerr[1]     = dqmStore_->book1D("tagErrZ","Uncertainty z Coordinate (tagged Vtx)",100,0.,100);
  xerrVsTrks[1]	 = dqmStore_->book2D("tagErrVsWeightX","Uncertainty x Coordinate vs. track weight (tagged Vtx)",100,0,100.,100,0.,100);
  yerrVsTrks[1]	 = dqmStore_->book2D("tagErrVsWeightY","Uncertainty y Coordinate vs. track weight (tagged Vtx)",100,0,100.,100,0.,100);
  zerrVsTrks[1]	 = dqmStore_->book2D("tagErrVsWeightZ","Uncertainty z Coordinate vs. track weight (tagged Vtx)",100,0,100.,100,0.,100);

  type[0] = dqmStore_->book1D("otherType","Vertex type (other Vtx)",3,-0.5,2.5);
  type[1] = dqmStore_->book1D("tagType","Vertex type (tagged Vtx)",3,-0.5,2.5);
  for (int i=0;i<2;++i){
    type[i]->getTH1F()->GetXaxis()->SetBinLabel(1,"Valid, real");
    type[i]->getTH1F()->GetXaxis()->SetBinLabel(2,"Valid, fake");
    type[i]->getTH1F()->GetXaxis()->SetBinLabel(3,"Invalid");
  }


  //  get the store
  dqmLabel = TopFolderName_+"/"+beamSpotInputTag_.label();
  dqmStore_->setCurrentFolder(dqmLabel);
  
  bsX 		= dqmStore_->book1D("bsX", "BeamSpot x0", 100,-0.1,0.1);
  bsY 		= dqmStore_->book1D("bsY", "BeamSpot y0", 100,-0.1,0.1);
  bsZ 		= dqmStore_->book1D("bsZ", "BeamSpot z0", 100,-2.,2.);
  bsSigmaZ 	= dqmStore_->book1D("bsSigmaZ", "BeamSpot sigmaZ", 100, 0., 10. );
  bsDxdz 	= dqmStore_->book1D("bsDxdz", "BeamSpot dxdz", 100, -0.0003, 0.0003);
  bsDydz 	= dqmStore_->book1D("bsDydz", "BeamSpot dydz", 100, -0.0003, 0.0003);
  bsBeamWidthX 	= dqmStore_->book1D("bsBeamWidthX", "BeamSpot BeamWidthX", 100, 0., 100.);
  bsBeamWidthY 	= dqmStore_->book1D("bsBeamWidthY", "BeamSpot BeamWidthY", 100, 0., 100.);
  bsType	= dqmStore_->book1D("bsType", "BeamSpot type", 4, -1.5, 2.5);
  bsType->getTH1F()->GetXaxis()->SetBinLabel(1,"Unknown");
  bsType->getTH1F()->GetXaxis()->SetBinLabel(2,"Fake");
  bsType->getTH1F()->GetXaxis()->SetBinLabel(3,"LHC");
  bsType->getTH1F()->GetXaxis()->SetBinLabel(4,"Tracker");

  
  //  get the store
  dqmLabel = TopFolderName_+"/"+AlignmentLabel_;
  dqmStore_->setCurrentFolder(dqmLabel);

  int    TKNoBin    = conf_.getParameter<int>(   "TkSizeBin");
  double TKNoMin    = conf_.getParameter<double>("TkSizeMin");
  double TKNoMax    = conf_.getParameter<double>("TkSizeMax");

  int    DxyBin     = conf_.getParameter<int>(   "DxyBin");
  double DxyMin     = conf_.getParameter<double>("DxyMin");
  double DxyMax     = conf_.getParameter<double>("DxyMax");
  
  int    DzBin      = conf_.getParameter<int>(   "DzBin");
  double DzMin      = conf_.getParameter<double>("DzMin");
  double DzMax      = conf_.getParameter<double>("DzMax");
  
  int    PhiBin     = conf_.getParameter<int>(   "PhiBin");
  double PhiMin     = conf_.getParameter<double>("PhiMin");
  double PhiMax     = conf_.getParameter<double>("PhiMax");

  int    EtaBin     = conf_.getParameter<int>(   "EtaBin");
  double EtaMin     = conf_.getParameter<double>("EtaMin");
  double EtaMax     = conf_.getParameter<double>("EtaMax");
  
      
  ntracks = dqmStore_->book1D("ntracks","number of PV tracks (p_{T} > 1 GeV)", 3*TKNoBin, TKNoMin, (TKNoMax+0.5)*3.-0.5);
  ntracks->setAxisTitle("Number of PV Tracks (p_{T} > 1 GeV) per Event", 1);
  ntracks->setAxisTitle("Number of Event", 2);

  weight = dqmStore_->book1D("weight","weight of PV tracks (p_{T} > 1 GeV)", 100, 0., 1.);
  weight->setAxisTitle("weight of PV Tracks (p_{T} > 1 GeV) per Event", 1);
  weight->setAxisTitle("Number of Event", 2);

  sumpt    = dqmStore_->book1D("sumpt",   "#Sum p_{T} of PV tracks (p_{T} > 1 GeV)",       100,-0.5,199.5); 
  chi2ndf  = dqmStore_->book1D("chi2ndf", "PV tracks (p_{T} > 1 GeV) #chi^{2}/ndof",       100, 0., 200. );
  chi2prob = dqmStore_->book1D("chi2prob","PV tracks (p_{T} > 1 GeV) #chi^{2} probability",100, 0.,   1. );
  dxy      = dqmStore_->book1D("dxy",     "PV tracks (p_{T} > 1 GeV) d_{xy} (cm)",         DxyBin, DxyMin, DxyMax);
  dz       = dqmStore_->book1D("dz",      "PV tracks (p_{T} > 1 GeV) d_{z} (cm)",          DzBin,  DzMin,  DzMax );
  dxyErr   = dqmStore_->book1D("dxyErr",  "PV tracks (p_{T} > 1 GeV) d_{xy} error (cm)",   100, 0.,   1. );
  dzErr    = dqmStore_->book1D("dzErr",   "PV tracks (p_{T} > 1 GeV) d_{z} error(cm)",     100, 0.,   1. );

  dxyVsPhi = dqmStore_->bookProfile("dxyVsPhi", "PV tracks (p_{T} > 1 GeV) d_{xy} (cm) VS track #phi",PhiBin, PhiMin, PhiMax, DxyBin, DxyMin, DxyMax,"");
  dxyVsPhi->setAxisTitle("PV track (p_{T} > 1 GeV) #phi",  1);
  dxyVsPhi->setAxisTitle("PV track (p_{T} > 1 GeV) d_{xy}",2);
  dzVsPhi  = dqmStore_->bookProfile("dzVsPhi",  "PV tracks (p_{T} > 1 GeV) d_{z} (cm) VS track #phi", PhiBin, PhiMin, PhiMax, DzBin,  DzMin,  DzMax, "");  
  dzVsPhi->setAxisTitle("PV track (p_{T} > 1 GeV) #phi", 1);
  dzVsPhi->setAxisTitle("PV track (p_{T} > 1 GeV) d_{z}",2);

  dxyVsEta = dqmStore_->bookProfile("dxyVsEta", "PV tracks (p_{T} > 1 GeV) d_{xy} (cm) VS track #eta",EtaBin, EtaMin, EtaMax, DxyBin, DxyMin, DxyMax,"");
  dxyVsEta->setAxisTitle("PV track (p_{T} > 1 GeV) #eta",  1);
  dxyVsEta->setAxisTitle("PV track (p_{T} > 1 GeV) d_{xy}",2);
  dzVsEta  = dqmStore_->bookProfile("dzVsEta",  "PV tracks (p_{T} > 1 GeV) d_{z} (cm) VS track #eta", EtaBin, EtaMin, EtaMax, DzBin,  DzMin,  DzMax, "");
  dzVsEta->setAxisTitle("PV track (p_{T} > 1 GeV) #eta", 1);
  dzVsEta->setAxisTitle("PV track (p_{T} > 1 GeV) d_{z}",2);

}


PrimaryVertexMonitor::~PrimaryVertexMonitor()
{}

void PrimaryVertexMonitor::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  Handle<reco::VertexCollection> recVtxs;
  iEvent.getByToken(vertexToken_, recVtxs);

  edm::Handle<reco::BeamSpot> beamSpotHandle;
  iEvent.getByToken(beamspotToken_,beamSpotHandle);

  //
  // check for absent products and simply "return" in that case
  //
  if (recVtxs.isValid() == false || beamSpotHandle.isValid()== false){
    edm::LogWarning("PrimaryVertexMonitor")
      <<" Some products not available in the event: VertexCollection "
      <<vertexInputTag_<<" " 
      <<recVtxs.isValid() <<" BeamSpot "
      <<beamSpotInputTag_<<" "
      <<beamSpotHandle.isValid()<<". Skipping plots for this event";
    return;
  }

  BeamSpot beamSpot = *beamSpotHandle;

  nbvtx->Fill(recVtxs->size()*1.);

  vertexPlots(recVtxs->front(), beamSpot, 1);

  // fill PV tracks MEs (as now, for alignment)
  pvTracksPlots(recVtxs->front());

  for(reco::VertexCollection::const_iterator v=recVtxs->begin()+1; 
      v!=recVtxs->end(); ++v){
    vertexPlots(*v, beamSpot, 0);
  }
  // Beamline plots:
  bsX->Fill(beamSpot.x0());
  bsY->Fill(beamSpot.y0());
  bsZ->Fill(beamSpot.z0());
  bsSigmaZ->Fill(beamSpot.sigmaZ());
  bsDxdz->Fill(beamSpot.dxdz());
  bsDydz->Fill(beamSpot.dydz());
  bsBeamWidthX->Fill(beamSpot.BeamWidthX()*10000);
  bsBeamWidthY->Fill(beamSpot.BeamWidthY()*10000);
  // bsType->Fill(beamSpot.type());

}

void
PrimaryVertexMonitor::pvTracksPlots(const Vertex & v)
{

  const math::XYZPoint myVertex(v.position().x(),v.position().y(),v.position().z());

  if ( !v.isValid() ) return;
  if (  v.isFake()  ) return;

  if ( v.tracksSize() == 0 ) {
    ntracks -> Fill ( 0 );
    return;
  }

  size_t nTracks = 0;
  float sumPT = 0.;
  for (reco::Vertex::trackRef_iterator t = v.tracks_begin(); t != v.tracks_end(); t++) {

    bool isHighPurity = (**t).quality(reco::TrackBase::highPurity);
    if ( !isHighPurity ) continue;

    float pt = (**t).pt();    
    if ( pt < 1. ) continue;

    nTracks++;

    float eta      = (**t).eta();
    float phi      = (**t).phi();

    float w        = v.trackWeight(*t);
    float chi2NDF  = (**t).normalizedChi2();
    float chi2Prob = TMath::Prob((**t).chi2(),(int)(**t).ndof());
    float Dxy      = (**t).dxy(myVertex);
    float Dz       = (**t).dz(myVertex);
    float DxyErr   = (**t).dxyError();
    float DzErr    = (**t).dzError();

    sumPT += pt*pt;

    
    // fill MEs
    weight   -> Fill (w);
    chi2ndf  -> Fill (chi2NDF);
    chi2prob -> Fill (chi2Prob);
    dxy      -> Fill (Dxy);
    dz       -> Fill (Dz);
    dxyErr   -> Fill (DxyErr);
    dzErr    -> Fill (DzErr);
    
    dxyVsPhi -> Fill (phi,Dxy);
    dzVsPhi  -> Fill (phi,Dz);
    dxyVsEta -> Fill (eta,Dxy);
    dzVsEta  -> Fill (eta,Dz);
  }
  ntracks -> Fill (float(nTracks));
  sumpt   -> Fill (sumPT);

}

void PrimaryVertexMonitor::vertexPlots(const Vertex & v, const BeamSpot& beamSpot, int i)
{

    if (i < 0 || i > 1) return;
    if (!v.isValid()) type[i]->Fill(2.);
    else if (v.isFake()) type[i]->Fill(1.);
    else type[i]->Fill(0.);

    if (v.isValid() && !v.isFake()) {
      float weight = 0;
      for(reco::Vertex::trackRef_iterator t = v.tracks_begin(); 
	  t!=v.tracks_end(); t++) weight+= v.trackWeight(*t);
      trksWeight[i]->Fill(weight);
      nbtksinvtx[i]->Fill(v.tracksSize());

      vtxchi2[i]->Fill(v.chi2());
      vtxndf[i]->Fill(v.ndof());
      vtxprob[i]->Fill(ChiSquaredProbability(v.chi2() ,v.ndof()));

      xrec[i]->Fill(v.position().x());
      yrec[i]->Fill(v.position().y());
      zrec[i]->Fill(v.position().z());

      float xb = beamSpot.x0() + beamSpot.dxdz() * (v.position().z() - beamSpot.z0());
      float yb = beamSpot.y0() + beamSpot.dydz() * (v.position().z() - beamSpot.z0());
      xDiff[i]->Fill((v.position().x() - xb)*10000);
      yDiff[i]->Fill((v.position().y() - yb)*10000);

      xerr[i]->Fill(v.xError()*10000);
      yerr[i]->Fill(v.yError()*10000);
      zerr[i]->Fill(v.zError()*10000);
      xerrVsTrks[i]->Fill(weight, v.xError()*10000);
      yerrVsTrks[i]->Fill(weight, v.yError()*10000);
      zerrVsTrks[i]->Fill(weight, v.zError()*10000);

      nans[i]->Fill(1.,edm::isNotFinite(v.position().x())*1.);
      nans[i]->Fill(2.,edm::isNotFinite(v.position().y())*1.);
      nans[i]->Fill(3.,edm::isNotFinite(v.position().z())*1.);

      int index = 3;
      for (int k = 0; k != 3; k++) {
	for (int j = k; j != 3; j++) {
	  index++;
	  nans[i]->Fill(index*1., edm::isNotFinite(v.covariance(k, j))*1.);
	  // in addition, diagonal element must be positive
	  if (j == k && v.covariance(k, j) < 0) {
	    nans[i]->Fill(index*1., 1.);
	  }
	}
      }
    }
}


void PrimaryVertexMonitor::endJob()
{
}


//define this as a plug-in
DEFINE_FWK_MODULE(PrimaryVertexMonitor);
