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
  , dxy2(NULL)
  , dz(NULL)
  , dxyErr(NULL)
  , dzErr(NULL)
  , dxyVsPhi_pt1(NULL)
  , dzVsPhi_pt1(NULL)
  , dxyVsEta_pt1(NULL)
  , dzVsEta_pt1(NULL)
  , dxyVsPhi_pt10(NULL)
  , dzVsPhi_pt10(NULL)
  , dxyVsEta_pt10(NULL)
  , dzVsEta_pt10(NULL)
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
PrimaryVertexMonitor::bookHistograms(DQMStore::IBooker &iBooker,
  edm::Run const &, edm::EventSetup const &) {

  std::string dqmLabel = "";

  //
  // Book all histograms.
  //

  //  get the store
  dqmLabel = TopFolderName_+"/"+vertexInputTag_.label();
  iBooker.setCurrentFolder(dqmLabel);

//   xPos = iBooker.book1D ("xPos","x Coordinate" ,100, -0.1, 0.1);

  nbvtx      = iBooker.book1D("vtxNbr","Reconstructed Vertices in Event",50,-0.5,49.5);
  nbgvtx      = iBooker.book1D("goodvtxNbr","Reconstructed Good Vertices in Event",50,-0.5,49.5);

  // to be configured each year...
  auto vposx = conf_.getParameter<double>("Xpos");
  auto vposy = conf_.getParameter<double>("Ypos");

  nbtksinvtx[0] = iBooker.book1D("otherVtxTrksNbr","Reconstructed Tracks in Vertex (other Vtx)",40,-0.5,99.5);
  ntracksVsZ[0]  = iBooker.bookProfile("otherVtxTrksVsZ","Reconstructed Tracks in Vertex (other Vtx) vs Z",80,-20.,20.,50,0,100,"");
  ntracksVsZ[0]->setAxisTitle("z-bs",1);
  ntracksVsZ[0]->setAxisTitle("#tracks",2);
 
  trksWeight[0] = iBooker.book1D("otherVtxTrksWeight","Total weight of Tracks in Vertex (other Vtx)",40,0,100.); 
  vtxchi2[0]    = iBooker.book1D("otherVtxChi2","#chi^{2} (other Vtx)",100,0.,200.);
  vtxndf[0]     = iBooker.book1D("otherVtxNdf","ndof (other Vtx)",100,0.,200.);
  vtxprob[0]    = iBooker.book1D("otherVtxProb","#chi^{2} probability (other Vtx)",100,0.,1.);
  nans[0]       = iBooker.book1D("otherVtxNans","Illegal values for x,y,z,xx,xy,xz,yy,yz,zz (other Vtx)",9,0.5,9.5);

  nbtksinvtx[1] = iBooker.book1D("tagVtxTrksNbr","Reconstructed Tracks in Vertex (tagged Vtx)",100,-0.5,99.5);
  ntracksVsZ[1]  = iBooker.bookProfile("tagVtxTrksVsZ","Reconstructed Tracks in Vertex (tagged Vtx) vs Z",80,-20.,20.,50,0,100,"");
  ntracksVsZ[1]->setAxisTitle("z-bs",1);
  ntracksVsZ[1]->setAxisTitle("#tracks",2);
 
  trksWeight[1] = iBooker.book1D("tagVtxTrksWeight","Total weight of Tracks in Vertex (tagged Vtx)",100,0,100.); 
  vtxchi2[1]    = iBooker.book1D("tagVtxChi2","#chi^{2} (tagged Vtx)",100,0.,200.);
  vtxndf[1]     = iBooker.book1D("tagVtxNdf","ndof (tagged Vtx)",100,0.,200.);
  vtxprob[1]    = iBooker.book1D("tagVtxProb","#chi^{2} probability (tagged Vtx)",100,0.,1.);
  nans[1]       = iBooker.book1D("tagVtxNans","Illegal values for x,y,z,xx,xy,xz,yy,yz,zz (tagged Vtx)",9,0.5,9.5);

  xrec[0]	 = iBooker.book1D("otherPosX","Position x Coordinate (other Vtx)",100,vposx-0.1,vposx+0.1);
  yrec[0]	 = iBooker.book1D("otherPosY","Position y Coordinate (other Vtx)",100,vposy-0.1,vposy+0.1);
  zrec[0]        = iBooker.book1D("otherPosZ","Position z Coordinate (other Vtx)",100,-20.,20.);
  xDiff[0]	 = iBooker.book1D("otherDiffX","X distance from BeamSpot (other Vtx)",100,-500,500);
  yDiff[0]	 = iBooker.book1D("otherDiffY","Y distance from BeamSpot (other Vtx)",100,-500,500);
  xerr[0]	 = iBooker.book1D("otherErrX","Uncertainty x Coordinate (other Vtx)",100,0.,100);
  yerr[0]	 = iBooker.book1D("otherErrY","Uncertainty y Coordinate (other Vtx)",100,0.,100);
  zerr[0]        = iBooker.book1D("otherErrZ","Uncertainty z Coordinate (other Vtx)",100,0.,100);
  xerrVsTrks[0]	 = iBooker.book2D("otherErrVsWeightX","Uncertainty x Coordinate vs. track weight (other Vtx)",100,0,100.,100,0.,100);
  yerrVsTrks[0]	 = iBooker.book2D("otherErrVsWeightY","Uncertainty y Coordinate vs. track weight (other Vtx)",100,0,100.,100,0.,100);
  zerrVsTrks[0]	 = iBooker.book2D("otherErrVsWeightZ","Uncertainty z Coordinate vs. track weight (other Vtx)",100,0,100.,100,0.,100);


  xrec[1]     = iBooker.book1D("tagPosX","Position x Coordinate (tagged Vtx)",100,vposx-0.1,vposx+0.1);
  yrec[1]     = iBooker.book1D("tagPosY","Position y Coordinate (tagged Vtx)",100,vposx-0.1,vposy+0.1);
  zrec[1]     = iBooker.book1D("tagPosZ","Position z Coordinate (tagged Vtx)",100,-20.,20.);
  xDiff[1]    = iBooker.book1D("tagDiffX","X distance from BeamSpot (tagged Vtx)",100,-500, 500);
  yDiff[1]    = iBooker.book1D("tagDiffY","Y distance from BeamSpot (tagged Vtx)",100,-500, 500);
  xerr[1]     = iBooker.book1D("tagErrX","Uncertainty x Coordinate (tagged Vtx)",100,0.,100);
  yerr[1]     = iBooker.book1D("tagErrY","Uncertainty y Coordinate (tagged Vtx)",100,0.,100);
  zerr[1]     = iBooker.book1D("tagErrZ","Uncertainty z Coordinate (tagged Vtx)",100,0.,100);
  xerrVsTrks[1]	 = iBooker.book2D("tagErrVsWeightX","Uncertainty x Coordinate vs. track weight (tagged Vtx)",100,0,100.,100,0.,100);
  yerrVsTrks[1]	 = iBooker.book2D("tagErrVsWeightY","Uncertainty y Coordinate vs. track weight (tagged Vtx)",100,0,100.,100,0.,100);
  zerrVsTrks[1]	 = iBooker.book2D("tagErrVsWeightZ","Uncertainty z Coordinate vs. track weight (tagged Vtx)",100,0,100.,100,0.,100);

  type[0] = iBooker.book1D("otherType","Vertex type (other Vtx)",3,-0.5,2.5);
  type[1] = iBooker.book1D("tagType","Vertex type (tagged Vtx)",3,-0.5,2.5);
  for (int i=0;i<2;++i){
    type[i]->getTH1F()->GetXaxis()->SetBinLabel(1,"Valid, real");
    type[i]->getTH1F()->GetXaxis()->SetBinLabel(2,"Valid, fake");
    type[i]->getTH1F()->GetXaxis()->SetBinLabel(3,"Invalid");
  }


  //  get the store
  dqmLabel = TopFolderName_+"/"+beamSpotInputTag_.label();
  iBooker.setCurrentFolder(dqmLabel);
  
  bsX 		= iBooker.book1D("bsX", "BeamSpot x0", 100,-0.1,0.1);
  bsY 		= iBooker.book1D("bsY", "BeamSpot y0", 100,-0.1,0.1);
  bsZ 		= iBooker.book1D("bsZ", "BeamSpot z0", 100,-2.,2.);
  bsSigmaZ 	= iBooker.book1D("bsSigmaZ", "BeamSpot sigmaZ", 100, 0., 10. );
  bsDxdz 	= iBooker.book1D("bsDxdz", "BeamSpot dxdz", 100, -0.0003, 0.0003);
  bsDydz 	= iBooker.book1D("bsDydz", "BeamSpot dydz", 100, -0.0003, 0.0003);
  bsBeamWidthX 	= iBooker.book1D("bsBeamWidthX", "BeamSpot BeamWidthX", 100, 0., 100.);
  bsBeamWidthY 	= iBooker.book1D("bsBeamWidthY", "BeamSpot BeamWidthY", 100, 0., 100.);
  bsType	= iBooker.book1D("bsType", "BeamSpot type", 4, -1.5, 2.5);
  bsType->getTH1F()->GetXaxis()->SetBinLabel(1,"Unknown");
  bsType->getTH1F()->GetXaxis()->SetBinLabel(2,"Fake");
  bsType->getTH1F()->GetXaxis()->SetBinLabel(3,"LHC");
  bsType->getTH1F()->GetXaxis()->SetBinLabel(4,"Tracker");

  
  //  get the store
  dqmLabel = TopFolderName_+"/"+AlignmentLabel_;
  iBooker.setCurrentFolder(dqmLabel);

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
  
      
  ntracks = iBooker.book1D("ntracks","number of PV tracks (p_{T} > 1 GeV)", TKNoBin, TKNoMin, TKNoMax);
  ntracks->setAxisTitle("Number of PV Tracks (p_{T} > 1 GeV) per Event", 1);
  ntracks->setAxisTitle("Number of Event", 2);

  weight = iBooker.book1D("weight","weight of PV tracks (p_{T} > 1 GeV)", 100, 0., 1.);
  weight->setAxisTitle("weight of PV Tracks (p_{T} > 1 GeV) per Event", 1);
  weight->setAxisTitle("Number of Event", 2);

  sumpt    = iBooker.book1D("sumpt",   "#Sum p_{T} of PV tracks (p_{T} > 1 GeV)",       100,-0.5,249.5); 
  chi2ndf  = iBooker.book1D("chi2ndf", "PV tracks (p_{T} > 1 GeV) #chi^{2}/ndof",       100, 0., 20. );
  chi2prob = iBooker.book1D("chi2prob","PV tracks (p_{T} > 1 GeV) #chi^{2} probability",100, 0.,   1. );
  dxy      = iBooker.book1D("dxy",     "PV tracks (p_{T} > 1 GeV) d_{xy} (cm)",         DxyBin, DxyMin, DxyMax);
  dxy2	   = iBooker.book1D("dxyzoom", "PV tracks (p_{T} > 1 GeV) d_{xy} (cm)",         DxyBin, DxyMin/5., DxyMax/5.);
  dz       = iBooker.book1D("dz",      "PV tracks (p_{T} > 1 GeV) d_{z} (cm)",          DzBin,  DzMin,  DzMax );
  dxyErr   = iBooker.book1D("dxyErr",  "PV tracks (p_{T} > 1 GeV) d_{xy} error (cm)",   100, 0.,   0.2 );
  dzErr    = iBooker.book1D("dzErr",   "PV tracks (p_{T} > 1 GeV) d_{z} error(cm)",     100, 0.,   1. );

  dxyVsPhi_pt1 = iBooker.bookProfile("dxyVsPhi_pt1", "PV tracks (p_{T} > 1 GeV) d_{xy} (cm) VS track #phi",PhiBin, PhiMin, PhiMax, DxyBin, DxyMin, DxyMax,"");
  dxyVsPhi_pt1->setAxisTitle("PV track (p_{T} > 1 GeV) #phi",  1);
  dxyVsPhi_pt1->setAxisTitle("PV track (p_{T} > 1 GeV) d_{xy}",2);
  dzVsPhi_pt1  = iBooker.bookProfile("dzVsPhi_pt1",  "PV tracks (p_{T} > 1 GeV) d_{z} (cm) VS track #phi", PhiBin, PhiMin, PhiMax, DzBin,  DzMin,  DzMax, "");  
  dzVsPhi_pt1->setAxisTitle("PV track (p_{T} > 1 GeV) #phi", 1);
  dzVsPhi_pt1->setAxisTitle("PV track (p_{T} > 1 GeV) d_{z}",2);

  dxyVsEta_pt1 = iBooker.bookProfile("dxyVsEta_pt1", "PV tracks (p_{T} > 1 GeV) d_{xy} (cm) VS track #eta",EtaBin, EtaMin, EtaMax, DxyBin, DxyMin, DxyMax,"");
  dxyVsEta_pt1->setAxisTitle("PV track (p_{T} > 1 GeV) #eta",  1);
  dxyVsEta_pt1->setAxisTitle("PV track (p_{T} > 1 GeV) d_{xy}",2);
  dzVsEta_pt1  = iBooker.bookProfile("dzVsEta_pt1",  "PV tracks (p_{T} > 1 GeV) d_{z} (cm) VS track #eta", EtaBin, EtaMin, EtaMax, DzBin,  DzMin,  DzMax, "");
  dzVsEta_pt1->setAxisTitle("PV track (p_{T} > 1 GeV) #eta", 1);
  dzVsEta_pt1->setAxisTitle("PV track (p_{T} > 1 GeV) d_{z}",2);

  dxyVsPhi_pt10 = iBooker.bookProfile("dxyVsPhi_pt10", "PV tracks (p_{T} > 1 GeV) d_{xy} (cm) VS track #phi",PhiBin, PhiMin, PhiMax, DxyBin, DxyMin, DxyMax,"");
  dxyVsPhi_pt10->setAxisTitle("PV track (p_{T} > 10 GeV) #phi",  1);
  dxyVsPhi_pt10->setAxisTitle("PV track (p_{T} > 10 GeV) d_{xy}",2);
  dzVsPhi_pt10  = iBooker.bookProfile("dzVsPhi_pt10",  "PV tracks (p_{T} > 10 GeV) d_{z} (cm) VS track #phi", PhiBin, PhiMin, PhiMax, DzBin,  DzMin,  DzMax, "");  
  dzVsPhi_pt10->setAxisTitle("PV track (p_{T} > 10 GeV) #phi", 1);
  dzVsPhi_pt10->setAxisTitle("PV track (p_{T} > 10 GeV) d_{z}",2);

  dxyVsEta_pt10 = iBooker.bookProfile("dxyVsEta_pt10", "PV tracks (p_{T} > 10 GeV) d_{xy} (cm) VS track #eta",EtaBin, EtaMin, EtaMax, DxyBin, DxyMin, DxyMax,"");
  dxyVsEta_pt10->setAxisTitle("PV track (p_{T} > 10 GeV) #eta",  1);
  dxyVsEta_pt10->setAxisTitle("PV track (p_{T} > 10 GeV) d_{xy}",2);
  dzVsEta_pt10  = iBooker.bookProfile("dzVsEta_pt10",  "PV tracks (p_{T} > 10 GeV) d_{z} (cm) VS track #eta", EtaBin, EtaMin, EtaMax, DzBin,  DzMin,  DzMax, "");
  dzVsEta_pt10->setAxisTitle("PV track (p_{T} > 10 GeV) #eta", 1);
  dzVsEta_pt10->setAxisTitle("PV track (p_{T} > 10 GeV) d_{z}",2);

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
  int ng=0;
  for (auto const & vx : (*recVtxs) )
    if (vx.isValid() && !vx.isFake()  && vx.ndof()>=4.) ++ng;
  nbgvtx->Fill(ng*1.);

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
    dxy2     -> Fill (Dxy); 
    dz       -> Fill (Dz);
    dxyErr   -> Fill (DxyErr);
    dzErr    -> Fill (DzErr);
    
    dxyVsPhi_pt1 -> Fill (phi,Dxy);
    dzVsPhi_pt1  -> Fill (phi,Dz);
    dxyVsEta_pt1 -> Fill (eta,Dxy);
    dzVsEta_pt1  -> Fill (eta,Dz);

    if ( pt < 10. ) continue;
    dxyVsPhi_pt10 -> Fill (phi,Dxy);
    dzVsPhi_pt10  -> Fill (phi,Dz);
    dxyVsEta_pt10 -> Fill (eta,Dxy);
    dzVsEta_pt10  -> Fill (eta,Dz);
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
      ntracksVsZ[i]->Fill(v.position().z()- beamSpot.z0(),v.tracksSize());
  
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

//define this as a plug-in
DEFINE_FWK_MODULE(PrimaryVertexMonitor);
