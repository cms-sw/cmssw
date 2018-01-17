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
  , TopFolderName_ ( pSet.getParameter<std::string>("TopFolderName")  )
  , AlignmentLabel_( pSet.getParameter<std::string>("AlignmentLabel") )
  , ndof_          ( pSet.getParameter<int>        ("ndof")           )
  , errorPrinted_  ( false )
{
  vertexInputTag_   = pSet.getParameter<InputTag>("vertexLabel");
  beamSpotInputTag_ = pSet.getParameter<InputTag>("beamSpotLabel");
  vertexToken_   = consumes<reco::VertexCollection>(vertexInputTag_);
  scoreToken_    = consumes<VertexScore>           (vertexInputTag_);
  beamspotToken_ = consumes<reco::BeamSpot>        (beamSpotInputTag_);
}

// -- BeginRun
//---------------------------------------------------------------------------------//
void
PrimaryVertexMonitor::bookHistograms(DQMStore::ConcurrentBooker & iBooker,
  edm::Run const &, edm::EventSetup const &, Histograms & histograms) const {

  std::string dqmLabel = "";

  //
  // Book all histograms.
  //

  //  get the store
  dqmLabel = TopFolderName_+"/"+vertexInputTag_.label();
  iBooker.setCurrentFolder(dqmLabel);

//   histograms.xPos = iBooker.book1D ("xPos","x Coordinate" ,100, -0.1, 0.1);

  histograms.nbvtx         = iBooker.book1D("vtxNbr","Reconstructed Vertices in Event",80,-0.5,79.5);
  histograms.nbgvtx        = iBooker.book1D("goodvtxNbr","Reconstructed Good Vertices in Event",80,-0.5,79.5);

  // to be configured each year...
  auto vposx = conf_.getParameter<double>("Xpos");
  auto vposy = conf_.getParameter<double>("Ypos");

  histograms.nbtksinvtx[0] = iBooker.book1D("otherVtxTrksNbr","Reconstructed Tracks in Vertex (other Vtx)",40,-0.5,99.5);
  histograms.ntracksVsZ[0] = iBooker.bookProfile("otherVtxTrksVsZ","Reconstructed Tracks in Vertex (other Vtx) vs Z",80,-20.,20.,50,0,100,"");
  histograms.ntracksVsZ[0].setAxisTitle("z-bs",1);
  histograms.ntracksVsZ[0].setAxisTitle("#tracks",2);

  histograms.score[0]      = iBooker.book1D("otherVtxScore","sqrt(score) (other Vtx)",100,0.,400.); 
  histograms.trksWeight[0] = iBooker.book1D("otherVtxTrksWeight","Total weight of Tracks in Vertex (other Vtx)",40,0,100.); 
  histograms.vtxchi2[0]    = iBooker.book1D("otherVtxChi2","#chi^{2} (other Vtx)",100,0.,200.);
  histograms.vtxndf[0]     = iBooker.book1D("otherVtxNdf","ndof (other Vtx)",100,0.,200.);
  histograms.vtxprob[0]    = iBooker.book1D("otherVtxProb","#chi^{2} probability (other Vtx)",100,0.,1.);
  histograms.nans[0]       = iBooker.book1D("otherVtxNans","Illegal values for x,y,z,xx,xy,xz,yy,yz,zz (other Vtx)",9,0.5,9.5);

  histograms.nbtksinvtx[1] = iBooker.book1D("tagVtxTrksNbr","Reconstructed Tracks in Vertex (tagged Vtx)",100,-0.5,99.5);
  histograms.ntracksVsZ[1] = iBooker.bookProfile("tagVtxTrksVsZ","Reconstructed Tracks in Vertex (tagged Vtx) vs Z",80,-20.,20.,50,0,100,"");
  histograms.ntracksVsZ[1].setAxisTitle("z-bs",1);
  histograms.ntracksVsZ[1].setAxisTitle("#tracks",2);
 
  histograms.score[1]      = iBooker.book1D("tagVtxScore","sqrt(score) (tagged Vtx)",100,0.,400.);
  histograms.trksWeight[1] = iBooker.book1D("tagVtxTrksWeight","Total weight of Tracks in Vertex (tagged Vtx)",100,0,100.); 
  histograms.vtxchi2[1]    = iBooker.book1D("tagVtxChi2","#chi^{2} (tagged Vtx)",100,0.,200.);
  histograms.vtxndf[1]     = iBooker.book1D("tagVtxNdf","ndof (tagged Vtx)",100,0.,200.);
  histograms.vtxprob[1]    = iBooker.book1D("tagVtxProb","#chi^{2} probability (tagged Vtx)",100,0.,1.);
  histograms.nans[1]       = iBooker.book1D("tagVtxNans","Illegal values for x,y,z,xx,xy,xz,yy,yz,zz (tagged Vtx)",9,0.5,9.5);

  histograms.xrec[0]       = iBooker.book1D("otherPosX","Position x Coordinate (other Vtx)",100,vposx-0.1,vposx+0.1);
  histograms.yrec[0]       = iBooker.book1D("otherPosY","Position y Coordinate (other Vtx)",100,vposy-0.1,vposy+0.1);
  histograms.zrec[0]       = iBooker.book1D("otherPosZ","Position z Coordinate (other Vtx)",100,-20.,20.);
  histograms.xDiff[0]      = iBooker.book1D("otherDiffX","X distance from BeamSpot (other Vtx)",100,-500,500);
  histograms.yDiff[0]      = iBooker.book1D("otherDiffY","Y distance from BeamSpot (other Vtx)",100,-500,500);
  histograms.xerr[0]       = iBooker.book1D("otherErrX","Uncertainty x Coordinate (other Vtx)",100,0.,100);
  histograms.yerr[0]       = iBooker.book1D("otherErrY","Uncertainty y Coordinate (other Vtx)",100,0.,100);
  histograms.zerr[0]       = iBooker.book1D("otherErrZ","Uncertainty z Coordinate (other Vtx)",100,0.,100);
  histograms.xerrVsTrks[0] = iBooker.book2D("otherErrVsWeightX","Uncertainty x Coordinate vs. track weight (other Vtx)",100,0,100.,100,0.,100);
  histograms.yerrVsTrks[0] = iBooker.book2D("otherErrVsWeightY","Uncertainty y Coordinate vs. track weight (other Vtx)",100,0,100.,100,0.,100);
  histograms.zerrVsTrks[0] = iBooker.book2D("otherErrVsWeightZ","Uncertainty z Coordinate vs. track weight (other Vtx)",100,0,100.,100,0.,100);


  histograms.xrec[1]       = iBooker.book1D("tagPosX","Position x Coordinate (tagged Vtx)",100,vposx-0.1,vposx+0.1);
  histograms.yrec[1]       = iBooker.book1D("tagPosY","Position y Coordinate (tagged Vtx)",100,vposy-0.1,vposy+0.1);
  histograms.zrec[1]       = iBooker.book1D("tagPosZ","Position z Coordinate (tagged Vtx)",100,-20.,20.);
  histograms.xDiff[1]      = iBooker.book1D("tagDiffX","X distance from BeamSpot (tagged Vtx)",100,-500, 500);
  histograms.yDiff[1]      = iBooker.book1D("tagDiffY","Y distance from BeamSpot (tagged Vtx)",100,-500, 500);
  histograms.xerr[1]       = iBooker.book1D("tagErrX","Uncertainty x Coordinate (tagged Vtx)",100,0.,100);
  histograms.yerr[1]       = iBooker.book1D("tagErrY","Uncertainty y Coordinate (tagged Vtx)",100,0.,100);
  histograms.zerr[1]       = iBooker.book1D("tagErrZ","Uncertainty z Coordinate (tagged Vtx)",100,0.,100);
  histograms.xerrVsTrks[1] = iBooker.book2D("tagErrVsWeightX","Uncertainty x Coordinate vs. track weight (tagged Vtx)",100,0,100.,100,0.,100);
  histograms.yerrVsTrks[1] = iBooker.book2D("tagErrVsWeightY","Uncertainty y Coordinate vs. track weight (tagged Vtx)",100,0,100.,100,0.,100);
  histograms.zerrVsTrks[1] = iBooker.book2D("tagErrVsWeightZ","Uncertainty z Coordinate vs. track weight (tagged Vtx)",100,0,100.,100,0.,100);

  histograms.type[0] = iBooker.book1D("otherType","Vertex type (other Vtx)",3,-0.5,2.5);
  histograms.type[1] = iBooker.book1D("tagType","Vertex type (tagged Vtx)",3,-0.5,2.5);
  for (int i=0;i<2;++i){
    histograms.type[i].setBinLabel(1,"Valid, real");
    histograms.type[i].setBinLabel(2,"Valid, fake");
    histograms.type[i].setBinLabel(3,"Invalid");
  }


  //  get the store
  dqmLabel = TopFolderName_+"/"+beamSpotInputTag_.label();
  iBooker.setCurrentFolder(dqmLabel);
  
  histograms.bsX           = iBooker.book1D("bsX", "BeamSpot x0", 100,-0.1,0.1);
  histograms.bsY           = iBooker.book1D("bsY", "BeamSpot y0", 100,-0.1,0.1);
  histograms.bsZ           = iBooker.book1D("bsZ", "BeamSpot z0", 100,-2.,2.);
  histograms.bsSigmaZ      = iBooker.book1D("bsSigmaZ", "BeamSpot sigmaZ", 100, 0., 10. );
  histograms.bsDxdz        = iBooker.book1D("bsDxdz", "BeamSpot dxdz", 100, -0.0003, 0.0003);
  histograms.bsDydz        = iBooker.book1D("bsDydz", "BeamSpot dydz", 100, -0.0003, 0.0003);
  histograms.bsBeamWidthX  = iBooker.book1D("bsBeamWidthX", "BeamSpot BeamWidthX", 100, 0., 100.);
  histograms.bsBeamWidthY  = iBooker.book1D("bsBeamWidthY", "BeamSpot BeamWidthY", 100, 0., 100.);
  histograms.bsType        = iBooker.book1D("bsType", "BeamSpot type", 4, -1.5, 2.5);
  histograms.bsType.setBinLabel(1, "Unknown");
  histograms.bsType.setBinLabel(2, "Fake");
  histograms.bsType.setBinLabel(3, "LHC");
  histograms.bsType.setBinLabel(4, "Tracker");

  
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
  
      
  histograms.ntracks = iBooker.book1D("ntracks","number of PV tracks (p_{T} > 1 GeV)", TKNoBin, TKNoMin, TKNoMax);
  histograms.ntracks.setAxisTitle("Number of PV Tracks (p_{T} > 1 GeV) per Event", 1);
  histograms.ntracks.setAxisTitle("Number of Event", 2);

  histograms.weight = iBooker.book1D("weight","weight of PV tracks (p_{T} > 1 GeV)", 100, 0., 1.);
  histograms.weight.setAxisTitle("weight of PV Tracks (p_{T} > 1 GeV) per Event", 1);
  histograms.weight.setAxisTitle("Number of Event", 2);

  histograms.sumpt    = iBooker.book1D("sumpt",   "#Sum p_{T} of PV tracks (p_{T} > 1 GeV)",       100,-0.5,249.5); 
  histograms.chi2ndf  = iBooker.book1D("chi2ndf", "PV tracks (p_{T} > 1 GeV) #chi^{2}/ndof",       100, 0., 20. );
  histograms.chi2prob = iBooker.book1D("chi2prob","PV tracks (p_{T} > 1 GeV) #chi^{2} probability",100, 0.,   1. );
 
  histograms.dxy      = iBooker.book1D("dxy",     "PV tracks (p_{T} > 1 GeV) d_{xy} (#mum)",       DxyBin, DxyMin, DxyMax);
  histograms.dxy2          = iBooker.book1D("dxyzoom", "PV tracks (p_{T} > 1 GeV) d_{xy} (#mum)",       DxyBin, DxyMin/5., DxyMax/5.);
  histograms.dxyErr   = iBooker.book1D("dxyErr",  "PV tracks (p_{T} > 1 GeV) d_{xy} error (#mum)", 100, 0.,   2000. );
  histograms.dz       = iBooker.book1D("dz",      "PV tracks (p_{T} > 1 GeV) d_{z} (#mum)",        DzBin,  DzMin,  DzMax );
  histograms.dzErr    = iBooker.book1D("dzErr",   "PV tracks (p_{T} > 1 GeV) d_{z} error(#mum)",   100, 0.,   10000. );

  histograms.dxyVsPhi_pt1 = iBooker.bookProfile("dxyVsPhi_pt1", "PV tracks (p_{T} > 1 GeV) d_{xy} (#mum) VS track #phi",PhiBin, PhiMin, PhiMax, DxyBin, DxyMin, DxyMax,"");
  histograms.dxyVsPhi_pt1.setAxisTitle("PV track (p_{T} > 1 GeV) #phi",  1);
  histograms.dxyVsPhi_pt1.setAxisTitle("PV track (p_{T} > 1 GeV) d_{xy} (#mum)",2);

  histograms.dzVsPhi_pt1  = iBooker.bookProfile("dzVsPhi_pt1",  "PV tracks (p_{T} > 1 GeV) d_{z} (#mum) VS track #phi", PhiBin, PhiMin, PhiMax, DzBin,  DzMin,  DzMax, "");  
  histograms.dzVsPhi_pt1.setAxisTitle("PV track (p_{T} > 1 GeV) #phi", 1);
  histograms.dzVsPhi_pt1.setAxisTitle("PV track (p_{T} > 1 GeV) d_{z} (#mum)",2);

  histograms.dxyVsEta_pt1 = iBooker.bookProfile("dxyVsEta_pt1", "PV tracks (p_{T} > 1 GeV) d_{xy} (#mum) VS track #eta",EtaBin, EtaMin, EtaMax, DxyBin, DxyMin, DxyMax,"");
  histograms.dxyVsEta_pt1.setAxisTitle("PV track (p_{T} > 1 GeV) #eta",  1);
  histograms.dxyVsEta_pt1.setAxisTitle("PV track (p_{T} > 1 GeV) d_{xy} (#mum)",2);

  histograms.dzVsEta_pt1  = iBooker.bookProfile("dzVsEta_pt1",  "PV tracks (p_{T} > 1 GeV) d_{z} (#mum) VS track #eta", EtaBin, EtaMin, EtaMax, DzBin,  DzMin,  DzMax, "");
  histograms.dzVsEta_pt1.setAxisTitle("PV track (p_{T} > 1 GeV) #eta", 1);
  histograms.dzVsEta_pt1.setAxisTitle("PV track (p_{T} > 1 GeV) d_{z} (#mum)",2);

  histograms.dxyVsPhi_pt10 = iBooker.bookProfile("dxyVsPhi_pt10", "PV tracks (p_{T} > 10 GeV) d_{xy} (#mum) VS track #phi",PhiBin, PhiMin, PhiMax, DxyBin, DxyMin, DxyMax,"");
  histograms.dxyVsPhi_pt10.setAxisTitle("PV track (p_{T} > 10 GeV) #phi",  1);
  histograms.dxyVsPhi_pt10.setAxisTitle("PV track (p_{T} > 10 GeV) d_{xy} (#mum)",2);

  histograms.dzVsPhi_pt10  = iBooker.bookProfile("dzVsPhi_pt10",  "PV tracks (p_{T} > 10 GeV) d_{z} (#mum) VS track #phi", PhiBin, PhiMin, PhiMax, DzBin,  DzMin,  DzMax, "");  
  histograms.dzVsPhi_pt10.setAxisTitle("PV track (p_{T} > 10 GeV) #phi", 1);
  histograms.dzVsPhi_pt10.setAxisTitle("PV track (p_{T} > 10 GeV) d_{z} (#mum)",2);

  histograms.dxyVsEta_pt10 = iBooker.bookProfile("dxyVsEta_pt10", "PV tracks (p_{T} > 10 GeV) d_{xy} (#mum) VS track #eta",EtaBin, EtaMin, EtaMax, DxyBin, DxyMin, DxyMax,"");
  histograms.dxyVsEta_pt10.setAxisTitle("PV track (p_{T} > 10 GeV) #eta",  1);
  histograms.dxyVsEta_pt10.setAxisTitle("PV track (p_{T} > 10 GeV) d_{xy} (#mum)",2);

  histograms.dzVsEta_pt10  = iBooker.bookProfile("dzVsEta_pt10",  "PV tracks (p_{T} > 10 GeV) d_{z} (#mum) VS track #eta", EtaBin, EtaMin, EtaMax, DzBin,  DzMin,  DzMax, "");
  histograms.dzVsEta_pt10.setAxisTitle("PV track (p_{T} > 10 GeV) #eta", 1);
  histograms.dzVsEta_pt10.setAxisTitle("PV track (p_{T} > 10 GeV) d_{z} (#mum)",2);
}


PrimaryVertexMonitor::~PrimaryVertexMonitor()
{}

void PrimaryVertexMonitor::dqmAnalyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, const Histograms & histograms) const
{

  Handle<reco::VertexCollection> recVtxs;
  iEvent.getByToken(vertexToken_, recVtxs);

  Handle<VertexScore> scores;
  iEvent.getByToken(scoreToken_, scores);


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

  // check upfront that refs to track are (likely) to be valid
  {
    bool ok = true;
    for(const auto& v: *recVtxs) {
      if(v.tracksSize() > 0) {
        const auto& ref = v.trackRefAt(0);
        if(ref.isNull() || !ref.isAvailable()) {
          if (not errorPrinted_.test_and_set())
            edm::LogWarning("PrimaryVertexMonitor")
              << "Skipping vertex collection: " << vertexInputTag_ << " since likely the track collection the vertex has refs pointing to is missing (at least the first TrackBaseRef is null or not available)";
          ok = false;
        }
      }
    }
    if(!ok)
      return;
  }

  BeamSpot beamSpot = *beamSpotHandle;

  histograms.nbvtx.fill(recVtxs->size()*1.);
  int ng=0;
  for (auto const & vx : (*recVtxs) )
    if (vx.isValid() && !vx.isFake()  && vx.ndof()>=ndof_) ++ng;
  histograms.nbgvtx.fill(ng*1.);

  if (scores.isValid() && !(*scores).empty()) {
    auto pvScore = (*scores).get(0);
    histograms.score[1].fill(std::sqrt(pvScore));
    for (unsigned int i=1; i<(*scores).size(); ++i) 
      histograms.score[0].fill(std::sqrt((*scores).get(i)));
  }

  // fill PV tracks MEs (as now, for alignment)
  if (!recVtxs->empty()) {

    vertexPlots  (histograms, recVtxs->front(), beamSpot, 1);
    pvTracksPlots(histograms, recVtxs->front());
    
    for(reco::VertexCollection::const_iterator v=recVtxs->begin()+1; 
        v!=recVtxs->end(); ++v)
      vertexPlots(histograms, *v, beamSpot, 0);
  }

  // Beamline plots:
  histograms.bsX.fill(beamSpot.x0());
  histograms.bsY.fill(beamSpot.y0());
  histograms.bsZ.fill(beamSpot.z0());
  histograms.bsSigmaZ.fill(beamSpot.sigmaZ());
  histograms.bsDxdz.fill(beamSpot.dxdz());
  histograms.bsDydz.fill(beamSpot.dydz());
  histograms.bsBeamWidthX.fill(beamSpot.BeamWidthX()*10000);
  histograms.bsBeamWidthY.fill(beamSpot.BeamWidthY()*10000);
  // histograms.bsType.fill(beamSpot.type());

}

void
PrimaryVertexMonitor::pvTracksPlots(const Histograms & histograms, const Vertex & v) const
{
  if ( !v.isValid() ) return;
  if (  v.isFake()  ) return;

  if ( v.tracksSize() == 0 ) {
    histograms.ntracks.fill ( 0 );
    return;
  }

  const math::XYZPoint myVertex(v.position().x(),v.position().y(),v.position().z());

  size_t nTracks = 0;
  float sumPT = 0.;
  const int cmToUm = 10000;

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
    float Dxy      = (**t).dxy(myVertex)*cmToUm;  // is it needed ?
    float Dz       = (**t).dz(myVertex)*cmToUm;   // is it needed ?
    float DxyErr   = (**t).dxyError()*cmToUm;
    float DzErr    = (**t).dzError()*cmToUm;

    sumPT += pt*pt;

    // fill MEs
    histograms.weight.fill (w);
    histograms.chi2ndf.fill (chi2NDF);
    histograms.chi2prob.fill (chi2Prob);
    histograms.dxy.fill (Dxy);
    histograms.dxy2.fill (Dxy); 
    histograms.dz.fill (Dz);
    histograms.dxyErr.fill (DxyErr);
    histograms.dzErr.fill (DzErr);
    
    histograms.dxyVsPhi_pt1.fill (phi,Dxy);
    histograms.dzVsPhi_pt1.fill (phi,Dz);
    histograms.dxyVsEta_pt1.fill (eta,Dxy);
    histograms.dzVsEta_pt1.fill (eta,Dz);

    if ( pt < 10. ) continue;
    histograms.dxyVsPhi_pt10.fill (phi,Dxy);
    histograms.dzVsPhi_pt10.fill (phi,Dz);
    histograms.dxyVsEta_pt10.fill (eta,Dxy);
    histograms.dzVsEta_pt10.fill (eta,Dz);
  }
  histograms.ntracks.fill (float(nTracks));
  histograms.sumpt.fill (sumPT);

}

void PrimaryVertexMonitor::vertexPlots(const Histograms & histograms, const Vertex & v, const BeamSpot& beamSpot, int i) const
{

    if (i < 0 || i > 1) return;
    if (!v.isValid()) histograms.type[i].fill(2.);
    else if (v.isFake()) histograms.type[i].fill(1.);
    else histograms.type[i].fill(0.);

    if (v.isValid() && !v.isFake()) {
      float weight = 0;
      for(reco::Vertex::trackRef_iterator t = v.tracks_begin(); 
          t!=v.tracks_end(); t++) weight+= v.trackWeight(*t);
      histograms.trksWeight[i].fill(weight);
      histograms.nbtksinvtx[i].fill(v.tracksSize());
      histograms.ntracksVsZ[i].fill(v.position().z()- beamSpot.z0(),v.tracksSize());
  
      histograms.vtxchi2[i].fill(v.chi2());
      histograms.vtxndf[i].fill(v.ndof());
      histograms.vtxprob[i].fill(ChiSquaredProbability(v.chi2() ,v.ndof()));

      histograms.xrec[i].fill(v.position().x());
      histograms.yrec[i].fill(v.position().y());
      histograms.zrec[i].fill(v.position().z());

      float xb = beamSpot.x0() + beamSpot.dxdz() * (v.position().z() - beamSpot.z0());
      float yb = beamSpot.y0() + beamSpot.dydz() * (v.position().z() - beamSpot.z0());
      histograms.xDiff[i].fill((v.position().x() - xb)*10000);
      histograms.yDiff[i].fill((v.position().y() - yb)*10000);

      histograms.xerr[i].fill(v.xError()*10000);
      histograms.yerr[i].fill(v.yError()*10000);
      histograms.zerr[i].fill(v.zError()*10000);
      histograms.xerrVsTrks[i].fill(weight, v.xError()*10000);
      histograms.yerrVsTrks[i].fill(weight, v.yError()*10000);
      histograms.zerrVsTrks[i].fill(weight, v.zError()*10000);

      histograms.nans[i].fill(1.,edm::isNotFinite(v.position().x())*1.);
      histograms.nans[i].fill(2.,edm::isNotFinite(v.position().y())*1.);
      histograms.nans[i].fill(3.,edm::isNotFinite(v.position().z())*1.);

      int index = 3;
      for (int k = 0; k != 3; k++) {
        for (int j = k; j != 3; j++) {
          index++;
          histograms.nans[i].fill(index*1., edm::isNotFinite(v.covariance(k, j))*1.);
          // in addition, diagonal element must be positive
          if (j == k && v.covariance(k, j) < 0) {
            histograms.nans[i].fill(index*1., 1.);
          }
        }
      }
    }
}

//define this as a plug-in
DEFINE_FWK_MODULE(PrimaryVertexMonitor);
