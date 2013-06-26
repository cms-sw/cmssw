#include "DQMOffline/RecoB/plugins/PrimaryVertexMonitor.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/isFinite.h"

#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"     

using namespace reco;
using namespace edm;

PrimaryVertexMonitor::PrimaryVertexMonitor(const edm::ParameterSet& pSet)
{
  moduleLabel = pSet.getParameter<InputTag>("vertexLabel");
  beamSpotLabel = pSet.getParameter<InputTag>("beamSpotLabel");

  //
  // Book all histograms.
  //

  //  get the store
  dqmStore_ = edm::Service<DQMStore>().operator->();
  dqmLabel = "OfflinePV/"+moduleLabel.label();
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
}


PrimaryVertexMonitor::~PrimaryVertexMonitor()
{}

void PrimaryVertexMonitor::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  Handle<reco::VertexCollection> recVtxs;
  iEvent.getByLabel(moduleLabel, recVtxs);

  edm::Handle<reco::BeamSpot> beamSpotHandle;
  iEvent.getByLabel(beamSpotLabel,beamSpotHandle);

  //
  // check for absent products and simply "return" in that case
  //
  if (recVtxs.isValid() == false || beamSpotHandle.isValid()== false){
    edm::LogWarning("PrimaryVertexMonitor")
      <<" Some products not available in the event: VertexCollection "
      <<moduleLabel<<" " 
      <<recVtxs.isValid() <<" BeamSpot "
      <<beamSpotLabel<<" "
      <<beamSpotHandle.isValid()<<". Skipping plots for this event";
    return;
  }

  BeamSpot beamSpot = *beamSpotHandle;

  nbvtx->Fill(recVtxs->size()*1.);

  vertexPlots(recVtxs->front(), beamSpot, 1);

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
