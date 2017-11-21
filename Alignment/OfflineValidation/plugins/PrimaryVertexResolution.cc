// -*- C++ -*-
//
// Package:    BeamSpotCalibration/PrimaryVertexResolution
// Class:      PrimaryVertexResolution
// 
/**\class PrimaryVertexResolution PrimaryVertexResolution.cc Alignment/OfflineValidation/plugins/PrimaryVertexResolution.cc

*/
//
// Original Author:  Marco Musich
//         Created:  Mon, 13 Jun 2016 15:07:11 GMT
//
//


// system include files
#include <memory>
#include <algorithm>    // std::sort
#include <vector>       // std::vector
#include "TRandom.h"
#include "TTree.h"
#include "TProfile.h"
#include "TF1.h"
#include "TMath.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "RecoVertex/AdaptiveVertexFit/interface/AdaptiveVertexFitter.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

#include "Alignment/OfflineValidation/interface/PVValidationHelpers.h"

//
// useful code
//

namespace statmode{

  using fitParams = std::pair<std::pair<double,double>, std::pair<double,double> >;
}

//
// class declaration
//

class PrimaryVertexResolution : public edm::one::EDAnalyzer<edm::one::SharedResources>  {
   public:
      explicit PrimaryVertexResolution(const edm::ParameterSet&);
      ~PrimaryVertexResolution() override;

      static  void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
      static  bool mysorter (reco::Track i, reco::Track j) { return (i.pt () > j.pt()); }

   private:
      void beginJob() override;
      void analyze(const edm::Event&, const edm::EventSetup&) override;
      void endJob() override;
  
      template<std::size_t SIZE> bool checkBinOrdering(std::array<float, SIZE>& bins);
      std::vector<TH1F*> bookResidualsHistogram(TFileDirectory dir,unsigned int theNOfBins,TString resType,TString varType);
  
      void fillTrendPlotByIndex(TH1F* trendPlot,std::vector<TH1F*>& h,  PVValHelper::estimator fitPar_);
      statmode::fitParams fitResiduals(TH1 *hist,bool singleTime=false);
      statmode::fitParams fitResiduals_v0(TH1 *hist);

      edm::InputTag      pvsTag_;
      edm::EDGetTokenT<reco::VertexCollection> pvsToken_;

      edm::InputTag      tracksTag_;
      edm::EDGetTokenT<reco::TrackCollection>  tracksToken_;
      
      double minVtxNdf_;
      double minVtxWgt_;

      const double cmToUm = 10000.;

      edm::Service<TFileService> outfile_;

      TH1F * h_diffX ;
      TH1F * h_diffY ;
      TH1F * h_diffZ ;
  
      TH1F * h_pullX ;
      TH1F * h_pullY ;
      TH1F * h_pullZ ;

      TH1F * h_ntrks ;
      TH1F * h_nhalftrks;
      TH1F * h_sumPt;
  
      TH1F * h_sumPt1;
      TH1F * h_sumPt2;

      TH1F * h_wTrks1 ;
      TH1F * h_wTrks2 ;

      TH1F * h_minWTrks1;
      TH1F * h_minWTrks2;
  
      TH1F * h_PVCL_subVtx1;
      TH1F * h_PVCL_subVtx2;

      TH1F * h_runNumber; 

      TH1I * h_nOfflineVertices;
      TH1I * h_nVertices;
      TH1I * h_nNonFakeVertices;
      TH1I * h_nFinalVertices;
  
      // subvertices positions and errors
  
      TH1F* h_X1;
      TH1F* h_Y1;
      TH1F* h_Z1;
            
      TH1F* h_errX1; 
      TH1F* h_errY1; 
      TH1F* h_errZ1;
            
      TH1F* h_X2;
      TH1F* h_Y2; 
      TH1F* h_Z2;
            
      TH1F* h_errX2; 
      TH1F* h_errY2;  
      TH1F* h_errZ2;  

      // resolutions 

      std::vector<TH1F*>  h_resolX_sumPt_;
      std::vector<TH1F*>  h_resolY_sumPt_;
      std::vector<TH1F*>  h_resolZ_sumPt_;

      std::vector<TH1F*>  h_resolX_Ntracks_;
      std::vector<TH1F*>  h_resolY_Ntracks_;
      std::vector<TH1F*>  h_resolZ_Ntracks_;
  
      std::vector<TH1F*>  h_resolX_Nvtx_;
      std::vector<TH1F*>  h_resolY_Nvtx_;
      std::vector<TH1F*>  h_resolZ_Nvtx_;

      TH1F * p_resolX_vsSumPt;
      TH1F * p_resolY_vsSumPt;
      TH1F * p_resolZ_vsSumPt;

      TH1F * p_resolX_vsNtracks;
      TH1F * p_resolY_vsNtracks;
      TH1F * p_resolZ_vsNtracks;

      TH1F * p_resolX_vsNvtx;
      TH1F * p_resolY_vsNvtx;
      TH1F * p_resolZ_vsNvtx;

      // pulls
      std::vector<TH1F*>  h_pullX_sumPt_;
      std::vector<TH1F*>  h_pullY_sumPt_;
      std::vector<TH1F*>  h_pullZ_sumPt_;

      std::vector<TH1F*>  h_pullX_Ntracks_;
      std::vector<TH1F*>  h_pullY_Ntracks_;
      std::vector<TH1F*>  h_pullZ_Ntracks_;

      std::vector<TH1F*>  h_pullX_Nvtx_;
      std::vector<TH1F*>  h_pullY_Nvtx_;
      std::vector<TH1F*>  h_pullZ_Nvtx_;

      TH1F * p_pullX_vsSumPt;
      TH1F * p_pullY_vsSumPt;
      TH1F * p_pullZ_vsSumPt;

      TH1F * p_pullX_vsNtracks;
      TH1F * p_pullY_vsNtracks;
      TH1F * p_pullZ_vsNtracks;

      TH1F * p_pullX_vsNvtx;
      TH1F * p_pullY_vsNvtx;
      TH1F * p_pullZ_vsNvtx;

      TRandom rand;

      // ----------member data ---------------------------
      static const int nPtBins_ = 30;      
      std::array<float, nPtBins_+1>  mypT_bins_ = PVValHelper::makeLogBins<float,nPtBins_>(1.,1e3); 

      static const int nTrackBins_ = 60;
      std::array<float,nTrackBins_+1>  myNTrack_bins_;
  
      static const int nVtxBins_  = 40;
      std::array<float,nVtxBins_+1>  myNVtx_bins_;
 
};

PrimaryVertexResolution::PrimaryVertexResolution(const edm::ParameterSet& iConfig):
  pvsTag_           (iConfig.getParameter<edm::InputTag>("vtxCollection")), 
  pvsToken_         (consumes<reco::VertexCollection>(pvsTag_)), 
  tracksTag_        (iConfig.getParameter<edm::InputTag>("trackCollection")), 
  tracksToken_      (consumes<reco::TrackCollection>(tracksTag_)),
  minVtxNdf_        (iConfig.getUntrackedParameter<double>("minVertexNdf")), 
  minVtxWgt_        (iConfig.getUntrackedParameter<double>("minVertexMeanWeight"))
{
  
  std::vector<float> vect = PVValHelper::generateBins(nTrackBins_+1,1.,120.);   
  std::copy(vect.begin(), vect.begin() + nTrackBins_+1, myNTrack_bins_.begin());  

  vect.clear();
  vect = PVValHelper::generateBins(nVtxBins_+1,1.,40.);
  std::copy(vect.begin(), vect.begin() + nVtxBins_+1, myNVtx_bins_.begin());  
}


PrimaryVertexResolution::~PrimaryVertexResolution()
{
}


// ------------ method called for each event  ------------
void
PrimaryVertexResolution::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  // Fill general info
  h_runNumber->Fill(iEvent.id().run());;
  
  edm::ESHandle<TransientTrackBuilder>            theB                ;
  edm::ESHandle<GlobalTrackingGeometry>           theTrackingGeometry ;
  iSetup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry) ;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",theB);

  edm::Handle<reco::VertexCollection> vertices; 
  iEvent.getByToken(pvsToken_, vertices);
  const reco::VertexCollection pvtx  = *(vertices.product())  ;    

  edm::Handle<reco::TrackCollection> tracks; 
  iEvent.getByToken(tracksToken_, tracks);
  
  int nOfflineVtx = pvtx.size();

  h_nOfflineVertices->Fill(nOfflineVtx);

  int counter       = 0;
  int noFakecounter = 0;
  int goodcounter   = 0;
  for (reco::VertexCollection::const_iterator pvIt = pvtx.begin(); pvIt!=pvtx.end(); pvIt++)        
  {
    reco::Vertex iPV = *pvIt;
    counter++;
    if (iPV.isFake()) continue;
    noFakecounter++;
    reco::Vertex::trackRef_iterator trki;

    // vertex selection as in bs code
    if ( iPV.ndof() < minVtxNdf_ || (iPV.ndof()+3.)/iPV.tracksSize()< 2*minVtxWgt_ )  continue;

    goodcounter++;
    reco::TrackCollection allTracks;
    reco::TrackCollection groupOne, groupTwo;
    for (trki  = iPV.tracks_begin(); trki != iPV.tracks_end(); ++trki) {
      if (trki->isNonnull()){
        reco::TrackRef trk_now(tracks, (*trki).key());
        allTracks.push_back(*trk_now);
      }
    }
    
    if(goodcounter>1) continue;

    // order with decreasing pt 
    std::sort (allTracks.begin(), allTracks.end(), mysorter);
    
    int ntrks = allTracks.size();
    h_ntrks -> Fill( ntrks );
    
    // discard lowest pt track
    uint even_ntrks;
    ntrks % 2 == 0 ? even_ntrks = ntrks : even_ntrks = ntrks - 1;
    
    // split into two sets equally populated 
    for (uint tracksIt =0 ;  tracksIt < even_ntrks; tracksIt = tracksIt+2)
    {
      reco::Track  firstTrk  = allTracks.at(tracksIt);      
      reco::Track  secondTrk = allTracks.at(tracksIt + 1);      
      double therand = rand.Uniform (0, 1);
      if (therand > 0.5) {
        groupOne.push_back(firstTrk);  
        groupTwo.push_back(secondTrk);        
      }    
      else {
        groupOne.push_back(secondTrk);  
        groupTwo.push_back(firstTrk);        
      }                                
    }
     
    if  (! (groupOne.size() >= 2 && groupTwo.size() >= 2) )   continue;

    float sumPt= 0, sumPt1 = 0, sumPt2=0;

    // refit the two sets of tracks
    std::vector<reco::TransientTrack> groupOne_ttks;
    groupOne_ttks.clear();
    for (reco::TrackCollection::const_iterator itrk = groupOne.begin(); itrk != groupOne.end(); itrk++)
    {
      reco::TransientTrack tmpTransientTrack = (*theB).build(*itrk); 
      groupOne_ttks.push_back(tmpTransientTrack);
      sumPt1 += itrk->pt(); 
      sumPt  += itrk->pt();
    }

    AdaptiveVertexFitter pvFitter;
    TransientVertex pvOne = pvFitter.vertex(groupOne_ttks);
    if (!pvOne.isValid()) continue;

    reco::Vertex onePV = pvOne;    

    std::vector<reco::TransientTrack> groupTwo_ttks;
    groupTwo_ttks.clear();
    for (reco::TrackCollection::const_iterator itrk = groupTwo.begin(); itrk != groupTwo.end(); itrk++)
    {
      reco::TransientTrack tmpTransientTrack = (*theB).build(*itrk); 
      groupTwo_ttks.push_back(tmpTransientTrack);
      sumPt2 += itrk->pt();
      sumPt  += itrk->pt();
    }

    TransientVertex pvTwo = pvFitter.vertex(groupTwo_ttks);
    if (!pvTwo.isValid()) continue;

    reco::Vertex twoPV = pvTwo;    

    float theminW1 = 1.;
    float theminW2 = 1.;
    for (std::vector<reco::TransientTrack>::const_iterator otrk  = pvOne.originalTracks().begin(); otrk != pvOne.originalTracks().end(); ++otrk) 
    {
      h_wTrks1 -> Fill( pvOne.trackWeight(*otrk));
      if (pvOne.trackWeight(*otrk) < theminW1) theminW1 = pvOne.trackWeight(*otrk); 
    } 
    for (std::vector<reco::TransientTrack>::const_iterator otrk  = pvTwo.originalTracks().begin(); otrk != pvTwo.originalTracks().end(); ++otrk) 
    {
      h_wTrks2 -> Fill( pvTwo.trackWeight(*otrk));
      if (pvTwo.trackWeight(*otrk) < theminW2) theminW2 = pvTwo.trackWeight(*otrk); 
    } 

    h_sumPt->Fill(sumPt);

    int half_trks = twoPV.nTracks();
    h_nhalftrks->Fill(half_trks);

    double deltaX = twoPV.x() - onePV.x();
    double deltaY = twoPV.y() - onePV.y();
    double deltaZ = twoPV.z() - onePV.z();
    
    h_diffX -> Fill(deltaX*cmToUm);
    h_diffY -> Fill(deltaY*cmToUm);
    h_diffZ -> Fill(deltaZ*cmToUm);
    
    double errX = sqrt( pow(twoPV.xError(),2) + pow(onePV.xError(),2) );
    double errY = sqrt( pow(twoPV.yError(),2) + pow(onePV.yError(),2) );
    double errZ = sqrt( pow(twoPV.zError(),2) + pow(onePV.zError(),2) );

    h_pullX -> Fill( (twoPV.x() - onePV.x()) / errX );
    h_pullY -> Fill( (twoPV.y() - onePV.y()) / errY );
    h_pullZ -> Fill( (twoPV.z() - onePV.z()) / errZ );
     
    // filling the pT-binned distributions
    
    for(int ipTBin=0; ipTBin<nPtBins_; ipTBin++){
      
      float pTF = mypT_bins_[ipTBin];
      float pTL = mypT_bins_[ipTBin+1];

      if(sumPt >= pTF && sumPt < pTL){
	
	PVValHelper::fillByIndex(h_resolX_sumPt_,ipTBin,deltaX*cmToUm,"1");
	PVValHelper::fillByIndex(h_resolY_sumPt_,ipTBin,deltaY*cmToUm,"2");
	PVValHelper::fillByIndex(h_resolZ_sumPt_,ipTBin,deltaZ*cmToUm,"3");

	PVValHelper::fillByIndex(h_pullX_sumPt_,ipTBin,deltaX/errX,"4");
	PVValHelper::fillByIndex(h_pullY_sumPt_,ipTBin,deltaY/errY,"5");
	PVValHelper::fillByIndex(h_pullZ_sumPt_,ipTBin,deltaZ/errZ,"6");
	
      } 
    }

    // filling the track multeplicity binned distributions

    for(int inTrackBin=0; inTrackBin<nTrackBins_; inTrackBin++){
      
      float nTrackF = myNTrack_bins_[inTrackBin];
      float nTrackL = myNTrack_bins_[inTrackBin+1];
      
      if(ntrks >= nTrackF && ntrks < nTrackL){

	PVValHelper::fillByIndex(h_resolX_Ntracks_,inTrackBin,deltaX*cmToUm,"7");
	PVValHelper::fillByIndex(h_resolY_Ntracks_,inTrackBin,deltaY*cmToUm,"8");
	PVValHelper::fillByIndex(h_resolZ_Ntracks_,inTrackBin,deltaZ*cmToUm,"9");

	PVValHelper::fillByIndex(h_pullX_Ntracks_,inTrackBin,deltaX/errX,"10");
	PVValHelper::fillByIndex(h_pullY_Ntracks_,inTrackBin,deltaY/errY,"11");
	PVValHelper::fillByIndex(h_pullZ_Ntracks_,inTrackBin,deltaZ/errZ,"12");
	
      }
    }    

    // filling the vertex multeplicity binned distributions

    for(int inVtxBin=0; inVtxBin<nVtxBins_; inVtxBin++){

      /*
	float nVtxF = myNVtx_bins_[inVtxBin];
	float nVtxL = myNVtx_bins_[inVtxBin+1];
      	if(nOfflineVtx >= nVtxF && nOfflineVtx < nVtxL){
      */

      if(nOfflineVtx==inVtxBin){

	PVValHelper::fillByIndex(h_resolX_Nvtx_,inVtxBin,deltaX*cmToUm,"7");
	PVValHelper::fillByIndex(h_resolY_Nvtx_,inVtxBin,deltaY*cmToUm,"8");
	PVValHelper::fillByIndex(h_resolZ_Nvtx_,inVtxBin,deltaZ*cmToUm,"9");

	PVValHelper::fillByIndex(h_pullX_Nvtx_,inVtxBin,deltaX/errX,"10");
	PVValHelper::fillByIndex(h_pullY_Nvtx_,inVtxBin,deltaY/errY,"11");
	PVValHelper::fillByIndex(h_pullZ_Nvtx_,inVtxBin,deltaZ/errZ,"12");
	
      }
    }    

    h_sumPt1->Fill(sumPt1);
    h_sumPt2->Fill(sumPt2);

    h_minWTrks1->Fill(theminW1);
    h_minWTrks2->Fill(theminW2);

    h_PVCL_subVtx1->Fill(TMath::Prob(pvOne.totalChiSquared(),(int)(pvOne.degreesOfFreedom())));
    h_PVCL_subVtx2->Fill(TMath::Prob(pvTwo.totalChiSquared(),(int)(pvTwo.degreesOfFreedom())));

    h_X1->Fill(onePV.x()*cmToUm); 
    h_Y1->Fill(onePV.y()*cmToUm); 
    h_Z1->Fill(onePV.z()); 
              	                
    h_errX1->Fill(onePV.xError()*cmToUm); 
    h_errY1->Fill(onePV.yError()*cmToUm); 
    h_errZ1->Fill(onePV.zError()*cmToUm); 
              	                
    h_X2->Fill(twoPV.x()*cmToUm); 
    h_Y2->Fill(twoPV.y()*cmToUm); 
    h_Z2->Fill(twoPV.z()); 
      		                
    h_errX2->Fill(twoPV.xError()*cmToUm);  
    h_errY2->Fill(twoPV.yError()*cmToUm);  
    h_errZ2->Fill(twoPV.zError()*cmToUm);  
    
  } // loop on the vertices

  // fill the histogram of vertices per event
  h_nVertices->Fill(counter);        
  h_nNonFakeVertices->Fill(noFakecounter); 
  h_nFinalVertices->Fill(goodcounter);

}


// ------------ method called once each job just before starting event loop  ------------
void 
PrimaryVertexResolution::beginJob()
{

  TH1F::SetDefaultSumw2(kTRUE);

  // resolutions

  if(!checkBinOrdering(mypT_bins_)) {
    edm::LogError("PrimaryVertexValidation")<<" Warning - the vector of pT bins is not ordered " << std::endl;    
  }

  if(!checkBinOrdering(myNTrack_bins_)) {
    edm::LogError("PrimaryVertexValidation")<<" Warning -the vector of n. tracks bins is not ordered " << std::endl;    
  }


  TFileDirectory xResolSumPt = outfile_->mkdir("xResolSumPt"); 
  h_resolX_sumPt_   = bookResidualsHistogram(xResolSumPt,nPtBins_,"resolX","sumPt");	   

  TFileDirectory yResolSumPt = outfile_->mkdir("yResolSumPt"); 
  h_resolY_sumPt_   = bookResidualsHistogram(yResolSumPt,nPtBins_,"resolY","sumPt");	   

  TFileDirectory zResolSumPt = outfile_->mkdir("zResolSumPt"); 
  h_resolZ_sumPt_   = bookResidualsHistogram(zResolSumPt,nPtBins_,"resolZ","sumPt");	   

  TFileDirectory xResolNtracks_ = outfile_->mkdir("xResolNtracks"); 
  h_resolX_Ntracks_   = bookResidualsHistogram(xResolNtracks_,nTrackBins_,"resolX","Ntracks");	   

  TFileDirectory yResolNtracks_ = outfile_->mkdir("yResolNtracks"); 
  h_resolY_Ntracks_   = bookResidualsHistogram(yResolNtracks_,nTrackBins_,"resolY","Ntracks");	   

  TFileDirectory zResolNtracks_ = outfile_->mkdir("zResolNtracks"); 
  h_resolZ_Ntracks_   = bookResidualsHistogram(zResolNtracks_,nTrackBins_,"resolZ","Ntracks");	  

  TFileDirectory xResolNvtx_ = outfile_->mkdir("xResolNvtx"); 
  h_resolX_Nvtx_   = bookResidualsHistogram(xResolNvtx_,nVtxBins_,"resolX","Nvtx");	   

  TFileDirectory yResolNvtx_ = outfile_->mkdir("yResolNvtx"); 
  h_resolY_Nvtx_   = bookResidualsHistogram(yResolNvtx_,nVtxBins_,"resolY","Nvtx");	   

  TFileDirectory zResolNvtx_ = outfile_->mkdir("zResolNvtx"); 
  h_resolZ_Nvtx_   = bookResidualsHistogram(zResolNvtx_,nVtxBins_,"resolZ","Nvtx");	  

  // pulls

  TFileDirectory xPullSumPt = outfile_->mkdir("xPullSumPt"); 
  h_pullX_sumPt_   = bookResidualsHistogram(xPullSumPt,nPtBins_,"pullX","sumPt");	   

  TFileDirectory yPullSumPt = outfile_->mkdir("yPullSumPt"); 
  h_pullY_sumPt_   = bookResidualsHistogram(yPullSumPt,nPtBins_,"pullY","sumPt");	   

  TFileDirectory zPullSumPt = outfile_->mkdir("zPullSumPt"); 
  h_pullZ_sumPt_   = bookResidualsHistogram(zPullSumPt,nPtBins_,"pullZ","sumPt");	   

  TFileDirectory xPullNtracks_ = outfile_->mkdir("xPullNtracks"); 
  h_pullX_Ntracks_   = bookResidualsHistogram(xPullNtracks_,nTrackBins_,"pullX","Ntracks");	   

  TFileDirectory yPullNtracks_ = outfile_->mkdir("yPullNtracks"); 
  h_pullY_Ntracks_   = bookResidualsHistogram(yPullNtracks_,nTrackBins_,"pullY","Ntracks");	   

  TFileDirectory zPullNtracks_ = outfile_->mkdir("zPullNtracks"); 
  h_pullZ_Ntracks_   = bookResidualsHistogram(zPullNtracks_,nTrackBins_,"pullZ","Ntracks");	  

  TFileDirectory xPullNvtx_ = outfile_->mkdir("xPullNvtx"); 
  h_pullX_Nvtx_   = bookResidualsHistogram(xPullNvtx_,nVtxBins_,"pullX","Nvtx");	   

  TFileDirectory yPullNvtx_ = outfile_->mkdir("yPullNvtx"); 
  h_pullY_Nvtx_   = bookResidualsHistogram(yPullNvtx_,nVtxBins_,"pullY","Nvtx");	   

  TFileDirectory zPullNvtx_ = outfile_->mkdir("zPullNvtx"); 
  h_pullZ_Nvtx_   = bookResidualsHistogram(zPullNvtx_,nVtxBins_,"pullZ","Nvtx");	  


  // control plots
  h_runNumber        = outfile_->make<TH1F>("h_runNumber","run number;run number;n_{events}",100000,250000.,350000.);	

  h_nOfflineVertices = outfile_->make<TH1I>("h_nOfflineVertices","n. of vertices;n. vertices; events",100,0,100);
  h_nVertices        = outfile_->make<TH1I>("h_nVertices","n. of vertices;n. vertices; events",100,0,100);
  h_nNonFakeVertices = outfile_->make<TH1I>("h_nRealVertices","n. of non-fake vertices;n. vertices; events",100,0,100);
  h_nFinalVertices   = outfile_->make<TH1I>("h_nSelectedVertices","n. of selected vertices vertices;n. vertices; events",100,0,100); ;  
  
  h_diffX = outfile_->make<TH1F>( "h_diffX"  , "x-coordinate vertex resolution;vertex resolution (x) [#mum];vertices", 100,  -300, 300. );
  h_diffY = outfile_->make<TH1F>( "h_diffY"  , "y-coordinate vertex resolution;vertex resolution (y) [#mum];vertices", 100,  -300, 300. );
  h_diffZ = outfile_->make<TH1F>( "h_diffZ"  , "z-coordinate vertex resolution;vertex resolution (z) [#mum];vertices", 100,  -500, 500. );

  h_pullX = outfile_->make<TH1F>( "h_pullX"  , "x-coordinate vertex pull;vertex pull (x);vertices", 500,  -10, 10. );
  h_pullY = outfile_->make<TH1F>( "h_pullY"  , "y-coordinate vertex pull;vertex pull (y);vertices", 500,  -10, 10. );
  h_pullZ = outfile_->make<TH1F>( "h_pullZ"  , "z-coordinate vertex pull;vertex pull (z);vertices", 500,  -10, 10. );

  h_ntrks = outfile_->make<TH1F>( "h_ntrks"  , "number of tracks in vertex;vertex multeplicity;vertices",myNTrack_bins_.size()-1 , myNTrack_bins_.data());  
  h_nhalftrks = outfile_->make<TH1F>( "h_halfntrks"  , "number of tracks in sub-vertex;sub-vertex multeplicity;vertices",myNTrack_bins_.size()-1 , myNTrack_bins_.data());

  h_sumPt = outfile_->make<TH1F>( "h_sumPt"  ,  "#Sigma p_{T};#sum p_{T} [GeV];vertices",mypT_bins_.size()-1 , mypT_bins_.data());
  h_sumPt1 = outfile_->make<TH1F>( "h_sumPt1"  , "#Sigma p_{T} sub-vertex 1;#sum p_{T} sub-vertex 1 [GeV];subvertices",mypT_bins_.size()-1 , mypT_bins_.data());
  h_sumPt2 = outfile_->make<TH1F>( "h_sumPt2"  , "#Sigma p_{T} sub-vertex 2;#sum p_{T} sub-vertex 2 [GeV];subvertices",mypT_bins_.size()-1 , mypT_bins_.data());

  h_wTrks1 = outfile_->make<TH1F>( "h_wTrks1"  , "weight of track for sub-vertex 1;track weight;subvertices", 500, 0.,   1. );
  h_wTrks2 = outfile_->make<TH1F>( "h_wTrks2"  , "weithg of track for sub-vertex 2;track weight;subvertices", 500, 0.,   1. );

  h_minWTrks1 = outfile_->make<TH1F>( "h_minWTrks1"  , "minimum weight of track for sub-vertex 1;minimum track weight;subvertices", 500, 0.,   1. );
  h_minWTrks2 = outfile_->make<TH1F>( "h_minWTrks2"  , "minimum weithg of track for sub-vertex 2;minimum track weight;subvertices", 500, 0.,   1. );

  h_PVCL_subVtx1 = outfile_->make<TH1F>( "h_PVCL_subVtx1"  , "#chi^{2} probability for sub-vertex 1;Prob(#chi^{2},ndof) for sub-vertex 1;subvertices", 100, 0.,   1 );
  h_PVCL_subVtx2 = outfile_->make<TH1F>( "h_PVCL_subVtx2"  , "#chi^{2} probability for sub-vertex 2;Prob(#chi^{2},ndof) for sub-vertex 2;subvertices", 100, 0.,   1 );

  // sub-vertices positions and errors

  TFileDirectory SubVertices = outfile_->mkdir("SubVertices"); 

  h_X1 = SubVertices.make<TH1F>( "h_X1"  , "x-coordinate sub-vertex 1;vertex1 x [#mum];vertices", 100,  -1000, 1000. );
  h_Y1 = SubVertices.make<TH1F>( "h_Y1"  , "y-coordinate sub-vertex 1;vertex1 y [#mum];vertices", 100,  -1000, 1000. );
  h_Z1 = SubVertices.make<TH1F>( "h_Z1"  , "z-coordinate sub-vertex 1;vertex1 z [cm];vertices"  , 100,  -30, 30. );

  h_errX1 = SubVertices.make<TH1F>( "h_errX1"  , "x-coordinate vertex uncertainty;vertex1 err_{x} [#mum];vertices", 250,0., 500. );
  h_errY1 = SubVertices.make<TH1F>( "h_errY1"  , "y-coordinate vertex uncertainty;vertex1 err_{y} [#mum];vertices", 250,0., 500. );
  h_errZ1 = SubVertices.make<TH1F>( "h_errZ1"  , "z-coordinate vertex uncertainty;vertex1 err_{z} [#mum];vertices", 250,0., 500. );

  h_X2 = SubVertices.make<TH1F>( "h_X2"  , "x-coordinate sub-vertex 2;vertex2 x [#mum];vertices", 100,  -1000, 1000. );
  h_Y2 = SubVertices.make<TH1F>( "h_Y2"  , "y-coordinate sub-vertex 2;vertex2 y [#mum];vertices", 100,  -1000, 1000. );
  h_Z2 = SubVertices.make<TH1F>( "h_Z2"  , "z-coordinate sub-vertex 2;vertex2 z [cm];vertices"  , 100,  -30, 30. );

  h_errX2 = SubVertices.make<TH1F>( "h_errX2"  , "x-coordinate vertex 2 uncertainty;vertex2 err_{x} [#mum];vertices", 250,0., 500. );
  h_errY2 = SubVertices.make<TH1F>( "h_errY2"  , "y-coordinate vertex 2 uncertainty;vertex2 err_{y} [#mum];vertices", 250,0., 500. );
  h_errZ2 = SubVertices.make<TH1F>( "h_errZ2"  , "z-coordinate vertex 2 uncertainty;vertex2 err_{z} [#mum];vertices", 250,0., 500. );


  // resolutions

  p_resolX_vsSumPt = outfile_->make<TH1F>( "p_resolX_vsSumPt"  , "x-resolution vs #Sigma p_{T};#sum p_{T}; x vertex resolution [#mum]", mypT_bins_.size()-1 , mypT_bins_.data() ); 
  p_resolY_vsSumPt = outfile_->make<TH1F>( "p_resolY_vsSumPt"  , "y-resolution vs #Sigma p_{T};#sum p_{T}; y vertex resolution [#mum]", mypT_bins_.size()-1 , mypT_bins_.data() ); 
  p_resolZ_vsSumPt = outfile_->make<TH1F>( "p_resolZ_vsSumPt"  , "z-resolution vs #Sigma p_{T};#sum p_{T}; z vertex resolution [#mum]", mypT_bins_.size()-1 , mypT_bins_.data() ); 
                
  p_resolX_vsNtracks = outfile_->make<TH1F>( "p_resolX_vsNtracks"  , "x-resolution vs n_{tracks};n_{tracks}; x vertex resolution [#mum]", myNTrack_bins_.size()-1 , myNTrack_bins_.data() );
  p_resolY_vsNtracks = outfile_->make<TH1F>( "p_resolY_vsNtracks"  , "y-resolution vs n_{tracks};n_{tracks}; y vertex resolution [#mum]", myNTrack_bins_.size()-1 , myNTrack_bins_.data() );
  p_resolZ_vsNtracks = outfile_->make<TH1F>( "p_resolZ_vsNtracks"  , "z-resolution vs n_{tracks};n_{tracks}; z vertex resolution [#mum]", myNTrack_bins_.size()-1 , myNTrack_bins_.data() );

  p_resolX_vsNvtx = outfile_->make<TH1F>( "p_resolX_vsNvtx"  , "x-resolution vs n_{vertices};n_{vertices}; x vertex resolution [#mum]", myNVtx_bins_.size()-1 , myNVtx_bins_.data() );
  p_resolY_vsNvtx = outfile_->make<TH1F>( "p_resolY_vsNvtx"  , "y-resolution vs n_{vertices};n_{vertices}; y vertex resolution [#mum]", myNVtx_bins_.size()-1 , myNVtx_bins_.data() );
  p_resolZ_vsNvtx = outfile_->make<TH1F>( "p_resolZ_vsNvtx"  , "z-resolution vs n_{vertices};n_{vertices}; z vertex resolution [#mum]", myNVtx_bins_.size()-1 , myNVtx_bins_.data() );

  // pulls

  p_pullX_vsSumPt = outfile_->make<TH1F>( "p_pullX_vsSumPt"  , "x-pull vs #Sigma p_{T};#sum p_{T}; x vertex pull", mypT_bins_.size()-1 , mypT_bins_.data() ); 
  p_pullY_vsSumPt = outfile_->make<TH1F>( "p_pullY_vsSumPt"  , "y-pull vs #Sigma p_{T};#sum p_{T}; y vertex pull", mypT_bins_.size()-1 , mypT_bins_.data() ); 
  p_pullZ_vsSumPt = outfile_->make<TH1F>( "p_pullZ_vsSumPt"  , "z-pull vs #Sigma p_{T};#sum p_{T}; z vertex pull", mypT_bins_.size()-1 , mypT_bins_.data() ); 
                
  p_pullX_vsNtracks = outfile_->make<TH1F>( "p_pullX_vsNtracks"  , "x-pull vs n_{tracks};n_{tracks}; x vertex pull", myNTrack_bins_.size()-1 , myNTrack_bins_.data() );
  p_pullY_vsNtracks = outfile_->make<TH1F>( "p_pullY_vsNtracks"  , "y-pull vs n_{tracks};n_{tracks}; y vertex pull", myNTrack_bins_.size()-1 , myNTrack_bins_.data() );
  p_pullZ_vsNtracks = outfile_->make<TH1F>( "p_pullZ_vsNtracks"  , "z-pull vs n_{tracks};n_{tracks}; z vertex pull", myNTrack_bins_.size()-1 , myNTrack_bins_.data() );

  p_pullX_vsNvtx = outfile_->make<TH1F>( "p_pullX_vsNvtx"  , "x-pull vs n_{vertices};n_{vertices}; x vertex pull", myNVtx_bins_.size()-1 , myNVtx_bins_.data() );
  p_pullY_vsNvtx = outfile_->make<TH1F>( "p_pullY_vsNvtx"  , "y-pull vs n_{vertices};n_{vertices}; y vertex pull", myNVtx_bins_.size()-1 , myNVtx_bins_.data() );
  p_pullZ_vsNvtx = outfile_->make<TH1F>( "p_pullZ_vsNvtx"  , "z-pull vs n_{vertices};n_{vertices}; z vertex pull", myNVtx_bins_.size()-1 , myNVtx_bins_.data() );

}

//*************************************************************
// Generic booker function
//*************************************************************
std::vector<TH1F*> PrimaryVertexResolution::bookResidualsHistogram(TFileDirectory dir,
								   unsigned int theNOfBins,
								   TString resType,
								   TString varType){
  TH1F::SetDefaultSumw2(kTRUE);
  
  double up   = 500.;
  double down = -500.;
  
  if(resType.Contains("pull")){
    up*=0.01;
    down*=0.01;
  }
  
  std::vector<TH1F*> h;
  h.reserve(theNOfBins);
  
  const char* auxResType = (resType.ReplaceAll("_","")).Data();
  
  for(unsigned int i=0; i<theNOfBins;i++){
    TH1F* htemp = dir.make<TH1F>(Form("histo_%s_%s_plot%i",resType.Data(),varType.Data(),i),
				 Form("%s vs %s - bin %i;%s;vertices",auxResType,varType.Data(),i,auxResType),
				 250,down,up); 
    h.push_back(htemp);
  }
  
  return h;
  
}


// ------------ method called once each job just after ending the event loop  ------------
void 
PrimaryVertexResolution::endJob() 
{
  // resolutions
  
  fillTrendPlotByIndex(p_resolX_vsSumPt,h_resolX_sumPt_,PVValHelper::WIDTH);
  fillTrendPlotByIndex(p_resolY_vsSumPt,h_resolY_sumPt_,PVValHelper::WIDTH); 
  fillTrendPlotByIndex(p_resolZ_vsSumPt,h_resolZ_sumPt_,PVValHelper::WIDTH);
  
  fillTrendPlotByIndex(p_resolX_vsNtracks,h_resolX_Ntracks_,PVValHelper::WIDTH);
  fillTrendPlotByIndex(p_resolY_vsNtracks,h_resolY_Ntracks_,PVValHelper::WIDTH); 
  fillTrendPlotByIndex(p_resolZ_vsNtracks,h_resolZ_Ntracks_,PVValHelper::WIDTH);

  fillTrendPlotByIndex(p_resolX_vsNvtx,h_resolX_Nvtx_,PVValHelper::WIDTH);
  fillTrendPlotByIndex(p_resolY_vsNvtx,h_resolY_Nvtx_,PVValHelper::WIDTH); 
  fillTrendPlotByIndex(p_resolZ_vsNvtx,h_resolZ_Nvtx_,PVValHelper::WIDTH);


  // pulls

  fillTrendPlotByIndex(p_pullX_vsSumPt,h_pullX_sumPt_,PVValHelper::WIDTH);
  fillTrendPlotByIndex(p_pullY_vsSumPt,h_pullY_sumPt_,PVValHelper::WIDTH); 
  fillTrendPlotByIndex(p_pullZ_vsSumPt,h_pullZ_sumPt_,PVValHelper::WIDTH);
  
  fillTrendPlotByIndex(p_pullX_vsNtracks,h_pullX_Ntracks_,PVValHelper::WIDTH);
  fillTrendPlotByIndex(p_pullY_vsNtracks,h_pullY_Ntracks_,PVValHelper::WIDTH); 
  fillTrendPlotByIndex(p_pullZ_vsNtracks,h_pullZ_Ntracks_,PVValHelper::WIDTH);

  fillTrendPlotByIndex(p_pullX_vsNvtx,h_pullX_Nvtx_,PVValHelper::WIDTH);
  fillTrendPlotByIndex(p_pullY_vsNvtx,h_pullY_Nvtx_,PVValHelper::WIDTH); 
  fillTrendPlotByIndex(p_pullZ_vsNvtx,h_pullZ_Nvtx_,PVValHelper::WIDTH);

}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
PrimaryVertexResolution::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}


//*************************************************************
void PrimaryVertexResolution::fillTrendPlotByIndex(TH1F* trendPlot,std::vector<TH1F*>& h,  PVValHelper::estimator fitPar_)
//*************************************************************
{  

  for(auto iterator = h.begin(); iterator != h.end(); iterator++) {
    
    unsigned int bin = std::distance(h.begin(),iterator)+1;
    //statmode::fitParams myFit = fitResiduals((*iterator));
    std::pair<Measurement1D, Measurement1D> myFit = PVValHelper::fitResiduals((*iterator));


    switch(fitPar_)
      {
      case PVValHelper::MEAN: 
	{   
	  float mean_      = myFit.first.value();
	  float meanErr_   = myFit.first.error();
	  trendPlot->SetBinContent(bin,mean_);
	  trendPlot->SetBinError(bin,meanErr_);
	  break;
	}
      case PVValHelper::WIDTH:
	{
	  float width_     = myFit.second.value();
	  float widthErr_  = myFit.second.error();
	  trendPlot->SetBinContent(bin,width_);
	  trendPlot->SetBinError(bin,widthErr_);
	  break;
	}
      case PVValHelper::MEDIAN:
	{
	  float median_    = PVValHelper::getMedian((*iterator)).value();
	  float medianErr_ = PVValHelper::getMedian((*iterator)).error();
	  trendPlot->SetBinContent(bin,median_);
	  trendPlot->SetBinError(bin,medianErr_);
	  break;
	}
      case PVValHelper::MAD:
	{
	  float mad_       = PVValHelper::getMAD((*iterator)).value(); 
	  float madErr_    = PVValHelper::getMAD((*iterator)).error();
	  trendPlot->SetBinContent(bin,mad_);
	  trendPlot->SetBinError(bin,madErr_);
	  break;
	}
      default:
	edm::LogWarning("PrimaryVertexResolution")<<"fillTrendPlotByIndex() "<<fitPar_<<" unknown estimator!"<<std::endl;
	break;
      }
  }
}

//*************************************************************
statmode::fitParams PrimaryVertexResolution::fitResiduals(TH1 *hist,bool singleTime)
//*************************************************************
{
  if (hist->GetEntries() < 10){ 
    // std::cout<<"hist name: "<<hist->GetName() << std::endl;
    return std::make_pair(std::make_pair(0.,0.),std::make_pair(0.,0.));
  }
  
  float maxHist = hist->GetXaxis()->GetXmax();
  float minHist = hist->GetXaxis()->GetXmin();
  float mean  = hist->GetMean();
  float sigma = hist->GetRMS();
  
  if(TMath::IsNaN(mean) || TMath::IsNaN(sigma)){  
    mean=0;
    //sigma= - hist->GetXaxis()->GetBinLowEdge(1) + hist->GetXaxis()->GetBinLowEdge(hist->GetNbinsX()+1);
    sigma = - minHist + maxHist;
    edm::LogWarning("PrimaryVertexResolution")<< "FitPVResiduals::fitResiduals(): histogram" << hist->GetName()  << " mean or sigma are NaN!!"<< std::endl;
  }

  TF1 func("tmp", "gaus", mean - 2.*sigma, mean + 2.*sigma); 
  if (0 == hist->Fit(&func,"QNR")) { // N: do not blow up file by storing fit!
    mean  = func.GetParameter(1);
    sigma = func.GetParameter(2);

    if(!singleTime){
      // second fit: three sigma of first fit around mean of first fit
      func.SetRange(std::max(mean - 3*sigma,minHist),std::min(mean + 3*sigma,maxHist));
      // I: integral gives more correct results if binning is too wide
      // L: Likelihood can treat empty bins correctly (if hist not weighted...)
      if (0 == hist->Fit(&func, "Q0LR")) {
	if (hist->GetFunction(func.GetName())) { // Take care that it is later on drawn:
	  hist->GetFunction(func.GetName())->ResetBit(TF1::kNotDraw);
	}
      }
    }
  }

  return std::make_pair(std::make_pair(func.GetParameter(1),func.GetParError(1)),std::make_pair(func.GetParameter(2),func.GetParError(2)));

}

//*************************************************************
statmode::fitParams PrimaryVertexResolution::fitResiduals_v0(TH1 *hist)
//*************************************************************
{
  //float fitResult(9999);
  //if (hist->GetEntries() < 20) return ;
  
  float mean  = hist->GetMean();
  float sigma = hist->GetRMS();
  
  TF1 func("tmp", "gaus", mean - 1.5*sigma, mean + 1.5*sigma); 
  if (0 == hist->Fit(&func,"QNR")) { // N: do not blow up file by storing fit!
    mean  = func.GetParameter(1);
    sigma = func.GetParameter(2);
    // second fit: three sigma of first fit around mean of first fit
    func.SetRange(mean - 2*sigma, mean + 2*sigma);
      // I: integral gives more correct results if binning is too wide
      // L: Likelihood can treat empty bins correctly (if hist not weighted...)
    if (0 == hist->Fit(&func, "Q0LR")) {
      if (hist->GetFunction(func.GetName())) { // Take care that it is later on drawn:
	hist->GetFunction(func.GetName())->ResetBit(TF1::kNotDraw);
      }
    }
  }

  float res_mean  = func.GetParameter(1);
  float res_width = func.GetParameter(2);
  
  float res_mean_err  = func.GetParError(1);
  float res_width_err = func.GetParError(2);

  std::pair<double,double> resultM;
  std::pair<double,double> resultW;

  resultM = std::make_pair(res_mean,res_mean_err);
  resultW = std::make_pair(res_width,res_width_err);

  statmode::fitParams result;
  
  result = std::make_pair(resultM,resultW);
  return result;
}

//*************************************************************
template<std::size_t SIZE>
bool PrimaryVertexResolution::checkBinOrdering(std::array<float, SIZE>& bins)
//*************************************************************
{
  int i=1;

  if(std::is_sorted(bins.begin(),bins.end())){
    return true;
  } else {

    for(const auto &bin : bins){
      std::cout<<"bin: "<<i<< " : "<<bin<<std::endl; 
      i++;
    }
    std::cout<<"--------------------------------"<<std::endl;
    return false;
  }
}


//define this as a plug-in
DEFINE_FWK_MODULE(PrimaryVertexResolution);
