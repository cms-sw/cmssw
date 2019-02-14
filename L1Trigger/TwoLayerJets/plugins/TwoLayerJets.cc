// -*- C++ -*-
//
// Package:    L1Trigger/TwoLayerJets
// Class:      TwoLayerJets
// 
/**\class TwoLayerJets TwoLayerJets.cc L1Trigger/TwoLayerJets/plugins/TwoLayerJets.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Rishi Patel
//         Created:  Wed, 01 Aug 2018 14:01:41 GMT
//
//


// system include files
#include <memory>
//DataFormat Files:
#include "DataFormats/Common/interface/Ref.h"
// L1 tracks
//#include "DataFormats/L1TrackTrigger/interface/L1TkJetParticleDisp.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkJetParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkJetParticleFwd.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "L1Trigger/TrackTrigger/interface/StubPtConsistency.h"
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <string>
#include <cstdlib>
#include "TH1D.h"
#include "TH2D.h"
#include <TMath.h>
//
#include "tracklet_em_disp.h"
#include "TwoLayerL1Jets.h"
// class declaration


//
using namespace std;
using namespace edm;
using namespace l1t;
class TwoLayerJets : public stream::EDProducer<> {
//enum TwoLayerTrkType {ttrk,tdtrk,ttdtrk, ptrk};//last entry is null just prompt track
/*
struct track_data {
  //this is a just a TTTrack
  float pT;
  float eta;
  float z;
  float phi;
  //this is a counter (status)
//use enumerate
  //this associates the track to a zbin
  //this can be a counter along with the vector of tracks
  int bincount;  //How many zbins it's gone into (to make sure it doesn't go into more than 2): start at 0

};
struct etaphibin {
  //these are track counters
  int numtracks; 
  int numttrks;
  int numtdtrks;
  int numttdtrks;
  bool used;


  //this is just a 2D cluster
  float pTtot;
   //average phi value (halfway b/t min and max)
   float phi;
   //average eta value
   float eta;
   //
   };

//store important information for plots
struct maxzbin {
      int znum;        //Numbered from 0 to nzbins (16, 32, or 64) in order.
//    //mc_data * mcd;   //array of jets from input file.
      int nclust;      //number of clusters in this bin.
//      int nttrks;   //number of tigh tracks.
//      int ntdtrks;  //number of tigh displaced tracks.
      etaphibin * clusters;     //list of all the clusters in this bin.
      float ht;   //sum of all cluster pTs--only the zbin with the maximum ht is stored.
          };
*/
     ////function to find all clusters, find zbin with max ht. In file find_clusters.cpp
   public:
      explicit TwoLayerJets(const ParameterSet&);
      ~TwoLayerJets();
      typedef TTTrack< Ref_Phase2TrackerDigi_ >  L1TTTrackType;
      typedef vector< L1TTTrackType > L1TTTrackCollectionType;
     
      static void fillDescriptions(ConfigurationDescriptions& descriptions);
      bool TrackQualityCuts(float trk_pt,int trk_nstub, double trk_chi2,double trk_bconsist);
      //bool TrackQualityCuts(float trk_pt,int trk_nstub, double trk_chi2);
      //Clustering Steps
      void L2_cluster(vector< Ptr< L1TTTrackType > > L1TrackPtrs, vector<int>ttrk, vector<int>tdtrk,vector<int>ttdtrk, maxzbin &mzb);
      virtual etaphibin * L1_cluster(etaphibin * phislice);
     // virtual void  L2_cluster(vector< Ptr< L1TTTrackType > > L1TrackPtrs, vector<int>ttrk, vector<int>tdtrk,vector<int>ttdtrk, int nzbins, int ntracks, maxzbin &mzb);
   private:
      virtual void beginStream(StreamID) override;
      virtual void produce(Event&, const EventSetup&) override;
      virtual void endStream() override;
      const EDGetTokenT<vector<TTTrack< Ref_Phase2TrackerDigi_ > > > trackToken;
       vector< Ptr< L1TTTrackType > > L1TrackPtrs;
       //vector<TwoLayerTrkType> TwoLayerTrackTypes;
       vector<int> zbincount;
       vector<int> ttrk;
       vector<int> tdtrk;
       vector<int> ttdtrk;
       int netabins = 24;
       float maxz = 15.0;
       int nphibins = 27;       
       int Zbins=60;
       float zstep=0.5;
      //etastep is the width of an etabin
      ////Any tracks with pT > 200 GeV should be capped at 200
      float pTmax = 200.0;
      const float maxeta = 2.4;
      const float etastep = 2.0 * maxeta / netabins;
         ////Upper bound on number of tracks per event.
      const int numtracks = 500; //Don't need this?
      ////phistep is the width of a phibin.
      const float phistep = 2*M_PI / nphibins;
      float TRK_PTMIN;      // [GeV]
      float TRK_ETAMAX;     // [rad]
      float CHI2_MAX;
      float BendConsistency_Cut;
      float D0_Cut;
      float NStubs4Chi2_rz_Loose;
      float NStubs4Chi2_rphi_Loose;
      float NStubs4Displacedbend_Loose;
      float  NStubs5Chi2_rz_Loose;
      float NStubs5Chi2_rphi_Loose;
      float NStubs5Displacedbend_Loose; 
      float NStubs5Chi2_rz_Tight;
      float NStubs5Chi2_rphi_Tight;
      float NStubs5Displacedbend_Tight;

      //virtual void beginRun(Run const&, EventSetup const&) override;
      //virtual void endRun(Run const&, EventSetup const&) override;
      //virtual void beginLuminosityBlock(LuminosityBlock const&, EventSetup const&) override;
      //virtual void endLuminosityBlock(LuminosityBlock const&, EventSetup const&) override;

      // ----------member data ---------------------------
};

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
TwoLayerJets::TwoLayerJets(const ParameterSet& iConfig):
trackToken(consumes< vector<TTTrack< Ref_Phase2TrackerDigi_> > > (iConfig.getParameter<InputTag>("L1TrackInputTag")))
{
     produces<L1TkJetParticleCollection>("L1TwoLayerJets");
   //now do what ever other initialization is needed
      maxz    = (float)iConfig.getParameter<double>("ZMAX"); 
      pTmax   =(float)iConfig.getParameter<double>("PTMAX");
      netabins=(int)iConfig.getParameter<int>("Etabins");
      nphibins=(int)iConfig.getParameter<int>("Phibins");
      Zbins=(int)iConfig.getParameter<int>("Zbins"); 
      TRK_PTMIN=(float)iConfig.getParameter<double>("TRK_PTMIN");
      TRK_ETAMAX=(float)iConfig.getParameter<double>("TRK_ETAMAX");      
      zstep = 2.0 * maxz / Zbins;
      CHI2_MAX=(float)iConfig.getParameter<double>("CHI2_MAX");
      BendConsistency_Cut=(float)iConfig.getParameter<double>("PromptBendConsistency");
      D0_Cut=(float)iConfig.getParameter<double>("D0_Cut");
      NStubs4Chi2_rz_Loose=(float)iConfig.getParameter<double>("NStubs4Chi2_rz_Loose");
      NStubs4Chi2_rphi_Loose=(float)iConfig.getParameter<double>("NStubs4Chi2_rphi_Loose");
      NStubs4Displacedbend_Loose=(float)iConfig.getParameter<double>("NStubs4Displacedbend_Loose");
      NStubs5Chi2_rz_Loose=(float)iConfig.getParameter<double>("NStubs5Chi2_rz_Loose");
      NStubs5Chi2_rphi_Loose=(float)iConfig.getParameter<double>("NStubs5Chi2_rphi_Loose");
      NStubs5Displacedbend_Loose=(float)iConfig.getParameter<double>("NStubs5Displacedbend_Loose") ; 
       NStubs5Chi2_rz_Tight=(float)iConfig.getParameter<double>("NStubs5Chi2_rz_Tight");
       NStubs5Chi2_rphi_Tight=(float)iConfig.getParameter<double>("NStubs5Chi2_rphi_Tight");
       NStubs5Displacedbend_Tight=(float)iConfig.getParameter<double>("NStubs5Displacedbend_Tight");
}


TwoLayerJets::~TwoLayerJets()
{
 
   // do anything here that needs to be done at destruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
TwoLayerJets::produce(Event& iEvent, const EventSetup& iSetup)
{
  // more for TTStubs
  //ESHandle<TrackerGeometry> geometryHandle;
  //iSetup.get<TrackerDigiGeometryRecord>().get(geometryHandle);

  ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);

  ESHandle<TrackerGeometry> tGeomHandle;
  iSetup.get<TrackerDigiGeometryRecord>().get(tGeomHandle);

  const TrackerTopology* const tTopo = tTopoHandle.product();
  const TrackerGeometry* const theTrackerGeom = tGeomHandle.product();

  edm::ESHandle<MagneticField> magneticFieldHandle;
  iSetup.get<IdealMagneticFieldRecord>().get(magneticFieldHandle);
//  if(!magneticFieldHandle.isValid())std::cout<<" Mag field not present "<<std::endl;
//  else {
  const MagneticField* theMagneticField = magneticFieldHandle.product();
  double mMagneticFieldStrength = theMagneticField->inTesla(GlobalPoint(0,0,0)).z();
  unique_ptr<L1TkJetParticleCollection> L1TwoLayerJets(new L1TkJetParticleCollection);
  // L1 tracks
   edm::Handle< vector< TTTrack< Ref_Phase2TrackerDigi_ > > > TTTrackHandle;
   iEvent.getByToken(trackToken, TTTrackHandle);
   vector< TTTrack< Ref_Phase2TrackerDigi_ > >::const_iterator iterL1Track;
   L1TrackPtrs.clear();
   //TwoLayerTrackTypes.clear(); 
  zbincount.clear();
  ttrk.clear();
  tdtrk.clear();
  ttdtrk.clear();
//std::cout<<"z eta, phi step "<<zstep <<", " <<phistep<<", "<<etastep<<std::endl;
   unsigned int this_l1track = 0;
  for ( iterL1Track = TTTrackHandle->begin(); iterL1Track != TTTrackHandle->end(); iterL1Track++ ) {
       edm::Ptr< L1TTTrackType > trkPtr(TTTrackHandle, this_l1track) ;
       ++this_l1track;
	//Quality Cuts
	float trackpT=trkPtr->getMomentum().perp();
	int tracknstubs=trkPtr->getStubRefs().size();
	float trackchi2=trkPtr->getChi2(4);
	float trackchi2z=trkPtr->getChi2(4);
	float trk_x0   = trkPtr->getPOCA(5).x();
        float trk_y0   = trkPtr->getPOCA(5).y();
	float trk_phi = trkPtr->getMomentum().phi();
        float trk_d0 = -trk_x0*sin(trk_phi) + trk_y0*cos(trk_phi);
	
	//check Trk Class
	float trk_bstubPt=StubPtConsistency::getConsistency(TTTrackHandle->at(this_l1track-1), theTrackerGeom, tTopo,mMagneticFieldStrength,4);//trkPtr->getStubPtConsistency(4)/tracknstubs;
	//float trk_bstubPt=trkPtr->getStubPtConsistency(4)/tracknstubs;
	//trk_stubPt=trk_stubPt/tracknstubs;
        //need min pT, eta, z cut
	if(!TrackQualityCuts(trackpT,tracknstubs,trackchi2/(2*tracknstubs-4),trk_bstubPt))continue;
    	if(fabs(iterL1Track->getPOCA(4).z())>maxz)continue;
    	if(fabs(iterL1Track->getMomentum(4).eta())>TRK_ETAMAX)continue;
    	if(iterL1Track->getMomentum(4).perp()<TRK_PTMIN)continue;
	L1TrackPtrs.push_back(trkPtr);	
	zbincount.push_back(0);	
	//flag as displaced tracks:NOTE Exclusive categories
	//int whichcase=0;
			
	if ((abs(trk_d0)>D0_Cut && tracknstubs>=5)||(tracknstubs==4 && fabs(trk_d0)>D0_Cut))tdtrk.push_back(1);
	else tdtrk.push_back(0);//displaced track
        if (( tracknstubs==4 && (trackchi2/(tracknstubs-3))<NStubs4Chi2_rphi_Loose && (trackchi2z/(tracknstubs-2))<NStubs4Chi2_rz_Loose && trk_bstubPt<NStubs4Displacedbend_Loose  )||( (trackchi2/(tracknstubs-3)) <NStubs5Chi2_rphi_Tight && (trackchi2z/(tracknstubs-2))<NStubs5Chi2_rz_Tight && tracknstubs>=5 && trk_bstubPt<NStubs5Displacedbend_Tight) || ( (trackchi2/(tracknstubs-3)) <NStubs5Chi2_rphi_Loose && (trackchi2z/(tracknstubs-2))<NStubs5Chi2_rz_Loose && tracknstubs>=5 && trk_bstubPt<NStubs5Displacedbend_Loose))ttrk.push_back(1);
	else ttrk.push_back(0); 
        if ((( tracknstubs==4 && (trackchi2/(tracknstubs-3))<NStubs4Chi2_rphi_Loose && (trackchi2z/(tracknstubs-2))<NStubs4Chi2_rz_Loose && trk_bstubPt<NStubs4Displacedbend_Loose  )||( (trackchi2/(tracknstubs-3)) <NStubs5Chi2_rphi_Tight && (trackchi2z/(tracknstubs-2))<NStubs5Chi2_rz_Tight && tracknstubs>=5 && trk_bstubPt<NStubs5Displacedbend_Tight) || ( (trackchi2/(tracknstubs-3)) <NStubs5Chi2_rphi_Loose && (trackchi2z/(tracknstubs-2))<NStubs5Chi2_rz_Loose && tracknstubs>=5 && trk_bstubPt<NStubs5Displacedbend_Loose)) && fabs(trk_d0)>D0_Cut)ttrk.push_back(1);
	//if ( (trackchi2/(tracknstubs-3)) <NStubs5Chi2_rphi_Loose && (trackchi2z/(tracknstubs-2))<NStubs5Chi2_rz_Loose && tracknstubs>=5 && trk_bstubPt<NStubs5Displacedbend_Tight && ((abs(trk_d0)>D0_Cut && tracknstubs>=5)||(tracknstubs==4 && abs(trk_d0)>D0_Cut)) )ttdtrk.push_back(1);
	else ttdtrk.push_back(0);
  } 
    if(L1TrackPtrs.size()>0){
    maxzbin mzb;

       L2_cluster(L1TrackPtrs, ttrk, tdtrk,ttdtrk,mzb);
       edm::Ref< JetBxCollection > jetRef ;//null no Calo Jet Ref
        vector< Ptr< L1TTTrackType > > L1TrackAssocJet; 
	if(mzb.clusters!=NULL){
        for(int j = 0; j < mzb.nclust; ++j){
		//FILL Two Layer Jets for Jet Collection
		if(mzb.clusters[j].pTtot<=0)continue;
		//if(mzb.nclust>mzb.clusters[j].numtracks)continue;
		if(mzb.clusters[j].numtracks<1)continue;
		if(mzb.clusters[j].numtracks>5000)continue;
		float jetEta=mzb.clusters[j].eta;
		float jetPhi=mzb.clusters[j].phi;
		float jetPt=mzb.clusters[j].pTtot;
		float jetPx=jetPt*cos(jetPhi);
		float jetPy=jetPt*sin(jetPhi);
		float jetPz=jetPt*sinh(jetEta);
		float jetP=jetPt*cosh(jetEta);
		math::XYZTLorentzVector jetP4(jetPx,jetPy,jetPz,jetP );
		L1TrackAssocJet.clear();
		for(unsigned int t=0; t<L1TrackPtrs.size(); ++t){
			if(L1TrackAssocJet.size()==(unsigned int)mzb.clusters[j].numtracks)break;
			float deta=L1TrackPtrs[t]->getMomentum().eta()-jetEta;
			float dphi=L1TrackPtrs[t]->getMomentum().phi()-jetPhi;
			float dZ=fabs(mzb.zbincenter-L1TrackPtrs[t]->getPOCA(5).z());
			//std::cout<<"Trk z "<<L1TrackPtrs[t]->getPOCA(5).z()<<" "<<L1TrackPtrs[t]->getPOCA(4).z()<<std::endl;
			//std::cout<<"Trk eta "<<L1TrackPtrs[t]->getMomentum().eta()<<" Jet eta "<<jetEta<<std::endl;
			if(dZ<zstep && fabs(deta)<etastep*2. && fabs(dphi)<phistep*2. ){
		//	std::cout<<"Match "<<std::endl;
		//		std::cout<<" Deta, Dphi, dZ "<<deta<<", "<<dphi<<", "<<dZ<<std::endl;
				L1TrackAssocJet.push_back(L1TrackPtrs[t]);
			}
		}
    		//if(mzb.clusters[j].numtracks!= (int)L1TrackAssocJet.size())std::cout<<"ntracks "<<mzb.clusters[j].numtracks<<" L1TrackAssocJet "<<L1TrackAssocJet.size()<<std::endl;
		int totalTighttrk=0;
		int totalDisptrk=0;
		int totalTightDisptrk=0;
		for(unsigned int t=0; t<ttrk.size(); ++t){
			if(ttrk[t]>0)++totalTighttrk;
			if(tdtrk[t]>0)++totalDisptrk;
			if(ttdtrk[t]>0)++totalTightDisptrk;
		}	
		///L1TkJetParticleDisp DispCounters(mzb.clusters[j].numtracks,totalTighttrk, totalDisptrk, totalTightDisptrk);
		L1TkJetParticle trkJet(jetP4,  L1TrackAssocJet, mzb.zbincenter,mzb.clusters[j].numtracks,totalTighttrk, totalDisptrk, totalTightDisptrk);
		//trkJet.setDispCounters(DispCounters);
    		if(L1TrackAssocJet.size()>0)L1TwoLayerJets->push_back(trkJet);		

          }       
	}
	iEvent.put( std::move(L1TwoLayerJets), "L1TwoLayerJets");
	}
}
void TwoLayerJets::L2_cluster(vector< Ptr< L1TTTrackType > > L1TrackPtrs, vector<int>ttrk, vector<int>tdtrk,vector<int>ttdtrk,maxzbin &mzb){
  const int nz = Zbins;
  //return;
  maxzbin  all_zbins[nz];
  if(all_zbins==NULL){ cout<<" all_zbins memory not assigned"<<endl;return;}
  //int best_ind=0;
  //          
  float zmin = -1.0*maxz;
  float zmax = zmin + 2*zstep;
  //                //Create grid of phibins! 
  etaphibin epbins[nphibins][netabins];
   float phi = -1.0 * M_PI;
  float eta;
  float etamin, etamax, phimin, phimax;
  for(int i = 0; i < nphibins; ++i){
      eta = -1.0 * maxeta;
            for(int j = 0; j < netabins; ++j){
    phimin = phi;
    phimax = phi + phistep;
    etamin = eta;
    eta = eta + etastep;
    etamax = eta;
    epbins[i][j].phi = (phimin + phimax) / 2;
    epbins[i][j].eta = (etamin + etamax) / 2;
       }//for each etabin
       phi = phi + phistep;
   } //for each phibin (finished creating epbins)
  mzb = all_zbins[0];

for(int zbin = 0; zbin < Zbins-1; ++zbin){
  
        //First initialize pT, numtracks, used to 0 (or false)
        for(int i = 0; i < nphibins; ++i){
             for(int j = 0; j < netabins; ++j){
                 epbins[i][j].pTtot = 0;
                 epbins[i][j].used = false;
                 epbins[i][j].numtracks = 0;
                 epbins[i][j].numttrks = 0;
                 epbins[i][j].numtdtrks = 0;
                 epbins[i][j].numttdtrks = 0;
                 }//for each etabin
           } //for each phibin

   for (unsigned int k=0; k<L1TrackPtrs.size(); ++k){
      float trketa=L1TrackPtrs[k]->getMomentum().eta();
      float trkphi=L1TrackPtrs[k]->getMomentum().phi();
      float trkZ=L1TrackPtrs[k]->getPOCA(5).z();
      for(int i = 0; i < nphibins; ++i){
        for(int j = 0; j < netabins; ++j){
          if((zmin <= trkZ && zmax >= trkZ) &&
            ((epbins[i][j].eta - etastep / 2 <= trketa && epbins[i][j].eta + etastep / 2 >= trketa) 
              && epbins[i][j].phi - phistep / 2 <= trkphi && epbins[i][j].phi + phistep / 2 >= trkphi && (zbincount[k] != 2))){
            zbincount.at(k)=zbincount.at(k)+1;
            if(L1TrackPtrs[k]->getMomentum().perp()<pTmax)epbins[i][j].pTtot += L1TrackPtrs[k]->getMomentum().perp();
	    else epbins[i][j].pTtot +=pTmax;
	   //add Displaced Track criteria
            epbins[i][j].numttrks += ttrk[k];
            epbins[i][j].numtdtrks += tdtrk[k];
            epbins[i][j].numttdtrks += ttdtrk[k];
            ++epbins[i][j].numtracks;
    //        cout << epbins[i][j].phi << "\t" << tracks[k].pT << endl;
             } //if right bin
       } //for each phibin: j loop
      }//for each phibin: i loop
     //new 
    }
  //  etaphibin ** L1clusters = (etaphibin**)malloc(nphibins*sizeof(etaphibin*));
      etaphibin *L1clusters[nphibins];
                for(int phislice = 0; phislice < nphibins; ++phislice){
      L1clusters[phislice] = L1_cluster(epbins[phislice]);
      for(int ind = 0; L1clusters[phislice][ind].pTtot != 0; ++ind){
        L1clusters[phislice][ind].used = false;
	//cout<<"L1 Clusters "<< L1clusters[phislice][ind].eta<<", "<< L1clusters[phislice][ind].phi<<", "<< L1clusters[phislice][ind].pTtot<<std::endl;
      }
    }
  //Create clusters array to hold output cluster data for Layer2; can't have more clusters than tracks.
    int ntracks=L1TrackPtrs.size();

    //etaphibin L2cluster[ntracks];//= (etaphibin *)malloc(ntracks * sizeof(etaphibin));
    etaphibin L2cluster[ntracks];// = (etaphibin *)malloc(ntracks * sizeof(etaphibin));
    //etaphibin * L2cluster = (etaphibin *)malloc(ntracks * sizeof(etaphibin));
    if(L2cluster==NULL) cout<<"L2cluster memory not assigned"<<endl;

  //Find eta-phibin with maxpT, make center of cluster, add neighbors if not already used.
    float hipT = 0;
    int nclust = 0;
    int phibin = 0;
    int imax=-1;
       //index of clusters array for each phislice.
    int index1;
    float E1 =0;
    float E0 =0;
    float E2 =0;
    int trx1, trx2;
    int ttrk1, ttrk2;
    int tdtrk1, tdtrk2;
    int ttdtrk1, ttdtrk2;
    int used1, used2, used3, used4;

      //Find eta-phibin with highest pT.
    for(phibin = 0; phibin < nphibins; ++phibin){
        while(true){
      hipT = 0;
      for(index1 = 0; L1clusters[phibin][index1].pTtot > 0; ++index1){
        if(!L1clusters[phibin][index1].used && L1clusters[phibin][index1].pTtot >= hipT){
          hipT = L1clusters[phibin][index1].pTtot;
          imax = index1;
        }
      }//for each index within the phibin
          //If highest pT is 0, all bins are used.
      if(hipT == 0){
        break;
      }
      E0 = hipT;   //E0 is pT of first phibin of the cluster.
      E1 = 0;
      E2 = 0;
      trx1 = 0;
      trx2 = 0;
      ttrk1 = 0;
      ttrk2 = 0;
      tdtrk1 = 0;
      tdtrk2 = 0;
      ttdtrk1 = 0;
      ttdtrk2 = 0;
      L2cluster[nclust] = L1clusters[phibin][imax];
      L1clusters[phibin][imax].used = true;
    //Add pT of upper neighbor.
    //E1 is pT of the middle phibin (should be highest pT)
      if(phibin != nphibins-1){
        used1 = -1;
        used2 = -1;
        for (index1 = 0; L1clusters[phibin+1][index1].pTtot != 0; ++index1){
          if(L1clusters[phibin+1][index1].used){
            continue;
          }
          if(fabs(L1clusters[phibin+1][index1].eta - L1clusters[phibin][imax].eta) <= 1.5*etastep){
            E1 += L1clusters[phibin+1][index1].pTtot;
            trx1 += L1clusters[phibin+1][index1].numtracks;
            ttrk1 += L1clusters[phibin+1][index1].numttrks;
            tdtrk1 += L1clusters[phibin+1][index1].numtdtrks;
            ttdtrk1 += L1clusters[phibin+1][index1].numttdtrks;
            if(used1 < 0)
              used1 = index1;
            else
              used2 = index1;
          }//if cluster is within one phibin
        } //for each cluster in above phibin
      //if E1 isn't higher, E0 and E1 are their own cluster.
        if(E1 < E0){
          L2cluster[nclust].pTtot += E1;   
          L2cluster[nclust].numtracks += trx1;
          L2cluster[nclust].numttrks += ttrk1;
          L2cluster[nclust].numtdtrks += tdtrk1;
          L2cluster[nclust].numttdtrks += ttdtrk1;
          if(used1 >= 0)
            L1clusters[phibin+1][used1].used = true;
          if(used2 >= 0)
            L1clusters[phibin+1][used2].used = true;
          ++nclust;
          continue;
        }
        
        if(phibin != nphibins-2){
                                      //E2 will be the pT of the third phibin (should be lower than E1).
          used3 = -1;
          used4 = -1;
          for (index1 = 0; L1clusters[phibin+2][index1].pTtot != 0; ++index1){
            if(L1clusters[phibin+2][index1].used){
              continue;
            }
            if(fabs(L1clusters[phibin+2][index1].eta - L1clusters[phibin][imax].eta) <= 1.5*etastep){
              E2 += L1clusters[phibin+2][index1].pTtot;
              trx2 += L1clusters[phibin+2][index1].numtracks;
              ttrk2 += L1clusters[phibin+2][index1].numttrks;
              tdtrk2 += L1clusters[phibin+2][index1].numtdtrks;
              ttdtrk2 += L1clusters[phibin+2][index1].numttdtrks;
              if(used3 < 0)
                used3 = index1;
              else
                used4 = index1;
            }
    
          }
             //if indeed E2 < E1, add E1 and E2 to E0, they're all a cluster together.
             //  otherwise, E0 is its own cluster.
          if(E2 < E1){
            L2cluster[nclust].pTtot += E1 + E2;
            L2cluster[nclust].numtracks += trx1 + trx2;
            L2cluster[nclust].numttrks += ttrk1 + ttrk2;
            L2cluster[nclust].numtdtrks += tdtrk1 + tdtrk2;
            L2cluster[nclust].numttdtrks += ttdtrk1 + ttdtrk2;
            L2cluster[nclust].phi = L1clusters[phibin+1][used1].phi;  
            if(used1 >= 0)
              L1clusters[phibin+1][used1].used = true;
            if(used2 >= 0)
              L1clusters[phibin+1][used2].used = true;
            if(used3 >= 0)
              L1clusters[phibin+2][used3].used = true;
            if(used4 >= 0)
              L1clusters[phibin+2][used4].used = true;
          }
          ++nclust;
          continue;
        } // end Not nphibins-2
        else{
          L2cluster[nclust].pTtot += E1;
          L2cluster[nclust].numtracks += trx1;
          L2cluster[nclust].numttrks += ttrk1;
          L2cluster[nclust].numtdtrks += tdtrk1;
          L2cluster[nclust].numttdtrks += ttdtrk1;
          L2cluster[nclust].phi = L1clusters[phibin+1][used1].phi;
          if(used1 >= 0)
            L1clusters[phibin+1][used1].used = true;
          if(used2 >= 0)
            L1clusters[phibin+1][used2].used = true;
          ++nclust;
          continue;
        }
      }//End not last phibin(23)
      else { //if it is phibin 23
        L1clusters[phibin][imax].used = true;
        ++nclust;
      }
        }//while hipT not 0
     //free(L1clusters[phislice]);
    }//for each phibin
    //for(int db=0;db<nclust;++db)cout<<L2cluster[db].phi<<"\t"<<L2cluster[db].pTtot<<"\t"<<L2cluster[db].numtracks<<endl;  
//for(phibin = 0; phibin < nphibins; ++phibin)free(L1clusters[phibin]);
  //Now merge clusters, if necessary
 for(int m = 0; m < nclust -1; ++m){
                     for(int n = m+1; n < nclust; ++n)
                        if(L2cluster[n].eta == L2cluster[m].eta && (fabs(L2cluster[n].phi - L2cluster[m].phi) < 1.5*phistep || fabs(L2cluster[n].phi - L2cluster[m].phi) > 6.0)){
                                if(L2cluster[n].pTtot > L2cluster[m].pTtot){
                                        L2cluster[m].phi = L2cluster[n].phi;
                                }
                                L2cluster[m].pTtot += L2cluster[n].pTtot;
                                L2cluster[m].numtracks += L2cluster[n].numtracks;
        L2cluster[m].numttrks += L2cluster[n].numttrks;
        L2cluster[m].numtdtrks += L2cluster[n].numtdtrks;
        L2cluster[m].numttdtrks += L2cluster[n].numttdtrks;
                                for(int m1 = n; m1 < nclust-1; ++m1){
                                        L2cluster[m1] = L2cluster[m1+1];
                                }
                                nclust--;
                                m = -1;
                                break; //?????
                        }//end if clusters neighbor in eta
                }//end for (m) loop     
          //sum up all pTs in this zbin to find ht.
    float ht = 0;
    for(int k = 0; k < nclust; ++k){
                        if(L2cluster[k].pTtot>50 && L2cluster[k].numtracks<2)continue;
                        if(L2cluster[k].pTtot>100 && L2cluster[k].numtracks<=4)continue;
                        if(L2cluster[k].pTtot>5){
      			ht += L2cluster[k].pTtot;
                }
	}
     //if ht is larger than previous max, this is the new vertex zbin.
      //all_zbins[zbin].mcd = mcd;
    all_zbins[zbin].znum = zbin;
    all_zbins[zbin].clusters = new etaphibin[nclust];// (etaphibin *)malloc(nclust*sizeof(etaphibin));
    all_zbins[zbin].nclust = nclust;
    all_zbins[zbin].zbincenter=(zmin+zmax)/2.0;
    for(int k = 0; k < nclust; ++k){
      all_zbins[zbin].clusters[k].phi = L2cluster[k].phi;                               
      all_zbins[zbin].clusters[k].eta = L2cluster[k].eta;                             
      all_zbins[zbin].clusters[k].pTtot = L2cluster[k].pTtot;
      all_zbins[zbin].clusters[k].numtracks = L2cluster[k].numtracks;
      all_zbins[zbin].clusters[k].numttrks = L2cluster[k].numttrks;
      all_zbins[zbin].clusters[k].numtdtrks = L2cluster[k].numtdtrks;
      all_zbins[zbin].clusters[k].numttdtrks = L2cluster[k].numttdtrks;
    }
  
  //  for(int db=0;db<nclust;++db)cout<<all_zbins[zbin].clusters[db].phi<<"\t"<<all_zbins[zbin].clusters[db].pTtot<<endl; 
    all_zbins[zbin].ht = ht;
    if(ht >= mzb.ht){
      mzb = all_zbins[zbin];
      mzb.zbincenter=(zmin+zmax)/2.0;
     // best_ind=zbin;
    }
    //Prepare for next zbin!
    zmin = zmin + zstep;
    zmax = zmax + zstep;
      
    //   for(int phislice = 0; phislice < nphibins; ++phislice){
    //   free(L1clusters[phislice]);      
    // }
   // free(all_zbins[zbin].clusters);

    //for(int k = 0; k < nclust; ++k)
//	free(L1clusters);

    } //for each zbin
   //std::cout<<"Chosen Z -bin "<<mzb.zbincenter<<std::endl; 
    //for(int zbin = 0; zbin < Zbins-1; ++zbin)free(all_zbins[zbin].clusters);
    //for(int k = 0; k < mzb.nclust; ++k)std::cout<<"L2 Eta, Phi "<<mzb.clusters[k].eta<<", "<<mzb.clusters[k].phi<<", "<<mzb.clusters[k].pTtot<<std::endl;
    for(int zbin = 0; zbin < Zbins-1; ++zbin){
      delete[] all_zbins[zbin].clusters;
    }
}

etaphibin *  TwoLayerJets::L1_cluster(etaphibin *phislice){
    etaphibin * clusters = new etaphibin[netabins/2];
    //etaphibin * clusters = (etaphibin *)malloc(netabins/2 * sizeof(etaphibin));
    //static etaphibin clusters[netabins]; //= (etaphibin *)malloc(netabins/2 * sizeof(etaphibin));
    if(clusters==NULL) cout<<"clusters memory not assigned"<<endl;
  //Find eta-phibin with maxpT, make center of cluster, add neighbors if not already used.
    float my_pt, left_pt, right_pt, right2pt;

    int nclust = 0;
    right2pt=0;
    for(int etabin = 0; etabin < netabins; ++etabin){
      //assign values for my pT and neighbors' pT
      if(phislice[etabin].used) continue;
      my_pt = phislice[etabin].pTtot;
      if(etabin > 0 && !phislice[etabin-1].used) {
        left_pt = phislice[etabin-1].pTtot;
        // if(etabin > 1 && !phislice[etabin-2].used) {
        //   left2pt = phislice[etabin-2].pTtot;
        // } else left2pt = 0;
      } else left_pt = 0;
      if(etabin < netabins - 1 && !phislice[etabin+1].used) {
        right_pt = phislice[etabin+1].pTtot;
        if(etabin < netabins - 2 && !phislice[etabin+2].used) {
          right2pt = phislice[etabin+2].pTtot;
        } else right2pt = 0;
      } else right_pt = 0;
    
    //if I'm not a cluster, move on.
      if(my_pt < left_pt || my_pt <= right_pt) {
         //if unused pT in the left neighbor, spit it out as a cluster.
              if(left_pt > 0) {
          clusters[nclust] = phislice[etabin-1];
          phislice[etabin-1].used = true;
          ++nclust;
        }
        continue;
      }

    //I guess I'm a cluster-- should I use my right neighbor?
    // Note: left neighbor will definitely be used because if it 
    //       didn't belong to me it would have been used already
      clusters[nclust] = phislice[etabin];
      phislice[etabin].used = true;
      if(left_pt > 0) {
        clusters[nclust].pTtot += left_pt;
        clusters[nclust].numtracks += phislice[etabin-1].numtracks;
        clusters[nclust].numttrks += phislice[etabin-1].numttrks;
        clusters[nclust].numtdtrks += phislice[etabin-1].numtdtrks;
        clusters[nclust].numttdtrks += phislice[etabin-1].numttdtrks;
      }
      if(my_pt >= right2pt && right_pt > 0) {
        clusters[nclust].pTtot += right_pt;
        clusters[nclust].numtracks += phislice[etabin+1].numtracks;
        clusters[nclust].numttrks += phislice[etabin+1].numttrks;
        clusters[nclust].numtdtrks += phislice[etabin+1].numtdtrks;
        clusters[nclust].numttdtrks += phislice[etabin+1].numttdtrks;
        phislice[etabin+1].used = true;
      }

      ++nclust;
    } //for each etabin                       
                           
  //Now merge clusters, if necessary
    for(int m = 0; m < nclust -1; ++m){
      if(fabs(clusters[m+1].eta - clusters[m].eta) < 1.5*etastep){
        if(clusters[m+1].pTtot > clusters[m].pTtot){
          clusters[m].eta = clusters[m+1].eta;
        }
        clusters[m].pTtot += clusters[m+1].pTtot;
        clusters[m].numtracks += clusters[m+1].numtracks;  //Previous version didn't add tracks when merging. 
        clusters[m].numttrks += clusters[m+1].numttrks;
        clusters[m].numtdtrks += clusters[m+1].numtdtrks;
        clusters[m].numttdtrks += clusters[m+1].numttdtrks;
        for(int m1 = m+1; m1 < nclust-1; ++m1){
          clusters[m1] = clusters[m1+1];
        }
        nclust--;
        m = -1;
      }//end if clusters neighbor in eta
    }//end for (m) loop
//  for(int i = 0; i < nclust; ++i) cout << clusters[i].phi << "\t" << clusters[i].pTtot << "\t" << clusters[i].numtracks << endl;
  //zero out remaining unused clusters.
  for(int i = nclust; i < netabins/2; ++i){
    clusters[i].pTtot = 0;
  }
  return clusters;
}
// ------------ method called once each stream before processing any runs, lumis or events  ------------
void
TwoLayerJets::beginStream(StreamID)
{
}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void
TwoLayerJets::endStream() {
}

// ------------ method called when starting to processes a run  ------------
/*
void
TwoLayerJets::beginRun(Run const&, EventSetup const&)
{
}
*/
 
// ------------ method called when ending the processing of a run  ------------
/*
void
TwoLayerJets::endRun(Run const&, EventSetup const&)
{
}
*/
 
// ------------ method called when starting to processes a luminosity block  ------------
/*
void
TwoLayerJets::beginLuminosityBlock(LuminosityBlock const&, EventSetup const&)
{
}
*/
 
// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
TwoLayerJets::endLuminosityBlock(LuminosityBlock const&, EventSetup const&)
{
}
*/
//etaphibin * TwoLayerJets::L1_cluster(etaphibin*phislice){


//}
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------

bool TwoLayerJets::TrackQualityCuts(float trk_pt,int trk_nstub, double trk_chi2,double trk_bconsist){
bool PassQuality=false;

if(trk_bconsist<BendConsistency_Cut && trk_chi2<CHI2_MAX && trk_nstub>=4)PassQuality=true;
return PassQuality; 
} 
void
TwoLayerJets::fillDescriptions(ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(TwoLayerJets);
