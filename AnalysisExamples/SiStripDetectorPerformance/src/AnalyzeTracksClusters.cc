// created by Livio Fano'
#include <memory>
#include <string>
#include <iostream>
#include <TMath.h>
#include "AnalysisExamples/SiStripDetectorPerformance/interface/AnalyzeTracksClusters.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "TRandom.h"
#include "AnalysisExamples/SiStripDetectorPerformance/interface/TrackLocalAngle.h"
//add trigger info
#include "DataFormats/LTCDigi/interface/LTCDigi.h"



using namespace std;
AnalyzeTracksClusters::AnalyzeTracksClusters(edm::ParameterSet const& conf) : 
  conf_(conf)
{
	anglefinder_=new  TrackLocalAngle(conf);  
}

AnalyzeTracksClusters::~AnalyzeTracksClusters() 
{  
}  

void AnalyzeTracksClusters::beginJob(const edm::EventSetup& conf){

  hFile = new TFile ( "trackcluster.root", "RECREATE" );
  nt = new TTree("nt","a Tree");

  nt->Branch("Ntk",&Ntk,"Ntk/I");
  nt->Branch("p_tk",p_tk,"p_tk[Ntk]/F");
  nt->Branch("pt_tk",pt_tk,"pt_tk[Ntk]/F");
  nt->Branch("eta_tk",eta_tk,"eta_tk[Ntk]/F");
  nt->Branch("phi_tk",phi_tk,"phi_tk[Ntk]/F");
  nt->Branch("nhits_tk",nhits_tk,"nhits_tk[Ntk]/I");
  nt->Branch("Nclu",&Nclu,"Nclu/I");
  nt->Branch("Nclu_matched",&Nclu_matched,"Nclu_Matched/I");
  nt->Branch("Subid",Subid,"Subid[Nclu]/I");
  nt->Branch("Layer",Layer,"Layer[Nclu]/I");
  nt->Branch("Clu_ch",Clu_ch,"Clu_ch[Nclu]/F");
  nt->Branch("Clu_ang",Clu_ang,"Clu_ang[Nclu]/F");
  nt->Branch("Clu_size",Clu_size,"Clu_size[Nclu]/F");
  nt->Branch("Clu_bar",Clu_bar,"Clu_bar[Nclu]/F");
  nt->Branch("Clu_1strip",Clu_1strip,"Clu_1strip[Nclu]/I");
  nt->Branch("Clu_rawid",Clu_rawid,"Clu_rawid[Nclu]/I");
  nt->Branch("Nclu_all",&Nclu_all,"Nclu_all/I");
  //  nt->Branch("Nclu_all_matched",&Nclu_all_matched,"Nclu_all_matched/I");
  //  nt->Branch("Nclu_all_st",&Nclu_all_st,"Nclu_all_st/I");
  //  nt->Branch("Nclu_all_rphi",&Nclu_all_rphi,"Nclu_all_rphi/I");
  nt->Branch("Subid_all",Subid_all,"Subid_all[Nclu_all]/I");
  nt->Branch("Layer_all",Layer_all,"Layer_all[Nclu_all]/I");
  nt->Branch("Clu_ch_all",Clu_ch_all,"Clu_ch_all[Nclu_all]/F");
  nt->Branch("Clu_ang_all",Clu_ang_all,"Clu_ang_all[Nclu_all]/F");
  nt->Branch("Clu_size_all",Clu_size_all,"Clu_size_all[Nclu_all]/F");
  nt->Branch("Clu_bar_all",Clu_bar_all,"Clu_bar_all[Nclu_all]/F");
  nt->Branch("Clu_1strip_all",Clu_1strip_all,"Clu_1strip_all[Nclu_all]/I");
  nt->Branch("Clu_rawid_all",Clu_rawid_all,"Clu_rawid_all[Nclu_all]/I");
  //nt->Branch("TrigBits",TrigBits,"TrigBits[6]/I");
  nt->Branch("nev",&nev,"nev/I");
  nt->Branch("DTtrig",&DTtrig,"DTtrig/I");
  nt->Branch("NoDTtrig",&NoDTtrig,"NoDTtrig/I");
  nt->Branch("DTOnlytrig",&DTOnlytrig,"DTOnlytrig/I");
  nt->Branch("CSCtrig",&CSCtrig,"CSCtrig/I");
  nt->Branch("Othertrig",&othertrig,"Othertrig/I");
  itk=0;

  nev=0;

}

void AnalyzeTracksClusters::endJob(){
  std::cout << "****************************************" << std::endl;
  std::cout << nev << " events processed" << std::endl;
  std::cout << itk << " tracks processed" << std::endl;
  nt->Write();
  hFile->Write();
  hFile->Close();
}

void AnalyzeTracksClusters::analyze(const edm::Event& e, const edm::EventSetup& es)
{

  nev++;
// get LTDC info ----------------------------------------------
                                                                                                            
  edm::Handle<LTCDigiCollection> ltcdigis;
  e.getByType(ltcdigis);

  
  DTtrig=0;
  NoDTtrig=0;
  DTOnlytrig=0;
  CSCtrig=0;
  othertrig=0;


  for(int i=0;i<6;i++)  TrigBits[i]=0;

  
  //  cout<<"[LTCDigiCollection]: size "<<ltcdigis->size()<<endl;
  for (std::vector<LTCDigi>::const_iterator ltc_it =
	 ltcdigis->begin(); ltc_it != ltcdigis->end(); ltc_it++){
    for (int i = 0; i < 6; i++){
      if((*ltc_it).HasTriggered(i)) TrigBits[i]++; //save the trig bit info
      //      cout<<"[LTCDigi]: bit "<<i <<" has Trigger? "<<(*ltc_it).HasTriggered(i)<<endl;
    }
    if ((*ltc_it).HasTriggered(0)) {
       DTtrig = 1;
    }
    if ((*ltc_it).HasTriggered(1)) {
      CSCtrig = 1;
    }
    if (!(*ltc_it).HasTriggered(0)){
      NoDTtrig = 1;
    }
    for (int i = 1; i < 6; i++){
      othertrig += int((*ltc_it).HasTriggered(i));
    }
    if ((*ltc_it).HasTriggered(0) && othertrig == 0){
      DTOnlytrig = 1;
    }
  }
  

  edm::Handle<reco::TrackCollection> trackCollection;
  edm::Handle<TrajectorySeedCollection> seedcoll;
  if(!(conf_.getParameter<bool>("MTCCtrack"))){
    edm::LogInfo("AnalyzeTracksClusters")<<"Analyze standard tracks ";
    std::string src=conf_.getParameter<std::string>( "src" );
    e.getByLabel(src, trackCollection);
  } else {
    edm::LogInfo("AnalyzeTracksClusters")<<"Analyze MTCC tracks ";
    e.getByType(seedcoll);
    LogDebug("AnalyzeTracksClusters::analyze")<<"Getting used rechit";
    e.getByType(trackCollection);
  }

  const reco::TrackCollection *tracks=trackCollection.product();
  anglefinder_->init(e,es);
  reco::TrackCollection::const_iterator tciter;

  Ntk=0;

  if(tracks->size()>0){

    for(tciter=tracks->begin();tciter!=tracks->end();tciter++)
      {
	itk++;
	p_tk[Ntk]=tciter->outerP();
	pt_tk[Ntk]=tciter->outerPt();
	eta_tk[Ntk]=tciter->outerEta();
	phi_tk[Ntk]=tciter->outerPhi();
	nhits_tk[Ntk]=tciter->found();
      
	TrajectorySeed seed = *(*seedcoll).begin();

	std::vector<std::pair<const TrackingRecHit *,float> > hitangle;

	hitangle=anglefinder_->findtrackangle(seed,*tciter);

	std::vector<std::pair<const TrackingRecHit *,float> >::iterator iter;

	int temp=0;


	// Loop on hits

	Nclu=0;
	Nclu_matched=0;

	for(iter=hitangle.begin();iter!=hitangle.end();iter++)
	  {

	    const TrackingRecHit* lo_hit;
	    lo_hit = iter->first;
	    temp++;
	    DetId detid = lo_hit->geographicalId();
	    Subid[Nclu]=detid.subdetId();
	    Clu_rawid[Nclu]=detid.rawId();
	    
	    int l=0;
	    switch(detid.subdetId())
	      {
	      case 3:
		l = TIBDetId(detid).layer();
		break;
	      case 5:
		l = TOBDetId(detid).layer();
		break;
	      case 6:
		l = TECDetId(detid).wheel();
		break;
	      default:
		cout << "WARNING!!! this detid does not belong to tracker" << endl;
	      }
	    Layer[Nclu]=l;

	    float angle=iter->second;
	    Clu_ang[Nclu]=angle;
	    
	    //check if it is a simple SiStripRecHit2D
	    if(const SiStripRecHit2D * singlehit = dynamic_cast<const SiStripRecHit2D *>(lo_hit))
	      {	  
		const edm::Ref<edm::DetSetVector<SiStripCluster>, SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster> > 
		  cluster=singlehit->cluster();
		const std::vector<uint16_t> amplitudes( cluster->amplitudes().begin(),
		                                        cluster->amplitudes().end());
		Clu_size[Nclu]=amplitudes.size();
		double cluscharge=0;
		for(size_t ia=0; ia<amplitudes.size();ia++)
		  {
		    cluscharge+=amplitudes[ia];             ;
		  }
		float barycenter=cluster->barycenter();
		int posi=cluster->firstStrip();
		Clu_bar[Nclu]=barycenter;
		Clu_1strip[Nclu]=posi;
		//		cout << " carica(ADC) "<< cluscharge << endl;
		Clu_ch[Nclu]=cluscharge;
		Nclu++;
	      }
	    //check if it is a matched SiStripMatchedRecHit2D
	    if(dynamic_cast<const SiStripMatchedRecHit2D *>(lo_hit))
	      {	  
		Nclu_matched++;
	      }
	  }// loop on hits
	Ntk++;
      } // end of loop on tracks



  }


  
  else edm::LogInfo("AnalyzeTracksClusters")<<"No track found in the event";


  // Get all the clusters in the event

  using namespace edm;
  edm::Handle< edm::DetSetVector<SiStripCluster> >  input;
  e.getByLabel("siStripClusters",input);
  edm::DetSetVector<SiStripCluster>::const_iterator DSViter=input->begin();
  Nclu_all=0;
  //  Nclu_all_matched=0;
  //  Nclu_all_st=0;
  //  Nclu_all_rphi=0;
  
  edm::ESHandle<TrackerGeometry> tkgeom;
  es.get<TrackerDigiGeometryRecord>().get( tkgeom );

  for (; DSViter!=input->end();DSViter++)
    {      
      float clusiz =0;
      
      for(edm::DetSet<SiStripCluster>::const_iterator ic = DSViter->data.begin(); ic!=DSViter->data.end(); ic++)
	{
 	  uint32_t detid = DSViter->id;
	  //	  const StripGeomDetUnit*_StripGeomDetUnit = dynamic_cast<const StripGeomDetUnit*>(tkgeom->idToDetUnit(DetId(detid)));
	  unsigned int subid=DetId(detid).subdetId();
	    int l=0;
	    switch(subid)
	      {
	      case 3:
		l = TIBDetId(detid).layer();
		break;
	      case 5:
		l = TOBDetId(detid).layer();
		break;
	      case 6:
		l = TECDetId(detid).wheel();
		break;
	      default:
		cout << "WARNING!!! this detid does not belong to tracker" << endl;
	      }
	    Layer_all[Nclu_all]=l;
	    Subid_all[Nclu_all]= DetId(detid).subdetId();
	    
	    Clu_rawid_all[Nclu_all]=DetId(detid).rawId();
	    clusiz = ic->amplitudes().size();
	    float barycenter=ic->barycenter();
	    float Signal=0;
	    int posi=ic->firstStrip();
	    
	    const std::vector<uint16_t> amplitudes( ic->amplitudes().begin(),
	                                            ic->amplitudes().end());
	    for(size_t i=0; i<amplitudes.size();i++)
	      {
		if (amplitudes[i]>0)
		  {
		    Signal+=amplitudes[i];
		  }
		
	      }
	    
	    Clu_ch_all[Nclu_all]= Signal;
	    Clu_size_all[Nclu_all]= clusiz;
	    Clu_bar_all[Nclu_all]=barycenter;
	    Clu_1strip_all[Nclu_all]=posi;
	    
	    Nclu_all++;
	    
	}
    }
  
  
  nt->Fill();
  
}
