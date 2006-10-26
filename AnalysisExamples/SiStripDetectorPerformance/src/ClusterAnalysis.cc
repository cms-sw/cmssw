#include "AnalysisExamples/SiStripDetectorPerformance/interface/ClusterAnalysis.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "sstream"

static const uint16_t _NUM_SISTRIP_SUBDET_ = 4;
static std::string SubDet[_NUM_SISTRIP_SUBDET_]={"TIB","TOB","TID","TEC"};

namespace cms{
  ClusterAnalysis::ClusterAnalysis(edm::ParameterSet const& conf): 
    conf_(conf),
    filename_(conf.getParameter<std::string>("fileName")), 
    psfilename_(conf.getParameter<std::string>("psfileName")), 
    psfiletype_(conf.getParameter<int32_t>("psfiletype")), 
    SiStripNoiseService_(conf),
    SiStripPedestalsService_(conf),
    Track_src_( conf.getParameter<edm::InputTag>( "Track_src" ) ),
    ClusterInfo_src_( conf.getParameter<edm::InputTag>( "ClusterInfo_src" ) ),
    Cluster_src_( conf.getParameter<edm::InputTag>( "Cluster_src" ) )
  {};

  ClusterAnalysis::~ClusterAnalysis(){};
  
  void ClusterAnalysis::beginJob( const edm::EventSetup& es ) {
    char name[128];
    
    fFile = new TFile(filename_.c_str(),"RECREATE");
    fFile->mkdir("BadStrips");
    fFile->mkdir("ClusterNoise");
    fFile->mkdir("ClusterSignal");
    fFile->mkdir("ClusterStoN");
    fFile->mkdir("ClusterEta");
    fFile->mkdir("ClusterWidth");
    fFile->mkdir("ClusterPos");
    fFile->mkdir("Tracks");
    
    fFile->cd();
    
    edm::ParameterSet Parameters;

    //Create histograms
    Hlist = new TObjArray();


    fFile->cd();fFile->cd("Tracks");
    Parameters =  conf_.getParameter<edm::ParameterSet>("TH1nTracks");
    Hlist->Add(new TH1F("nTracks","nTracks",
			Parameters.getParameter<int32_t>("Nbinx"),
			Parameters.getParameter<double>("xmin"),
			Parameters.getParameter<double>("xmax")
			)
	       );
    Parameters =  conf_.getParameter<edm::ParameterSet>("TH1nRecHits");
    Hlist->Add(new TH1F("nRecHits","nRecHits",
			Parameters.getParameter<int32_t>("Nbinx"),
			Parameters.getParameter<double>("xmin"),
			Parameters.getParameter<double>("xmax")
			)
	       );

    //get geom    
    es.get<TrackerDigiGeometryRecord>().get( tkgeom );
    edm::LogInfo("ClusterAnalysis") << "[ClusterAnalysis::beginJob] There are "<<tkgeom->detUnits().size() <<" detectors instantiated in the geometry" << std::endl;  

    es.get<SiStripDetCablingRcd>().get( SiStripDetCabling_ );

    
    // get list of active detectors from SiStripDetCabling - this will change and be taken from a SiStripDetControl object
    std::vector<uint32_t> vdetId_;
    SiStripDetCabling_->addActiveDetectorsRawIds(vdetId_);
    
    for (std::vector<uint32_t>::const_iterator detid_iter=vdetId_.begin();detid_iter!=vdetId_.end();detid_iter++){
      
      uint32_t detid = *detid_iter;
      
      if (detid < 1){
	edm::LogError("SiStripCondObjDisplay")<< "[SiStripCondObjDisplay::geometry_Vs_Cabling] invalid detid " << detid<< std::endl;
	continue;
      }
      const StripGeomDetUnit* _StripGeomDetUnit = dynamic_cast<const StripGeomDetUnit*>(tkgeom->idToDetUnit(DetId(detid)));
      if (_StripGeomDetUnit==0){
	edm::LogError("SiStripCondObjDisplay")<< "[SiStripCondObjDisplay::beginJob] the detID " << detid << " doesn't seem to belong to Tracker" << std::endl; 
	continue;
      }     
    

      //&&&&&&&&&&&&&&&&&
      // Insert here code to instantiate histos per detector
      //eg

//       unsigned int nstrips = _StripGeomDetUnit->specificTopology().nstrips();
      
//       sprintf(name,"Pedestals_%s_%d",_StripGeomDetUnit->type().name().c_str(),detid);
//       fFile->cd();fFile->cd("Pedestals");
//       _TH1F_PedestalsProfile_m[detid] = new TH1F(name,name,nstrips,-0.5,nstrips-0.5);


      //&&&&&&&&&&&&&&&&&
    
    }

    std::string flags[3] = {"onTrack","offTrack","All"};
    for (int j=0;j<3;j++){
      for (int i=0;i<_NUM_SISTRIP_SUBDET_;i++){

	sprintf(name,"cNoise_%s_%s",SubDet[i].c_str(),flags[j].c_str());
	Parameters =  conf_.getParameter<edm::ParameterSet>("TH1ClusterNoise");
	fFile->cd();fFile->cd("ClusterNoise");
	Hlist->Add(new TH1F(name,name,
			    Parameters.getParameter<int32_t>("Nbinx"),
			    Parameters.getParameter<double>("xmin"),
			    Parameters.getParameter<double>("xmax")
			    )
		   );

	sprintf(name,"cSignal_%s_%s",SubDet[i].c_str(),flags[j].c_str());
	Parameters =  conf_.getParameter<edm::ParameterSet>("TH1ClusterSignal");
	fFile->cd();fFile->cd("ClusterSignal");
	Hlist->Add(new TH1F(name,name,
			    Parameters.getParameter<int32_t>("Nbinx"),
			    Parameters.getParameter<double>("xmin"),
			    Parameters.getParameter<double>("xmax")
			    )
		   );

	sprintf(name,"cStoN_%s_%s",SubDet[i].c_str(),flags[j].c_str());
	Parameters =  conf_.getParameter<edm::ParameterSet>("TH1ClusterStoN");
	fFile->cd();fFile->cd("ClusterStoN");
	Hlist->Add(new TH1F(name,name,
			    Parameters.getParameter<int32_t>("Nbinx"),
			    Parameters.getParameter<double>("xmin"),
			    Parameters.getParameter<double>("xmax")
			    )
		   );

	sprintf(name,"cEta_%s_%s",SubDet[i].c_str(),flags[j].c_str());
	Parameters =  conf_.getParameter<edm::ParameterSet>("TH2ClusterEta");
	fFile->cd();fFile->cd("ClusterEta");
	Hlist->Add(new TH2F(name,name,
			    Parameters.getParameter<int32_t>("Nbinx"),
			    Parameters.getParameter<double>("xmin"),
			    Parameters.getParameter<double>("xmax"),
			    Parameters.getParameter<int32_t>("Nbiny"),
			    Parameters.getParameter<double>("ymin"),
			    Parameters.getParameter<double>("ymax")
			    )
		   );

	sprintf(name,"cWidth_%s_%s",SubDet[i].c_str(),flags[j].c_str());
	Parameters =  conf_.getParameter<edm::ParameterSet>("TH1ClusterWidth");
	fFile->cd();fFile->cd("ClusterWidth");
	Hlist->Add(new TH1F(name,name,
			    Parameters.getParameter<int32_t>("Nbinx"),
			    Parameters.getParameter<double>("xmin"),
			    Parameters.getParameter<double>("xmax")
			    )
		   );


	sprintf(name,"cPos_%s_%s",SubDet[i].c_str(),flags[j].c_str());
	Parameters =  conf_.getParameter<edm::ParameterSet>("TH1ClusterPos");
	fFile->cd();fFile->cd("ClusterPos");
	Hlist->Add(new TH1F(name,name,
			    Parameters.getParameter<int32_t>("Nbinx"),
			    Parameters.getParameter<double>("xmin"),
			    Parameters.getParameter<double>("xmax")
			    )
		   );


	//       sprintf(name,"BadStrips_Cumulative_%s_%s",SubDet[i].c_str(),flags[j].c_str());
	//       Parameters =  conf_.getParameter<edm::ParameterSet>("TH1BadStrips");
	//       fFile->cd();fFile->cd("BadStrips");
	//       _TH1F_BadStrips_v.push_back(new TH1F(name,name,
	// 					   Parameters.getParameter<int32_t>("Nbinx"),
	// 					   Parameters.getParameter<double>("xmin"),
	// 					   Parameters.getParameter<double>("xmax")
	// 					   )
	// 				  );
      
    
      }
    }
  }

  //------------------------------------------------------------------------------------------

  void ClusterAnalysis::endJob() {  
    edm::LogInfo("ClusterAnalysis") << "[ClusterAnalysis::endJob] >>> saving histograms" << std::endl;
    
    fFile->cd();
    
    
    edm::LogInfo("ClusterAnalysis")  << "... And now write on ps file " << psfiletype_ << std::endl;
    TPostScript ps(psfilename_.c_str(),psfiletype_);
    TCanvas Canvas("c","c");//("c","c",600,300);
    for (int ih=0; ih<Hlist->GetEntries();ih++){
      edm::LogInfo("ClusterAnalysis") << "Histos " << ih << " name " << (*Hlist)[ih]->GetName() << " title " <<  (*Hlist)[ih]->GetTitle() << std::endl;
      (*Hlist)[ih]->Draw();
      Canvas.Update();
      ps.NewPage();
    }
    ps.Close();

    fFile->ls();
    fFile->Write();
    fFile->Close();

  }

  //------------------------------------------------------------------------------------------

  void ClusterAnalysis::analyze(const edm::Event& e, const edm::EventSetup& es) {
    edm::LogInfo("ClusterAnalysis") << "[ClusterAnalysis::analyse]  " << "Run " << e.id().run() << " Event " << e.id().event() << std::endl;

    runNb   = e.id().run();
    eventNb = e.id().event();
    edm::LogInfo("ClusterAnalysis") << "Processing run " << runNb << " event " << eventNb << std::endl;

    SiStripNoiseService_.setESObjects(es);
    SiStripPedestalsService_.setESObjects(es);
    
    //Get input 
    e.getByLabel( ClusterInfo_src_, dsv_SiStripClusterInfo);
    e.getByLabel( Cluster_src_, dsv_SiStripCluster);    
    e.getByLabel(Track_src_, trackCollection);

    vPSiStripCluster.clear();
    countOn=0;
    countOff=0;
    countAll=0;

    //Perform track study
    trackStudy();
    
    std::stringstream ss;
    ss << "\nList of SiStripClusterPointer\n";
    for (std::vector<const SiStripCluster*>::iterator iter=vPSiStripCluster.begin();iter!=vPSiStripCluster.end();iter++)
      ss << *iter << "\n";    
    LogTrace("ClusterAnalysis") << "\n["<<__PRETTY_FUNCTION__<<"] \n vPSiStripCluster.size()=" << vPSiStripCluster.size()<< ss.str() << std::endl;	
    
    //Perform Cluster Study (irrespectively to tracks)
    AllClusters();

    if (countAll != countOn+countOff)
      edm::LogWarning("ClusterAnalysis") << "\n["<<__PRETTY_FUNCTION__<<"] Counts (on, off, all) do not match" << countOn << " " << countOff << " " << countAll; 

}

  //------------------------------------------------------------------------
  
  void ClusterAnalysis::trackStudy(){
    LogTrace("ClusterAnalysis") << "\n["<<__PRETTY_FUNCTION__<<"]" << std::endl;
    const reco::TrackCollection tC = *(trackCollection.product());

    int nTracks=tC.size();
 
    edm::LogInfo("ClusterAnalysis") << "Reconstructed "<< nTracks << " tracks" << std::endl ;
    ((TH1F*) Hlist->FindObject("nTracks"))->Fill(nTracks);

    int i=1;
    for (reco::TrackCollection::const_iterator track=tC.begin(); track!=tC.end(); track++){
      LogTrace("ClusterAnalysis")
	<< "Track number "<< i 
	<< "\n\tmomentum: " << track->momentum()
	<< "\n\tPT: " << track->pt()
	<< "\n\tvertex: " << track->vertex()
	<< "\n\timpact parameter: " << track->d0()
	<< "\n\tcharge: " << track->charge()
	<< "\n\tnormalizedChi2: " << track->normalizedChi2() 
	<<"\n\tFrom EXTRA : "
	<<"\n\t\touter PT "<< track->outerPt()<<std::endl;
      i++;
  
      //
      // try and access Hits
      //
      
      int recHitsSize=track->recHitsSize();
      edm::LogInfo("ClusterAnalysis") <<"\t\tNumber of RecHits "<<recHitsSize<<std::endl;
      ((TH1F*) Hlist->FindObject("nRecHits"))->Fill(recHitsSize);
  
      for (trackingRecHit_iterator it = track->recHitsBegin();  it != track->recHitsEnd(); it++){
	const TrackingRecHit* trh = &(**it);
	const uint32_t& detid = trh->geographicalId().rawId();

	if (trh->isValid()){
	  LogTrace("ClusterAnalysis")
	    <<"\n\t\tRecHit on det "<<trh->geographicalId().rawId()
	    <<"\n\t\tRecHit in LP "<<trh->localPosition()
	    <<"\n\t\tRecHit in GP "<<tkgeom->idToDet(trh->geographicalId())->surface().toGlobal(trh->localPosition()) <<std::endl;

	  //Get SiStripCluster from SiStripRecHit
	  const SiStripRecHit2D* hit=dynamic_cast<const SiStripRecHit2D*>(trh);
	  //const edm::Ref<edm::DetSetVector<SiStripCluster>, SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster> > clust=hit.cluster();
	  if ( hit != NULL ){
	    LogTrace("ClusterAnalysis") << "GOOD hit" << std::endl;
	    const SiStripCluster* SiStripCluster_ = &*(hit->cluster());
	    
	    const SiStripClusterInfo* SiStripClusterInfo_ = MatchClusterInfo(SiStripCluster_,detid);
	    clusterInfos(SiStripClusterInfo_,detid,"_onTrack");
	    vPSiStripCluster.push_back(SiStripCluster_);
	    countOn++;
	  }else{
	    LogTrace("ClusterAnalysis") << "NULL hit" << std::endl;
	  }
	  
	}else{
	  LogTrace("ClusterAnalysis") <<"\t\t Invalid Hit On "<<detid<<std::endl;
	}
      }
    }
  }

  //------------------------------------------------------------------------

  void ClusterAnalysis::AllClusters(){
    LogTrace("ClusterAnalysis") << "\n["<<__PRETTY_FUNCTION__<<"]" << std::endl;
    //Loop on Dets
    edm::DetSetVector<SiStripCluster>::const_iterator DSViter=dsv_SiStripCluster->begin();
    for (; DSViter!=dsv_SiStripCluster->end();DSViter++){
      uint32_t detid=DSViter->id;
      //Loop on Clusters
      LogTrace("ClusterAnalysis") << "\n["<<__PRETTY_FUNCTION__<<"] \n on detid "<< detid << " N Cluster= " << DSViter->data.size() <<std::endl;
      
      edm::DetSet<SiStripCluster>::const_iterator ClusIter = DSViter->data.begin();
      for(; ClusIter!=DSViter->data.end(); ClusIter++) {
	const SiStripClusterInfo* SiStripClusterInfo_=MatchClusterInfo(&*ClusIter,detid);
	clusterInfos(SiStripClusterInfo_, detid,"_All");
	countAll++;
	//LogTrace("ClusterAnalysis") << "\n["<<__PRETTY_FUNCTION__<<"] ClusIter " << &*ClusIter << 
	//  "\t " << std::find(vPSiStripCluster.begin(),vPSiStripCluster.end(),&*ClusIter)-vPSiStripCluster.begin() << std::endl;
	if (std::find(vPSiStripCluster.begin(),vPSiStripCluster.end(),&*ClusIter) == vPSiStripCluster.end()){
	  clusterInfos(SiStripClusterInfo_,detid,"_offTrack");
	  countOff++;
	}
      }       
    }
  }
  
  //------------------------------------------------------------------------

  const SiStripClusterInfo* ClusterAnalysis::MatchClusterInfo(const SiStripCluster* cluster, const uint32_t& detid){
    LogTrace("ClusterAnalysis") << "\n["<<__PRETTY_FUNCTION__<<"]" << std::endl;
    edm::DetSetVector<SiStripClusterInfo>::const_iterator DSViter = dsv_SiStripClusterInfo->find(detid);
    edm::DetSet<SiStripClusterInfo>::const_iterator ClusIter = DSViter->data.begin();
    for(; ClusIter!=DSViter->data.end(); ClusIter++) {
      if ( 
	  (ClusIter->firstStrip() == cluster->firstStrip())
	  &&
	  (ClusIter->stripAmplitudes().size() == cluster->amplitudes().size())
	  )
	return &(*ClusIter);
    }
    return 0;
  }


  //------------------------------------------------------------------------

//   void ClusterAnalysis::AllClusters(){
//     LogTrace("ClusterAnalysis") << "\n["<<__PRETTY_FUNCTION__<<"]" << std::endl;
//     //Loop on Dets
//     edm::DetSetVector<SiStripCluster>::const_iterator DSViter=dsv_SiStripCluster->begin();
//     for (; DSViter!=dsv_SiStripClusterInfo->end();DSViter++){
//       //Loop on Clusters
//       edm::DetSet<SiStripClusterInfo>::const_iterator ClusIter = DSViter->data.begin();
//       for(; ClusIter!=DSViter->data.end(); ClusIter++) {
// 	clusterInfos(&*ClusIter, DSViter->id,"_All");
//       }       
//     }
//   }

  //------------------------------------------------------------------------

  void ClusterAnalysis::clusterInfos(const SiStripClusterInfo* cluster, const uint32_t& detid,TString flag){

    const StripGeomDetUnit*_StripGeomDetUnit = dynamic_cast<const StripGeomDetUnit*>(tkgeom->idToDetUnit(DetId(detid)));

    //GeomDetEnumerators::SubDetector SubDet_enum=_StripGeomDetUnit->specificType().subDetector();
    int SubDet_enum=_StripGeomDetUnit->specificType().subDetector() -2;

    LogTrace("ClusterAnalysis") 
      << "\n["<<__PRETTY_FUNCTION__<<"]"
      << "\n\t\tcluster detid "       << cluster->geographicalId()  
      << "\n\t\tcluster first strip " << cluster->firstStrip()
      << "\n\t\tcluster charge "      << cluster->charge()    
      << "\n\t\tcluster noise "       << cluster->noise()     
      << "\n\t\tcluster position "    << cluster->position()  
      << "\n\t\tcluster width "       << cluster->width()     
      << "\n\t\tcluster maxCharge "   << cluster->maxCharge() 
      << "\n\t\tcluster maxPos "      << cluster->maxPos()       
      << "\n\t\tcluster chargeL "     << cluster->chargeL()      
      << "\n\t\tcluster chargeR "     << cluster->chargeR()      
      << std::endl;

      ((TH1F*) Hlist->FindObject("cSignal_"+TString(SubDet[SubDet_enum])+flag))
	->Fill(cluster->charge());
      
      ((TH1F*) Hlist->FindObject("cNoise_"+TString(SubDet[SubDet_enum])+flag))
	->Fill(cluster->noise());

      if (cluster->noise()){
	((TH1F*) Hlist->FindObject("cStoN_"+TString(SubDet[SubDet_enum])+flag))
	  ->Fill(cluster->charge()/cluster->noise());
      }
      
      ((TH1F*) Hlist->FindObject("cWidth_" +TString(SubDet[SubDet_enum])+flag))
	->Fill(cluster->width());

      ((TH2F*) Hlist->FindObject("cEta_"   +TString(SubDet[SubDet_enum])+flag))
	->Fill((cluster->chargeL()-cluster->chargeR())/cluster->charge(),(cluster->chargeL()+cluster->chargeR())/cluster->charge());
  }
}


