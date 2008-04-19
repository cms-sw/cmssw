/*
 * $Date: 2008/03/04 13:44:35 $
 * $Revision: 1.14 $
 *
 * \author: D. Giordano, domenico.giordano@cern.ch
 * Modified: M.De Mattia 2/3/2007 & R.Castello 5/4/2007 & Susy Borgia 15/11/07
 */

#include "AnalysisExamples/SiStripDetectorPerformance/plugins/ClusterAnalysis.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"


#include "AnalysisDataFormats/TrackInfo/src/TrackInfo.cc"

#include "sstream"

#include "TH3S.h"
#include "TProfile.h"
#include "TCanvas.h"
#include "TPostScript.h"

static const uint8_t _NUM_SISTRIP_SUBDET_ = 4;
static TString SubDet[_NUM_SISTRIP_SUBDET_]={"_TIB","_TOB","_TID","_TEC"};
static TString flags[3] = {"_onTrack","_offTrack","_All"};
static TString width_flags[5] = {"","_width_1","_width_2","_width_3","_width_ge_4"};


namespace cms{
  ClusterAnalysis::ClusterAnalysis(edm::ParameterSet const& conf): 
    conf_(conf),
    psfilename_(conf.getParameter<std::string>("psfileName")), 
    psfiletype_(conf.getParameter<int32_t>("psfiletype")),
    psfilemode_(conf.getUntrackedParameter<int32_t>("psfilemode",1)),
    Filter_src_( conf.getParameter<edm::InputTag>( "Filter_src" ) ),
    Track_src_( conf.getParameter<edm::InputTag>( "Track_src" ) ),
    Cluster_src_( conf.getParameter<edm::InputTag>( "Cluster_src" ) ),
    ModulesToBeExcluded_(conf.getParameter< std::vector<uint32_t> >("ModulesToBeExcluded")),
    EtaAlgo_(conf.getParameter<int32_t>("EtaAlgo")),
    NeighStrips_(conf.getParameter<int32_t>("NeighStrips")),
    not_the_first_event(false),
    tracksCollection_in_EventTree(true),
    trackAssociatorCollection_in_EventTree(true)
  {
  }

  ClusterAnalysis::~ClusterAnalysis(){
    std::cout << "Destructing object" << std::endl;
    //    delete Hlist;
  }
  
  void ClusterAnalysis::beginRun(const edm::Run& run, const edm::EventSetup& es ) {

    //get geom    
    es.get<TrackerDigiGeometryRecord>().get( tkgeom );
    edm::LogInfo("ClusterAnalysis") << "[ClusterAnalysis::beginJob] There are "<<tkgeom->detUnits().size() <<" detectors instantiated in the geometry" << std::endl;  

    es.get<SiStripDetCablingRcd>().get( SiStripDetCabling_ );

    es.get<SiStripQualityRcd>().get(SiStripQuality_);

    book();
  }

  void ClusterAnalysis::book() {
    
    TFileDirectory ClusterNoise = fFile->mkdir( "ClusterNoise" );
    TFileDirectory ClusterSignal = fFile->mkdir("ClusterSignal");
    TFileDirectory ClusterStoN = fFile->mkdir("ClusterStoN");
    TFileDirectory ClusterEta = fFile->mkdir("ClusterEta");
    TFileDirectory ClusterWidth = fFile->mkdir("ClusterWidth");
    TFileDirectory ClusterPos = fFile->mkdir("ClusterPos");
    TFileDirectory Tracks = fFile->mkdir("Tracks");
    TFileDirectory Layer = fFile->mkdir("Layer");

    const edm::ParameterSet mapSet = conf_.getParameter<edm::ParameterSet>("MapFlag");
    if( mapSet.getParameter<bool>("Map_ClusOccOn") ){
      tkMap_ClusOcc[0]=new TrackerMap( "ClusterOccupancy_onTrack" );
      tkMap_ClusOcc[1]=new TrackerMap( "ClusterOccupancy_offTrack" );
      tkMap_ClusOcc[2]=new TrackerMap( "ClusterOccupancy_All" );
    }  
    if( mapSet.getParameter<bool>("Map_InvHit") ) 
      tkInvHit=new TrackerMap("Invalid_Hit");

    //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // get list of active detectors from SiStripDetCabling 

    std::vector<uint32_t> vdetId_;
    SiStripDetCabling_->addActiveDetectorsRawIds(vdetId_);
    //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    //Create histograms
    Hlist = new THashList();

    //Display 3D
    name = "ClusterGlobalPos";
    bookHlist("TH3","TH3ClusterGlobalPos",ClusterPos, name, "z (cm)","x (cm)","y (cm)"); 

    //&&&&&&&&&&&&&&&&&&&&&&&&

    // bookHlist(TObjArray, Name of the parameterset in the cfg, name of the TFileDirectory, name of the hitsogram, xbinnumber, xmin, xmax )

    name = "nTracks";
    bookHlist("TH1","TH1nTracks", Tracks, name, "N Tracks" );
    name = "nRecHits";
    bookHlist("TH1","TH1nRecHits",Tracks,  name, "N RecHits" );

    // Loop on onTrack, offTrack, All
    for (int j=0;j<3;j++){      
      //Number of Cluster 
      name="nClusters"+flags[j];
      bookHlist("TH1","TH1nClusters",Tracks, name, "N Clusters" );

      for (int i=0;i<_NUM_SISTRIP_SUBDET_;i++) {
   	//Number of Cluster on each det
	name="nClusters"+SubDet[i]+flags[j];
	bookHlist("TH1","TH1nClusters",Tracks, name, "N Clusters" );
      }
    }

    // Loop on onTrack, offTrack, All
    for (int j=0;j<3;j++){
      //Histos for detector type
      for (int i=0;i<_NUM_SISTRIP_SUBDET_;i++){
	
    	TString appString=SubDet[i]+flags[j];

	//Cluster Width
    	name="cWidth"+appString;
    	bookHlist("TH1","TH1ClusterWidth",ClusterWidth, name, "Nstrip" );

    	//Loop for cluster width
    	for (int iw=0;iw<5;iw++){
	  
    	  appString=SubDet[i]+flags[j]+width_flags[iw];
	
     	  //Cluster Noise
     	  name="cNoise"+appString;
     	  bookHlist("TH1","TH1ClusterNoise",ClusterNoise, name, "ADC count" );

     	  //Cluster Signal
     	  name="cSignal"+appString;
    	  bookHlist("TH1","TH1ClusterSignal",ClusterSignal, name, "ADC count" );	  	 

	  //Cluster Signal corrected
	  if(j==0 && iw==0 ){
	    name="cSignalCorr"+appString;
	    bookHlist("TH1","TH1ClusterSignalCorr",ClusterSignal, name, "ADC count" );  
	  }
	  
     	  //Cluster StoN
     	  name="cStoN"+appString;
     	  bookHlist("TH1","TH1ClusterStoN",ClusterStoN, name );

	  //Cluster SignaltoNoise corrected
	  if(j==0 && iw==0 ){	     
	    name="cStoNCorr"+appString;
	    bookHlist("TH1","TH1ClusterStoNCorr",ClusterStoN, name );  
	  }

     	  //Cluster Position
     	  name="cPos"+appString;
     	  bookHlist("TH1","TH1ClusterPos",ClusterPos, name, "strip Num" );

	  //Cluster StoN Vs Cluster Position
	  name="cStoNVsPos"+appString;
     	  bookHlist("TH2","TH2ClusterStoNVsPos",ClusterPos, name, "strip Num");

     	  //Cluster Charge Division (only for study on Raw Data Runs)
     	  name="cEta"+appString;
     	  bookHlist("TH1","TH1ClusterEta",ClusterEta, name, "" );

     	  name="cEta_scatter"+appString;
     	  bookHlist("TH2","TH2ClusterEta",ClusterEta, name, "" , "");
     	}//end loop on width 

	//cWidth Vs Angle
	name = "ClusterWidthVsAngle"+appString;
	bookHlist("TProfile","TProfileWidthAngle",ClusterWidth, name, "cos(angle_xz)", "clusWidth");

	//Residual Vs Angle
	name = "ResidualVsAngle"+appString;
	bookHlist("TProfile","TProfileResidualAngle",ClusterWidth, name, "Angle" , "Residual");
      
      } //end loop on det type 
    }//end loop on onTrack,offTrack,all


    //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    //Detector Detail Plots

    //Histos for each detector
    for (std::vector<uint32_t>::const_iterator detid_iter=vdetId_.begin();detid_iter!=vdetId_.end();detid_iter++){  
      uint32_t detid = *detid_iter;
      
      if (detid < 1){
	edm::LogError("ClusterAnalysis")<< "[" <<__PRETTY_FUNCTION__ << "] invalid detid " << detid<< std::endl;
	continue;
      }
      const StripGeomDetUnit* _StripGeomDetUnit = dynamic_cast<const StripGeomDetUnit*>(tkgeom->idToDetUnit(DetId(detid)));
      if (_StripGeomDetUnit==0){
	edm::LogError("SiStripCondObjDisplay")<< "[SiStripCondObjDisplay::beginJob] the detID " << detid << " doesn't seem to belong to Tracker" << std::endl; 
	continue;
      }
      
      
      //&&&&&&&&&&&&&&&&&
      // Insert here code to instantiate histos per detector
      
      unsigned int nstrips = _StripGeomDetUnit->specificTopology().nstrips();
      
      //      edm::LogError("ClusterAnalysis") << " Detid " << detid << " SubDet " << GetSubDetAndLayer(detid).first << " Layer " << GetSubDetAndLayer(detid).second << std::endl;   
      if (DetectedLayers.find(GetSubDetAndLayer(detid)) == DetectedLayers.end()){

	DetectedLayers[GetSubDetAndLayer(detid)]=true;
      }
 
      //&&&&&&&&&&&&&&
      // Retrieve information for the module
      //&&&&&&&&&&&&&&&&&&     
      char cdetid[128];
      sprintf(cdetid,"%d",detid);
      char aname[128];
      sprintf(aname,"%s_%d",_StripGeomDetUnit->type().name().c_str(),detid);
      char SubStr[128];

      SiStripDetId a(detid);
      if ( a.subdetId() == 3 ){
	TIBDetId b(detid);
	sprintf(SubStr,"_SingleDet_%d_TIB_%d_%d_%d_%d",detid,b.layer(),b.string()[0],b.string()[1],b.glued());
      } else if ( a.subdetId() == 4 ) {
	TIDDetId b(detid);
	sprintf(SubStr,"_SingleDet_%d_TID_%d_%d_%d_%d",detid,b.wheel(),b.ring(),b.side(),b.glued());
      } else if ( a.subdetId() == 5 ) {
	TOBDetId b(detid);
	sprintf(SubStr,"_SingleDet_%d_TOB_%d_%d_%d_%d",detid,b.layer(),b.rod()[0],b.rod()[1],b.glued());
      } else if ( a.subdetId() == 6 ) {
	TECDetId b(detid);
	sprintf(SubStr,"_SingleDet_%d_TEC_%d_%d_%d_%d_%d",detid,b.wheel(),b.ring(),b.side(),b.glued(),b.stereo());
      }
      
      TString appString=TString(SubStr);//+"_"+cdetid;

      TFileDirectory detid_dir = fFile->mkdir( cdetid );     

      //Cluster Noise
      name="cNoise"+appString;
      bookHlist("TH1","TH1ClusterNoise",detid_dir, name, "ADC count" );

      //Cluster Signal
      name="cSignal"+appString;
      bookHlist("TH1","TH1ClusterSignal",detid_dir, name, "ADC count" );

      //Cluster StoN
      name="cStoN"+appString;
      bookHlist("TH1","TH1ClusterStoN",detid_dir, name, "" );

      //Cluster Signal x Fiber
      name="cSignalxFiber"+appString+"_onTrack";
      bookHlist("TProfile","TProfileSignalxFiber",detid_dir, name, "ApvPair", "ADC count" );

      //Cluster Width
      name="cWidth"+appString;
      bookHlist("TH1","TH1ClusterWidth",detid_dir, name, "Nstrip" );

      //Cluster Position
      name="cPos"+appString;
      //bookHlist("TH1","TH1ClusterPos", name, "Nbinx", "xmin", "xmax" );
      Hlist->Add(new TH1F(name,name,nstrips,0,nstrips));
		
      //Cluster StoN Vs Cluster Position
      name="cStoNVsPos"+appString;
      char labeln[128];
      sprintf(labeln,"strip Num (Ntot=%d)",nstrips);
      bookHlist("TH2","TH2ClusterStoNVsPos",detid_dir, name, labeln);
 
      //Cluster Charge Division (only for study on Raw Data Runs)
      name="cEta"+appString;
      bookHlist("TH1","TH1ClusterEta", detid_dir,name, "" );

      name="cEta_scatter"+appString;
      bookHlist("TH2","TH2ClusterEta", detid_dir,name, "" ,  "" );

      //      const StripGeomDetUnit* _StripGeomDetUnit = dynamic_cast<const StripGeomDetUnit*>(tkgeom->idToDetUnit(DetId(detid)));
      //      if (_StripGeomDetUnit==0){
      //	continue;
      //      }
          
      //      unsigned int nstrips = _StripGeomDetUnit->specificTopology().nstrips();

      name="DBPedestals"+appString;
      TH1F* pPed = detid_dir.make<TH1F>(name,name,nstrips,-0.5,nstrips-0.5);
      Hlist->Add(pPed);
    
      name="DBNoise"+appString;
      TH1F* pNoi = detid_dir.make<TH1F>(name,name,nstrips,-0.5,nstrips-0.5);
      Hlist->Add(pNoi);
    
      name="DBBadStrips"+appString;
      TH1F* pBad = detid_dir.make<TH1F>(name,name,nstrips,-0.5,nstrips-0.5);  
      Hlist->Add(pBad);

      //&&&&&&&&&&&&&&&&&
    }//end loop on detector	


    //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    //Layer Detail Plots
    
    for (std::map<std::pair<std::string,uint32_t>,bool>::const_iterator iter=DetectedLayers.begin(); iter!=DetectedLayers.end();iter++){
     
      char cApp[64];
      sprintf(cApp,"_Layer_%d",iter->first.second);
      TFileDirectory Layer = fFile->mkdir("Layer");
      // Loop on onTrack, offTrack, All
      for (int j=0;j<3;j++){
	
	TString appString="_"+TString(iter->first.first)+cApp+flags[j];
     
	//Cluster Noise
	name="cNoise"+appString;
	bookHlist("TH1","TH1ClusterNoise", Layer,name, "ADC count" );

	//Cluster Signal
	name="cSignal"+appString;
	bookHlist("TH1","TH1ClusterSignal",Layer, name, "ADC count" );
	
	//Cluster Signal corrected
	if(j==0){
	  name="cSignalCorr"+appString;
	  bookHlist("TH1","TH1ClusterSignalCorr",Layer, name, "ADC count" );

	  //Cluster Signal Vs Angle
	  name = "cSignalVsAngle"+appString;
	  bookHlist("TProfile","TProfilecSignalVsAngle",Layer, name, "cos(angle_rz)" , "ADC count");
	  name = "cSignalVsAngleH"+appString;
	  bookHlist("TH2","TH2cSignalVsAngle",Layer, name, "cos(angle_rz)" , "ADC count");
	}

	//Cluster StoN
	name="cStoN"+appString;
	bookHlist("TH1","TH1ClusterStoN",Layer, name, "" );
	
	//Cluster SignaltoNoise corrected
	if(j==0){
	  name="cStoNCorr"+appString;
	  bookHlist("TH1","TH1ClusterStoNCorr",Layer, name, "" );
	}

	//Cluster Width
	name="cWidth"+appString;
	bookHlist("TH1","TH1ClusterWidth",Layer, name, "Nstrip" );

	//Cluster Position
	name="cPos"+appString;
	bookHlist("TH1","TH1ClusterPos",Layer, name, "strip Num" );

	//Cluster StoN Vs Cluster Position
	name="cStoNVsPos"+appString;
	bookHlist("TH2","TH2ClusterStoNVsPos",Layer, name, "strip Num");

	//residual
	name="res_x"+appString;
	bookHlist("TH1","TH1Residual_x",Layer, name, "" );
	
	//residual y
	name="res_y"+appString;
	bookHlist("TH1","TH1Residual_y",Layer, name, "" );
      
	//cWidth Vs Angle
	if(j==0){
	  name = "ClusterWidthVsAngle"+appString;
	  bookHlist("TProfile","TProfileWidthAngle",Layer, name, "angle_xz" , "clusWidth");
	
	  //Residuals Vs Angle
	  name = "ResidualVsAngle"+appString;
	  bookHlist("TProfile","TProfileResidualAngle",Layer, name, "cos(angle_rz)" , "Residual");

	  //Angle Vs phi
	  name = "AngleVsPhi"+appString;
	  bookHlist("TProfile","TProfileAngleVsPhi",Layer, name, "Phi (deg)" , "Impact angle angle_xz (deg)");
	}
      }
    }
  }

  //------------------------------------------------------------------------------------------

  void ClusterAnalysis::endJob() {  
    edm::LogInfo("ClusterAnalysis") << "[ClusterAnalysis::endJob] >>> saving histograms" << std::endl;
    
    if (psfilemode_>0){    
      
      edm::LogInfo("ClusterAnalysis")  << "... And now write on ps file " << psfiletype_ << std::endl;
      TPostScript ps(psfilename_.c_str(),psfiletype_);
      TCanvas Canvas("c","c");//("c","c",600,300);
      TIter lIter(Hlist);
      TObject *th = (TObject*)lIter.Next(); 
      while (th!=NULL){
	if (psfilemode_>1 || (strstr(th->GetName(),"SingleDet_")==NULL && strstr(th->GetName(),"_width_")==NULL)){  
	  edm::LogInfo("ClusterAnalysis") << "Histos name " << th->GetName() << " title " <<  th->GetTitle() << std::endl;
	  if (dynamic_cast<TH1F*>(th) !=NULL){
	    if (dynamic_cast<TH1F*>(th)->GetEntries() != 0)
	      th->Draw();
	  }
	  if (dynamic_cast<TH2F*>(th) !=NULL){
	    if (dynamic_cast<TH2F*>(th)->GetEntries() != 0)
	      th->Draw();
	  }
	  else if (dynamic_cast<TProfile*>(th) !=NULL){
	    if (dynamic_cast<TProfile*>(th)->GetEntries() != 0)
	      th->Draw();
	  }   
	  else if (dynamic_cast<TH3S*>(th) !=NULL){
	    if (dynamic_cast<TH3S*>(th)->GetEntries() != 0)
	      th->Draw();
	  }
	  Canvas.Update();
	  ps.NewPage();
	}
	
      th = (TObject*)lIter.Next(); 
      }      
      ps.Close();      
    }    

    //TkMap->Save() and Print()

    const edm::ParameterSet mapSett = conf_.getParameter<edm::ParameterSet>("MapFlag");
    if( mapSett.getParameter<bool>("Map_ClusOccOn") ){
      tkMap_ClusOcc[0]->save(true,0,0,"ClusterOccupancyMap_onTrack.png");
      tkMap_ClusOcc[1]->save(true,0,0,"ClusterOccupancyMap_offTrack.png");
      tkMap_ClusOcc[2]->save(true,0,0,"ClusterOccupancyMap_All.png");
      
      tkMap_ClusOcc[0]->print(true,0,0,"ClusterOccupancyMap_onTrack");   
      tkMap_ClusOcc[1]->print(true,0,0,"ClusterOccupancyMap_offTrack");   
      tkMap_ClusOcc[2]->print(true,0,0,"ClusterOccupancyMap_All");
    }

    if( mapSett.getParameter<bool>("Map_InvHit")) {
      tkInvHit->save(true,0,0,"Inv_Hit_map.png");
      tkInvHit->print(true,0,0,"Inv_Hit_map");
    }
    
  }  
  //------------------------------------------------------------------------------------------

  void ClusterAnalysis::analyze(const edm::Event& e, const edm::EventSetup& es) {

    tracksCollection_in_EventTree=true;
    trackAssociatorCollection_in_EventTree=true;
    
    LogTrace("ClusterAnalysis") << "[ClusterAnalysis::analyse]  " << "Run " << e.id().run() << " Event " << e.id().event() << std::endl;
    runNb   = e.id().run();
    eventNb = e.id().event();
    edm::LogInfo("ClusterAnalysis") << "Processing run " << runNb << " event " << eventNb << std::endl;

    es.get<SiStripDetCablingRcd>().get( SiStripDetCabling_ );

    //get Pedestal and Noise  ES handle
    es.get<SiStripPedestalsRcd>().get(pedestalHandle);
    es.get<SiStripNoisesRcd>().get(noiseHandle);

    if (!not_the_first_event){
      fillPedNoiseFromDB();
      not_the_first_event=true;
    }
    
    //Get input
    e.getByLabel( Cluster_src_, dsv_SiStripCluster);    
    
    e.getByLabel(Track_src_, trackCollection);
    if(trackCollection.isValid()){
    }else{
      edm::LogError("ClusterAnalysis")<<"trackCollection not found "<<std::endl;
      tracksCollection_in_EventTree=false;
    }
    
    edm::InputTag TkiTag = conf_.getParameter<edm::InputTag>( "TrackInfo" );  
    
    e.getByLabel(TkiTag,tkiTkAssCollection); 
    if(tkiTkAssCollection.isValid()){
    }else{
      edm::LogError("ClusterAnalysis")<<"trackInfo not found "<<std::endl;
      trackAssociatorCollection_in_EventTree=false;
    }
    vPSiStripCluster.clear();
    countOn=0;
    countOff=0;
    countAll=0;
    // istart=oXZHitAngle.size();  
  
    // get geometry to evaluate local angles
    edm::ESHandle<TrackerGeometry> estracker;
    es.get<TrackerDigiGeometryRecord>().get(estracker);
    _tracker=&(* estracker);

    //Perform track study
    if (tracksCollection_in_EventTree || trackAssociatorCollection_in_EventTree) {
      LogTrace("ClusterAnalysis") << "\n["<<__PRETTY_FUNCTION__<<"] \n Executing trackStudy for Event = " << eventNb << std::endl; 
      trackStudy(es);
    }
   
    std::stringstream ss;
    ss << "\nList of SiStripClusterPointer\n";
    for (std::vector<const SiStripCluster*>::iterator iter=vPSiStripCluster.begin();iter!=vPSiStripCluster.end();iter++)
      ss << *iter << "\n";    
    LogTrace("ClusterAnalysis") << "\n["<<__PRETTY_FUNCTION__<<"] \n vPSiStripCluster.size()=" << vPSiStripCluster.size()<< ss.str() << std::endl;	
    

    //Perform Cluster Study (irrespectively to tracks)
    AllClusters(es);

    if (countAll != countOn+countOff)
      edm::LogWarning("ClusterAnalysis") << "\n["<<__PRETTY_FUNCTION__<<"] Counts (on, off, all) do not match" << countOn << " " << countOff << " " << countAll; 

    for (int j=0;j<3;j++){
      int nTot=0;
      for (int i=0;i<4;i++){
	((TH1F*) Hlist->FindObject("nClusters"+SubDet[i]+flags[j]))
	  ->Fill(NClus[i][j]);
	nTot+=NClus[i][j];
	NClus[i][j]=0;
      }
      ((TH1F*) Hlist->FindObject("nClusters"+flags[j]))->Fill(nTot);
    }
  }
  
  //------------------------------------------------------------------------
  
  void ClusterAnalysis::trackStudy( const edm::EventSetup& es){
    LogTrace("ClusterAnalysis") << "\n["<<__PRETTY_FUNCTION__<<"]" << std::endl;
    
    const reco::TrackCollection tC = *(trackCollection.product());
    
    int nTracks=tC.size();
    
    edm::LogInfo("ClusterAnalysis") << "Reconstructed "<< nTracks << " tracks" << std::endl ;
    
    int i=0;
    
    edm::LogInfo("ClusterAnalysis") <<"\t\t TrackCollection size "<< trackCollection->size() << std::endl;
    if(trackCollection->size()==0){
      ((TH1F*) Hlist->FindObject("nTracks"))->Fill(trackCollection->size());
      edm::LogInfo("ClusterAnalysis") <<"\t\t TrackCollection size zero!! Histo filled with a "<< trackCollection->size() << std::endl;
    }
    
    for (unsigned int track=0;track<trackCollection->size();++track){
      reco::TrackRef trackref = reco::TrackRef(trackCollection, track);
      
      //get the ref to the trackinfo
      reco::TrackInfoRef trackinforef=(*tkiTkAssCollection.product())[trackref];
      // loop on all the track hits
      edm:: LogInfo("ClusterAnalysis")
	<< "Track number "<< i+1 
	<< "\n\tmomentum: " << trackref->momentum()
	<< "\n\tPT: " << trackref->pt()
	<< "\n\tvertex: " << trackref->vertex()
	<< "\n\timpact parameter: " << trackref->d0()
	<< "\n\tcharge: " << trackref->charge()
	<< "\n\tnormalizedChi2: " << trackref->normalizedChi2() 
	<<"\n\tFrom EXTRA : "
	<<"\n\t\touter PT "<< trackref->outerPt()<<std::endl;
      i++;
      
      int recHitsSize=trackref->recHitsSize();
      edm::LogInfo("ClusterAnalysis") <<"\t\tNumber of RecHits "<<recHitsSize<<std::endl;
      
      const edm::ParameterSet psett = conf_.getParameter<edm::ParameterSet>("TrackThresholds");
      if( trackref->normalizedChi2() < psett.getParameter<double>("maxChi2") && recHitsSize > psett.getParameter<double>("minRecHit") ){
	((TH1F*) Hlist->FindObject("nRecHits"))->Fill(recHitsSize);
	((TH1F*) Hlist->FindObject("nTracks"))->Fill(nTracks);
      }else{
	LogTrace("ClusterAnalysis") <<"\t\tSkipped track "<< i << std::endl;
	//continue;
      }
      
      //------------------------------RESIDUAL at the layer level ------------
      /*	  
		 LocalPoint stateposition= 
		 LocalPoint rechitposition= 
		 fillTH1( stateposition.x() - rechitposition.x(),"res_x"+appString,0);
		 fillTH1( stateposition.y() - rechitposition.y(),"res_y"+appString,0);
		 ((TProfile*) Hlist->FindObject("ResidualVsAngle"))->Fill(angle,stateposition.x()- rechitposition.x(),1);
		 ((TProfile*) Hlist->FindObject("ResidualVsAngle"+appString+"_onTrack"))->Fill(angle,stateposition.x()- rechitposition.x(),1);
		 
      */
      
      TrackingRecHitRef rechitref=trackref->recHit(2);
      
      const uint32_t& ndetid = rechitref->geographicalId().rawId();
      if (find(ModulesToBeExcluded_.begin(),ModulesToBeExcluded_.end(),ndetid)!=ModulesToBeExcluded_.end()){
	LogTrace("ClusterAnalysis") << "Modules Excluded" << std::endl;
	return;
      }	
      
      const edm::ParameterSet mappSet = conf_.getParameter<edm::ParameterSet>("MapFlag");
      if( mappSet.getParameter<bool>("Map_InvHit") ){
	edm::LogInfo("TrackInfoAnalyzerExample") << "RecHit type: " << rechitref->getType() << std::endl;  
	if(!rechitref->isValid()){
	  edm::LogInfo("TrackInfoAnalyzerExample") <<"The recHit is not valid"<< std::endl;
	  //inactive??      
	  if(rechitref->getType() == 2) {
	    edm::LogInfo("TrackInfoAnalyzerExample") << "inactive rechit found on detid " << ndetid << std::endl;
	    tkInvHit->fill(ndetid,1); 
	    tkInvHit->fill(ndetid+1,1);
	    tkInvHit->fill(ndetid+2,1);
	  }  
	}else{
	  edm::LogInfo("TrackInfoAnalyzerExample") <<"The recHit is valid"<< std::endl;
	}     
      }  
      reco::TrackInfo::TrajectoryInfo::const_iterator iter;

      for(iter=trackinforef->trajStateMap().begin();iter!=trackinforef->trajStateMap().end();iter++){
	
	//trajectory local direction and position on detector
	LocalVector statedirection=(trackinforef->stateOnDet(Updated,(*iter).first)->parameters()).momentum();
	LocalPoint  stateposition=(trackinforef->stateOnDet(Updated,(*iter).first)->parameters()).position();
       	
	std::stringstream ss;
	ss <<"LocalMomentum: "<<statedirection
	   <<"\nLocalPosition: "<<stateposition
	   << "\nLocal x-z plane angle: "<<atan2(statedirection.x(),statedirection.z());

	if(trackinforef->type((*iter).first)==Matched){ // get the direction for the components
	  
	  const SiStripMatchedRecHit2D* matchedhit=dynamic_cast<const SiStripMatchedRecHit2D*>(&(*(iter)->first));
	  if (matchedhit!=0){
	    ss<<"\nMatched recHit found"<< std::endl;	  
	    //mono side
	    statedirection= trackinforef->localTrackMomentumOnMono(Updated,(*iter).first);
	    if(statedirection.mag() != 0)	  RecHitInfo(es, matchedhit->monoHit(),statedirection,trackref);
	    //stereo side
	    statedirection= trackinforef->localTrackMomentumOnStereo(Updated,(*iter).first);
	    if(statedirection.mag() != 0)	  RecHitInfo(es, matchedhit->stereoHit(),statedirection,trackref);
	    ss<<"\nLocalMomentum (stereo): "<<trackinforef->localTrackMomentumOnStereo(Updated,(*iter).first);
	  }
	}
	else if (trackinforef->type((*iter).first)==Projected){//one should be 0
	  ss<<"\nProjected recHit found"<< std::endl;
	  const ProjectedSiStripRecHit2D* phit=dynamic_cast<const ProjectedSiStripRecHit2D*>(&(*(iter)->first));
	  if(phit!=0){
	    //mono side
	    statedirection= trackinforef->localTrackMomentumOnMono(Updated,(*iter).first);
	    if(statedirection.mag() != 0) RecHitInfo(es, &(phit->originalHit()),statedirection,trackref);
	    //stereo side
	    statedirection= trackinforef->localTrackMomentumOnStereo(Updated,(*iter).first);
	    if(statedirection.mag() != 0)  RecHitInfo(es, &(phit->originalHit()),statedirection,trackref);
	  }	
	  
	}
	else {
	  const SiStripRecHit2D* hit=dynamic_cast<const SiStripRecHit2D*>(&(*(iter)->first));
	  if(hit!=0){
	    ss<<"\nSingle recHit found"<< std::endl;	  
	    statedirection=(trackinforef->stateOnDet(Updated,(*iter).first)->parameters()).momentum();
	    if(statedirection.mag() != 0) RecHitInfo(es, hit,statedirection,trackref);

	  }
 	}
	LogTrace("TrackInfoAnalyzerExample") <<ss.str() << std::endl;
      }
    }
  }

  void ClusterAnalysis::RecHitInfo( const edm::EventSetup& es, const SiStripRecHit2D* tkrecHit, LocalVector LV,reco::TrackRef track_ref ){
    
    if(!tkrecHit->isValid()){
      LogTrace("ClusterAnalysis") <<"\t\t Invalid Hit " << std::endl;
      return;  
    }
    
    const uint32_t& detid = tkrecHit->geographicalId().rawId();
    if (find(ModulesToBeExcluded_.begin(),ModulesToBeExcluded_.end(),detid)!=ModulesToBeExcluded_.end()){
      LogTrace("ClusterAnalysis") << "Modules Excluded" << std::endl;
      return;
    }
    
    LogTrace("ClusterAnalysis")
      <<"\n\t\tRecHit on det "<<tkrecHit->geographicalId().rawId()
      <<"\n\t\tRecHit in LP "<<tkrecHit->localPosition()
      <<"\n\t\tRecHit in GP "<<tkgeom->idToDet(tkrecHit->geographicalId())->surface().toGlobal(tkrecHit->localPosition()) 
      <<"\n\t\tRecHit trackLocal vector "<<LV.x() << " " << LV.y() << " " << LV.z() <<std::endl; 
    
    //Get SiStripCluster from SiStripRecHit
    if ( tkrecHit != NULL ){
      LogTrace("ClusterAnalysis") << "GOOD hit" << std::endl;
      const SiStripCluster* SiStripCluster_ = &*(tkrecHit->cluster());
      //const SiStripClusterInfo* SiStripClusterInfo_ = MatchClusterInfo(SiStripCluster_,detid);
      
      const edm::ParameterSet pset = conf_.getParameter<edm::ParameterSet>("TrackThresholds");
      if( track_ref->normalizedChi2() < pset.getParameter<double>("maxChi2") && track_ref->recHitsSize() > pset.getParameter<double>("minRecHit") ){	  	    
	
	SiStripClusterInfo* clusterInfo = new SiStripClusterInfo( detid, *SiStripCluster_, es);

	if ( clusterInfos(clusterInfo,detid,"_onTrack", LV ) ) {
	  vPSiStripCluster.push_back(SiStripCluster_);
	  countOn++;
	}
	delete clusterInfo ; //CHECKME
      }
    }else{
      LogTrace("ClusterAnalysis") << "NULL hit" << std::endl;
    }	  
  }
  
  //------------------------------------------------------------------------
  
  void ClusterAnalysis::AllClusters( const edm::EventSetup& es){

    LogTrace("ClusterAnalysis") << "Executing AllClusters" << std::endl;
    //Loop on Dets
    edm::DetSetVector<SiStripCluster>::const_iterator DSViter=dsv_SiStripCluster->begin();
    for (; DSViter!=dsv_SiStripCluster->end();DSViter++){
      uint32_t detid=DSViter->id;

      if (find(ModulesToBeExcluded_.begin(),ModulesToBeExcluded_.end(),detid)!=ModulesToBeExcluded_.end())
	continue;
      
      //Loop on Clusters
      LogTrace("ClusterAnalysis") << "\n["<<__PRETTY_FUNCTION__<<"] \n on detid "<< detid << " N Cluster= " << DSViter->data.size() <<std::endl;
      
      edm::DetSet<SiStripCluster>::const_iterator ClusIter = DSViter->data.begin();
      for(; ClusIter!=DSViter->data.end(); ClusIter++) {

	//const SiStripClusterInfo* SiStripClusterInfo_=MatchClusterInfo(&*ClusIter,detid);
        
	SiStripClusterInfo* clusterInfo = new SiStripClusterInfo( detid, *ClusIter, es);

	if ( clusterInfos(clusterInfo,detid,"_All", LV) ){ 
	  countAll++;

	  LogTrace("ClusterAnalysis") << "\n["<<__PRETTY_FUNCTION__<<"] ClusIter " << &*ClusIter << 
	    "\t " << std::find(vPSiStripCluster.begin(),vPSiStripCluster.end(),&*ClusIter)-vPSiStripCluster.begin() << std::endl;

	  if (std::find(vPSiStripCluster.begin(),vPSiStripCluster.end(),&*ClusIter) == vPSiStripCluster.end()){
	    if ( clusterInfos(clusterInfo, detid,"_offTrack", LV) ) 
	      countOff++;
	  }
	}
	delete clusterInfo; // CHECKME
      }       
    }
  }
  
  //------------------------------------------------------------------------

 
  //------------------------------------------------------------------------
  
  bool ClusterAnalysis::clusterInfos( SiStripClusterInfo* clusterInfo, const uint32_t& detid,TString flag , const LocalVector LV ){
    LogTrace("ClusterAnalysis") << "\n["<<__PRETTY_FUNCTION__<<"]" << std::endl;
    
    if (clusterInfo==0) 
      return false;
   

    const  edm::ParameterSet ps_b = conf_.getParameter<edm::ParameterSet>("BadModuleStudies");
    if  ( ps_b.getParameter<bool>("Bad") ) {//it will perform Bad modules discrimination
      short n_Apv;
      switch((int)clusterInfo->getFirstStrip()/128){
      case 0:
	n_Apv=0;
	break;
      case 1:
	n_Apv=1;
	break;
      case 2:
	n_Apv=2;
	break;
      case 3:
	n_Apv=3;
	break;
      case 4:
	n_Apv=4;
	break;
      case 5:
	n_Apv=5;
	break;
      }
      if ( ps_b.getParameter<bool>("justGood") ){//it will analyze just good modules 
	LogTrace("SiStripQuality") << "Just good module selected " << std::endl;
	if(SiStripQuality_->IsModuleBad(detid)){
	  LogTrace("SiStripQuality") << "\n Excluding bad module " << detid << std::endl;
	  return false;
	}else if(SiStripQuality_->IsApvBad(detid, n_Apv)){
	  LogTrace("SiStripQuality") << "\n Excluding bad module and APV " << detid << n_Apv << std::endl;
	  return false;
	}
      }else{
	LogTrace("SiStripQuality") << "Just bad module selected " << std::endl;
	if(!SiStripQuality_->IsModuleBad(detid) || !SiStripQuality_->IsApvBad(detid, n_Apv)){
	  LogTrace("SiStripQuality") << "\n Skipping good module " << detid << std::endl;
	  return false;
	}
      }
    }

    const  edm::ParameterSet ps = conf_.getParameter<edm::ParameterSet>("clusterInfoConditions");
    if  ( ps.getParameter<bool>("On") 
	  &&
	  ( 
	   //CHECKME
	   clusterInfo->getCharge()/clusterInfo->getNoise() < ps.getParameter<double>("minStoN") 
	   ||
	   clusterInfo->getCharge()/clusterInfo->getNoise() > ps.getParameter<double>("maxStoN") 
	   ||
	   clusterInfo->getWidth() < ps.getParameter<double>("minWidth") 
	   ||
	   clusterInfo->getWidth() > ps.getParameter<double>("maxWidth") 
	   )
	  )
      return false;

    LogTrace("ClusterAnalysis") << "Executing clusterInfos for module " << detid << std::endl;

    const StripGeomDetUnit*_StripGeomDetUnit = dynamic_cast<const StripGeomDetUnit*>(tkgeom->idToDetUnit(DetId(detid)));
    //GeomDetEnumerators::SubDetector SubDet_enum=_StripGeomDetUnit->specificType().subDetector();
    int SubDet_enum=_StripGeomDetUnit->specificType().subDetector() -2;
    
    //&&&&&&&&&&&&&&&& GLOBAL POS &&&&&&&&&&&&&&&&&&&&&&&&
    const StripTopology &topol=(StripTopology&)_StripGeomDetUnit->topology();
    MeasurementPoint mp(clusterInfo->getPosition(),rnd.Uniform(-0.5,0.5));
    LocalPoint localPos = topol.localPosition(mp);
    GlobalPoint globalPos=(_StripGeomDetUnit->surface()).toGlobal(localPos);
    //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    
    //char cdetid[128];
    //sprintf(cdetid,"_%d",detid);
    int iflag;
    if (flag=="_onTrack")
      iflag=0;
    else if (flag=="_offTrack")
      iflag=1;
    else
      iflag=2;
    
    NClus[SubDet_enum][iflag]++;

    LogTrace("ClusterAnalysis") << "NClus on detid = " << detid << " " << flag << " is " << NClus[SubDet_enum][iflag] << std::endl;
    //    TrackerMap filling for each flag
    const edm::ParameterSet _mapSet = conf_.getParameter<edm::ParameterSet>("MapFlag");
    if( _mapSet.getParameter<bool>("Map_ClusOccOn") )
      tkMap_ClusOcc[iflag]->fill(detid,1);
 
 /* CHECKME
    std::stringstream ss;
    const_cast<SiStripClusterInfo*>(cluster)->print(ss);
    LogTrace("ClusterAnalysis") 
      << "\n["<<__PRETTY_FUNCTION__<<"]\n"
      << ss.str() 
      << "\n\t\tcluster LocalPos "     << localPos
      << "\n\t\tcluster GlobalPos "     << globalPos
      << std::endl;
*/
    long double tanXZ = -999;
    float cosRZ = -2;
    long double atanXZ = -200;
    LogTrace("ClusterAnalysis")<< "\n\tLV " << LV.x() << " " << LV.y() << " " << LV.z() << " " << LV.mag() << std::endl;
    if (LV.mag()!=0){
      double proj_yZ=_StripGeomDetUnit->surface().toGlobal(LocalVector(0.,1.,0.)).z();
      tanXZ= LV.x() /LV.z() * (-1) * proj_yZ/fabs(proj_yZ);
      cosRZ= fabs(LV.z())/LV.mag();
      atanXZ=atan(tanXZ)*189/Geom::pi();

      LogTrace("ClusterAnalysis")<< "\n\t tanXZ " << tanXZ << " cosRZ " << cosRZ << std::endl;
    }

    //Display
    ((TH3S*) Hlist->FindObject("ClusterGlobalPos"))
      ->Fill(globalPos.z(),globalPos.x(),globalPos.y());

    //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    //Cumulative Plots

    TString appString=SubDet[SubDet_enum]+flag;    
  
    fillTH1(clusterInfo->getCharge(),"cSignal"+appString,1,clusterInfo->getWidth());

    if (LV.mag()!=0){
      
      fillTH1(clusterInfo->getCharge()*cosRZ,"cSignalCorr"+appString,0); //Filled only for ontrack
      
      fillTProfile(atanXZ,clusterInfo->getWidth(),"ClusterWidthVsAngle"+appString,0); //Filled only for ontrack

      fillTH1((clusterInfo->getCharge()/clusterInfo->getNoise())*cosRZ,"cStoNCorr"+appString,0); //Filled only for ontrack
    } 
       
    fillTH1(clusterInfo->getNoise(),"cNoise"+appString,1,clusterInfo->getWidth());
    
    if (clusterInfo->getNoise()){
      fillTH1(clusterInfo->getCharge()/clusterInfo->getNoise(),"cStoN"+appString,1,clusterInfo->getWidth());
      
    }
      
    fillTH1(clusterInfo->getWidth(),"cWidth"+appString,0);

    fillTH1(clusterInfo->getPosition(),"cPos"+appString,1,clusterInfo->getWidth());

    fillTH2(clusterInfo->getPosition(),clusterInfo->getCharge()/clusterInfo->getNoise(),"cStoNVsPos"+appString,1,clusterInfo->getWidth());
   
   /*
    // to retrieve the (raw)digiamplitudesL and (raw)digiamplitudesR, one needs to provide
    // the rawdigicollection via e.g. a cfg-file. An example can be developped upon request
    // if needed by contacting the developper of SiStripClusterInfo(E.Delmeire)
     
    int neighbourStripNumber=3;
    std::string rawdigiProducer = "SiStripDigis"; 
    std::string rawdigiLabel = "VirginRaw"// "ProcessedRaw"
    edm::Handle< edm::DetSetVector<SiStripRawDigi> > rawDigiHandle;
    es.getByLabel(rawdigiProducer,rawdigiLabel,rawDigiHandle);
    
    vector<float> amplitudesL, amplitudesR;
    amplitudesL = clusterInfo->getRawDigiAmplitudesLR(neighbourStripNumber, 
                                                            *rawDigiHandle, 
                                                               dsv_SiStripCluster,
                                                              rawDigiLabel).first;
							      
    amplitudesR = clusterInfo->getRawDigiAmplitudesLR(neighbourStripNumber, 
                                                            *rawDigiHandle, 
                                                                dsv_SiStripCluster,
                                                              rawDigiLabel).second;
   					      
    if (amplitudesL.size()!=0 ||  amplitudesR.size()!=0){
	
      float Ql=0;
      float Qr=0;
      float Qt=0;

      if (EtaAlgo_==1){
	Ql=clusterInfo->getChargeLR().first;
	Qr=clusterInfo->getChargeLR().second;
      
  	for (std::vector<float>::const_iterator it=amplitudesL.begin(); it !=amplitudesL.end() && it-amplitudesL.begin()<NeighStrips_; it ++)
  	  { Ql += (*it);}

	
  	for (std::vector<float>::const_iterator it=amplitudesR.begin(); it !=amplitudesR.end() && it-amplitudesR.begin()<NeighStrips_; it ++)
  	  { Qr += (*it);}
	
	Qt=Ql+Qr+clusterInfo->getMaxCharge();
      }
      else{
	
	int Nstrip=clusterInfo->getStripAmplitudes().size();
	float pos=clusterInfo->getPosition()-0.5;
	for(int is=0;is<Nstrip && clusterInfo->getFirstStrip()+is<=pos;is++)
	  Ql+=clusterInfo->getStripAmplitudes()[is];
	
	Qr=clusterInfo->getCharge()-Ql;

  	for (std::vector<float>::const_iterator it=amplitudesL.begin(); it !=amplitudesL.end() && it-amplitudesL.begin()<NeighStrips_; it ++)
  	  { Ql += (*it);}
	
  	for (std::vector<float>::const_iterator it=amplitudesR.begin(); it !=amplitudesR.end() && it-amplitudesR.begin()<NeighStrips_; it ++)
  	  { Qr += (*it);}

	
	Qt=Ql+Qr;
      }
      
      LogTrace("ClusterAnalysis") << "\n["<<__PRETTY_FUNCTION__<<"] \n on detid "<< detid << " Ql=" << Ql << " Qr="<< Qr << " Qt="<<Qt<< " eta="<< Ql/Qt<< std::endl;
      
      fillTH1(Ql/Qt,"cEta"+appString,1,clusterInfo->getWidth());
    
      fillTH2((Ql-Qr)/Qt,(Ql+Qr)/Qt,"cEta_scatter"+appString,1,clusterInfo->getWidth());
      
    }
    */
    //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    //Detector Detail Plots    
    char aname[128];
    //sprintf(aname,"%s_%d",_StripGeomDetUnit->type().name().c_str(),detid);
    SiStripDetId a(detid);
    if ( a.subdetId() == 3 ){
      TIBDetId b(detid);
      sprintf(aname,"_SingleDet_%d_TIB_%d_%d_%d_%d",detid,b.layer(),b.string()[0],b.string()[1],b.glued());
    } else if ( a.subdetId() == 4 ) {
      TIDDetId b(detid);
      sprintf(aname,"_SingleDet_%d_TID_%d_%d_%d_%d",detid,b.wheel(),b.ring(),b.side(),b.glued());
    } else if ( a.subdetId() == 5 ) {
      TOBDetId b(detid);
      sprintf(aname,"_SingleDet_%d_TOB_%d_%d_%d_%d",detid,b.layer(),b.rod()[0],b.rod()[1],b.glued());
    } else if ( a.subdetId() == 6 ) {
      TECDetId b(detid);
      sprintf(aname,"_SingleDet_%d_TEC_%d_%d_%d_%d_%d",detid,b.wheel(),b.ring(),b.side(),b.glued(),b.stereo());
    }        
    appString=TString(aname);
        
    if(flag=="_All"){

      fillTH1(clusterInfo->getCharge(),"cSignal"+appString,0);

      fillTH1(clusterInfo->getNoise(),"cNoise"+appString,0);

      if (clusterInfo->getNoise()){
	fillTH1(clusterInfo->getCharge()/clusterInfo->getNoise(),"cStoN"+appString,0);
      }
      
      fillTH1(clusterInfo->getWidth(),"cWidth"+appString,0);

      fillTH1(clusterInfo->getPosition(),"cPos"+appString,0);

      fillTH2(clusterInfo->getPosition(),clusterInfo->getCharge()/clusterInfo->getNoise(),"cStoNVsPos"+appString,0);
    }

    if(flag=="_onTrack" && cosRZ>-2){
      fillTProfile((int)(clusterInfo->getPosition()-.5)/256,clusterInfo->getCharge()*cosRZ,"cSignalxFiber"+appString,0);
    }

    //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // Layer Detail Plots
    char cApp[64];
    sprintf(cApp,"_Layer_%d",GetSubDetAndLayer(detid).second);
    appString="_"+TString(GetSubDetAndLayer(detid).first)+cApp+flag;

    fillTH1(clusterInfo->getCharge(),"cSignal"+appString,0);
  
    fillTH1(clusterInfo->getNoise(),"cNoise"+appString,0);

    if (clusterInfo->getNoise()){
      fillTH1(clusterInfo->getCharge()/clusterInfo->getNoise(),"cStoN"+appString,0);
    }
      
    fillTH1(clusterInfo->getWidth(),"cWidth"+appString,0);
    
    fillTH1(clusterInfo->getPosition(),"cPos"+appString,0);      
    
    if(flag=="_onTrack" && LV.mag()!=0){
      fillTProfile(atanXZ,clusterInfo->getWidth(),"ClusterWidthVsAngle"+appString+"_onTrack",0);
      
      fillTH1(clusterInfo->getCharge()*cosRZ,"cSignalCorr"+appString,0);
      
      fillTProfile(cosRZ,clusterInfo->getCharge(),"cSignalVsAngle"+appString,0);
      fillTH2(cosRZ,clusterInfo->getCharge(),"cSignalVsAngleH"+appString,0);
      
      if (clusterInfo->getNoise()){
	
	fillTH1((clusterInfo->getCharge()/clusterInfo->getNoise())*cosRZ,"cStoNCorr"+appString,0); 
      }
      
      
      //***Only for TIB and TOB rphi modules***//      
      if ( StripSubdetector(detid).stereo() == 0 ){
	
	fillTProfile(
		     tkgeom->idToDet(DetId(detid))->surface().toGlobal(LV).phi().degrees(),
		     atanXZ,
		     "AngleVsPhi"+appString,0
		     );
	
	//std::cout << " detid phi sinxz " <<  appString << " R " << tkgeom->idToDet(DetId(detid))->surface().toGlobal(_HitDir._TrackingRecHit->localPosition()).mag() << "\t |  phi " << tkgeom->idToDet(DetId(detid))->surface().toGlobal(_HitDir._TrackingRecHit->localPosition()).phi().degrees() << " anglexz " << atanXZ << " | tanxz " << tanXZ << " x " << _HitDir._LV.x() << " z " << _HitDir._LV.z() << " mag " << sqrt(_HitDir._LV.x()*_HitDir._LV.x()+_HitDir._LV.z()*_HitDir._LV.z()) << " \t | " << tkgeom->idToDet(DetId(detid))->surface().toGlobal(LocalVector(0,0,1)) << " " << tkgeom->idToDet(DetId(detid))->surface().toGlobal(LocalVector(1,0,0)) << std::endl;
	
	//LogTrace("ClusterAnalysis") << " det " << appString << " angle Ph " << tkgeom->idToDet(DetId(detid))->surface().toGlobal(_HitDir._TrackingRecHit->localPosition()).phi().degrees() << " " << _HitDir._TrackingRecHit->localPosition() << "   " <<  tkgeom->idToDet(DetId(detid))->surface().toGlobal(_HitDir._TrackingRecHit->localPosition()) << std::endl; 
      }      
    }
    return true;
  }

  //--------------------------------------------------------------------------------
  std::pair<std::string,uint32_t> ClusterAnalysis::GetSubDetAndLayer(const uint32_t& detid){
    
    std::string cSubDet;
    uint32_t layer=0;
    const StripGeomDetUnit* _StripGeomDetUnit = dynamic_cast<const StripGeomDetUnit*>(tkgeom->idToDetUnit(DetId(detid)));
    switch(_StripGeomDetUnit->specificType().subDetector())
      {
      case GeomDetEnumerators::TIB:
	cSubDet="TIB";
	layer=TIBDetId(detid).layer();
	break;
      case GeomDetEnumerators::TOB:
	cSubDet="TOB";
	layer=TOBDetId(detid).layer();
	break;
      case GeomDetEnumerators::TID:
	cSubDet="TID";
	layer=TIDDetId(detid).ring();
	// SUSY Modified me on 11/9/07
	LogTrace("ClusterAnalysis") << "SubDet found TID ring" << layer << std::endl;
	break;
      case GeomDetEnumerators::TEC:
	cSubDet="TEC";
	layer=TECDetId(detid).ring();
	LogTrace("ClusterAnalysis") << "SubDet found TEC ring" << layer << std::endl;
	break;
      default:
	edm::LogWarning("ClusterAnalysis") << "WARNING!!! this detid does not belong to tracker" << std::endl;
      }
    return std::make_pair(cSubDet,layer);
  }


  void ClusterAnalysis::fillTH1(float value,TString name,bool widthFlag,float cwidth){

    for (int iw=0;iw<5;iw++){
      if ( iw==0 || (iw==4 && cwidth>3) || ( iw>0 && iw<4 && cwidth==iw) ){     
	TH1F* hh = (TH1F*) Hlist->FindObject(name+width_flags[iw]);
	if (hh!=0)  
	  hh->Fill(value);
      }
      if (!widthFlag)
	break;
    }
  }

  void ClusterAnalysis::fillTProfile(float xvalue,float yvalue,TString name,bool widthFlag,float cwidth){

    for (int iw=0;iw<5;iw++){
      if ( iw==0 || (iw==4 && cwidth>3) || ( iw>0 && iw<4 && cwidth==iw) ){     
	TProfile* hh = (TProfile*) Hlist->FindObject(name+width_flags[iw]);
	if (hh!=0)  
	  hh->Fill(xvalue,yvalue);
      }
      if (!widthFlag)
	break;
    }
  }

  void ClusterAnalysis::fillTH2(float xvalue,float yvalue,TString name,bool widthFlag, float cwidth){

    for (int iw=0;iw<5;iw++){
      if ( iw==0 || (iw==4 && cwidth>3) || ( iw>0 && iw<4 && cwidth==iw) ){     
	TH2F* hh = (TH2F*) Hlist->FindObject(name+width_flags[iw]);
	if (hh!=0)  
	  hh->Fill(xvalue,yvalue);
      }
      if (!widthFlag)
	break;
    }
  }

  void ClusterAnalysis::bookHlist(char* HistoType, char* ParameterSetLabel, TFileDirectory subDir, TString & HistoName, char* xTitle, char* yTitle, char* zTitle){
    if ( HistoType == "TH1" ) {
      Parameters =  conf_.getParameter<edm::ParameterSet>(ParameterSetLabel);
      TH1F* p = subDir.make<TH1F>(HistoName,HistoName,
			 Parameters.getParameter<int32_t>("Nbinx"),
			 Parameters.getParameter<double>("xmin"),
			 Parameters.getParameter<double>("xmax")
			 );
      if ( xTitle != "" )
	p->SetXTitle(xTitle);
      if ( yTitle != "" )
	p->SetYTitle(yTitle);
      Hlist->Add(p);
    }
    else if ( HistoType == "TH2" ) {
      Parameters =  conf_.getParameter<edm::ParameterSet>(ParameterSetLabel);
      TH2F* p = subDir.make<TH2F>(HistoName,HistoName,
			 Parameters.getParameter<int32_t>("Nbinx"),
			 Parameters.getParameter<double>("xmin"),
			 Parameters.getParameter<double>("xmax"),
			 Parameters.getParameter<int32_t>("Nbiny"),
			 Parameters.getParameter<double>("ymin"),
			 Parameters.getParameter<double>("ymax")
			 );
      if ( xTitle != "" )
	p->SetXTitle(xTitle);
      if ( yTitle != "" )
	p->SetYTitle(yTitle);
      if ( zTitle != "" )
	p->SetZTitle(zTitle);
      Hlist->Add(p);
    }
    else if ( HistoType == "TH3" ){
      Parameters =  conf_.getParameter<edm::ParameterSet>(ParameterSetLabel);
      TH3S* p = subDir.make<TH3S>(HistoName,HistoName,
			 Parameters.getParameter<int32_t>("Nbinx"),
			 Parameters.getParameter<double>("xmin"),
			 Parameters.getParameter<double>("xmax"),
			 Parameters.getParameter<int32_t>("Nbiny"),
			 Parameters.getParameter<double>("ymin"),
			 Parameters.getParameter<double>("ymax"),
			 Parameters.getParameter<int32_t>("Nbinz"),
			 Parameters.getParameter<double>("zmin"),
			 Parameters.getParameter<double>("zmax")
			 );
      if ( xTitle != "" )
	p->SetXTitle(xTitle);
      if ( yTitle != "" )
	p->SetYTitle(yTitle);
      if ( zTitle != "" )
	p->SetZTitle(zTitle);
      Hlist->Add(p);
    }
    else if ( HistoType == "TProfile" ){
      Parameters =  conf_.getParameter<edm::ParameterSet>(ParameterSetLabel);
      TProfile* p = subDir.make<TProfile>(HistoName,HistoName,
				  Parameters.getParameter<int32_t>("Nbinx"),
				  Parameters.getParameter<double>("xmin"),
				  Parameters.getParameter<double>("xmax"),
				  Parameters.getParameter<double>("ymin"),
				  Parameters.getParameter<double>("ymax")
				  );
      if ( xTitle != "" )
	p->SetXTitle(xTitle);
      if ( yTitle != "" )
	p->SetYTitle(yTitle);
      Hlist->Add(p);
    }
    else{
      edm::LogError("ClusterAnalysis")<< "[" <<__PRETTY_FUNCTION__ << "] invalid HistoType " << HistoType << std::endl;
    }
  }



  void ClusterAnalysis::fillPedNoiseFromDB(){
    std::vector<uint32_t> vdetId_;
    SiStripDetCabling_->addActiveDetectorsRawIds(vdetId_);
    
    for (std::vector<uint32_t>::const_iterator detid_iter=vdetId_.begin();detid_iter!=vdetId_.end();detid_iter++){
      
      uint32_t detid = *detid_iter;
      
      if (detid < 1){
 	edm::LogError("ClusterAnalysis")<< "[" <<__PRETTY_FUNCTION__ << "] invalid detid " << detid<< std::endl;
	continue;
      }
      const StripGeomDetUnit* _StripGeomDetUnit = dynamic_cast<const StripGeomDetUnit*>(tkgeom->idToDetUnit(DetId(detid)));
      if (_StripGeomDetUnit==0){
	continue;
      }
      
      
      unsigned int nstrips = _StripGeomDetUnit->specificTopology().nstrips();

      //&&&&&&&&&&&&&&
      // Retrieve information for the module
      //&&&&&&&&&&&&&&&&&&     
      char cdetid[128];
      sprintf(cdetid,"%d",detid);
      char aname[128];
      sprintf(aname,"%s_%d",_StripGeomDetUnit->type().name().c_str(),detid);
      char SubStr[128];
      
      SiStripDetId a(detid);
      if ( a.subdetId() == 3 ){
	TIBDetId b(detid);
	sprintf(SubStr,"_SingleDet_%d_TIB_%d_%d_%d_%d",detid,b.layer(),b.string()[0],b.string()[1],b.glued());
      } else if ( a.subdetId() == 4 ) {
	TIDDetId b(detid);
	sprintf(SubStr,"_SingleDet_%d_TID_%d_%d_%d_%d",detid,b.wheel(),b.ring(),b.side(),b.glued());
      } else if ( a.subdetId() == 5 ) {
	TOBDetId b(detid);
	sprintf(SubStr,"_SingleDet_%d_TOB_%d_%d_%d_%d",detid,b.layer(),b.rod()[0],b.rod()[1],b.glued());
      } else if ( a.subdetId() == 6 ) {
	TECDetId b(detid);
	sprintf(SubStr,"_SingleDet_%d_TEC_%d_%d_%d_%d_%d",detid,b.wheel(),b.ring(),b.side(),b.glued(),b.stereo());
      }
      
      TString appString=TString(SubStr);

      SiStripNoises::Range noiseRange = noiseHandle->getRange(detid);
      SiStripPedestals::Range pedRange = pedestalHandle->getRange(detid);
      
      SiStripQuality::Range detQualityRange = SiStripQuality_->getRange(detid);
     
      for(size_t istrip=0;istrip<nstrips;istrip++){
	try{
	  //Fill Pedestals
	  TH1F* hh1 = (TH1F*) Hlist->FindObject("DBPedestals"+appString);
	  if (hh1!=0) 
	    hh1->Fill(istrip,pedestalHandle->getPed(istrip,pedRange)); 
	  
	  //Fill Noises
	  TH1F* hh2 = (TH1F*) Hlist->FindObject("DBNoise"+appString);
	  if (hh2!=0)   
	    hh2->Fill(istrip,noiseHandle->getNoise(istrip,noiseRange));	    
	  
	  //Fill BadStripsNoise
	  TH1F* hh3 = (TH1F*) Hlist->FindObject("DBBadStrips"+appString);
	  if (hh3!=0)  
	    hh3->Fill(istrip,SiStripQuality_->IsStripBad(detQualityRange,istrip)?1.:0.);
	  
	}catch(cms::Exception& e){
	  edm::LogError("SiStripCondObjDisplay") << "[SiStripCondObjDisplay::endJob]  cms::Exception:  DetName " << name << " " << e.what() ;
	}
      }
      
    }
  }
}
