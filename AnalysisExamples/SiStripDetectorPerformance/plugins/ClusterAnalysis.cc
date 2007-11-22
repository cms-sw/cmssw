/*
 * $Date: 2007/11/20 08:20:40 $
 * $Revision: 1.2 $
 *
 * \author: D. Giordano, domenico.giordano@cern.ch
 * Modified: M.De Mattia 2/3/2007 & R.Castello 5/4/2007
 */

#include "AnalysisExamples/SiStripDetectorPerformance/plugins/ClusterAnalysis.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"

#include "AnalysisDataFormats/TrackInfo/src/TrackInfo.cc"
#include "sstream"

#include "TTree.h"
#include "TBranch.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TH3S.h"
#include "TProfile.h"
#include "TCanvas.h"
#include "TPostScript.h"

static const uint16_t _NUM_SISTRIP_SUBDET_ = 4;
static TString SubDet[_NUM_SISTRIP_SUBDET_]={"_TIB","_TOB","_TID","_TEC"};
static TString flags[3] = {"_onTrack","_offTrack","_All"};
static TString width_flags[5] = {"","_width_1","_width_2","_width_3","_width_ge_4"};


namespace cms{
  ClusterAnalysis::ClusterAnalysis(edm::ParameterSet const& conf): 
    conf_(conf),
    filename_(conf.getParameter<std::string>("fileName")), 
    psfilename_(conf.getParameter<std::string>("psfileName")), 
    psfiletype_(conf.getParameter<int32_t>("psfiletype")),
    psfilemode_(conf.getUntrackedParameter<int32_t>("psfilemode",1)),
    Filter_src_( conf.getParameter<edm::InputTag>( "Filter_src" ) ),
    Track_src_( conf.getParameter<edm::InputTag>( "Track_src" ) ),
    ClusterInfo_src_( conf.getParameter<edm::InputTag>( "ClusterInfo_src" ) ),
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

    book();
  }

  void ClusterAnalysis::book() {

    fFile = new TFile(filename_.c_str(),"RECREATE");
    fFile->mkdir("ClusterNoise");
    fFile->mkdir("ClusterSignal");
    fFile->mkdir("ClusterStoN");
    fFile->mkdir("ClusterEta");
    fFile->mkdir("ClusterWidth");
    fFile->mkdir("ClusterPos");
    fFile->mkdir("Tracks");
    fFile->mkdir("Trigger");
    fFile->mkdir("Layer");
    fFile->cd();

    //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // get list of active detectors from SiStripDetCabling 

    std::vector<uint32_t> vdetId_;
    SiStripDetCabling_->addActiveDetectorsRawIds(vdetId_);
    //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    //Create histograms
    Hlist = new THashList();

    //Display 3D
    name = "ClusterGlobalPos";
    bookHlist("TH3","TH3ClusterGlobalPos", name, "z (cm)","x (cm)","y (cm)"); 

    std::cout << "added TH3D " << std::endl;
    //&&&&&&&&&&&&&&&&&&&&&&&&

    fFile->cd();fFile->cd("Trigger");

    Hlist->Add(new TH1F("FilterBits","FilterBits",10,-0.5,9.5));

    // bookHlist(TObjArray, Name of the parameterset in the cfg, name of the hitsogram, xbinnumber, xmin, xmax )
    name = "TriggerBits";
    bookHlist("TH1", "TH1TriggerBits", name);
    fFile->cd();fFile->cd("Tracks");


    name = "nTracks";
    bookHlist("TH1","TH1nTracks", name, "N Tracks" );
    name = "nRecHits";
    bookHlist("TH1","TH1nRecHits", name, "N RecHits" );

    // Loop on onTrack, offTrack, All
    for (int j=0;j<3;j++){      
      //Number of Cluster 
      name="nClusters"+flags[j];
      fFile->cd();fFile->cd("Tracks");
      bookHlist("TH1","TH1nClusters", name, "N Clusters" );

      for (int i=0;i<_NUM_SISTRIP_SUBDET_;i++) {
   	//Number of Cluster on each det
	name="nClusters"+SubDet[i]+flags[j];
	bookHlist("TH1","TH1nClusters", name, "N Clusters" );
      }
    }

    // Loop on onTrack, offTrack, All
    for (int j=0;j<3;j++){
      //Histos for detector type
      for (int i=0;i<_NUM_SISTRIP_SUBDET_;i++){
	
    	TString appString=SubDet[i]+flags[j];

	//Cluster Width
    	name="cWidth"+appString;
    	fFile->cd();fFile->cd("ClusterWidth");
    	bookHlist("TH1","TH1ClusterWidth", name, "Nstrip" );

    	//Loop for cluster width
    	for (int iw=0;iw<5;iw++){
	  
    	  appString=SubDet[i]+flags[j]+width_flags[iw];
	
     	  //Cluster Noise
     	  name="cNoise"+appString;
     	  fFile->cd();fFile->cd("ClusterNoise");
     	  bookHlist("TH1","TH1ClusterNoise", name, "ADC count" );

     	  //Cluster Signal
     	  name="cSignal"+appString;
     	  fFile->cd();fFile->cd("ClusterSignal");
     	  bookHlist("TH1","TH1ClusterSignal", name, "ADC count" );	  	 

	  //Cluster Signal corrected
	  if(j==0 && iw==0 ){
	    name="cSignalCorr"+appString;
	    fFile->cd();fFile->cd("ClusterSignal");
	    bookHlist("TH1","TH1ClusterSignalCorr", name, "ADC count" );  
	  }
	  
     	  //Cluster StoN
     	  name="cStoN"+appString;
     	  fFile->cd();fFile->cd("ClusterStoN");
     	  bookHlist("TH1","TH1ClusterStoN", name );

	  //Cluster SignaltoNoise corrected
	  if(j==0 && iw==0 ){	     
	    name="cStoNCorr"+appString;
	    fFile->cd();fFile->cd("ClusterStoN");
	    bookHlist("TH1","TH1ClusterStoNCorr", name );  
	  }

     	  //Cluster Position
     	  name="cPos"+appString;
     	  fFile->cd();fFile->cd("ClusterPos");
     	  bookHlist("TH1","TH1ClusterPos", name, "strip Num" );

	  //Cluster StoN Vs Cluster Position
	  name="cStoNVsPos"+appString;
     	  fFile->cd();fFile->cd("ClusterPos");
     	  bookHlist("TH2","TH2ClusterStoNVsPos", name, "strip Num");

     	  //Cluster Charge Division (only for study on Raw Data Runs)
     	  name="cEta"+appString;
     	  fFile->cd();fFile->cd("ClusterEta");
     	  bookHlist("TH1","TH1ClusterEta", name, "" );

     	  name="cEta_scatter"+appString;
     	  fFile->cd();fFile->cd("ClusterEta");
     	  bookHlist("TH2","TH2ClusterEta", name, "" , "");
     	}//end loop on width 

	//cWidth Vs Angle
	name = "ClusterWidthVsAngle"+appString;
	bookHlist("TProfile","TProfileWidthAngle", name, "cos(angle_xz)", "clusWidth");

	//Residual Vs Angle
	name = "ResidualVsAngle"+appString;
	bookHlist("TProfile","TProfileResidualAngle", name, "Angle" , "Residual");
      
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
      //eg
      
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

      fFile->cd();
      fFile->mkdir(cdetid);    
      fFile->cd(cdetid);    

      //Cluster Noise
      name="cNoise"+appString;
      bookHlist("TH1","TH1ClusterNoise", name, "ADC count" );

      //Cluster Signal
      name="cSignal"+appString;
      bookHlist("TH1","TH1ClusterSignal", name, "ADC count" );

      //Cluster StoN
      name="cStoN"+appString;
      bookHlist("TH1","TH1ClusterStoN", name, "" );

      //Cluster Signal x Fiber
      name="cSignalxFiber"+appString+"_onTrack";
      bookHlist("TProfile","TProfileSignalxFiber", name, "ApvPair", "ADC count" );

      //Cluster Width
      name="cWidth"+appString;
      bookHlist("TH1","TH1ClusterWidth", name, "Nstrip" );

      //Cluster Position
      name="cPos"+appString;
      //bookHlist("TH1","TH1ClusterPos", name, "Nbinx", "xmin", "xmax" );
      Hlist->Add(new TH1F(name,name,nstrips,0,nstrips));
		
      //Cluster StoN Vs Cluster Position
      name="cStoNVsPos"+appString;
      char labeln[128];
      sprintf(labeln,"strip Num (Ntot=%d)",nstrips);
      bookHlist("TH2","TH2ClusterStoNVsPos", name, labeln);
 
      //Cluster Charge Division (only for study on Raw Data Runs)
      name="cEta"+appString;
      bookHlist("TH1","TH1ClusterEta", name, "" );

      name="cEta_scatter"+appString;
      bookHlist("TH2","TH2ClusterEta", name, "" ,  "" );

      //&&&&&&&&&&&&&&&&&
    }//end loop on detector	


    //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    //Layer Detail Plots
    
    for (std::map<std::pair<std::string,uint32_t>,bool>::const_iterator iter=DetectedLayers.begin(); iter!=DetectedLayers.end();iter++){
     
      char cApp[64];
      sprintf(cApp,"_Layer_%d",iter->first.second);
      fFile->cd(); fFile->cd("Layer");    

      // Loop on onTrack, offTrack, All
      for (int j=0;j<3;j++){
	
	TString appString="_"+TString(iter->first.first)+cApp+flags[j];
     
	//Cluster Noise
	name="cNoise"+appString;
	bookHlist("TH1","TH1ClusterNoise", name, "ADC count" );

	//Cluster Signal
	name="cSignal"+appString;
	bookHlist("TH1","TH1ClusterSignal", name, "ADC count" );
	
	//Cluster Signal corrected
	if(j==0){
	  name="cSignalCorr"+appString;
	  bookHlist("TH1","TH1ClusterSignalCorr", name, "ADC count" );

	  //Cluster Signal Vs Angle
	  name = "cSignalVsAngle"+appString;
	  bookHlist("TProfile","TProfilecSignalVsAngle", name, "cos(angle_rz)" , "ADC count");
	  name = "cSignalVsAngleH"+appString;
	  bookHlist("TH2","TH2cSignalVsAngle", name, "cos(angle_rz)" , "ADC count");
	}

	//Cluster StoN
	name="cStoN"+appString;
	bookHlist("TH1","TH1ClusterStoN", name, "" );
	
	//Cluster SignaltoNoise corrected
	if(j==0){
	  name="cStoNCorr"+appString;
	  bookHlist("TH1","TH1ClusterStoNCorr", name, "" );
	}

	//Cluster Width
	name="cWidth"+appString;
	bookHlist("TH1","TH1ClusterWidth", name, "Nstrip" );

	//Cluster Position
	name="cPos"+appString;
	bookHlist("TH1","TH1ClusterPos", name, "strip Num" );

	//Cluster StoN Vs Cluster Position
	name="cStoNVsPos"+appString;
	bookHlist("TH2","TH2ClusterStoNVsPos", name, "strip Num");

	//residual
	name="res_x"+appString;
	bookHlist("TH1","TH1Residual_x", name, "" );
	
	//residual y
	name="res_y"+appString;
	bookHlist("TH1","TH1Residual_y", name, "" );
      
	//cWidth Vs Angle
	if(j==0){
	  name = "ClusterWidthVsAngle"+appString;
	  bookHlist("TProfile","TProfileWidthAngle", name, "angle_xz" , "clusWidth");
	
	  //Residuals Vs Angle
	  name = "ResidualVsAngle"+appString;
	  bookHlist("TProfile","TProfileResidualAngle", name, "cos(angle_rz)" , "Residual");

	  //Angle Vs phi
	  name = "AngleVsPhi"+appString;
	  bookHlist("TProfile","TProfileAngleVsPhi", name, "Phi (deg)" , "Impact angle angle_xz (deg)");
	}
      }
    }
  }

  //------------------------------------------------------------------------------------------

  void ClusterAnalysis::endJob() {  
    edm::LogInfo("ClusterAnalysis") << "[ClusterAnalysis::endJob] >>> saving histograms" << std::endl;
    
    if (psfilemode_>0){    
      fFile->cd();
      
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
      fFile->ls();
    }    
    fFile->Write();
    fFile->Close();
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
    e.getByLabel( ClusterInfo_src_, dsv_SiStripClusterInfo);
    e.getByLabel( Cluster_src_, dsv_SiStripCluster);    
    
    try{
      e.getByLabel(Track_src_, trackCollection);
    } catch ( cms::Exception& er ) {
      LogTrace("ClusterAnalysis")<<"caught std::exception "<<er.what()<<std::endl;
      tracksCollection_in_EventTree=false;
    } catch ( ... ) {
      LogTrace("ClusterAnalysis")<<" funny error " <<std::endl;
      tracksCollection_in_EventTree=false;
    }
    
    edm::InputTag TkiTag = conf_.getParameter<edm::InputTag>( "TrackInfo" );  
    try{
      e.getByLabel(TkiTag,tkiTkAssCollection); 
    } catch ( cms::Exception& er ) {
      LogTrace("ClusterAnalysis")<<"caught std::exception "<<er.what()<<std::endl;
      trackAssociatorCollection_in_EventTree=false;
    } catch ( ... ) {
      LogTrace("ClusterAnalysis")<<" funny error " <<std::endl;
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
      edm::LogInfo("ClusterAnalysis") << "\n["<<__PRETTY_FUNCTION__<<"] \n Executing trackStudy for Event = " << eventNb << std::endl; 
      trackStudy();
    }
   
    std::stringstream ss;
    ss << "\nList of SiStripClusterPointer\n";
    for (std::vector<const SiStripCluster*>::iterator iter=vPSiStripCluster.begin();iter!=vPSiStripCluster.end();iter++)
      ss << *iter << "\n";    
    LogTrace("ClusterAnalysis") << "\n["<<__PRETTY_FUNCTION__<<"] \n vPSiStripCluster.size()=" << vPSiStripCluster.size()<< ss.str() << std::endl;	
    

    //Perform Cluster Study (irrespectively to tracks)
    AllClusters();

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
  
  void ClusterAnalysis::trackStudy(){
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
      edm::LogInfo("ClusterAnalysis") <<"\t\tSkipped track "<< i << std::endl;
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
      reco::TrackInfo::TrajectoryInfo::const_iterator iter;
       for(iter=trackinforef->trajStateMap().begin();iter!=trackinforef->trajStateMap().end();iter++){
         
         //trajectory local direction and position on detector
	LocalVector statedirection;

	LocalPoint  stateposition=(trackinforef->stateOnDet(Updated,(*iter).first)->parameters()).position();
       	edm::LogInfo("TrackInfoAnalyzerExample") <<"LocalMomentum initialized: "<<statedirection;
       	edm::LogInfo("TrackInfoAnalyzerExample") <<"LocalPosition: "<<stateposition;
      	edm::LogInfo("TrackInfoAnalyzerExample") <<"Local x-z plane angle: "<<atan2(statedirection.x(),statedirection.z());
	
	if(trackinforef->type((*iter).first)==Matched){ // get the direction for the components
	  
	  const SiStripMatchedRecHit2D* matchedhit=dynamic_cast<const SiStripMatchedRecHit2D*>(&(*(iter)->first));
	  if (matchedhit!=0){
	    edm::LogInfo("TrackInfoAnalyzerExample")<<"Matched recHit found"<< std::endl;	  
	    //mono side
	    statedirection= trackinforef->localTrackMomentumOnMono(Updated,(*iter).first);
	    if(statedirection.mag() != 0)	  RecHitInfo(matchedhit->monoHit(),statedirection,trackref);
	    //stereo side
	    statedirection= trackinforef->localTrackMomentumOnStereo(Updated,(*iter).first);
	    if(statedirection.mag() != 0)	  RecHitInfo(matchedhit->stereoHit(),statedirection,trackref);
	    edm::LogInfo("TrackInfoAnalyzerExample") <<"LocalMomentum (stereo): "<<trackinforef->localTrackMomentumOnStereo(Updated,(*iter).first);
	  }
	}
	else if (trackinforef->type((*iter).first)==Projected){//one should be 0
	  edm::LogInfo("TrackInfoAnalyzerExample")<<"Projected recHit found"<< std::endl;
	  const ProjectedSiStripRecHit2D* phit=dynamic_cast<const ProjectedSiStripRecHit2D*>(&(*(iter)->first));
	  if(phit!=0){
	    //mono side
	    statedirection= trackinforef->localTrackMomentumOnMono(Updated,(*iter).first);
	    if(statedirection.mag() != 0) RecHitInfo(&(phit->originalHit()),statedirection,trackref);
	    //stereo side
	    statedirection= trackinforef->localTrackMomentumOnStereo(Updated,(*iter).first);
	    if(statedirection.mag() != 0) RecHitInfo(&(phit->originalHit()),statedirection,trackref);
	  }	
	}
	else {
	  const SiStripRecHit2D* hit=dynamic_cast<const SiStripRecHit2D*>(&(*(iter)->first));
	 if(hit!=0){
	   edm::LogInfo("TrackInfoAnalyzerExample")<<"Single recHit found"<< std::endl;	  
	   statedirection=(trackinforef->stateOnDet(Updated,(*iter).first)->parameters()).momentum();
	   if(statedirection.mag() != 0) RecHitInfo(hit,statedirection,trackref);
       
	 }
	}
      }
    }
  }

void ClusterAnalysis::RecHitInfo(const SiStripRecHit2D* tkrecHit, LocalVector LV,reco::TrackRef track_ref ){
  if(!tkrecHit->isValid()){
    edm::LogInfo("ClusterAnalysis") <<"\t\t Invalid Hit "<<std::endl;
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
    const SiStripClusterInfo* SiStripClusterInfo_ = MatchClusterInfo(SiStripCluster_,detid);
      
    const edm::ParameterSet pset = conf_.getParameter<edm::ParameterSet>("TrackThresholds");
    if( track_ref->normalizedChi2() < pset.getParameter<double>("maxChi2") && track_ref->recHitsSize() > pset.getParameter<double>("minRecHit") ){	  	    
	
      if ( clusterInfos(SiStripClusterInfo_,detid,"_onTrack", LV ) ) {
	vPSiStripCluster.push_back(SiStripCluster_);
	countOn++;
      }
    }
  }else{
    LogTrace("ClusterAnalysis") << "NULL hit" << std::endl;
  }	  
}

  
  //------------------------------------------------------------------------
  
  void ClusterAnalysis::AllClusters(){
    LogTrace("ClusterAnalysis") << "\n["<<__PRETTY_FUNCTION__<<"]" << std::endl;
    
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

	const SiStripClusterInfo* SiStripClusterInfo_=MatchClusterInfo(&*ClusIter,detid);
	if ( clusterInfos(SiStripClusterInfo_,detid,"_All", LV) ){ 
	  countAll++;

	  LogTrace("ClusterAnalysis") << "\n["<<__PRETTY_FUNCTION__<<"] ClusIter " << &*ClusIter << 
	    "\t " << std::find(vPSiStripCluster.begin(),vPSiStripCluster.end(),&*ClusIter)-vPSiStripCluster.begin() << std::endl;

	  if (std::find(vPSiStripCluster.begin(),vPSiStripCluster.end(),&*ClusIter) == vPSiStripCluster.end()){
	    if ( clusterInfos(SiStripClusterInfo_,detid,"_offTrack", LV) ) 
	      countOff++;
	  }
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
    edm::LogError("ClusterAnalysis") << "\n["<<__PRETTY_FUNCTION__<<"]\n\t" << "Matching of SiStripCluster and SiStripClusterInfo is failed for cluster on detid "<< detid << "\n\tReturning NULL pointer" <<std::endl;
    return 0;
  }

  //------------------------------------------------------------------------
  
  bool ClusterAnalysis::clusterInfos(const SiStripClusterInfo* cluster, const uint32_t& detid,TString flag , const LocalVector LV){
    LogTrace("ClusterAnalysis") << "\n["<<__PRETTY_FUNCTION__<<"]" << std::endl;
    
    if (cluster==0) 
      return false;
    
    const  edm::ParameterSet ps = conf_.getParameter<edm::ParameterSet>("ClusterConditions");
    if  ( ps.getParameter<bool>("On") 
	  &&
	  ( 
	   cluster->charge()/cluster->noise() < ps.getParameter<double>("minStoN") 
	   ||
	   cluster->charge()/cluster->noise() > ps.getParameter<double>("maxStoN") 
	   ||
	   cluster->width() < ps.getParameter<double>("minWidth") 
	   ||
	   cluster->width() > ps.getParameter<double>("maxWidth") 
	   )
	  )
      return false;
    
    const StripGeomDetUnit*_StripGeomDetUnit = dynamic_cast<const StripGeomDetUnit*>(tkgeom->idToDetUnit(DetId(detid)));
    //GeomDetEnumerators::SubDetector SubDet_enum=_StripGeomDetUnit->specificType().subDetector();
    int SubDet_enum=_StripGeomDetUnit->specificType().subDetector() -2;

    //&&&&&&&&&&&&&&&& GLOBAL POS &&&&&&&&&&&&&&&&&&&&&&&&
    const StripTopology &topol=(StripTopology&)_StripGeomDetUnit->topology();
    MeasurementPoint mp(cluster->position(),rnd.Uniform(-0.5,0.5));
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

    std::stringstream ss;
    const_cast<SiStripClusterInfo*>(cluster)->print(ss);
    LogTrace("ClusterAnalysis") 
      << "\n["<<__PRETTY_FUNCTION__<<"]\n"
      << ss.str() 
      << "\n\t\tcluster LocalPos "     << localPos
      << "\n\t\tcluster GlobalPos "     << globalPos
      << std::endl;

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
  
    fillTH1(cluster->charge(),"cSignal"+appString,1,cluster->width());

    if (LV.mag()!=0){
      
      fillTH1(cluster->charge()*cosRZ,"cSignalCorr"+appString,0); //Filled only for ontrack
      
      fillTProfile(atanXZ,cluster->width(),"ClusterWidthVsAngle"+appString,0); //Filled only for ontrack

      fillTH1((cluster->charge()/cluster->noise())*cosRZ,"cStoNCorr"+appString,0); //Filled only for ontrack
    } 
       
    fillTH1(cluster->noise(),"cNoise"+appString,1,cluster->width());
    
    if (cluster->noise()){
      fillTH1(cluster->charge()/cluster->noise(),"cStoN"+appString,1,cluster->width());
      
    }
      
    fillTH1(cluster->width(),"cWidth"+appString,0);

    fillTH1(cluster->position(),"cPos"+appString,1,cluster->width());

    fillTH2(cluster->position(),cluster->charge()/cluster->noise(),"cStoNVsPos"+appString,1,cluster->width());
   
    if (cluster->rawdigiAmplitudesL().size()!=0 ||  cluster->rawdigiAmplitudesR().size()!=0){
	
      float Ql=0;
      float Qr=0;
      float Qt=0;

      if (EtaAlgo_==1){
	Ql=cluster->chargeL();
	Qr=cluster->chargeR();
      
  	for (std::vector<float>::const_iterator it=cluster->rawdigiAmplitudesL().begin(); it !=cluster->rawdigiAmplitudesL().end() && it-cluster->rawdigiAmplitudesL().begin()<NeighStrips_; it ++)
  	  { Ql += (*it);}

	
  	for (std::vector<float>::const_iterator it=cluster->rawdigiAmplitudesR().begin(); it !=cluster->rawdigiAmplitudesR().end() && it-cluster->rawdigiAmplitudesR().begin()<NeighStrips_; it ++)
  	  { Qr += (*it);}
	
	Qt=Ql+Qr+cluster->maxCharge();
      }
      else{
	
	int Nstrip=cluster->stripAmplitudes().size();
	float pos=cluster->position()-0.5;
	for(int is=0;is<Nstrip && cluster->firstStrip()+is<=pos;is++)
	  Ql+=cluster->stripAmplitudes()[is];
	
	Qr=cluster->charge()-Ql;

  	for (std::vector<float>::const_iterator it=cluster->rawdigiAmplitudesL().begin(); it !=cluster->rawdigiAmplitudesL().end() && it-cluster->rawdigiAmplitudesL().begin()<NeighStrips_; it ++)
  	  { Ql += (*it);}
	
  	for (std::vector<float>::const_iterator it=cluster->rawdigiAmplitudesR().begin(); it !=cluster->rawdigiAmplitudesR().end() && it-cluster->rawdigiAmplitudesR().begin()<NeighStrips_; it ++)
  	  { Qr += (*it);}

	
	Qt=Ql+Qr;
      }
      
      LogTrace("ClusterAnalysis") << "\n["<<__PRETTY_FUNCTION__<<"] \n on detid "<< detid << " Ql=" << Ql << " Qr="<< Qr << " Qt="<<Qt<< " eta="<< Ql/Qt<< std::endl;
      
      fillTH1(Ql/Qt,"cEta"+appString,1,cluster->width());
    
      fillTH2((Ql-Qr)/Qt,(Ql+Qr)/Qt,"cEta_scatter"+appString,1,cluster->width());
    }

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

      fillTH1(cluster->charge(),"cSignal"+appString,0);

      fillTH1(cluster->noise(),"cNoise"+appString,0);

      if (cluster->noise()){
	fillTH1(cluster->charge()/cluster->noise(),"cStoN"+appString,0);
      }
      
      fillTH1(cluster->width(),"cWidth"+appString,0);

      fillTH1(cluster->position(),"cPos"+appString,0);

      fillTH2(cluster->position(),cluster->charge()/cluster->noise(),"cStoNVsPos"+appString,0);
    }

    if(flag=="_onTrack" && cosRZ>-2){
      fillTProfile((int)(cluster->position()-.5)/256,cluster->charge()*cosRZ,"cSignalxFiber"+appString,0);
    }

    //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // Layer Detail Plots
    char cApp[64];
    sprintf(cApp,"_Layer_%d",GetSubDetAndLayer(detid).second);
    appString="_"+TString(GetSubDetAndLayer(detid).first)+cApp+flag;

    fillTH1(cluster->charge(),"cSignal"+appString,0);
  
    fillTH1(cluster->noise(),"cNoise"+appString,0);

    if (cluster->noise()){
      fillTH1(cluster->charge()/cluster->noise(),"cStoN"+appString,0);
    }
      
    fillTH1(cluster->width(),"cWidth"+appString,0);
    
    fillTH1(cluster->position(),"cPos"+appString,0);      
    
    if(flag=="_onTrack" && LV.mag()!=0){
      fillTProfile(atanXZ,cluster->width(),"ClusterWidthVsAngle"+appString+"_onTrack",0);
      
      fillTH1(cluster->charge()*cosRZ,"cSignalCorr"+appString,0);
      
      fillTProfile(cosRZ,cluster->charge(),"cSignalVsAngle"+appString,0);
      fillTH2(cosRZ,cluster->charge(),"cSignalVsAngleH"+appString,0);
      
      if (cluster->noise()){
	
	fillTH1((cluster->charge()/cluster->noise())*cosRZ,"cStoNCorr"+appString,0); 
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

  void ClusterAnalysis::bookHlist(char* HistoType, char* ParameterSetLabel, TString & HistoName, char* xTitle, char* yTitle, char* zTitle){
    if ( HistoType == "TH1" ) {
      Parameters =  conf_.getParameter<edm::ParameterSet>(ParameterSetLabel);
      TH1F* p = new TH1F(HistoName,HistoName,
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
      TH2F* p = new TH2F(HistoName,HistoName,
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
    else if ( HistoType == "TH2" ) {
      Parameters =  conf_.getParameter<edm::ParameterSet>(ParameterSetLabel);
      TH2F* p = new TH2F(HistoName,HistoName,
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
      TH3S* p = new TH3S(HistoName,HistoName,
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
      TProfile* p =  new TProfile(HistoName,HistoName,
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
      
      fFile->cd();fFile->cd(cdetid);

      name="DBPedestals"+appString;
      TH1F* pPed = new TH1F(name,name,nstrips,-0.5,nstrips-0.5);
      Hlist->Add(pPed);
    
      name="DBNoise"+appString;
      TH1F* pNoi = new TH1F(name,name,nstrips,-0.5,nstrips-0.5);
      Hlist->Add(pNoi);
    
      name="DBBadStrips"+appString;
      TH1F* pBad = new TH1F(name,name,nstrips,-0.5,nstrips-0.5);
      Hlist->Add(pBad);

      SiStripNoises::Range noiseRange = noiseHandle->getRange(detid);
      SiStripPedestals::Range pedRange = pedestalHandle->getRange(detid);
	
      for(size_t istrip=0;istrip<nstrips;istrip++){
	try{
	  //Fill Pedestals
	  pPed->Fill(istrip,pedestalHandle->getPed(istrip,pedRange));
	  
	  //Fill Noises
	  pNoi->Fill(istrip,noiseHandle->getNoise(istrip,noiseRange));
	  
	  //Fill BadStripsNoise
	  pBad->Fill(istrip,noiseHandle->getDisable(istrip,noiseRange)?1.:0.);
	}catch(cms::Exception& e){
	  edm::LogError("SiStripCondObjDisplay") << "[SiStripCondObjDisplay::endJob]  cms::Exception:  DetName " << name << " " << e.what() ;
	}
      }
    }
  }
}
