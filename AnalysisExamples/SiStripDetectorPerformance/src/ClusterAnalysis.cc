/*
 * $Date: 2007/03/14 08:55:51 $
 * $Revision: 1.12 $
 *
 * \author: D. Giordano, domenico.giordano@cern.ch
 * Modified: M.De Mattia 2/3/2007 & R.Castello 5/4/2007
 */

#include "AnalysisExamples/SiStripDetectorPerformance/interface/ClusterAnalysis.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "sstream"

#include "Geometry/CommonDetAlgo/interface/MeasurementPoint.h"

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
    SiStripNoiseService_(conf),
    SiStripPedestalsService_(conf),
    Filter_src_( conf.getParameter<edm::InputTag>( "Filter_src" ) ),
    Track_src_( conf.getParameter<edm::InputTag>( "Track_src" ) ),
    ClusterInfo_src_( conf.getParameter<edm::InputTag>( "ClusterInfo_src" ) ),
    Cluster_src_( conf.getParameter<edm::InputTag>( "Cluster_src" ) ),
    ModulesToBeExcluded_(conf.getParameter< std::vector<uint32_t> >("ModulesToBeExcluded")),
    EtaAlgo_(conf.getParameter<int32_t>("EtaAlgo")),
    NeighStrips_(conf.getParameter<int32_t>("NeighStrips")),
    tracksCollection_in_EventTree(true),
    trackAssociatorCollection_in_EventTree(true),
    ltcdigisCollection_in_EventTree(true)
  {
    //     // Create TrackLocalAngle object to evaluate the local angles and separate the matched rechits in clusters
    //     Anglefinder = new TrackLocalAngleTIF();
  }

  ClusterAnalysis::~ClusterAnalysis(){
    std::cout << "Destructing object" << std::endl;
    //Hlist->Delete();
    delete Hlist;
    //     delete Anglefinder;
  }
  
  void ClusterAnalysis::beginJob( const edm::EventSetup& es ) {

    //get geom    
    es.get<TrackerDigiGeometryRecord>().get( tkgeom );
    edm::LogInfo("ClusterAnalysis") << "[ClusterAnalysis::beginJob] There are "<<tkgeom->detUnits().size() <<" detectors instantiated in the geometry" << std::endl;  

    es.get<SiStripDetCablingRcd>().get( SiStripDetCabling_ );

    book();
  }

  void ClusterAnalysis::book() {

    fFile = new TFile(filename_.c_str(),"RECREATE");
    fFile->mkdir("BadStrips");
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
    Hlist = new TObjArray();

    //Display 3D
    name = "ClusterGlobalPos";
    bookHlist( "TH3ClusterGlobalPos", name, "Nbinx", "xmin", "xmax", "Nbiny", "ymin", "ymax", "Nbinz", "zmin", "zmax" );
    // Take the pointer (dynamic cast since it is a TObject pointer (polymorphism))
    TH3S * temp_TH3S_ptr = dynamic_cast<TH3S*>(Hlist->Last());
    temp_TH3S_ptr->SetXTitle("z (cm)");
    temp_TH3S_ptr->SetYTitle("x (cm)");
    temp_TH3S_ptr->SetZTitle("y (cm)");

    std::cout << "added TH3D " << std::endl;
    //&&&&&&&&&&&&&&&&&&&&&&&&

    //Display TProfile
    name = "ClusterWidhtVsAngle";
    Parameters =  conf_.getParameter<edm::ParameterSet>("TProfileWidthAngle");
    Hlist->Add(new TProfile(name,name,
			    Parameters.getParameter<int32_t>("Nbinx"),
			    Parameters.getParameter<double>("xmin"),
			    Parameters.getParameter<double>("xmax"),
			    Parameters.getParameter<double>("ymin"),
			    Parameters.getParameter<double>("ymax"))
	       );
    
    //bookHlist( "TProfileWidthAngle", name, "Nbinx", "xmin", "xmax", "ymin", "ymax" );
    // Take the pointer (dynamic cast since it is a TObject pointer (polymorphism))
    TProfile * temp_TProfile_ptrA = dynamic_cast<TProfile*>(Hlist->Last());
    temp_TProfile_ptrA->SetXTitle("Angle");
    temp_TProfile_ptrA->SetYTitle("cWidth");

    std::cout << "added TProfile " << std::endl;

    //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    //Display TProfile
    name = "ResidualVsAngle";
    Parameters =  conf_.getParameter<edm::ParameterSet>("TProfileResidualAngle");
    Hlist->Add(new TProfile(name,name,
			    Parameters.getParameter<int32_t>("Nbinx"),
			    Parameters.getParameter<double>("xmin"),
			    Parameters.getParameter<double>("xmax"),
			    Parameters.getParameter<double>("ymin"),
			    Parameters.getParameter<double>("ymax"))
	       );
    
    //bookHlist( "TProfileWidthAngle", name, "Nbinx", "xmin", "xmax", "ymin", "ymax" );
    // Take the pointer (dynamic cast since it is a TObject pointer (polymorphism))
    TProfile * temp_TProfile_ptrB = dynamic_cast<TProfile*>(Hlist->Last());
    temp_TProfile_ptrB->SetXTitle("Angle");
    temp_TProfile_ptrB->SetYTitle("Residual");
    
    std::cout << "added TProfile " << std::endl;
    
    //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&



    fFile->cd();fFile->cd("Trigger");

    Hlist->Add(new TH1F("FilterBits","FilterBits",10,-0.5,9.5));

    // bookHlist( TObjArray, Name of the parameterset in the cfg, name of the hitsogram, xbinnumber, xmin, xmax )
    name = "TriggerBits";
    bookHlist( "TH1TriggerBits", name, "Nbinx", "xmin", "xmax" );
    fFile->cd();fFile->cd("Tracks");


    name = "nTracks";
    bookHlist( "TH1nTracks", name, "Nbinx", "xmin", "xmax" );
    name = "nRecHits";
    bookHlist( "TH1nRecHits", name, "Nbinx", "xmin", "xmax" );

    // Loop on onTrack, offTrack, All
    for (int j=0;j<3;j++){      
      //Number of Cluster 
      name="nClusters"+flags[j];
      fFile->cd();fFile->cd("Tracks");
      bookHlist( "TH1nClusters", name, "Nbinx", "xmin", "xmax" );

      for (int i=0;i<_NUM_SISTRIP_SUBDET_;i++) {
   	//Number of Cluster on each det
	name="nClusters"+SubDet[i]+flags[j];
	bookHlist( "TH1nClusters", name, "Nbinx", "xmin", "xmax" );
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
    	bookHlist( "TH1ClusterWidth", name, "Nbinx", "xmin", "xmax" );

	//Cluster Signal corrected
	if(j==0 )   	   
	  {
	    name="cSignalCorr"+SubDet[i]+flags[j];
	    fFile->cd();fFile->cd("ClusterSignal");
	    bookHlist( "TH1ClusterSignalCorr", name, "Nbinx", "xmin", "xmax" );
	  }

    	//Loop for cluster width
    	for (int iw=0;iw<5;iw++){
	  
    	  appString=SubDet[i]+flags[j]+width_flags[iw];
	
     	  //Cluster Noise
     	  name="cNoise"+appString;
     	  fFile->cd();fFile->cd("ClusterNoise");
     	  bookHlist( "TH1ClusterNoise", name, "Nbinx", "xmin", "xmax" );

     	  //Cluster Signal
     	  name="cSignal"+appString;
     	  fFile->cd();fFile->cd("ClusterSignal");
     	  bookHlist( "TH1ClusterSignal", name, "Nbinx", "xmin", "xmax" );
	  
	  
     	  //Cluster StoN
     	  name="cStoN"+appString;
     	  fFile->cd();fFile->cd("ClusterStoN");
     	  bookHlist( "TH1ClusterStoN", name, "Nbinx", "xmin", "xmax" );
	  
     	  //Cluster Position
     	  name="cPos"+appString;
     	  fFile->cd();fFile->cd("ClusterPos");
     	  bookHlist( "TH1ClusterPos", name, "Nbinx", "xmin", "xmax" );

     	  //Cluster Charge Division (only for study on Raw Data Runs)
     	  name="cEta"+appString;
     	  fFile->cd();fFile->cd("ClusterEta");
     	  bookHlist( "TH1ClusterEta", name, "Nbinx", "xmin", "xmax" );

     	  name="cEta_scatter"+appString;
     	  fFile->cd();fFile->cd("ClusterEta");
     	  bookHlist( "TH2ClusterEta", name, "Nbinx", "xmin", "xmax", "Nbiny", "ymin", "ymax" );

	  //       sprintf(name,"BadStrips_Cumulative_%s_%s",SubDet[i].c_str(),flags[j].c_str());
	  //       Parameters =  conf_.getParameter<edm::ParameterSet>("TH1BadStrips");
	  //       fFile->cd();fFile->cd("BadStrips");
	  //       _TH1F_BadStrips_v.push_back(new TH1F(name,name,
	  // 					   Parameters.getParameter<int32_t>("Nbinx"),
	  // 					   Parameters.getParameter<double>("xmin"),
	  // 					   Parameters.getParameter<double>("xmax")
	  // 					   )
	  // 				  );
     	}//end loop on width 
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

      //std::cout << "Confronto " << _StripGeomDetUnit->specificType().subDetector() << " " << DetId(detid).subdetId() << std::endl;
      
      edm::LogError("ClusterAnalysis") << " Detid " << detid << " SubDet " << GetSubDetAndLayer(detid).first << " Layer " << GetSubDetAndLayer(detid).second << std::endl;   
      if (DetectedLayers.find(GetSubDetAndLayer(detid)) == DetectedLayers.end()){
	DetectedLayers[GetSubDetAndLayer(detid)]=true;
      }
      //       sprintf(name,"Pedestals_%s_%d",_StripGeomDetUnit->type().name().c_str(),detid);
      //       fFile->cd();fFile->cd("Pedestals");
      //       _TH1F_PedestalsProfile_m[detid] = new TH1F(name,name,nstrips,-0.5,nstrips-0.5);
      
      char cdetid[128];
      sprintf(cdetid,"%d",detid);
           
      fFile->cd();
      fFile->mkdir(cdetid);    
      fFile->cd(cdetid);    
      char aname[128];
      sprintf(aname,"%s_%d",_StripGeomDetUnit->type().name().c_str(),detid);
      char SubStr[128];
      //      char * ptr = strchr(aname,":");
      sprintf(SubStr,"%s",strstr(aname,":"));
      //TString appString=TString(_StripGeomDetUnit->type().name()).ReplaceAll("FieldParameters:","_")+"_"+cdetid;
      //TString appString=TString(aname);//+"_"+cdetid;
      TString appString=TString(SubStr);//+"_"+cdetid;
      
      //Cluster Noise
      name="cNoise"+appString;
      bookHlist( "TH1ClusterNoise", name, "Nbinx", "xmin", "xmax" );

      //Cluster Position
      name="cPos"+appString;
      fFile->cd();fFile->cd("ClusterPos");
      bookHlist( "TH1ClusterPos", name, "Nbinx", "xmin", "xmax" );

      //Cluster Charge Division (only for study on Raw Data Runs)
      name="cEta"+appString;
      fFile->cd();fFile->cd("ClusterEta");
      bookHlist( "TH1ClusterEta", name, "Nbinx", "xmin", "xmax" );

      name="cEta_scatter"+appString;
      fFile->cd();fFile->cd("ClusterEta");
      bookHlist( "TH2ClusterEta", name, "Nbinx", "xmin", "xmax", "Nbiny", "ymin", "ymax" );

      //Cluster Signal
      name="cSignal"+appString;
      bookHlist( "TH1ClusterSignal", name, "Nbinx", "xmin", "xmax" );

      //Cluster Signal corrected
      // name="cSignalCorr"+appString;
      //bookHlist( "TH1ClusterSignalCorr", name, "Nbinx", "xmin", "xmax" );

      //Cluster StoN
      name="cStoN"+appString;
      bookHlist( "TH1ClusterStoN", name, "Nbinx", "xmin", "xmax" );

      //Cluster Width
      name="cWidth"+appString;
      bookHlist( "TH1ClusterWidth", name, "Nbinx", "xmin", "xmax" );
      //       Parameters =  conf_.getParameter<edm::ParameterSet>("TH1ClusterWidth");
      //       Hlist->Add(new TH1F(name,name,
      // 			  Parameters.getParameter<int32_t>("Nbinx"),
      // 			  Parameters.getParameter<double>("xmin"),
      // 			  Parameters.getParameter<double>("xmax")
      // 			  )
      // 		 );

      //Cluster Position
      name="cPos"+appString;
      Hlist->Add(new TH1F(name,name,nstrips,-0.5,nstrips-.5));
      
      //&&&&&&&&&&&&&&&&&
    }//end loop on detector	


    //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    //Layer Detail Plots
    
    for (std::map<std::pair<std::string,uint32_t>,bool>::const_iterator iter=DetectedLayers.begin(); iter!=DetectedLayers.end();iter++){

      char cApp[64];
      sprintf(cApp,"_Layer_%d",iter->first.second);
      TString appString=TString(iter->first.first)+cApp;
     
      fFile->cd(); fFile->cd("Layer");    

      //residual
      name="res_x"+appString;
      bookHlist( "TH1Residual_x", name, "Nbinx", "xmin", "xmax" );
      
      
      //residual y
      name="res_y"+appString;
      bookHlist( "TH1Residual_y", name, "Nbinx", "xmin", "xmax" );
      
      
      //Cluster Noise
      name="cNoise"+appString;
      bookHlist( "TH1ClusterNoise", name, "Nbinx", "xmin", "xmax" );

      //Cluster Signal
      name="cSignal"+appString;
      bookHlist( "TH1ClusterSignal", name, "Nbinx", "xmin", "xmax" );

      //Cluster Signal corrected
      name="cSignalCorr"+appString;
      bookHlist( "TH1ClusterSignalCorr", name, "Nbinx", "xmin", "xmax" );

      //Cluster StoN
      name="cStoN"+appString;
      bookHlist( "TH1ClusterStoN", name, "Nbinx", "xmin", "xmax" );

      //Cluster Width
      name="cWidth"+appString;
      bookHlist( "TH1ClusterWidth", name, "Nbinx", "xmin", "xmax" );

      //Cluster Position
      name="cPos"+appString;
      bookHlist( "TH1ClusterPos", name, "Nbinx", "xmin", "xmax" );
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
      if (dynamic_cast<TH1F*>((*Hlist)[ih]) !=NULL){
	if (dynamic_cast<TH1F*>((*Hlist)[ih])->GetEntries() != 0)
	  (*Hlist)[ih]->Draw();
      }
      else if (dynamic_cast<TProfile*>((*Hlist)[ih]) !=NULL){
	(*Hlist)[ih]->Draw();
      }   
      else if (dynamic_cast<TH3S*>((*Hlist)[ih]) !=NULL){
	(*Hlist)[ih]->Draw();
      }
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

    //SiStripNoiseService_.setESObjects(es);
    //SiStripPedestalsService_.setESObjects(es);
    
    //Get input
    e.getByLabel( ClusterInfo_src_, dsv_SiStripClusterInfo);
    e.getByLabel( Cluster_src_, dsv_SiStripCluster);    
    
    //    e.getByLabel( Filter_src_, filterWord);
    
    try{
      e.getByType(ltcdigis);
    } catch ( cms::Exception& er ) {
      LogTrace("ClusterAnalysis")<<"caught std::exception "<<er.what()<<std::endl;
      ltcdigisCollection_in_EventTree=false;
    }catch ( ... ) {
      LogTrace("ClusterAnalysis")<< " funny error " <<std::endl;
      ltcdigisCollection_in_EventTree=false;
    }
    
    try{
      e.getByLabel(Track_src_, trackCollection);
    } catch ( cms::Exception& er ) {
      LogTrace("ClusterAnalysis")<<"caught std::exception "<<er.what()<<std::endl;
      tracksCollection_in_EventTree=false;
    } catch ( ... ) {
      LogTrace("ClusterAnalysis")<<" funny error " <<std::endl;
      tracksCollection_in_EventTree=false;
    }
    
    // // TrackInfoAssociator Collections
//     edm::InputTag TkiTag = conf_.getParameter<edm::InputTag>( "TrackInfoLabel" );
//     try{
//       e.getByLabel(TkiTag,TItkAssociatorCollection);
//     } catch ( cms::Exception& er ) {
//       LogTrace("ClusterAnalysis")<<"caught std::exception "<<er.what()<<std::endl;
//       trackAssociatorCollection_in_EventTree=false;
//     } catch ( ... ) {
//       LogTrace("ClusterAnalysis")<<" funny error " <<std::endl;
//       trackAssociatorCollection_in_EventTree=false;
//     }
    
    //-------------------------------------------

    edm::InputTag TkiTagCmb = conf_.getParameter<edm::InputTag>( "TrackInfoLabelCmb" );  
    try{
      e.getByLabel(TkiTagCmb, tkiTkAssCollectionCmb); 
    } catch ( cms::Exception& er ) {
      LogTrace("ClusterAnalysis")<<"caught std::exception "<<er.what()<<std::endl;
     trackAssociatorCollection_in_EventTree=false;
    } catch ( ... ) {
      LogTrace("ClusterAnalysis")<<" funny error " <<std::endl;
      trackAssociatorCollection_in_EventTree=false;
    }
    
    
    edm::InputTag TkiTagUpd = conf_.getParameter<edm::InputTag>( "TrackInfoLabelUpd" );  
    try{
      e.getByLabel(TkiTagUpd, tkiTkAssCollectionUpd); 
    } catch ( cms::Exception& er ) {
      LogTrace("ClusterAnalysis")<<"caught std::exception "<<er.what()<<std::endl;
      trackAssociatorCollection_in_EventTree=false;
    } catch ( ... ) {
      LogTrace("ClusterAnalysis")<<" funny error " <<std::endl;
      trackAssociatorCollection_in_EventTree=false;
    }
    
    //--------------------------------------------

    //edm::InputTag TkiTag = conf_.getParameter<edm::InputTag>( "TrackInfoLabel" );
    //e.getByLabel( TkiTag, TItkAssociatorCollection );

    vPSiStripCluster.clear();
    countOn=0;
    countOff=0;
    countAll=0;
    // istart=oXZHitAngle.size();  
    
    //     Anglefinder->init(es);
    //
    // get geometry to evaluate local angles
    //
    edm::ESHandle<TrackerGeometry> estracker;
    es.get<TrackerDigiGeometryRecord>().get(estracker);
    _tracker=&(* estracker);

    //     //Filter bit word
    //     TH1F * HFilt = (TH1F*) Hlist->FindObject("FilterBits");
    //     for (int i=0;i<10;i++){
    //       if( *(filterWord.product()) >> i & 0x1u )
    // 	HFilt->Fill(i);
    //     }
    
    //Trigger bits
    if (ltcdigisCollection_in_EventTree){
      TH1F * Htrig = (TH1F*) Hlist->FindObject("TriggerBits");
      for (std::vector<LTCDigi>::const_iterator ltc_it =
	     ltcdigis->begin(); ltc_it != ltcdigis->end(); ltc_it++){
	for (int i=0;i<6;i++)
	  if ((*ltc_it).HasTriggered(i))
	    Htrig->Fill(i);
      }
    }

    
    //Perform track study
    if (tracksCollection_in_EventTree || trackAssociatorCollection_in_EventTree)
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
    ((TH1F*) Hlist->FindObject("nTracks"))->Fill(nTracks);

    int i=0;
    for (reco::TrackCollection::const_iterator track=tC.begin(); track!=tC.end(); track++){
      LogTrace("ClusterAnalysis")
	<< "Track number "<< i+1 
	<< "\n\tmomentum: " << track->momentum()
	<< "\n\tPT: " << track->pt()
	<< "\n\tvertex: " << track->vertex()
	<< "\n\timpact parameter: " << track->d0()
	<< "\n\tcharge: " << track->charge()
	<< "\n\tnormalizedChi2: " << track->normalizedChi2() 
	<<"\n\tFrom EXTRA : "
	<<"\n\t\touter PT "<< track->outerPt()<<std::endl;

      // TrackInfo Map, extract TrackInfo for this track
      reco::TrackRef trackref = reco::TrackRef(trackCollection, i);
      
      //reco::TrackInfoRef trackinforef=(*TItkAssociatorCollection.product())[trackref];
      reco::TrackInfoRef trackinforefUpd=(*tkiTkAssCollectionUpd.product())[trackref];      
      reco::TrackInfoRef trackinforefCmb=(*tkiTkAssCollectionCmb.product())[trackref];

      std::vector<std::pair<const TrackingRecHit*, float> > hitangle;
      hitangle = SeparateHits(trackinforefUpd);
      i++;
      
      //------------------------------------------------------------------------
      //
      // try and access Hits
      //
      // WORK IN PROGRESS
      // Continue from here: when the vPSiStripCluster is filled, could fill also a vPSiStripClusterAngle vector, of maybe
      // turn vPSiStripCluster into a map.
      // ---------------------------------

      int recHitsSize=track->recHitsSize();
      edm::LogInfo("ClusterAnalysis") <<"\t\tNumber of RecHits "<<recHitsSize<<std::endl;
      ((TH1F*) Hlist->FindObject("nRecHits"))->Fill(recHitsSize);



      //------------------------------RESIDUAL at the layer level ------------
	  
      for(_tkinfoCmbiter=trackinforefCmb->trajStateMap().begin();_tkinfoCmbiter!=trackinforefCmb->trajStateMap().end();_tkinfoCmbiter++){
	
	const SiStripMatchedRecHit2D* matchedhit=dynamic_cast<const SiStripMatchedRecHit2D*>(&(*(_tkinfoCmbiter->first)));
	const SiStripRecHit2D* hit=dynamic_cast<const SiStripRecHit2D*>(&(*(_tkinfoCmbiter->first)));

	if(matchedhit)  
	  {
	  
	    //track angle
	    LocalVector trackdirection=(_tkinfoCmbiter->second.parameters()).momentum();
	    GluedGeomDet * gdet=(GluedGeomDet *)_tracker->idToDet(matchedhit->geographicalId());
	    GlobalVector gtrkdir=gdet->toGlobal(trackdirection);
	    float angle = atan(gtrkdir.x()/gtrkdir.z())*180/TMath::Pi();
	    
	    const SiStripRecHit2D *monohit = matchedhit->monoHit();
	    //const SiStripRecHit2D *stereohit = matchedhit->stereoHit();
	    const uint32_t& detid = monohit->geographicalId().rawId();
	    
	    char cApp[64];
	    sprintf(cApp,"_Layer_%d",GetSubDetAndLayer(detid).second);
	    TString appString=TString(GetSubDetAndLayer(detid).first)+cApp; 
	    //std::cout << "MATCHED::subdet_layer->"<<appString << std::endl;
	    
	    //---------------------------
	    
	    LocalPoint stateposition= (_tkinfoCmbiter->second.parameters()).position();	
	    LocalPoint rechitposition= matchedhit->localPosition();
	    fillTH1( stateposition.x() - rechitposition.x(),"res_x"+appString,0);
	    fillTH1( stateposition.y() - rechitposition.y(),"res_y"+appString,0);
	    ((TProfile*) Hlist->FindObject("ResidualVsAngle"))->Fill(angle,stateposition.x()- rechitposition.x(),1);
	    
	  }
	//  std::cout << "detidCmb from track" <<detidCmb << "versus" <<(&(*(*_tkinfoCmbiter).first))->geographicalId().rawId()<< std::endl;
	else if ( hit ){
	  
	  //track angle
	  LocalVector trackdirection=(_tkinfoCmbiter->second.parameters()).momentum();
	  GluedGeomDet * gdet=(GluedGeomDet *)_tracker->idToDet(hit->geographicalId());
	  GlobalVector gtrkdir=gdet->toGlobal(trackdirection);
	  float angle = atan(gtrkdir.x()/gtrkdir.z())*180/TMath::Pi();
	  
	  char cApp[64];
	  const uint32_t& detid = hit->geographicalId().rawId();
	  //const unsigned int detid = ((*_tkinfoCmbiter).second).detId();
	  sprintf(cApp,"_Layer_%d",GetSubDetAndLayer(detid).second);
	  TString appString=TString(GetSubDetAndLayer(detid).first)+cApp;
	  //std::cout << " NOT_MATCHED::subdet_layer->"<<appString << std::endl;
	  
	  LocalPoint  stateposition= (_tkinfoCmbiter->second.parameters()).position(); //trajectory position on the detector
	  LocalPoint  rechitposition = (_tkinfoCmbiter->first)->localPosition();// rechit position on the detector
	  fillTH1( stateposition.x()- rechitposition.x(),"res_x"+appString,0);
	  fillTH1( stateposition.y()- rechitposition.y(),"res_y"+appString,0);
	  ((TProfile*) Hlist->FindObject("ResidualVsAngle"))->Fill(angle,stateposition.x()- rechitposition.x(),1);
	}
	else {std::cout <<"---No RecHit found-----------"<< std::endl;}
      }
      
      //---------------------------------------------------------------------
  
      //       for (trackingRecHit_iterator it = track->recHitsBegin();  it != track->recHitsEnd(); it++){
      // Loop directly on the vector
      // We are using clusters now, so no matched hits
      std::vector<std::pair<const TrackingRecHit*, float> >::const_iterator tkangle_iter;
      // int iter=0;
      for ( tkangle_iter = hitangle.begin(); tkangle_iter != hitangle.end(); ++tkangle_iter ) {
	// 	const TrackingRecHit* trh = &(**it);
	const TrackingRecHit* trh = tkangle_iter->first;
	const uint32_t& detid = trh->geographicalId().rawId();
	if (find(ModulesToBeExcluded_.begin(),ModulesToBeExcluded_.end(),detid)!=ModulesToBeExcluded_.end())
	  continue;
	
	if (trh->isValid()){
	  LogTrace("ClusterAnalysis")
	    <<"\n\t\tRecHit on det "<<trh->geographicalId().rawId()
	    <<"\n\t\tRecHit in LP "<<trh->localPosition()
	    <<"\n\t\tRecHit track angle "<<tkangle_iter->second
	    <<"\n\t\tRecHit in GP "<<tkgeom->idToDet(trh->geographicalId())->surface().toGlobal(trh->localPosition()) <<std::endl; 
	  
	  
	  //Get SiStripCluster from SiStripRecHit
	  const SiStripRecHit2D* hit=dynamic_cast<const SiStripRecHit2D*>(trh);
	  //const edm::Ref<edm::DetSetVector<SiStripCluster>, SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster> > clust=hit.cluster();
	  if ( hit != NULL ){
	    LogTrace("ClusterAnalysis") << "GOOD hit" << std::endl;
	    const SiStripCluster* SiStripCluster_ = &*(hit->cluster());
	    const SiStripClusterInfo* SiStripClusterInfo_ = MatchClusterInfo(SiStripCluster_,detid);
	    
	    // WORK IN PROGRESS
	    // Pass the angle here
	    //std::cout << "tkangle_iter->second " << tkangle_iter->second<<std::endl;
	    
	    if ( clusterInfos(SiStripClusterInfo_,detid,"_onTrack", tkangle_iter->second ) ) {
	      vPSiStripCluster.push_back(SiStripCluster_);
	      countOn++;
	      
	    }
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

      if (find(ModulesToBeExcluded_.begin(),ModulesToBeExcluded_.end(),detid)!=ModulesToBeExcluded_.end())
	continue;
      
      //Loop on Clusters
      LogTrace("ClusterAnalysis") << "\n["<<__PRETTY_FUNCTION__<<"] \n on detid "<< detid << " N Cluster= " << DSViter->data.size() <<std::endl;
      
      edm::DetSet<SiStripCluster>::const_iterator ClusIter = DSViter->data.begin();
      for(; ClusIter!=DSViter->data.end(); ClusIter++) {

	const SiStripClusterInfo* SiStripClusterInfo_=MatchClusterInfo(&*ClusIter,detid);
	if ( clusterInfos(SiStripClusterInfo_,detid,"_All") ){
	  countAll++;

	  LogTrace("ClusterAnalysis") << "\n["<<__PRETTY_FUNCTION__<<"] ClusIter " << &*ClusIter << 
	    "\t " << std::find(vPSiStripCluster.begin(),vPSiStripCluster.end(),&*ClusIter)-vPSiStripCluster.begin() << std::endl;

	  if (std::find(vPSiStripCluster.begin(),vPSiStripCluster.end(),&*ClusIter) == vPSiStripCluster.end()){
	    if ( clusterInfos(SiStripClusterInfo_,detid,"_offTrack") )
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
  
  bool ClusterAnalysis::clusterInfos(const SiStripClusterInfo* cluster, const uint32_t& detid,TString flag , float angle){
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

    //std::cout<<"angle1-> "<<angle<<std::endl;

    //Display
    ((TH3S*) Hlist->FindObject("ClusterGlobalPos"))
      ->Fill(globalPos.z(),globalPos.x(),globalPos.y());

    //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    //Cumulative Plots

    TString appString=SubDet[SubDet_enum]+flag;    
    //std::cout << "appString "<<appString << std::endl;

    
    fillTH1(cluster->charge(),"cSignal"+appString,1,cluster->width());


    if(flag=="_onTrack"){
      	fillTH1(cluster->charge()*cos((angle/180)* TMath::Pi()),"cSignalCorr"+appString,0);
	((TProfile*) Hlist->FindObject("ClusterWidhtVsAngle"))
	  ->Fill(angle,cluster->width(),1);
    }
    
    fillTH1(cluster->noise(),"cNoise"+appString,1,cluster->width());

    if (cluster->noise()){
      fillTH1(cluster->charge()/cluster->noise(),"cStoN"+appString,1,cluster->width());
    }
      
    fillTH1(cluster->width(),"cWidth"+appString,0);

    fillTH1(cluster->position(),"cPos"+appString,1,cluster->width());
   
     
    //---- ClusterCharge corrected by angle (Layer)-----
    char cAppL[64];
    sprintf(cAppL,"_Layer_%d",GetSubDetAndLayer(detid).second);
    TString appStringL=TString(GetSubDetAndLayer(detid).first)+cAppL;
    if (flag=="_onTrack")fillTH1(cluster->charge()*cos((angle/180)* TMath::Pi()),"cSignalCorr"+appStringL,1,cluster->width());
    //---------------------------------------------------

    if (cluster->rawdigiAmplitudesL().size()!=0 ||  cluster->rawdigiAmplitudesR().size()!=0){
	
      float Ql=0;
      float Qr=0;
      float Qt=0;

      if (EtaAlgo_==1){
	Ql=cluster->chargeL();
	Qr=cluster->chargeR();
      
  	for (std::vector<int16_t>::const_iterator it=cluster->rawdigiAmplitudesL().begin(); it !=cluster->rawdigiAmplitudesL().end() && it-cluster->rawdigiAmplitudesL().begin()<NeighStrips_; it ++)
  	  { Ql += (*it);}

	
  	for (std::vector<int16_t>::const_iterator it=cluster->rawdigiAmplitudesR().begin(); it !=cluster->rawdigiAmplitudesR().end() && it-cluster->rawdigiAmplitudesR().begin()<NeighStrips_; it ++)
  	  { Qr += (*it);}
	
	Qt=Ql+Qr+cluster->maxCharge();
      }
      else{

	std::cout << "possible error part" << std::endl;

	int Nstrip=cluster->stripAmplitudes().size();
	float pos=cluster->position()-0.5;
	for(int is=0;is<Nstrip && cluster->firstStrip()+is<=pos;is++)
	  Ql+=cluster->stripAmplitudes()[is];
	
	Qr=cluster->charge()-Ql;

  	for (std::vector<int16_t>::const_iterator it=cluster->rawdigiAmplitudesL().begin(); it !=cluster->rawdigiAmplitudesL().end() && it-cluster->rawdigiAmplitudesL().begin()<NeighStrips_; it ++)
  	  { Ql += (*it);}
	
  	for (std::vector<int16_t>::const_iterator it=cluster->rawdigiAmplitudesR().begin(); it !=cluster->rawdigiAmplitudesR().end() && it-cluster->rawdigiAmplitudesR().begin()<NeighStrips_; it ++)
  	  { Qr += (*it);}

	std::cout << "after possible error part" << std::endl;
	
	Qt=Ql+Qr;
      }
      
      LogTrace("ClusterAnalysis") << "\n["<<__PRETTY_FUNCTION__<<"] \n on detid "<< detid << " Ql=" << Ql << " Qr="<< Qr << " Qt="<<Qt<< " eta="<< Ql/Qt<< std::endl;
      
      fillTH1(Ql/Qt,"cEta"+appString,1,cluster->width());
    
      fillTH2((Ql-Qr)/Qt,(Ql+Qr)/Qt,"cEta_scatter"+appString,1,cluster->width());
    }
    
    if(flag=="_All"){
      //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
      //Detector Detail Plots
      char aname[128];
      sprintf(aname,"%s_%d",_StripGeomDetUnit->type().name().c_str(),detid);
      TString appString=TString(strstr(aname,":"));
      //appString=TString(_StripGeomDetUnit->type().name()).ReplaceAll("FieldParameters:","_")+cdetid;

      fillTH1(cluster->charge(),"cSignal"+appString,0);
      //fillTH1(cluster->charge()*TMath::Cos((angle/180)* TMath::Pi()),"cSignalCorr"+appString,0);

      fillTH1(cluster->noise(),"cNoise"+appString,0);

      if (cluster->noise()){
	fillTH1(cluster->charge()/cluster->noise(),"cStoN"+appString,0);
      }
      
      fillTH1(cluster->width(),"cWidth"+appString,0);


      fillTH1(cluster->position(),"cPos"+appString,0);

      //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
      // Layer Detail Plots

      char cApp[64];
      sprintf(cApp,"_Layer_%d",GetSubDetAndLayer(detid).second);
      appString=TString(GetSubDetAndLayer(detid).first)+cApp;
      
      fillTH1(cluster->charge(),"cSignal"+appString,0);

      //fillTH1(cluster->charge()*TMath::Cos((angle/180)* TMath::Pi()),"cSignalCorr"+appString,0);

      fillTH1(cluster->noise(),"cNoise"+appString,0);

      if (cluster->noise()){
	fillTH1(cluster->charge()/cluster->noise(),"cStoN"+appString,0);
      }
      
      fillTH1(cluster->width(),"cWidth"+appString,0);

      fillTH1(cluster->position(),"cPos"+appString,0);
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
	layer=TIDDetId(detid).wheel();
	break;
      case GeomDetEnumerators::TEC:
	cSubDet="TEC";
	layer=TECDetId(detid).wheel();
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

  // To book histograms
  void ClusterAnalysis::bookHlist(char* ParameterSetLabel, TString & HistoName,
				   char* Nbinx, char* xmin, char* xmax,
				   char* Nbiny, char* ymin, char* ymax,
				   char* Nbinz, char* zmin, char* zmax ) {
    if ( Nbiny == "" ) {
      Parameters =  conf_.getParameter<edm::ParameterSet>(ParameterSetLabel);
      Hlist->Add(new TH1F(HistoName,HistoName,
			  Parameters.getParameter<int32_t>(Nbinx),
			  Parameters.getParameter<double>(xmin),
			  Parameters.getParameter<double>(xmax)
			  )
		 );

    }
    else if ( Nbinz == "" ) {
      Parameters =  conf_.getParameter<edm::ParameterSet>(ParameterSetLabel);
      Hlist->Add(new TH2F(HistoName,HistoName,
			  Parameters.getParameter<int32_t>(Nbinx),
			  Parameters.getParameter<double>(xmin),
			  Parameters.getParameter<double>(xmax),
			  Parameters.getParameter<int32_t>(Nbiny),
			  Parameters.getParameter<double>(ymin),
			  Parameters.getParameter<double>(ymax)
			  )
		 );


    }
    else {
      Parameters =  conf_.getParameter<edm::ParameterSet>(ParameterSetLabel);
      Hlist->Add(new TH3S(HistoName,HistoName,
			  Parameters.getParameter<int32_t>(Nbinx),
			  Parameters.getParameter<double>(xmin),
			  Parameters.getParameter<double>(xmax),
			  Parameters.getParameter<int32_t>(Nbiny),
			  Parameters.getParameter<double>(ymin),
			  Parameters.getParameter<double>(ymax),
			  Parameters.getParameter<int32_t>(Nbinz),
			  Parameters.getParameter<double>(zmin),
			  Parameters.getParameter<double>(zmax)
			  )
		 );

    }
  }
  
  // Method to separate matched rechits in single clusters and to take
  // the cluster from projected rechits and evaluate the track angle 
  
  std::vector<std::pair<const TrackingRecHit*,float> > ClusterAnalysis::SeparateHits(reco::TrackInfoRef & trackinforef) {
    std::vector<std::pair<const TrackingRecHit*,float> >hitangleassociation;
    for(_tkinfoiter=trackinforef->trajStateMap().begin();_tkinfoiter!=trackinforef->trajStateMap().end();++_tkinfoiter) {
      const ProjectedSiStripRecHit2D* phit=dynamic_cast<const ProjectedSiStripRecHit2D*>(&(*(_tkinfoiter->first)));
      const SiStripMatchedRecHit2D* matchedhit=dynamic_cast<const SiStripMatchedRecHit2D*>(&(*(_tkinfoiter->first)));
      const SiStripRecHit2D* hit=dynamic_cast<const SiStripRecHit2D*>(&(*(_tkinfoiter->first)));
      LocalVector trackdirection=(_tkinfoiter->second.parameters()).momentum();
      

      if (phit) {
 	//phit = POINTER TO THE PROJECTED RECHIT
 	hit=&(phit->originalHit());
	std::cout << "ProjectedHit found" << std::endl;
      }
      if(matchedhit){//if matched hit...
	std::cout<<"MatchedHit found"<<std::endl;
 	GluedGeomDet * gdet=(GluedGeomDet *)_tracker->idToDet(matchedhit->geographicalId());

 	GlobalVector gtrkdir=gdet->toGlobal(trackdirection);
 	std::cout<<"Track direction trasformed in global direction"<<std::endl;
	
 	//cluster and trackdirection on mono det
	
 	// THIS THE POINTER TO THE MONO HIT OF A MATCHED HIT 
 	const SiStripRecHit2D *monohit=matchedhit->monoHit();
	
 	const edm::Ref<edm::DetSetVector<SiStripCluster>, SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster> > monocluster=monohit->cluster();
 	const GeomDetUnit * monodet=gdet->monoDet();
	
 	LocalVector monotkdir=monodet->toLocal(gtrkdir);
 	//size=(monocluster->amplitudes()).size();
 	if(monotkdir.z()!=0){
	  
 	  // THE LOCAL ANGLE (MONO)
 	  float angle = atan(monotkdir.x()/ monotkdir.z())*180/TMath::Pi();
 	  //
 	  hitangleassociation.push_back(std::make_pair(monohit, angle)); 
 	  oXZHitAngle.push_back( std::make_pair( monohit, atan( monotkdir.x()/ monotkdir.z())));
 	  oYZHitAngle.push_back( std::make_pair( monohit, atan( monotkdir.y()/ monotkdir.z())));
 	  oLocalDir.push_back( std::make_pair( monohit, monotkdir));
 	  oGlobalDir.push_back( std::make_pair( monohit, gtrkdir));
 	  //std::cout<<"Angle="<<atan(monotkdir.x(), monotkdir.z())*180/TMath::Pi()<<std::endl;
	  
 	  //cluster and trackdirection on stereo det
	  
 	  // THIS THE POINTER TO THE STEREO HIT OF A MATCHED HIT 
 	  const SiStripRecHit2D *stereohit=matchedhit->stereoHit();
	  
 	  const edm::Ref<edm::DetSetVector<SiStripCluster>, SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster> > stereocluster=stereohit->cluster();
 	  const GeomDetUnit * stereodet=gdet->stereoDet(); 
 	  LocalVector stereotkdir=stereodet->toLocal(gtrkdir);
 	  //size=(stereocluster->amplitudes()).size();
	  if(stereotkdir.z()!=0){
	    
 	    // THE LOCAL ANGLE (STEREO)
 	    float angle = atan(stereotkdir.x()/ stereotkdir.z())*180/TMath::Pi();
 	    hitangleassociation.push_back(std::make_pair(stereohit, angle)); 
 	    oXZHitAngle.push_back( std::make_pair( stereohit, atan( stereotkdir.x()/ stereotkdir.z())));
 	    oYZHitAngle.push_back( std::make_pair( stereohit, atan( stereotkdir.y()/ stereotkdir.z())));
 	    oLocalDir.push_back( std::make_pair( stereohit, stereotkdir));
 	    oGlobalDir.push_back( std::make_pair( stereohit, gtrkdir));
 	  }
 	}
      }
      else if(hit) {
	//  hit= POINTER TO THE RECHIT
	const edm::Ref<edm::DetSetVector<SiStripCluster>, SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster> > cluster=hit->cluster();
	GeomDet * gdet=(GeomDet *)_tracker->idToDet(hit->geographicalId());
	//size=(cluster->amplitudes()).size();
	  
	if(trackdirection.z()!=0){
	    
	  // THE LOCAL ANGLE (STEREO)
	  float angle = atan(trackdirection.x()/ trackdirection.z())*180/TMath::Pi();
	  hitangleassociation.push_back(std::make_pair(hit, angle)); 
	  oXZHitAngle.push_back( std::make_pair( hit, atan( trackdirection.x()/ trackdirection.z())));
	  oYZHitAngle.push_back( std::make_pair( hit, atan( trackdirection.y()/ trackdirection.z())));
	  oLocalDir.push_back( std::make_pair( hit, trackdirection));
	  GlobalVector gtrkdir=gdet->toGlobal(trackdirection);
	  oGlobalDir.push_back( std::make_pair( hit, gtrkdir));
	}
      }
      else {
	std::cout << "not matched, mono or projected rechit" << std::endl;
      }
    } // end loop on rechits
    return (hitangleassociation);
  }
}
