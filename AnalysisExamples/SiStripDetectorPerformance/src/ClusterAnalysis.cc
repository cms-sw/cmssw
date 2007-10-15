/*
* $Date: 2007/02/02 18:42:37 $
* $Revision: 1.10 $
*
* \author: D. Giordano, domenico.giordano@cern.ch
*/

#include "AnalysisExamples/SiStripDetectorPerformance/interface/ClusterAnalysis.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "sstream"

#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"

#include "TTree.h"
#include "TBranch.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TH3S.h"
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
    ltcdigisCollection_in_EventTree(true)
  {}

  ClusterAnalysis::~ClusterAnalysis(){
    std::cout << "Destructing object" << std::endl;
    //Hlist->Delete();
    delete Hlist;
  }
  
  void ClusterAnalysis::beginJob( const edm::EventSetup& es ) {

    //get geom    
    es.get<TrackerDigiGeometryRecord>().get( tkgeom );
    edm::LogInfo("ClusterAnalysis") << "[ClusterAnalysis::beginJob] There are "<<tkgeom->detUnits().size() <<" detectors instantiated in the geometry" << std::endl;  

    es.get<SiStripDetCablingRcd>().get( SiStripDetCabling_ );

    book();
  }

  void ClusterAnalysis::book() {
    
    TString name;

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
    
    edm::ParameterSet Parameters;

    //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // get list of active detectors from SiStripDetCabling 

    std::vector<uint32_t> vdetId_;
    SiStripDetCabling_->addActiveDetectorsRawIds(vdetId_);
    //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    //Create histograms
    Hlist = new TObjArray();

    //Display 3D
    Parameters =  conf_.getParameter<edm::ParameterSet>("TH3ClusterGlobalPos");
    TH3S * h3=new TH3S("ClusterGlobalPos","ClusterGlobalPos",
			Parameters.getParameter<int32_t>("Nbinz"),
			Parameters.getParameter<double>("zmin"),
			Parameters.getParameter<double>("zmax"),
			Parameters.getParameter<int32_t>("Nbinx"),
			Parameters.getParameter<double>("xmin"),
			Parameters.getParameter<double>("xmax"),
			Parameters.getParameter<int32_t>("Nbiny"),
			Parameters.getParameter<double>("ymin"),
			Parameters.getParameter<double>("ymax")
		       );
    h3->SetXTitle("z (cm)");
    h3->SetYTitle("x (cm)");
    h3->SetZTitle("y (cm)");
    Hlist->Add(h3);

    std::cout << "added TH3D " << std::endl;
    //&&&&&&&&&&&&&&&&&&&&&&&&

    fFile->cd();fFile->cd("Trigger");

    Hlist->Add(new TH1F("FilterBits","FilterBits",10,-0.5,9.5));

    Parameters =  conf_.getParameter<edm::ParameterSet>("TH1TriggerBits");
    Hlist->Add(new TH1F("TriggerBits","TriggerBits",
			Parameters.getParameter<int32_t>("Nbinx"),
			Parameters.getParameter<double>("xmin"),
			Parameters.getParameter<double>("xmax")
			)
	       );


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

    // Loop on onTrack, offTrack, All
    for (int j=0;j<3;j++){      
      //Number of Cluster 
      name="nClusters"+flags[j];
      Parameters =  conf_.getParameter<edm::ParameterSet>("TH1nClusters");
      fFile->cd();fFile->cd("Tracks");
      Hlist->Add(new TH1F(name,name,
			  Parameters.getParameter<int32_t>("Nbinx"),
			  Parameters.getParameter<double>("xmin"),
			  Parameters.getParameter<double>("xmax")
			  )
		 );

      for (int i=0;i<_NUM_SISTRIP_SUBDET_;i++){
	
	//Number of Cluster on each det
	name="nClusters"+SubDet[i]+flags[j];
	Parameters =  conf_.getParameter<edm::ParameterSet>("TH1nClusters");
	fFile->cd();fFile->cd("Tracks");
	Hlist->Add(new TH1F(name,name,
			    Parameters.getParameter<int32_t>("Nbinx"),
			    Parameters.getParameter<double>("xmin"),
			    Parameters.getParameter<double>("xmax")
			    )
		   );
      }
    }

    // Loop on onTrack, offTrack, All
    for (int j=0;j<3;j++){
      //Histos for detector type
      for (int i=0;i<_NUM_SISTRIP_SUBDET_;i++){
	
	TString appString=SubDet[i]+flags[j];

	//Cluster Width
	name="cWidth"+appString;
	Parameters =  conf_.getParameter<edm::ParameterSet>("TH1ClusterWidth");
	fFile->cd();fFile->cd("ClusterWidth");
	  Hlist->Add(new TH1F(name,name,
			      Parameters.getParameter<int32_t>("Nbinx"),
			      Parameters.getParameter<double>("xmin"),
			      Parameters.getParameter<double>("xmax")
			      )
		     );

	//Loop for cluster width
	for (int iw=0;iw<5;iw++){
	  
	  appString=SubDet[i]+flags[j]+width_flags[iw];
	
	  //Cluster Noise
	  name="cNoise"+appString;
	  Parameters =  conf_.getParameter<edm::ParameterSet>("TH1ClusterNoise");
	  fFile->cd();fFile->cd("ClusterNoise");
	  Hlist->Add(new TH1F(name,name,
			      Parameters.getParameter<int32_t>("Nbinx"),
			      Parameters.getParameter<double>("xmin"),
			      Parameters.getParameter<double>("xmax")
			      )
		     );

	  //Cluster Signal
	  name="cSignal"+appString;
	  Parameters =  conf_.getParameter<edm::ParameterSet>("TH1ClusterSignal");
	  fFile->cd();fFile->cd("ClusterSignal");
	  Hlist->Add(new TH1F(name,name,
			      Parameters.getParameter<int32_t>("Nbinx"),
			      Parameters.getParameter<double>("xmin"),
			      Parameters.getParameter<double>("xmax")
			      )
		     );
	  
	  //Cluster StoN
	  name="cStoN"+appString;
	  Parameters =  conf_.getParameter<edm::ParameterSet>("TH1ClusterStoN");
	  fFile->cd();fFile->cd("ClusterStoN");
	  Hlist->Add(new TH1F(name,name,
			      Parameters.getParameter<int32_t>("Nbinx"),
			      Parameters.getParameter<double>("xmin"),
			      Parameters.getParameter<double>("xmax")
			      )
		     );
	  
	  //Cluster Position
	  name="cPos"+appString;
	  Parameters =  conf_.getParameter<edm::ParameterSet>("TH1ClusterPos");
	  fFile->cd();fFile->cd("ClusterPos");
	  Hlist->Add(new TH1F(name,name,
			      Parameters.getParameter<int32_t>("Nbinx"),
			      Parameters.getParameter<double>("xmin"),
			      Parameters.getParameter<double>("xmax")
			      )
		     );

	  //Cluster Charge Division (only for study on Raw Data Runs)
	  name="cEta"+appString;
	  Parameters =  conf_.getParameter<edm::ParameterSet>("TH1ClusterEta");
	  fFile->cd();fFile->cd("ClusterEta");
	  Hlist->Add(new TH1F(name,name,
			      Parameters.getParameter<int32_t>("Nbinx"),
			      Parameters.getParameter<double>("xmin"),
			      Parameters.getParameter<double>("xmax")
			      )
		     );

	  name="cEta_scatter"+appString;
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
      Parameters =  conf_.getParameter<edm::ParameterSet>("TH1ClusterNoise");
      Hlist->Add(new TH1F(name,name,
			  Parameters.getParameter<int32_t>("Nbinx"),
			  Parameters.getParameter<double>("xmin"),
			  Parameters.getParameter<double>("xmax")
			  )
		 );

      //Cluster Signal
      name="cSignal"+appString;
      Parameters =  conf_.getParameter<edm::ParameterSet>("TH1ClusterSignal");
      Hlist->Add(new TH1F(name,name,
			  Parameters.getParameter<int32_t>("Nbinx"),
			  Parameters.getParameter<double>("xmin"),
			  Parameters.getParameter<double>("xmax")
			  )
		 );

      //Cluster StoN
      name="cStoN"+appString;
      Parameters =  conf_.getParameter<edm::ParameterSet>("TH1ClusterStoN");
      Hlist->Add(new TH1F(name,name,
			  Parameters.getParameter<int32_t>("Nbinx"),
			  Parameters.getParameter<double>("xmin"),
			  Parameters.getParameter<double>("xmax")
			  )
		 );

      //Cluster Width
      name="cWidth"+appString;
      Parameters =  conf_.getParameter<edm::ParameterSet>("TH1ClusterWidth");
      Hlist->Add(new TH1F(name,name,
			  Parameters.getParameter<int32_t>("Nbinx"),
			  Parameters.getParameter<double>("xmin"),
			  Parameters.getParameter<double>("xmax")
			  )
		 );

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
      
      //Cluster Noise
      name="cNoise"+appString;
      Parameters =  conf_.getParameter<edm::ParameterSet>("TH1ClusterNoise");
      Hlist->Add(new TH1F(name,name,
			  Parameters.getParameter<int32_t>("Nbinx"),
			  Parameters.getParameter<double>("xmin"),
			  Parameters.getParameter<double>("xmax")
			  )
		 );

      //Cluster Signal
      name="cSignal"+appString;
      Parameters =  conf_.getParameter<edm::ParameterSet>("TH1ClusterSignal");
      Hlist->Add(new TH1F(name,name,
			  Parameters.getParameter<int32_t>("Nbinx"),
			  Parameters.getParameter<double>("xmin"),
			  Parameters.getParameter<double>("xmax")
			  )
		 );

      //Cluster StoN
      name="cStoN"+appString;
      Parameters =  conf_.getParameter<edm::ParameterSet>("TH1ClusterStoN");
      Hlist->Add(new TH1F(name,name,
			  Parameters.getParameter<int32_t>("Nbinx"),
			  Parameters.getParameter<double>("xmin"),
			  Parameters.getParameter<double>("xmax")
			  )
		 );

      //Cluster Width
      name="cWidth"+appString;
      Parameters =  conf_.getParameter<edm::ParameterSet>("TH1ClusterWidth");
      Hlist->Add(new TH1F(name,name,
			  Parameters.getParameter<int32_t>("Nbinx"),
			  Parameters.getParameter<double>("xmin"),
			  Parameters.getParameter<double>("xmax")
			  )
		 );

      //Cluster Position
      name="cPos"+appString;
      Parameters =  conf_.getParameter<edm::ParameterSet>("TH1ClusterPos");
      Hlist->Add(new TH1F(name,name,
			  Parameters.getParameter<int32_t>("Nbinx"),
			  Parameters.getParameter<double>("xmin"),
			  Parameters.getParameter<double>("xmax")
			  )
		 );
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
    
    e.getByLabel( Filter_src_, filterWord);
    
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
    
    vPSiStripCluster.clear();
    countOn=0;
    countOff=0;
    countAll=0;


    
    //Filter bit word
    TH1F * HFilt = (TH1F*) Hlist->FindObject("FilterBits");
    for (int i=0;i<10;i++){
      if( *(filterWord.product()) >> i & 0x1u )
	HFilt->Fill(i);
    }
    
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
    if (tracksCollection_in_EventTree)
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

	if (find(ModulesToBeExcluded_.begin(),ModulesToBeExcluded_.end(),detid)!=ModulesToBeExcluded_.end())
	  continue;

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
	    if ( clusterInfos(SiStripClusterInfo_,detid,"_onTrack") ){
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
	if ( clusterInfos(SiStripClusterInfo_, detid,"_All") ){
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

  bool ClusterAnalysis::clusterInfos(const SiStripClusterInfo* cluster, const uint32_t& detid,TString flag){
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


    //Display
    ((TH3S*) Hlist->FindObject("ClusterGlobalPos"))
      ->Fill(globalPos.z(),globalPos.x(),globalPos.y());
    //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    //Cumulative Plots

    TString appString=SubDet[SubDet_enum]+flag;

    fillTH1(cluster->charge(),"cSignal"+appString,1,cluster->width());

    fillTH1(cluster->noise(),"cNoise"+appString,1,cluster->width());

    if (cluster->noise()){
      fillTH1(cluster->charge()/cluster->noise(),"cStoN"+appString,1,cluster->width());
    }
      
    fillTH1(cluster->width(),"cWidth"+appString,0);

    fillTH1(cluster->position(),"cPos"+appString,1,cluster->width());

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

	int Nstrip=cluster->stripAmplitudes().size();
	float pos=cluster->position()-0.5;
	for(int is=0;is<Nstrip && cluster->firstStrip()+is<=pos;is++)
	  Ql+=cluster->stripAmplitudes()[is];
	
	Qr=cluster->charge()-Ql;

	for (std::vector<int16_t>::const_iterator it=cluster->rawdigiAmplitudesL().begin(); it !=cluster->rawdigiAmplitudesL().end() && it-cluster->rawdigiAmplitudesL().begin()<NeighStrips_; it ++)
	  { Ql += (*it);}
	
	for (std::vector<int16_t>::const_iterator it=cluster->rawdigiAmplitudesR().begin(); it !=cluster->rawdigiAmplitudesR().end() && it-cluster->rawdigiAmplitudesR().begin()<NeighStrips_; it ++)
	  { Qr += (*it);}
	
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
}


