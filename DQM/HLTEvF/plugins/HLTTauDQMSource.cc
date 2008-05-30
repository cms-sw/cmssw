#include "DQM/HLTEvF/interface/HLTTauDQMSource.h"


using namespace std;
using namespace edm;
using namespace reco;
using namespace l1extra;
using namespace trigger;

//
// constructors and destructor
//
HLTTauDQMSource::HLTTauDQMSource( const edm::ParameterSet& ps ) :counterEvt_(0)
{

  //Get General Monitoring Parameters
  ParameterSet mainParams = ps.getParameter<edm::ParameterSet>("MonitorSetup");
  outputFile_             = ps.getUntrackedParameter < std::string > ("outputFile", "");
  disable_                = ps.getUntrackedParameter < bool > ("disableROOToutput", false);
  prescaleEvt_            = ps.getUntrackedParameter<int>("prescaleEvt", -1);
  verbose_                = ps.getUntrackedParameter < bool > ("verbose", false);
  EtMin_                  = ps.getUntrackedParameter < double > ("HistEtMin", 0);
  EtMax_                  = ps.getUntrackedParameter < double > ("HistEtMax", 100);
  NEtBins_                = ps.getUntrackedParameter < int > ("HistNEtBins",20 );
  NEtaBins_               = ps.getUntrackedParameter < int > ("HistNEtaBins",20 );
  mainFolder_             = ps.getUntrackedParameter < std::string > ("DQMFolder","HLT/HLTMonTau" );


  //MONITOR SETUP
  
  monitorName_            = mainParams.getUntrackedParameter<string>("monitorName","DoubleTau");
  nTriggeredTaus_         = mainParams.getUntrackedParameter < unsigned > ("NTriggeredTaus", 2);
  triggerInfo_            = mainParams.getParameter<InputTag>("TriggerEventObject");
  l1Filter_               = mainParams.getUntrackedParameter<string>("L1SeedFilter","");
  l2Filter_               = mainParams.getUntrackedParameter<string>("L2EcalIsolFilter","");
  l25Filter_              = mainParams.getUntrackedParameter<string>("L25PixelIsolFilter","");
  l3Filter_               = mainParams.getUntrackedParameter<string>("L3SiliconIsolFilter","");

  refFilters_             = mainParams.getUntrackedParameter<std::vector<string> >("refFilters");
  refFilterDesc_          = mainParams.getUntrackedParameter<std::vector<string> >("refFilterDescriptions");
  refIDs_                 = mainParams.getUntrackedParameter<std::vector<int> >("refFilterIDs");
  PtCut_                  = mainParams.getUntrackedParameter<std::vector<double> >("refObjectPtCut"); 
  corrDeltaR_             = mainParams.getUntrackedParameter<double>("matchingDeltaR",0.5);


  //L2 DQM Setup
  ParameterSet l2Params  = ps.getParameter<edm::ParameterSet>("L2Monitoring");
  doL2Monitoring_        = l2Params.getUntrackedParameter < bool > ("doL2Monitoring", false);
  l2AssocMap_            = l2Params.getUntrackedParameter<InputTag>("L2AssociationMap");

 
  //L25 DQM Setup
  ParameterSet l25Params  = ps.getParameter<edm::ParameterSet>("L25Monitoring");
  doL25Monitoring_        = l25Params.getUntrackedParameter < bool > ("doL25Monitoring", false);
  l25IsolInfo_            = l25Params.getUntrackedParameter<InputTag>("L25IsolatedTauTagInfo");
  l25LeadTrackDeltaR_     = l25Params.getUntrackedParameter < double > ("L25LeadingTrackCone", 0.1);
  l25LeadTrackPt_         = l25Params.getUntrackedParameter < double > ("L25MinLeadTrackPt", 1);






     dbe_ = Service < DQMStore > ().operator->();
     dbe_->setVerbose(0);
  
   if (disable_) {
     outputFile_ = "";
   }

   if (dbe_) {
     dbe_->setCurrentFolder(mainFolder_+"/"+monitorName_);

   }


  
  


}


HLTTauDQMSource::~HLTTauDQMSource()
{
   
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  
}


//--------------------------------------------------------
void HLTTauDQMSource::beginJob(const EventSetup& context){
   
 
 
   if (dbe_) 
     {
     

       //Book Histograms
       dbe_->setCurrentFolder(mainFolder_);
       triggerBitInfoSum_ = dbe_->book1D((monitorName_+"Summary").c_str(),("TriggerAcceptance ("+monitorName_+")").c_str() ,8,0,8);
       triggerBitInfoSum_->setAxisTitle("#tau Trigger Paths");
       triggerBitInfoSum_->setBinLabel(1,"");
       triggerBitInfoSum_->setBinLabel(2,"L1Seed");
       triggerBitInfoSum_->setBinLabel(3,"");
       triggerBitInfoSum_->setBinLabel(4,"L2");
       triggerBitInfoSum_->setBinLabel(5,"");
       triggerBitInfoSum_->setBinLabel(6,"L25");
       triggerBitInfoSum_->setBinLabel(7,"");
       triggerBitInfoSum_->setBinLabel(8,"L3");
       formatHistogram(triggerBitInfoSum_,1);


       dbe_->setCurrentFolder(mainFolder_+"/"+monitorName_+"/PathSummary");
       //Book Static Histos
       triggerBitInfo_ = dbe_->book1D("triggerBitInfoTau","Trigger Bit (#tau inclusive) ",8,0,8);
       triggerBitInfo_->setAxisTitle("#tau Trigger Paths");
       triggerBitInfo_->setBinLabel(1,"");
       triggerBitInfo_->setBinLabel(2,"L1Seed");
       triggerBitInfo_->setBinLabel(3,"");
       triggerBitInfo_->setBinLabel(4,"L2");
       triggerBitInfo_->setBinLabel(5,"");
       triggerBitInfo_->setBinLabel(6,"L25");
       triggerBitInfo_->setBinLabel(7,"");
       triggerBitInfo_->setBinLabel(8,"L3");
       formatHistogram(triggerBitInfo_,1);






       triggerEfficiencyL1_ = dbe_->book1D("triggerEfficiencyRefToL1","#tau Path Efficiency with ref to L1",7,0,7);
       triggerEfficiencyL1_->setAxisTitle("#tau Trigger Paths");
       triggerEfficiencyL1_->setBinLabel(1,"");
       triggerEfficiencyL1_->setBinLabel(2,"L2");
       triggerEfficiencyL1_->setBinLabel(3,"");
       triggerEfficiencyL1_->setBinLabel(4,"L25");
       triggerEfficiencyL1_->setBinLabel(5,"");
       triggerEfficiencyL1_->setBinLabel(6,"L3");
       triggerEfficiencyL1_->setAxisRange(0,1,2);
       formatHistogram(triggerEfficiencyL1_,2);
       

       //Book Reference Histos

       for(size_t i=0;i<refFilters_.size();++i)
	 {


	   triggerBitInfoRef_.push_back(dbe_->book1D(("triggerBitInfoRef"+refFilterDesc_[i]).c_str(),("Trigger Bit (match to "+refFilterDesc_[i] + ") ").c_str(),8,0,8));

	   triggerBitInfoRef_[i]->setAxisTitle("#tau Trigger Paths");
	   triggerBitInfoRef_[i]->setBinLabel(1,"");
	   triggerBitInfoRef_[i]->setBinLabel(2,"L1Seed");
	   triggerBitInfoRef_[i]->setBinLabel(3,"");
	   triggerBitInfoRef_[i]->setBinLabel(4,"L2");
	   triggerBitInfoRef_[i]->setBinLabel(5,"");
	   triggerBitInfoRef_[i]->setBinLabel(6,"L25");
	   triggerBitInfoRef_[i]->setBinLabel(7,"");
	   triggerBitInfoRef_[i]->setBinLabel(8,"L3");
	   formatHistogram(triggerBitInfoRef_[i],1);


	   triggerEfficiencyRef_.push_back(dbe_->book1D(("triggerEfficiencyRefTo"+refFilterDesc_[i]).c_str(),("#tau Path Efficiency with ref to "+refFilterDesc_[i]).c_str() ,8,0,8));
	   triggerEfficiencyRef_[i]->setAxisTitle("#tau Trigger Paths");
	   triggerEfficiencyRef_[i]->setBinLabel(1,"");
	   triggerEfficiencyRef_[i]->setBinLabel(2,"L1Seed");
	   triggerEfficiencyRef_[i]->setBinLabel(3,"");
	   triggerEfficiencyRef_[i]->setBinLabel(4,"L2");
	   triggerEfficiencyRef_[i]->setBinLabel(5,"");
	   triggerEfficiencyRef_[i]->setBinLabel(6,"L25");
	   triggerEfficiencyRef_[i]->setBinLabel(7,"");
	   triggerEfficiencyRef_[i]->setBinLabel(8,"L3");
	   triggerEfficiencyRef_[i]->setAxisRange(0,1,2);
	   formatHistogram(triggerEfficiencyRef_[i],2);


	 }

	   
     }
 


     //Initialize Counters;
     NEventsPassedL1     =0;
     NEventsPassedL2     =0;
     NEventsPassedL25    =0;
     NEventsPassedL3     =0;
     
     for(size_t i=0;i<refFilters_.size();++i)
       {
	 NEventsPassedRefL1.push_back(0);
	 NEventsPassedRefL2.push_back(0);
	 NEventsPassedRefL25.push_back(0);
	 NEventsPassedRefL3.push_back(0);
	 NRefEvents.push_back(0);
       }

     //Book TH1F Denominators
     //     for(size_t i=0;i<refFilters_.size();++i)
     //  {
     /// EtRef_.push_back(new TH1F(("EtRef"+i).c_str(),"Reference Et",NEtBins_,EtMin_,EtMax_));
     //	 EtaRef_.push_back(new TH1F(("EtaRef"+i).c_str(),"Reference #eta",NEtaBins_,-2.5,2.5));
     //
     //  }     

     //Book L2 Histos

     if(doL2Monitoring_)
       {  


	 dbe_->setCurrentFolder(mainFolder_+"/"+monitorName_+"/L2CutDistos/"+"TauInclusive");
	 L2JetEt_             = dbe_->book1D("L2JetEt","L2 Jet E_{t}",NEtBins_,EtMin_,EtMax_);
	 L2JetEt_->setAxisTitle("Jet Candidate E_{t} [GeV]");
	 formatHistogram(L2JetEt_,1);
	 
	 L2JetEta_            = dbe_->book1D("L2JetEta","L2 Jet #eta",NEtaBins_,-2.5,2.5);
	 L2JetEta_->setAxisTitle("Jet Candidate #eta");
	 formatHistogram(L2JetEta_,1);

	 L2EcalIsolEt_        = dbe_->book1D("L2EcalIsolEt","L2 ECAL Isol. E_{t}",20,0,20);
	 L2EcalIsolEt_->setAxisTitle("ECAL Isolated E_{t} [GeV]");
	 formatHistogram(L2EcalIsolEt_,1);

	 L2TowerIsolEt_       = dbe_->book1D("L2TowerIsolEt","L2 Tower Isol. E_{t}",20,0,20);
	 L2TowerIsolEt_->setAxisTitle("Tower isolation E_{t} [GeV]");
	 formatHistogram(L2TowerIsolEt_,1);

	 L2SeedTowerEt_       = dbe_->book1D("L2SeedTowerEt","L2 Seed Tower E_{t}",20,0,60);
	 L2SeedTowerEt_->setAxisTitle("Seed Tower E_{t} [GeV]");
	 formatHistogram(L2SeedTowerEt_,1);

	 L2NClusters_         = dbe_->book1D("L2NClusters","L2 Number of Clusters",20,0,20);
	 L2NClusters_->setAxisTitle("Number of Clusters");
	 formatHistogram(L2NClusters_,1);

	 L2ClusterEtaRMS_     = dbe_->book1D("L2ClusterEtaRMS","L2 Cluster #eta RMS",20,0,1);
	 L2ClusterEtaRMS_->setAxisTitle("Cluster #eta RMS");
	 formatHistogram(L2ClusterEtaRMS_,1);

	 L2ClusterPhiRMS_     = dbe_->book1D("L2ClusterPhiRMS","L2 Cluster #phi RMS",20,0,1);
	 L2ClusterPhiRMS_->setAxisTitle("Cluster #phi RMS");
	 formatHistogram(L2ClusterPhiRMS_,1);
   
	 L2ClusterDeltaRRMS_  = dbe_->book1D("L2ClusterDRRMS","L2 Cluster #Delta R RMS",20,0,1);
	 L2ClusterDeltaRRMS_->setAxisTitle("Cluster #Delta R RMS");
	 formatHistogram(L2ClusterDeltaRRMS_,1);



	 //Book L2 Reference Histos
	 for(size_t i=0;i<refFilters_.size();++i)
	   {
	     dbe_->setCurrentFolder(mainFolder_+"/"+monitorName_+"/L2CutDistos/"+refFilterDesc_[i]);
	     L2JetEtRef_.push_back(dbe_->book1D("L2JetEt","L2 Jet E_{t}",NEtBins_,EtMin_,EtMax_));
	
	     L2JetEtaRef_.push_back(dbe_->book1D("L2JetEta","L2 Jet #eta",NEtaBins_,-2.5,2.5));

	     L2EcalIsolEtRef_.push_back(dbe_->book1D("L2EcalIsolEt","L2 ECAL Isol. E_{t}",20,0,20));
	     
	     L2TowerIsolEtRef_.push_back(dbe_->book1D("L2TowerIsolEt","L2 Tower Isol. E_{t}",20,0,20));
	     
	     L2SeedTowerEtRef_.push_back(dbe_->book1D("L2SeedTowerEt","L2 Seed Tower E_{t}",20,0,60));
	     
	     L2NClustersRef_.push_back(dbe_->book1D("L2NClusters","L2 Number of Clusters",20,0,20));
	     
	     L2ClusterEtaRMSRef_.push_back(dbe_->book1D("L2ClusterEtaRMS","L2 Cluster #eta RMS",20,0,1));
	     
	     L2ClusterPhiRMSRef_.push_back(dbe_->book1D("L2ClusterPhiRMS","L2 Cluster #phi RMS",20,0,1));
	     
	     L2ClusterDeltaRRMSRef_.push_back(dbe_->book1D("L2ClusterDRRMS","L2 Cluster #Delta R RMS",20,0,1));
	 
	     //Make Them beautiful

	     L2JetEtRef_[i]->setAxisTitle("Jet Candidate E_{t} [GeV]");
	     formatHistogram(L2JetEtRef_[i],1);
	     L2JetEtaRef_[i]->setAxisTitle("Jet Candidate #eta");
	     formatHistogram(L2JetEtaRef_[i],1);
	     L2EcalIsolEtRef_[i]->setAxisTitle("ECAL Isolated E_{t} [GeV]");
	     formatHistogram(L2EcalIsolEtRef_[i],1);
	     L2TowerIsolEtRef_[i]->setAxisTitle("Tower isolation E_{t} [GeV]");
	     formatHistogram(L2TowerIsolEtRef_[i],1);
	     L2SeedTowerEtRef_[i]->setAxisTitle("Seed Tower E_{t} [GeV]");
	     formatHistogram(L2SeedTowerEtRef_[i],1);
	     L2NClustersRef_[i]->setAxisTitle("Number of Clusters");
	     formatHistogram(L2NClustersRef_[i],1);
	     L2ClusterEtaRMSRef_[i]->setAxisTitle("Cluster #eta RMS");
	     formatHistogram(L2ClusterEtaRMSRef_[i],1);
	     L2ClusterPhiRMSRef_[i]->setAxisTitle("Cluster #phi RMS");
	     formatHistogram(L2ClusterPhiRMSRef_[i],1);
	     L2ClusterDeltaRRMSRef_[i]->setAxisTitle("Cluster #Delta R RMS");
	     formatHistogram(L2ClusterDeltaRRMSRef_[i],1);


	   }

       }


     //Book L25 histos
     if(doL25Monitoring_)
       {
	 dbe_->setCurrentFolder(mainFolder_+"/"+monitorName_+"/L25CutDistos/"+"TauInclusive");
	 //Book Inclusive
	 L25JetEt_           = dbe_->book1D("L25JetEt","L25 Jet Candidate E_{t}",NEtBins_,EtMin_,EtMax_);
	 L25JetEt_->setAxisTitle("L25 Jet Candidate E_{t}");
         formatHistogram(L25JetEt_,1);

	 L25JetEta_          = dbe_->book1D("L25JetEta","L25 Jet Candidate #eta",NEtaBins_,-2.5,2.5);
	 L25JetEta_->setAxisTitle("L25 Jet Candidate #eta");
	 formatHistogram(L25JetEta_,1);

	 L25NPixelTracks_    = dbe_->book1D("L25NPixelTracks","L25 Number of Pixel Tracks",20,0,20);
 	 L25NPixelTracks_->setAxisTitle("L25 # of Pixel Tracks");
	 formatHistogram(L25NPixelTracks_,1);

	 L25NQPixelTracks_   = dbe_->book1D("L25NQPixelTracks","L25 Number of Pixel Tracks(After Q test)",20,0,20);
	 L25NQPixelTracks_->setAxisTitle("L25 # of Quality Pixel Tracks");
	 formatHistogram(L25NQPixelTracks_,1);
	 
	 L25HasLeadingTrack_ = dbe_->book1D("L25HasLeadTrack","Leading Track(?)",2,0,2);
	 L25HasLeadingTrack_->setBinLabel(1,"YES");
	 L25HasLeadingTrack_->setBinLabel(2,"NO");
	 formatHistogram(L25HasLeadingTrack_,1);
	 
	 L25LeadTrackPt_     = dbe_->book1D("L25LeadTrackPt","L25 Leading Track P_{t}",60,0,60);
 	 L25LeadTrackPt_->setAxisTitle("L25 Leading Track P_{t}");
	 formatHistogram(L25LeadTrackPt_,1);

	 L25SumTrackPt_     = dbe_->book1D("L25SumTrackPt","L25 #Sigma Track P_{t}",100,0,100);
 	 L25LeadTrackPt_->setAxisTitle("L25 #Sigma Track P_{t}");
	 formatHistogram(L25SumTrackPt_,1);

       

	 //Book reference Histos
         for(size_t i=0;i<refFilters_.size();++i)
	   {

	     dbe_->setCurrentFolder(mainFolder_+"/"+monitorName_+"/L25CutDistos/"+refFilterDesc_[i]);

	     L25JetEtRef_.push_back(dbe_->book1D("L25JetEt","L25 Jet Candidate E_{t}",NEtBins_,EtMin_,EtMax_));
	     L25JetEtRef_[i]->setAxisTitle("L25 Jet Candidate E_{t}");
	     formatHistogram(L25JetEtRef_[i],1);

	     L25JetEtaRef_.push_back(dbe_->book1D("L25JetEta","L25 Jet Candidate #eta",NEtaBins_,-2.5,2.5));
	     L25JetEtaRef_[i]->setAxisTitle("L25 Jet Candidate #eta");
	     formatHistogram(L25JetEtaRef_[i],1);
	     
	     L25NPixelTracksRef_.push_back(dbe_->book1D("L25NPixelTracks","L25 Number of Pixel Tracks",20,0,20));
	     L25NPixelTracksRef_[i]->setAxisTitle("L25 # of Pixel Tracks");
	     formatHistogram(L25NPixelTracksRef_[i],1);

	     L25NQPixelTracksRef_.push_back(dbe_->book1D("L25NQPixelTracks","L25 Number of Pixel Tracks(After Q test)",20,0,20));
	     L25NQPixelTracksRef_[i]->setAxisTitle("L25 # of Quality Pixel Tracks");
	     formatHistogram(L25NQPixelTracksRef_[i],1);
	 
	     L25HasLeadingTrackRef_.push_back(dbe_->book1D("L25HasLeadTrack","Leading Track(?)",2,0,2));
	     formatHistogram(L25HasLeadingTrackRef_[i],1);
	      L25HasLeadingTrackRef_[i]->setBinLabel(1,"YES");
	      L25HasLeadingTrackRef_[i]->setBinLabel(2,"NO");

	     L25LeadTrackPtRef_.push_back(dbe_->book1D("L25LeadTrackPt","L25 Leading Track P_{t}",60,0,60));
	     L25LeadTrackPtRef_[i]->setAxisTitle("L25 Leading Track P_{t}");
	     formatHistogram(L25LeadTrackPtRef_[i],1);

	     L25SumTrackPtRef_.push_back(dbe_->book1D("L25SumTrackPt","L25 #SigmaTrack P_{t}",60,0,60));
	     L25SumTrackPtRef_[i]->setAxisTitle("L25 #Sigma Track P_{t}");
	     formatHistogram(L25SumTrackPtRef_[i],1);
	     
	   }


       }

     //Book L2 Efficiency histos with ref to L1
     // dbeq_->setCurrentFolder(monitorName_+"L2/L2Performance");
     //L2EtEff_ =  dbe_->book1D("L2EtEff","L2 E_{t} Efficiency(ref to L1)",NEtBins,EtMin_,EtMax_);
     //L2EtaEff_ =  dbe_->book1D("L2EtaEff","L2 #eta Efficiency(ref to L1)",NEtaBins,-2.5,2.5);


     //Book Reference Efficiency Histos
     //for(size_t i=0;i<refFilters_.size();++i)
     //  {
       //	 L2EtEffRef_.push_back(dbe_->book1D(("L2EtEff_"+refFilterDesc_[i]).c_str(),("L2 E_{t} Efficiency(ref to"+refFilterDesc_[i] + ")").c_str(),NEtBins,EtMin_,EtMax_));
       // L2EtaEffRef_.push_back(dbe_->book1D(("L2EtaEff_"+refFilterDesc_[i]).c_str(),("L2 #eta Efficiency(ref to"+refFilterDesc_[i] + ")").c_str(),NEtaBins,-2.5,2.5));

       // }





}

//--------------------------------------------------------
void HLTTauDQMSource::beginRun(const edm::Run& r, const EventSetup& context) {

}

//--------------------------------------------------------
void HLTTauDQMSource::beginLuminosityBlock(const LuminosityBlock& lumiSeg, 
				      const EventSetup& context) {
  
}

// ----------------------------------------------------------
void 
HLTTauDQMSource::analyze(const Event& iEvent, const EventSetup& iSetup )
{  
  //Apply the prescaler
  if(counterEvt_ > prescaleEvt_)
    {

      //Do Summary Analysis
      doSummary(iEvent,iSetup);

      //Do L2 Analysis
      if(doL2Monitoring_)
	doL2(iEvent,iSetup);

      //Do L25 Analysis
      if(doL25Monitoring_)
	doL25(iEvent,iSetup);
      
      counterEvt_ = 0;
    }
  else
      counterEvt_++;

}




//--------------------------------------------------------
void HLTTauDQMSource::endLuminosityBlock(const LuminosityBlock& lumiSeg, 
				    const EventSetup& context) {
}
//--------------------------------------------------------
void HLTTauDQMSource::endRun(const Run& r, const EventSetup& context){
}
//--------------------------------------------------------
void HLTTauDQMSource::endJob(){


 
   if (outputFile_.size() != 0 && dbe_)
   dbe_->save(outputFile_);
 
  return;


}


void 
HLTTauDQMSource::doSummary(const Event& iEvent, const EventSetup& iSetup)
{

  Handle<TriggerEventWithRefs> trigEv;
  iEvent.getByLabel(triggerInfo_,trigEv);

  if (trigEv.isValid()) 
    {


      //make The reference Denominators
      //LOOP ON the Different Triggers
      
      
      LVColl refObjects;
      
      for(size_t i=0;i<refFilters_.size();++i)
	{
	 
	  refObjects.clear();
	  refObjects = importReferenceObjects(refFilters_[i],refIDs_[i],*trigEv);
		  size_t object_counter = 0;
		  for(size_t j = 0 ; j< refObjects.size();++j)
		    {
		      if(refObjects[j].Pt()>PtCut_[i])
			{
			  object_counter++;
			  //EtRef_[i]->Fill(refObjects[j].Et());
			  //EtaRef_[i]->Fill(refObjects[j].Eta());

			}
		    }
		  if(object_counter>=nTriggeredTaus_)
		    {
		      NRefEvents[i]++;

		    }

	   
	  
	}

      //Look at the L1Trigger
      
      size_t L1ID; //L1 Trigger ID;
      L1ID = trigEv->filterIndex(l1Filter_);
      VRl1jet L1Taus;
      
      if(L1ID!=trigEv->size())
	{
  	  trigEv->getObjects(L1ID,86,L1Taus);
	  if(L1Taus.size()>=nTriggeredTaus_)
	    {
	      NEventsPassedL1++;
	    
	    }
	  //Match With REFERENCE OBjects
	  for(size_t i = 0;i<refFilters_.size();++i)
	    {
	      refObjects.clear();
	      refObjects = importReferenceObjects(refFilters_[i],refIDs_[i],*trigEv);

	      size_t match_counter = 0;
	       for(size_t j = 0;j<L1Taus.size();++j)
		{
		  if(match(*L1Taus[j],refObjects,corrDeltaR_,PtCut_[i]))
		    {
		      match_counter++;
		    }
		}
	      if(match_counter>=nTriggeredTaus_)
		{
		  NEventsPassedRefL1[i]++;
		
		}
	  

	    }

	}



      //Look at the L2Trigger
      
      size_t L2ID; //L2 Trigger ID;
      L2ID = trigEv->filterIndex(l2Filter_);
      VRjet L2Taus;
      
      if(L2ID!=trigEv->size())
	{
  	  trigEv->getObjects(L2ID,94,L2Taus);
	  if(L2Taus.size()>=nTriggeredTaus_)
	    {
	      NEventsPassedL2++;
	    
	    }
	  //Match With REFERENCE OBjects
	  for(size_t i = 0;i<refFilters_.size();++i)
	    {
	      refObjects.clear();
	      refObjects = importReferenceObjects(refFilters_[i],refIDs_[i],*trigEv);

	      size_t match_counter = 0;
	       for(size_t j = 0;j<L2Taus.size();++j)
		{
		  if(match(*L2Taus[j],refObjects,corrDeltaR_,PtCut_[i]))
		    {
		      match_counter++;
		    }
		}
	      if(match_counter>=nTriggeredTaus_)
		{
		  NEventsPassedRefL2[i]++;
		
		}
	  

	    }

	}

      //Look at the L25Trigger
      
      size_t L25ID; //L25 Trigger ID;
      L25ID = trigEv->filterIndex(l25Filter_);
      VRjet L25Taus;
      
      if(L25ID!=trigEv->size())
	{
  	  trigEv->getObjects(L25ID,94,L25Taus);
	  if(L25Taus.size()>=nTriggeredTaus_)
	    {
	      NEventsPassedL25++;
	    
	    }
	  //Match With REFERENCE OBjects
	  for(size_t i = 0;i<refFilters_.size();++i)
	    {
	      refObjects.clear();
	      refObjects = importReferenceObjects(refFilters_[i],refIDs_[i],*trigEv);

	      size_t match_counter = 0;
	       for(size_t j = 0;j<L25Taus.size();++j)
		{
		  if(match(*L25Taus[j],refObjects,corrDeltaR_,PtCut_[i]))
		    {
		      match_counter++;
		    }
		}
	      if(match_counter>=nTriggeredTaus_)
		{
		  NEventsPassedRefL25[i]++;
		
		}
	  

	    }

	}


      //Look at the L3Trigger
      
      size_t L3ID; //L3 Trigger ID;
      L3ID = trigEv->filterIndex(l3Filter_);
      VRjet L3Taus;
      
      if(L3ID!=trigEv->size())
	{
  	  trigEv->getObjects(L3ID,94,L3Taus);
	  if(L3Taus.size()>=nTriggeredTaus_)
	    {
	      NEventsPassedL3++;
	    
	    }
	  //Match With REFERENCE OBjects
	  for(size_t i = 0;i<refFilters_.size();++i)
	    {
	      refObjects.clear();
	      refObjects = importReferenceObjects(refFilters_[i],refIDs_[i],*trigEv);

	      size_t match_counter = 0;
	       for(size_t j = 0;j<L3Taus.size();++j)
		{
		  if(match(*L3Taus[j],refObjects,corrDeltaR_,PtCut_[i]))
		    {
		      match_counter++;
		    }
		}
	      if(match_counter>=nTriggeredTaus_)
		{
		  NEventsPassedRefL3[i]++;
		
		}
	  

	    }

	}



      //Fill Histogram Information

      triggerBitInfoSum_->setBinContent(2,NEventsPassedL1);
      triggerBitInfoSum_->setBinContent(4,NEventsPassedL2);
      triggerBitInfoSum_->setBinContent(6,NEventsPassedL25);
      triggerBitInfoSum_->setBinContent(8,NEventsPassedL3);


      triggerBitInfo_->setBinContent(2,NEventsPassedL1);
      triggerBitInfo_->setBinContent(4,NEventsPassedL2);
      triggerBitInfo_->setBinContent(6,NEventsPassedL25);
      triggerBitInfo_->setBinContent(8,NEventsPassedL3);
      
      //Efficiency with ref to L1
     
      triggerEfficiencyL1_->setBinContent(2,calcEfficiency(NEventsPassedL2,NEventsPassedL1)[0]);
      triggerEfficiencyL1_->setBinError(2,calcEfficiency(NEventsPassedL2,NEventsPassedL1)[1]);
      triggerEfficiencyL1_->setBinContent(4,calcEfficiency(NEventsPassedL25,NEventsPassedL1)[0]);
      triggerEfficiencyL1_->setBinError(4,calcEfficiency(NEventsPassedL25,NEventsPassedL1)[1]);
      triggerEfficiencyL1_->setBinContent(6,calcEfficiency(NEventsPassedL3,NEventsPassedL1)[0]);
      triggerEfficiencyL1_->setBinError(6,calcEfficiency(NEventsPassedL3,NEventsPassedL1)[1]);


      //REFERENCE TRIGGER STUFF
      for(size_t i=0;i<refFilters_.size();++i)
	{
	  triggerBitInfoRef_[i]->setBinContent(2,NEventsPassedRefL1[i]);
	  triggerBitInfoRef_[i]->setBinContent(4,NEventsPassedRefL2[i]);
	  triggerBitInfoRef_[i]->setBinContent(6,NEventsPassedRefL25[i]);
	  triggerBitInfoRef_[i]->setBinContent(8,NEventsPassedRefL3[i]);

	  //Efficiency With Ref to Electron trigger
	  triggerEfficiencyRef_[i]->setBinContent(2,calcEfficiency(NEventsPassedRefL1[i],NRefEvents[i])[0]);
	  triggerEfficiencyRef_[i]->setBinError(2,calcEfficiency(NEventsPassedRefL1[i],NRefEvents[i])[1]);
	  triggerEfficiencyRef_[i]->setBinContent(4,calcEfficiency(NEventsPassedRefL2[i],NRefEvents[i])[0]);
	  triggerEfficiencyRef_[i]->setBinError(4,calcEfficiency(NEventsPassedRefL2[i],NRefEvents[i])[1]);
	  triggerEfficiencyRef_[i]->setBinContent(6,calcEfficiency(NEventsPassedRefL25[i],NRefEvents[i])[0]);
	  triggerEfficiencyRef_[i]->setBinError(6,calcEfficiency(NEventsPassedRefL25[i],NRefEvents[i])[1]);
	  triggerEfficiencyRef_[i]->setBinContent(8,calcEfficiency(NEventsPassedRefL3[i],NRefEvents[i])[0]);
	  triggerEfficiencyRef_[i]->setBinError(8,calcEfficiency(NEventsPassedRefL3[i],NRefEvents[i])[1]);

	}


    


    }

}
void 
HLTTauDQMSource::doL2(const Event& iEvent, const EventSetup& iSetup)
{
  //get The trigger Element
  Handle<TriggerEventWithRefs> trigEv;
  iEvent.getByLabel(triggerInfo_,trigEv);

  if (trigEv.isValid()) 
    {
      //See if the event passed L1 Trigger
            size_t L1ID; //L1 Trigger ID;
	    L1ID = trigEv->filterIndex(l1Filter_);
	    if(L1ID!=trigEv->size())
	      {
		//That means that the L2 Trigger Has run ...So Access the isolation 
		//information
		Handle<L2TauInfoAssociation> l2TauInfoAssoc; //Handle to the input (L2 Tau Info Association)


		if(iEvent.getByLabel(l2AssocMap_,l2TauInfoAssoc))//get the Association class
	
		//If the Collection exists do work
		if(l2TauInfoAssoc->size()>0)
		  for(L2TauInfoAssociation::const_iterator p = l2TauInfoAssoc->begin();p!=l2TauInfoAssoc->end();++p)
		    {
		      //Retrieve The L2TauIsolationInfo Class from the AssociationMap
		      const L2TauIsolationInfo l2info = p->val;
		      //Retrieve the Jet From the AssociationMap
		      const Jet& jet =*(p->key);
		      
		      //Fill Inclusive Histos
		      L2JetEt_->Fill(jet.et());
		      L2JetEta_->Fill(jet.eta());
		      L2EcalIsolEt_->Fill(l2info.ECALIsolConeCut);
		      L2TowerIsolEt_->Fill(l2info.TowerIsolConeCut);
		      L2SeedTowerEt_->Fill(l2info.SeedTowerEt);
		      L2NClusters_->Fill(l2info.ECALClusterNClusters);
		      L2ClusterEtaRMS_->Fill(l2info.ECALClusterEtaRMS);
		      L2ClusterPhiRMS_->Fill(l2info.ECALClusterPhiRMS);
		      L2ClusterDeltaRRMS_->Fill(l2info.ECALClusterDRRMS);

		      //Fill Reference Histos
	            for(size_t i=0;i<refFilters_.size();++i)
		      {
			//Get reference Objects
			LVColl refObjects = importReferenceObjects(refFilters_[i],refIDs_[i],*trigEv);
			//Match Jet
			if(match(jet,refObjects,corrDeltaR_,PtCut_[i]))
			  {
			    //Fill the other histos!!
			    L2JetEtRef_[i]->Fill(jet.et());
			    L2JetEtaRef_[i]->Fill(jet.eta());
			    L2EcalIsolEtRef_[i]->Fill(l2info.ECALIsolConeCut);
			    L2TowerIsolEtRef_[i]->Fill(l2info.TowerIsolConeCut);
			    L2SeedTowerEtRef_[i]->Fill(l2info.SeedTowerEt);
			    L2NClustersRef_[i]->Fill(l2info.ECALClusterNClusters);
			    L2ClusterEtaRMSRef_[i]->Fill(l2info.ECALClusterEtaRMS);
			    L2ClusterPhiRMSRef_[i]->Fill(l2info.ECALClusterPhiRMS);
			    L2ClusterDeltaRRMSRef_[i]->Fill(l2info.ECALClusterDRRMS);
			    

			  }

		      }


		    }
	      }
    }
}
      

      


    
void 
HLTTauDQMSource::doL25(const Event& iEvent, const EventSetup& iSetup)
{

  Handle<TriggerEventWithRefs> trigEv;
  iEvent.getByLabel(triggerInfo_,trigEv);

  if (trigEv.isValid())
    {
      //Let's see if the tau jet passed L2
      size_t L2ID; //L2 Trigger ID;
      L2ID = trigEv->filterIndex(l2Filter_);
      if(L2ID!=trigEv->size())
	{
	  //Access the isolation class and do L25 analysis
	  Handle<IsolatedTauTagInfoCollection> tauTagInfo;
	  if(iEvent.getByLabel(l25IsolInfo_, tauTagInfo))
	    for(IsolatedTauTagInfoCollection::const_iterator tauTag  = tauTagInfo->begin();tauTag!=tauTagInfo->end();++tauTag)
	    {
	      //Fill Inclusive histograms
	      L25JetEt_->Fill(tauTag->jet()->et());
	      L25JetEta_->Fill(tauTag->jet()->eta());
	      L25NPixelTracks_->Fill(tauTag->allTracks().size());
	      L25NQPixelTracks_->Fill(tauTag->selectedTracks().size());
	      const TrackRef lTrack =tauTag->leadingSignalTrack(l25LeadTrackDeltaR_,l25LeadTrackPt_); 
	      if(!lTrack)
		{
		  L25HasLeadingTrack_->Fill(1.5);
		}
	      else
		{
		  L25HasLeadingTrack_->Fill(0.5);
		  L25LeadTrackPt_->Fill(lTrack->pt());

		}

	      //Calculate Sum of Track pt
	      double sumTrackPt = 0.;
		for(size_t k=0;k<tauTag->allTracks().size();++k)
		  {
		    sumTrackPt+=tauTag->allTracks()[k]->pt();
		  }

	      L25SumTrackPt_->Fill(sumTrackPt);


	      //Fill Matching Information
    	       for(size_t i=0;i<refFilters_.size();++i)
	      {
		//Get reference Objects
		LVColl refObjects = importReferenceObjects(refFilters_[i],refIDs_[i],*trigEv);
		//Match Jet
		if(match(*(tauTag->jet()),refObjects,corrDeltaR_,PtCut_[i]))
		  {
		     L25JetEtRef_[i]->Fill(tauTag->jet()->et());
		     L25JetEtaRef_[i]->Fill(tauTag->jet()->eta());
		     L25NPixelTracksRef_[i]->Fill(tauTag->allTracks().size());
		     L25NQPixelTracksRef_[i]->Fill(tauTag->selectedTracks().size());
		     
		     
		     if(!lTrack)
		       {
			  L25HasLeadingTrackRef_[i]->Fill(1.5);
	
			
		       }
		     else
		       {
			 L25HasLeadingTrackRef_[i]->Fill(0.5);
			 L25LeadTrackPtRef_[i]->Fill(lTrack->pt());
			 L25SumTrackPtRef_[i]->Fill(sumTrackPt);

		       }
		  }
	      }
	    }
	}
    }
}





bool 
HLTTauDQMSource::match(const reco::Candidate& cand,const  LVColl&  electrons/*VRelectron& electrons*/,double deltaR,double  ptMin)
{

 
  bool matched=false;

  for(size_t i = 0;i<electrons.size();++i)
      {
	double delta = ROOT::Math::VectorUtil::DeltaR(cand.p4().Vect(),electrons[i].Vect());
	if((delta<deltaR)&&electrons[i].Pt()>ptMin)
	  {
	    matched=true;
	  }
      }

  return matched;



}


std::vector<double> 
HLTTauDQMSource::calcEfficiency(int num,int denom)
{
  std::vector<double> a;

  if(denom==0)
    {
      a.push_back(0.);
      a.push_back(0.);
    }
  else
    {    
      a.push_back(((double)num)/((double)denom));
      a.push_back(sqrt(a[0]*(1-a[0])/((double)denom)));
    }
  return a;
}

LVColl 
HLTTauDQMSource::importReferenceObjects(std::string filter,int id,const trigger::TriggerEventWithRefs& trigEv)
{
  LVColl out;

  size_t ID; //Object Trigger ID
  ID =trigEv.filterIndex(filter);
  if(ID!=trigEv.size())
    {
      //Look at all Different triggers
      if(id==92) //HLT electron
	{
	  VRelectron obj;
	  trigEv.getObjects(ID,92,obj);
	  for(size_t i=0;i<obj.size();++i)
	    out.push_back(obj[i]->p4());

	}


    }

  return out;
}



void 
HLTTauDQMSource::formatHistogram(MonitorElement* m ,int type)
{
  if(type==1) //It is a simple histo
    {
      m->getTH1F()->SetFillColor(kBlue);
      // m->getTH1F()->SetLineColor(kBlue);
      //m->getTH1F()->SetLineWidth(2);
      //      m->getTH1F()->SetFillStyle(3001);
      //      m->getTH1F()->SetOptStat(111111); 
    }
  if(type==2) //It is an efficiency Histo
    {
      m->getTH1F()->SetLineColor(kBlack);
      //m->getTH1F()->SetLineWidth(2);
      m->getTH1F()->SetMarkerColor(kBlue);
      m->getTH1F()->SetMarkerStyle(20);
      //m->getTH1F()->SetOptStat(111111); 
      

    }






}
