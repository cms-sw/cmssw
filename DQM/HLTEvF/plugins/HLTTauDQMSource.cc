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
  l1Filter_               = mainParams.getUntrackedParameter<InputTag>("L1Seed");
  l2Filter_               = mainParams.getUntrackedParameter<InputTag>("L2EcalIsolJets");
  l25Filter_              = mainParams.getUntrackedParameter<InputTag>("L25PixelIsolJets");
  l3Filter_               = mainParams.getUntrackedParameter<InputTag>("L3SiliconIsolJets");

  refFilters_             = mainParams.getUntrackedParameter<std::vector<InputTag> >("refFilters");
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

  //L3 DQM Setup
  ParameterSet l3Params  = ps.getParameter<edm::ParameterSet>("L3Monitoring");
  doL3Monitoring_        = l3Params.getUntrackedParameter < bool > ("doL3Monitoring", false);
  l25IsolInfo_            = l3Params.getUntrackedParameter<InputTag>("L3IsolatedTauTagInfo");
  l25LeadTrackDeltaR_     = l3Params.getUntrackedParameter < double > ("L3LeadingTrackCone", 0.1);
  l25LeadTrackPt_         = l3Params.getUntrackedParameter < double > ("L3MinLeadTrackPt", 1);






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
void 
HLTTauDQMSource::beginJob(const EventSetup& context){

 
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






     //Book L3 histos
     if(doL3Monitoring_)
       {
	 dbe_->setCurrentFolder(mainFolder_+"/"+monitorName_+"/L3CutDistos/"+"TauInclusive");
	 //Book Inclusive
	 L3JetEt_           = dbe_->book1D("L3JetEt","L3 Jet Candidate E_{t}",NEtBins_,EtMin_,EtMax_);
	 L3JetEt_->setAxisTitle("L3 Jet Candidate E_{t}");
         formatHistogram(L3JetEt_,1);

	 L3JetEta_          = dbe_->book1D("L3JetEta","L3 Jet Candidate #eta",NEtaBins_,-2.5,2.5);
	 L3JetEta_->setAxisTitle("L3 Jet Candidate #eta");
	 formatHistogram(L3JetEta_,1);

	 L3NPixelTracks_    = dbe_->book1D("L3NPixelTracks","L3 Number of Pixel Tracks",20,0,20);
 	 L3NPixelTracks_->setAxisTitle("L3 # of Pixel Tracks");
	 formatHistogram(L3NPixelTracks_,1);

	 L3NQPixelTracks_   = dbe_->book1D("L3NQPixelTracks","L3 Number of Pixel Tracks(After Q test)",20,0,20);
	 L3NQPixelTracks_->setAxisTitle("L3 # of Quality Pixel Tracks");
	 formatHistogram(L3NQPixelTracks_,1);
	 
	 L3HasLeadingTrack_ = dbe_->book1D("L3HasLeadTrack","Leading Track(?)",2,0,2);
	 L3HasLeadingTrack_->setBinLabel(1,"YES");
	 L3HasLeadingTrack_->setBinLabel(2,"NO");
	 formatHistogram(L3HasLeadingTrack_,1);
	 
	 L3LeadTrackPt_     = dbe_->book1D("L3LeadTrackPt","L3 Leading Track P_{t}",60,0,60);
 	 L3LeadTrackPt_->setAxisTitle("L3 Leading Track P_{t}");
	 formatHistogram(L3LeadTrackPt_,1);

	 L3SumTrackPt_     = dbe_->book1D("L3SumTrackPt","L3 #Sigma Track P_{t}",100,0,100);
 	 L3LeadTrackPt_->setAxisTitle("L3 #Sigma Track P_{t}");
	 formatHistogram(L3SumTrackPt_,1);

       

	 //Book reference Histos
         for(size_t i=0;i<refFilters_.size();++i)
	   {

	     dbe_->setCurrentFolder(mainFolder_+"/"+monitorName_+"/L3CutDistos/"+refFilterDesc_[i]);

	     L3JetEtRef_.push_back(dbe_->book1D("L3JetEt","L3 Jet Candidate E_{t}",NEtBins_,EtMin_,EtMax_));
	     L3JetEtRef_[i]->setAxisTitle("L3 Jet Candidate E_{t}");
	     formatHistogram(L3JetEtRef_[i],1);

	     L3JetEtaRef_.push_back(dbe_->book1D("L3JetEta","L3 Jet Candidate #eta",NEtaBins_,-2.5,2.5));
	     L3JetEtaRef_[i]->setAxisTitle("L3 Jet Candidate #eta");
	     formatHistogram(L3JetEtaRef_[i],1);
	     
	     L3NPixelTracksRef_.push_back(dbe_->book1D("L3NPixelTracks","L3 Number of Pixel Tracks",20,0,20));
	     L3NPixelTracksRef_[i]->setAxisTitle("L3 # of Pixel Tracks");
	     formatHistogram(L3NPixelTracksRef_[i],1);

	     L3NQPixelTracksRef_.push_back(dbe_->book1D("L3NQPixelTracks","L3 Number of Pixel Tracks(After Q test)",20,0,20));
	     L3NQPixelTracksRef_[i]->setAxisTitle("L3 # of Quality Pixel Tracks");
	     formatHistogram(L3NQPixelTracksRef_[i],1);
	 
	     L3HasLeadingTrackRef_.push_back(dbe_->book1D("L3HasLeadTrack","Leading Track(?)",2,0,2));
	     formatHistogram(L3HasLeadingTrackRef_[i],1);
	      L3HasLeadingTrackRef_[i]->setBinLabel(1,"YES");
	      L3HasLeadingTrackRef_[i]->setBinLabel(2,"NO");

	     L3LeadTrackPtRef_.push_back(dbe_->book1D("L3LeadTrackPt","L3 Leading Track P_{t}",60,0,60));
	     L3LeadTrackPtRef_[i]->setAxisTitle("L3 Leading Track P_{t}");
	     formatHistogram(L3LeadTrackPtRef_[i],1);

	     L3SumTrackPtRef_.push_back(dbe_->book1D("L3SumTrackPt","L3 #SigmaTrack P_{t}",60,0,60));
	     L3SumTrackPtRef_[i]->setAxisTitle("L3 #Sigma Track P_{t}");
	     formatHistogram(L3SumTrackPtRef_[i],1);
	     
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
    
      //Do L3 Analysis
      if(doL3Monitoring_)
	doL3(iEvent,iSetup);
    
  
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


      //Look if the Event passed any one form the Reference Triggers and Tag this
      //Book Reference Object Collection

      LVColl refObjects;
      
      for(size_t i=0;i<refFilters_.size();++i)
	{
	 
          refObjects.clear();
          refObjects = importObjectColl(refFilters_[i],refIDs_[i],iEvent);
	  size_t object_counter = 0;
	  for(size_t j = 0 ; j< refObjects.size();++j)
	    {
	      if(refObjects[j].Pt()>PtCut_[i])
		{
		  object_counter++;
		}
	    }
	  if(object_counter>=nTriggeredTaus_)
	    {
	      NRefEvents[i]++;
	    }
  
	}

      //Look at the L1Trigger
      
      LVColl L1Taus = importObjectColl(l1Filter_,trigger::TriggerTau,iEvent);

	  if(L1Taus.size()>=nTriggeredTaus_)
	    {
	      NEventsPassedL1++;
	    
	    }

	  //Match With REFERENCE OBjects
	  for(size_t i = 0;i<refFilters_.size();++i)
	    {
	      refObjects.clear();
	      refObjects = importObjectColl(refFilters_[i],refIDs_[i],iEvent);

	      size_t match_counter = 0;
	       for(size_t j = 0;j<L1Taus.size();++j)
		{
		  if(match(L1Taus[j],refObjects,corrDeltaR_,PtCut_[i]))
		    {
		      match_counter++;
		    }
		}
	      if(match_counter>=nTriggeredTaus_)
		{
		  NEventsPassedRefL1[i]++;
		
		}
	  

	    }

	  //Look at the L2Trigger
      
	  LVColl L2Taus = importObjectColl(l2Filter_,trigger::TriggerTau,iEvent);
	  if(L2Taus.size()>=nTriggeredTaus_)
	    {
	      NEventsPassedL2++;
	    
	    }
	  //Match With REFERENCE OBjects
	  for(size_t i = 0;i<refFilters_.size();++i)
	    {
	      refObjects.clear();
	      refObjects = importObjectColl(refFilters_[i],refIDs_[i],iEvent);

	      size_t match_counter = 0;
	       for(size_t j = 0;j<L2Taus.size();++j)
		{
		  if(match(L2Taus[j],refObjects,corrDeltaR_,PtCut_[i]))
		    {
		      match_counter++;
		    }
		}
	      if(match_counter>=nTriggeredTaus_)
		{
		  NEventsPassedRefL2[i]++;
		
		}
	  

	    }

	

      //Look at the L25Trigger
      
	  LVColl L25Taus = importObjectColl(l25Filter_,trigger::TriggerTau,iEvent);
	  if(L25Taus.size()>=nTriggeredTaus_)
	    {
	      NEventsPassedL25++;
	    
	    }
	  //Match With REFERENCE OBjects
	  for(size_t i = 0;i<refFilters_.size();++i)
	    {
	      refObjects.clear();
	      refObjects = importObjectColl(refFilters_[i],refIDs_[i],iEvent);

	      size_t match_counter = 0;
	       for(size_t j = 0;j<L25Taus.size();++j)
		{
		  if(match(L25Taus[j],refObjects,corrDeltaR_,PtCut_[i]))
		    {
		      match_counter++;
		    }
		}
	      if(match_counter>=nTriggeredTaus_)
		{
		  NEventsPassedRefL25[i]++;
		
		}
	  

	    }

	


      //Look at the L3Trigger
	  LVColl L3Taus = importObjectColl(l3Filter_,trigger::TriggerTau,iEvent);

	  if(L3Taus.size()>=nTriggeredTaus_)
	    {
	      NEventsPassedL3++;
	    
	    }
	  //Match With REFERENCE OBjects
	  for(size_t i = 0;i<refFilters_.size();++i)
	    {
	      refObjects.clear();
	      refObjects = importObjectColl(refFilters_[i],refIDs_[i],iEvent);

	      size_t match_counter = 0;
	       for(size_t j = 0;j<L3Taus.size();++j)
		{
		  if(match(L3Taus[j],refObjects,corrDeltaR_,PtCut_[i]))
		    {
		      match_counter++;
		    }
		}
	      if(match_counter>=nTriggeredTaus_)
		{
		  NEventsPassedRefL3[i]++;
		
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

void 
HLTTauDQMSource::doL2(const Event& iEvent, const EventSetup& iSetup)
{

       Handle<L2TauInfoAssociation> l2TauInfoAssoc; //Handle to the input (L2 Tau Info Association)


       if(iEvent.getByLabel(l2AssocMap_,l2TauInfoAssoc))
	 {

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
			LVColl refObjects = importObjectColl(refFilters_[i],refIDs_[i],iEvent);
			//Match Jet
			if(match(jet.p4(),refObjects,corrDeltaR_,PtCut_[i]))
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



void 
HLTTauDQMSource::doL25(const Event& iEvent, const EventSetup& iSetup)
{



	  //Access the isolation class and do L25 analysis
	  Handle<IsolatedTauTagInfoCollection> tauTagInfo;
	  if(iEvent.getByLabel(l25IsolInfo_, tauTagInfo))
	    {
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
		      LVColl refObjects = importObjectColl(refFilters_[i],refIDs_[i],iEvent);
		      //Match Jet
		      if(match(tauTag->jet()->p4(),refObjects,corrDeltaR_,PtCut_[i]))
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
	    
	


void 
HLTTauDQMSource::doL3(const Event& iEvent, const EventSetup& iSetup)
{



	  //Access the isolation class and do L3 analysis
	  Handle<IsolatedTauTagInfoCollection> tauTagInfo;
	  if(iEvent.getByLabel(l3IsolInfo_, tauTagInfo))
	    {
	      for(IsolatedTauTagInfoCollection::const_iterator tauTag  = tauTagInfo->begin();tauTag!=tauTagInfo->end();++tauTag)
		{
		  //Fill Inclusive histograms
		  L3JetEt_->Fill(tauTag->jet()->et());
		  L3JetEta_->Fill(tauTag->jet()->eta());
		  L3NPixelTracks_->Fill(tauTag->allTracks().size());
		  L3NQPixelTracks_->Fill(tauTag->selectedTracks().size());
		  const TrackRef lTrack =tauTag->leadingSignalTrack(l3LeadTrackDeltaR_,l3LeadTrackPt_); 
		  if(!lTrack)
		    {
		      L3HasLeadingTrack_->Fill(1.5);
		    }
		  else
		    {
		      L3HasLeadingTrack_->Fill(0.5);
		      L3LeadTrackPt_->Fill(lTrack->pt());

		    }

		  //Calculate Sum of Track pt
		  double sumTrackPt = 0.;
		  for(size_t k=0;k<tauTag->allTracks().size();++k)
		    {
		      sumTrackPt+=tauTag->allTracks()[k]->pt();
		    }

		  L3SumTrackPt_->Fill(sumTrackPt);


		  //Fill Matching Information
		  for(size_t i=0;i<refFilters_.size();++i)
		    {
		      //Get reference Objects
		      LVColl refObjects = importObjectColl(refFilters_[i],refIDs_[i],iEvent);
		      //Match Jet
		      if(match(tauTag->jet()->p4(),refObjects,corrDeltaR_,PtCut_[i]))
			{
			  L3JetEtRef_[i]->Fill(tauTag->jet()->et());
			  L3JetEtaRef_[i]->Fill(tauTag->jet()->eta());
			  L3NPixelTracksRef_[i]->Fill(tauTag->allTracks().size());
			  L3NQPixelTracksRef_[i]->Fill(tauTag->selectedTracks().size());
		   		     
			  if(!lTrack)
			    {
			      L3HasLeadingTrackRef_[i]->Fill(1.5);
			    }
			  else
			    {
			      L3HasLeadingTrackRef_[i]->Fill(0.5);
			      L3LeadTrackPtRef_[i]->Fill(lTrack->pt());
			      L3SumTrackPtRef_[i]->Fill(sumTrackPt);
			    }
			
			}
		    }
		}
	    }
}	      
	    
	
    






bool 
HLTTauDQMSource::match(const LV& cand,const  LVColl&  electrons/*VRelectron& electrons*/,double deltaR,double  ptMin)
{

 
  bool matched=false;

  for(size_t i = 0;i<electrons.size();++i)
      {
	double delta = ROOT::Math::VectorUtil::DeltaR(cand.Vect(),electrons[i].Vect());
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
HLTTauDQMSource::importObjectColl(edm::InputTag& filter,int id,const Event& iEvent)
{
      //Create output Collection
      LVColl out;

      //Look at all Different triggers
      
      //If we have a L1 Jet
      if(id==84||id==85||id==86) 
	{
	  edm::Handle<trigger::TriggerFilterObjectWithRefs> f;
	  if(iEvent.getByLabel(filter,f))
	    {
	      VRl1jet jets; 
	      f->getObjects(id,jets);
	      for(size_t i = 0; i<jets.size();++i)
		out.push_back(jets[i]->p4());

	    }

	}

      //If we have an HLT Jet
      if(id==94||id==95||id==96) 
	{
	  edm::Handle<reco::CaloJetCollection> obj;
	  if(iEvent.getByLabel(filter,obj))
	    {
	     
	       for(size_t i = 0; i<obj->size();++i)
		out.push_back((*obj)[i].p4());

	    }

	}

     


      //If we have a gamma Collection(HLT Jets, Taus and bs)
      if(id==91) 
	{
	  edm::Handle<reco::RecoEcalCandidateCollection> gamma;
	  if(iEvent.getByLabel(filter,gamma))
	    {
	      for(size_t i = 0; i<gamma->size();++i)
		out.push_back((*gamma)[i].p4());

	    }

	}

      //If we have an electron Collection(Electrons)
      if(id==92) 
	{
	  edm::Handle<reco::ElectronCollection> obj;
	  if(iEvent.getByLabel(filter,obj))
	    {
	      for(size_t i = 0; i<obj->size();++i)
		out.push_back((*obj)[i].p4());

	    }

	}

      //If we have a Muon  Collection
      if(id==93) 
	{
	  edm::Handle<RecoChargedCandidateCollection> obj;
	  if(iEvent.getByLabel(filter,obj))
	    {
	      for(size_t i = 0; i<obj->size();++i)
		out.push_back((*obj)[i].p4());

	    }

	}

      //If we have a Track Collection
      if(id==99) 
	{
	  edm::Handle<std::vector<reco::IsolatedPixelTrackCandidate> > obj;
	  if(iEvent.getByLabel(filter,obj))
	    {
	      for(size_t i = 0; i<obj->size();++i)
		out.push_back((*obj)[i].p4());

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
