/*
 *  See header file for a description of this class.
 *
 *  \author Loic Quertenmont 
 */
#include "DQM/TrackingMonitor/interface/dEdxAnalyzer.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include <string>
#include "TMath.h"

dEdxAnalyzer::dEdxAnalyzer(const edm::ParameterSet& iConfig) 
  : dqmStore_( edm::Service<DQMStore>().operator->() )
  , fullconf_( iConfig )
  , conf_    (fullconf_.getParameter<edm::ParameterSet>("dEdxParameters") )
  , doAllPlots_  ( conf_.getParameter<bool>("doAllPlots") )
  , doDeDxPlots_ ( conf_.getParameter<bool>("doDeDxPlots") )    
  , genTriggerEventFlag_( new GenericTriggerEventFlag(conf_,consumesCollector(), *this) )
{

  trackInputTag_ = edm::InputTag(conf_.getParameter<std::string>("TracksForDeDx") );
  trackToken_ = consumes<reco::TrackCollection>(trackInputTag_);

  dEdxInputList_ = conf_.getParameter<std::vector<std::string> >("deDxProducers");
  for (auto const& tag : dEdxInputList_) {
    dEdxTokenList_.push_back(consumes<reco::DeDxDataValueMap>(edm::InputTag(tag) ) );
  }
}

dEdxAnalyzer::~dEdxAnalyzer() 
{ 

  if (genTriggerEventFlag_)      delete genTriggerEventFlag_;
}

// ------------ method called once each job just after ending the event loop  ------------
void 
dEdxAnalyzer::endJob() 
{
    bool outputMEsInRootFile   = conf_.getParameter<bool>("OutputMEsInRootFile");
    std::string outputFileName = conf_.getParameter<std::string>("OutputFileName");
    if(outputMEsInRootFile)
    {
      dqmStore_->showDirStructure();
      dqmStore_->save(outputFileName);
    }
}

/*
// -- BeginRun
//---------------------------------------------------------------------------------//
void dEdxAnalyzer::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{
 
  // Initialize the GenericTriggerEventFlag
  if ( genTriggerEventFlag_->on() ) genTriggerEventFlag_->initRun( iRun, iSetup );
}
*/

void dEdxAnalyzer::bookHistograms(DQMStore::IBooker & ibooker,
				  edm::Run const & iRun,
				  edm::EventSetup const & iSetup ) 
{

  // Initialize the GenericTriggerEventFlag
  if ( genTriggerEventFlag_->on() ) genTriggerEventFlag_->initRun( iRun, iSetup );
  

    // parameters from the configuration
    std::string MEFolderName   = conf_.getParameter<std::string>("FolderName"); 

    // get binning from the configuration
    TrackHitMin  = conf_.getParameter<double>("TrackHitMin");
    HIPdEdxMin   = conf_.getParameter<double>("HIPdEdxMin");
    HighPtThreshold =  conf_.getParameter<double>("HighPtThreshold");

    dEdxK      = conf_.getParameter<double>("dEdxK");
    dEdxC      = conf_.getParameter<double>("dEdxC");


    int    dEdxNHitBin     = conf_.getParameter<int>(   "dEdxNHitBin");
    double dEdxNHitMin     = conf_.getParameter<double>("dEdxNHitMin");
    double dEdxNHitMax     = conf_.getParameter<double>("dEdxNHitMax");

    int    dEdxBin      = conf_.getParameter<int>(   "dEdxBin");
    double dEdxMin      = conf_.getParameter<double>("dEdxMin");
    double dEdxMax      = conf_.getParameter<double>("dEdxMax");

    int    dEdxHIPmassBin  = conf_.getParameter<int>(   "dEdxHIPmassBin");
    double dEdxHIPmassMin  = conf_.getParameter<double>("dEdxHIPmassMin"); 
    double dEdxHIPmassMax  = conf_.getParameter<double>("dEdxHIPmassMax");

    int    dEdxMIPmassBin  = conf_.getParameter<int>(   "dEdxMIPmassBin");
    double dEdxMIPmassMin  = conf_.getParameter<double>("dEdxMIPmassMin"); 
    double dEdxMIPmassMax  = conf_.getParameter<double>("dEdxMIPmassMax");

    ibooker.setCurrentFolder(MEFolderName);

    // book the Hit Property histograms
    // ---------------------------------------------------------------------------------//

    if ( doDeDxPlots_ || doAllPlots_ ){
      for(unsigned int i=0;i<dEdxInputList_.size();i++){
         ibooker.setCurrentFolder(MEFolderName+"/"+ dEdxInputList_[i]);
         dEdxMEsVector.push_back(dEdxMEs() );

         histname = "MIP_dEdxPerTrack_"; 
         dEdxMEsVector[i].ME_MipDeDx = ibooker.book1D(histname, histname, dEdxBin, dEdxMin, dEdxMax);
         dEdxMEsVector[i].ME_MipDeDx->setAxisTitle("dEdx of each MIP Track (MeV/cm)");
         dEdxMEsVector[i].ME_MipDeDx->setAxisTitle("Number of Tracks", 2);

         histname =  "MIP_NumberOfdEdxHitsPerTrack_";
         dEdxMEsVector[i].ME_MipDeDxNHits = ibooker.book1D(histname, histname, dEdxNHitBin, dEdxNHitMin, dEdxNHitMax);
         dEdxMEsVector[i].ME_MipDeDxNHits->setAxisTitle("Number of dEdxHits of each MIP Track");
         dEdxMEsVector[i].ME_MipDeDxNHits->setAxisTitle("Number of Tracks", 2);

         histname =  "MIP_FractionOfSaturateddEdxHitsPerTrack_"; 
         dEdxMEsVector[i].ME_MipDeDxNSatHits = ibooker.book1D(histname, histname,2*dEdxNHitBin, 0, 1);
         dEdxMEsVector[i].ME_MipDeDxNSatHits->setAxisTitle("Fraction of Saturated dEdxHits of each MIP Track");
         dEdxMEsVector[i].ME_MipDeDxNSatHits->setAxisTitle("Number of Tracks", 2);

         histname =  "MIP_MassPerTrack_";
         dEdxMEsVector[i].ME_MipDeDxMass = ibooker.book1D(histname, histname, dEdxMIPmassBin, dEdxMIPmassMin, dEdxMIPmassMax);
         dEdxMEsVector[i].ME_MipDeDxMass->setAxisTitle("dEdx Mass of each MIP Track (GeV/c^{2})");
         dEdxMEsVector[i].ME_MipDeDxMass->setAxisTitle("Number of Tracks", 2);

         histname =  "HIP_MassPerTrack_";
         dEdxMEsVector[i].ME_HipDeDxMass = ibooker.book1D(histname, histname, dEdxHIPmassBin, dEdxHIPmassMin, dEdxHIPmassMax);
         dEdxMEsVector[i].ME_HipDeDxMass->setAxisTitle("dEdx Mass of each HIP Track (GeV/c^{2})");
         dEdxMEsVector[i].ME_HipDeDxMass->setAxisTitle("Number of Tracks", 2);

         histname = "MIPOfHighPt_dEdxPerTrack_";
         dEdxMEsVector[i].ME_MipHighPtDeDx = ibooker.book1D(histname, histname, dEdxBin, dEdxMin, dEdxMax);
         dEdxMEsVector[i].ME_MipHighPtDeDx->setAxisTitle("dEdx of each MIP (of High pT) Track (MeV/cm)");
         dEdxMEsVector[i].ME_MipHighPtDeDx->setAxisTitle("Number of Tracks", 2);

         histname =  "MIPOfHighPt_NumberOfdEdxHitsPerTrack_";
         dEdxMEsVector[i].ME_MipHighPtDeDxNHits = ibooker.book1D(histname, histname, dEdxNHitBin, dEdxNHitMin, dEdxNHitMax);
         dEdxMEsVector[i].ME_MipHighPtDeDxNHits->setAxisTitle("Number of dEdxHits of each MIP (of High pT) Track");
         dEdxMEsVector[i].ME_MipHighPtDeDxNHits->setAxisTitle("Number of Tracks", 2);
       }
    }

}

void dEdxAnalyzer::beginJob()
{
}


double dEdxAnalyzer::mass(double P, double I){
   if(I-dEdxC<0)return -1;
   return sqrt((I-dEdxC)/dEdxK)*P;
}

// -- Analyse
// ---------------------------------------------------------------------------------//
void dEdxAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  // Filter out events if Trigger Filtering is requested
  if (genTriggerEventFlag_->on()&& ! genTriggerEventFlag_->accept( iEvent, iSetup) ) return;


   if ( doDeDxPlots_ || doAllPlots_ ){
      edm::Handle<reco::TrackCollection> trackCollectionHandle;
      iEvent.getByToken(trackToken_, trackCollectionHandle );
      if(!trackCollectionHandle.isValid())return;

      for(unsigned int i=0;i<dEdxInputList_.size();i++){
         edm::Handle<reco::DeDxDataValueMap> dEdxObjectHandle;
	 iEvent.getByToken(dEdxTokenList_[i], dEdxObjectHandle );
         if(!dEdxObjectHandle.isValid())continue;
         const edm::ValueMap<reco::DeDxData> dEdxColl = *dEdxObjectHandle.product();
 
              
         for(unsigned int t=0; t<trackCollectionHandle->size(); t++){
            reco::TrackRef track = reco::TrackRef( trackCollectionHandle, t );


            if(track->quality(reco::TrackBase::highPurity) ) {
              //MIPs  
              if( track->pt() >= 5.0 && track->numberOfValidHits()>TrackHitMin){
                 dEdxMEsVector[i].ME_MipDeDx        ->Fill(dEdxColl[track].dEdx());
                 dEdxMEsVector[i].ME_MipDeDxNHits   ->Fill(dEdxColl[track].numberOfMeasurements());
		 if (dEdxColl[track].numberOfMeasurements()!=0)
		   dEdxMEsVector[i].ME_MipDeDxNSatHits->Fill((1.0*dEdxColl[track].numberOfSaturatedMeasurements())/dEdxColl[track].numberOfMeasurements());
                 dEdxMEsVector[i].ME_MipDeDxMass    ->Fill(mass(track->p(), dEdxColl[track].dEdx()));


                 if(track->pt() >= HighPtThreshold){
                    dEdxMEsVector[i].ME_MipHighPtDeDx        ->Fill(dEdxColl[track].dEdx());
                    dEdxMEsVector[i].ME_MipHighPtDeDxNHits   ->Fill(dEdxColl[track].numberOfMeasurements());
                 }

              //HighlyIonizing particles
              }else if(track->pt()<2 && dEdxColl[track].dEdx()>HIPdEdxMin){
                 dEdxMEsVector[i].ME_HipDeDxMass    ->Fill(mass(track->p(), dEdxColl[track].dEdx()));
              }
            }
         }
      }
   }
}

 

void 
dEdxAnalyzer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void 
dEdxAnalyzer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
dEdxAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(dEdxAnalyzer);
