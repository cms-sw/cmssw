/*
 *  See header file for a description of this class.
 *
 *  \author Loic Quertenmont 
 */
#include "DQM/TrackingMonitor/interface/dEdxHitAnalyzer.h"

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

dEdxHitAnalyzer::dEdxHitAnalyzer(const edm::ParameterSet& iConfig) 
  : fullconf_( iConfig )
  , conf_    (fullconf_.getParameter<edm::ParameterSet>("dEdxParameters") )
  , doAllPlots_  ( conf_.getParameter<bool>("doAllPlots") )
  , doDeDxPlots_ ( conf_.getParameter<bool>("doDeDxPlots") )    
  , genTriggerEventFlag_( new GenericTriggerEventFlag(conf_,consumesCollector()) )
{

  trackInputTag_ = edm::InputTag(conf_.getParameter<std::string>("TracksForDeDx") );
  trackToken_ = consumes<reco::TrackCollection>(trackInputTag_);

  dEdxInputList_ = conf_.getParameter<std::vector<std::string> >("deDxHitProducers");
  for (auto const& tag : dEdxInputList_) {
    dEdxTokenList_.push_back(consumes<reco::DeDxHitInfoAss>(edm::InputTag(tag) ) );
  }

  // parameters from the configuration
  MEFolderName   = conf_.getParameter<std::string>("FolderName"); 

  dEdxNHitBin     = conf_.getParameter<int>(   "dEdxNHitBin");
  dEdxNHitMin     = conf_.getParameter<double>("dEdxNHitMin");
  dEdxNHitMax     = conf_.getParameter<double>("dEdxNHitMax");

  dEdxStripBin    = conf_.getParameter<int>(   "dEdxStripBin");
  dEdxStripMin    = conf_.getParameter<double>("dEdxStripMin");
  dEdxStripMax    = conf_.getParameter<double>("dEdxStripMax");

  dEdxPixelBin    = conf_.getParameter<int>(   "dEdxPixelBin");
  dEdxPixelMin    = conf_.getParameter<double>("dEdxPixelMin");
  dEdxPixelMax    = conf_.getParameter<double>("dEdxPixelMax");

  dEdxHarm2Bin    = conf_.getParameter<int>(   "dEdxHarm2Bin");
  dEdxHarm2Min    = conf_.getParameter<double>("dEdxHarm2Min");
  dEdxHarm2Max    = conf_.getParameter<double>("dEdxHarm2Max");
}

dEdxHitAnalyzer::~dEdxHitAnalyzer() 
{ 

  if (genTriggerEventFlag_)      delete genTriggerEventFlag_;
}

// -- BeginRun
//---------------------------------------------------------------------------------//
void dEdxHitAnalyzer::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{
  // Initialize the GenericTriggerEventFlag
  if ( genTriggerEventFlag_->on() ) genTriggerEventFlag_->initRun( iRun, iSetup );
}


void dEdxHitAnalyzer::bookHistograms(DQMStore::IBooker & ibooker,
				  edm::Run const & iRun,
				  edm::EventSetup const & iSetup ) 
{

    ibooker.setCurrentFolder(MEFolderName);

    // book the Hit Property histograms
    // ---------------------------------------------------------------------------------//

    if ( doDeDxPlots_ || doAllPlots_ ){
      for(unsigned int i=0;i<dEdxInputList_.size();i++){
         ibooker.setCurrentFolder(MEFolderName+"/"+ dEdxInputList_[i]);
         dEdxMEsVector.push_back(dEdxMEs() );

         histname = "Strip_dEdxPerCluster_"; 
         dEdxMEsVector[i].ME_StripHitDeDx = ibooker.book1D(histname, histname, dEdxStripBin, dEdxStripMin, dEdxStripMax);
         dEdxMEsVector[i].ME_StripHitDeDx->setAxisTitle("dEdx of on-track strip cluster (ADC)");
         dEdxMEsVector[i].ME_StripHitDeDx->setAxisTitle("Number of Strip clusters", 2);

         histname = "Pixel_dEdxPerCluster_"; 
         dEdxMEsVector[i].ME_PixelHitDeDx = ibooker.book1D(histname, histname, dEdxPixelBin, dEdxPixelMin, dEdxPixelMax);
         dEdxMEsVector[i].ME_PixelHitDeDx->setAxisTitle("dEdx of on-track pixel cluster (ADC)");
         dEdxMEsVector[i].ME_PixelHitDeDx->setAxisTitle("Number of Pixel clusters", 2);

         histname =  "NumberOfdEdxHitsPerTrack_";
         dEdxMEsVector[i].ME_NHitDeDx = ibooker.book1D(histname, histname, dEdxNHitBin, dEdxNHitMin, dEdxNHitMax);
         dEdxMEsVector[i].ME_NHitDeDx->setAxisTitle("Number of dEdxHits per Track");
         dEdxMEsVector[i].ME_NHitDeDx->setAxisTitle("Number of Tracks", 2);

         histname =  "Harm2_dEdxPerTrack_"; 
         dEdxMEsVector[i].ME_Harm2DeDx = ibooker.book1D(histname, histname,dEdxHarm2Bin, dEdxHarm2Min, dEdxHarm2Max);
         dEdxMEsVector[i].ME_Harm2DeDx->setAxisTitle("Harmonic2 dEdx estimator for each Track");
         dEdxMEsVector[i].ME_Harm2DeDx->setAxisTitle("Number of Tracks", 2);
       }
    }
}

double dEdxHitAnalyzer::harmonic2(const reco::DeDxHitInfo* dedxHits){
     if(!dedxHits)return -1;
     std::vector<double> vect;
     for(unsigned int h=0;h<dedxHits->size();h++){
        DetId detid(dedxHits->detId(h));  
        double Norm = (detid.subdetId()<3)?3.61e-06:3.61e-06*265;
        double ChargeOverPathlength = Norm*dedxHits->charge(h)/dedxHits->pathlength(h);
        vect.push_back(ChargeOverPathlength); //save charge
     }

     int size = vect.size();
     if(size<=0)return -1;
     double result=0;
     double expo = -2;
     for(int i = 0; i< size; i ++){
        result+=pow(vect[i],expo); 
     }
     return pow(result/size,1./expo);
}


// -- Analyse
// ---------------------------------------------------------------------------------//
void dEdxHitAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  // Filter out events if Trigger Filtering is requested
  if (genTriggerEventFlag_->on()&& ! genTriggerEventFlag_->accept( iEvent, iSetup) ) return;


   if ( doDeDxPlots_ || doAllPlots_ ){
      edm::Handle<reco::TrackCollection> trackCollectionHandle;
      iEvent.getByToken(trackToken_, trackCollectionHandle );
      if(!trackCollectionHandle.isValid())return;

      for(unsigned int i=0;i<dEdxInputList_.size();i++){
         edm::Handle<reco::DeDxHitInfoAss> dEdxObjectHandle;
	 iEvent.getByToken(dEdxTokenList_[i], dEdxObjectHandle );
         if(!dEdxObjectHandle.isValid())continue;
              
         for(unsigned int t=0; t<trackCollectionHandle->size(); t++){
            reco::TrackRef track = reco::TrackRef( trackCollectionHandle, t );

            if(track->quality(reco::TrackBase::highPurity) ) {
               const reco::DeDxHitInfo* dedxHits = NULL;
               if(!track.isNull()) {
                  reco::DeDxHitInfoRef dedxHitsRef = (*dEdxObjectHandle)[track];
                  if(!dedxHitsRef.isNull())dedxHits = &(*dedxHitsRef);
               }
               if(!dedxHits)continue;

               for(unsigned int h=0;h<dedxHits->size();h++){
                  DetId detid(dedxHits->detId(h));
                  if(detid.subdetId()>=3)dEdxMEsVector[i].ME_StripHitDeDx   ->Fill(dedxHits->charge(h));
                  if(detid.subdetId()<3 )dEdxMEsVector[i].ME_PixelHitDeDx   ->Fill(dedxHits->charge(h));
               }
               dEdxMEsVector[i].ME_NHitDeDx->Fill(dedxHits->size());
               dEdxMEsVector[i].ME_Harm2DeDx->Fill(harmonic2(dedxHits));
            }
         }
      }
   }
}

 


// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
dEdxHitAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(dEdxHitAnalyzer);
