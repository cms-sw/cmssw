/*
 *  See header file for a description of this class.
 *
 *  $Date: 2010/01/17 20:00:42 $
 *  $Revision: 1.10 $
 *  \author Suchandra Dutta , Giorgia Mila
 */

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h" 
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h" 
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQM/TrackingMonitor/interface/TrackBuildingAnalyzer.h"
#include "DQM/TrackingMonitor/interface/TrackAnalyzer.h"
#include "DQM/TrackingMonitor/plugins/TrackingMonitor.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h" 
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include <string>

// TrackingMonitor 
// ----------------------------------------------------------------------------------//

TrackingMonitor::TrackingMonitor(const edm::ParameterSet& iConfig) 
    : dqmStore_( edm::Service<DQMStore>().operator->() )
    , conf_ ( iConfig )
    , theTrackAnalyzer( new TrackAnalyzer(conf_) )
    , theTrackBuildingAnalyzer( new TrackBuildingAnalyzer(conf_) )
    , NumberOfTracks(NULL)
    , NumberOfMeanRecHitsPerTrack(NULL)
    , NumberOfMeanLayersPerTrack(NULL)
    , NumberOfSeeds(NULL)
    , NumberOfTrackCandidates(NULL)
    , builderName( conf_.getParameter<std::string>("TTRHBuilder") )
{
}


TrackingMonitor::~TrackingMonitor() 
{
    delete theTrackAnalyzer;
    delete theTrackBuildingAnalyzer;
}


void TrackingMonitor::beginJob(void) 
{
    using namespace edm;
    using std::string;
    using namespace std; 

    // parameters from the configuration
    std::string Quality      = conf_.getParameter<std::string>("Quality");
    std::string AlgoName     = conf_.getParameter<std::string>("AlgoName");
    std::string MEFolderName = conf_.getParameter<std::string>("FolderName"); 

    // test for the Quality veriable validity
    if( Quality != "")
    {
        if( Quality != "highPurity" && Quality != "tight" && Quality != "loose") 
        {
            edm::LogWarning("TrackingMonitor")  << "Qualty Name is invalid, using no quality criterea by default";
            Quality = "";
        }
    }

    // use the AlgoName and Quality Name
    string CatagoryName = Quality != "" ? AlgoName + "_" + Quality : AlgoName;

    // get binning from the configuration
    int    TKNoBin     = conf_.getParameter<int>(   "TkSizeBin");
    double TKNoMin     = conf_.getParameter<double>("TkSizeMin");
    double TKNoMax     = conf_.getParameter<double>("TkSizeMax");

    int    TCNoBin     = conf_.getParameter<int>(   "TCSizeBin");
    double TCNoMin     = conf_.getParameter<double>("TCSizeMin");
    double TCNoMax     = conf_.getParameter<double>("TCSizeMax");

    int    TKNoSeedBin = conf_.getParameter<int>(   "TkSeedSizeBin");
    double TKNoSeedMin = conf_.getParameter<double>("TkSeedSizeMin");
    double TKNoSeedMax = conf_.getParameter<double>("TkSeedSizeMax");

    int    MeanHitBin  = conf_.getParameter<int>(   "MeanHitBin");
    double MeanHitMin  = conf_.getParameter<double>("MeanHitMin");
    double MeanHitMax  = conf_.getParameter<double>("MeanHitMax");

    int    MeanLayBin  = conf_.getParameter<int>(   "MeanLayBin");
    double MeanLayMin  = conf_.getParameter<double>("MeanLayMin");
    double MeanLayMax  = conf_.getParameter<double>("MeanLayMax");

    string StateName = conf_.getParameter<string>("MeasurementState");
    if
    (
        StateName != "OuterSurface" &&
        StateName != "InnerSurface" &&
        StateName != "ImpactPoint"  &&
        StateName != "default"      &&
        StateName != "All"
    )
    {
        // print warning
        edm::LogWarning("TrackingMonitor")  << "State Name is invalid, using 'ImpactPoint' by default";
    }

    dqmStore_->setCurrentFolder(MEFolderName);

    // book the General Property histograms
    // ---------------------------------------------------------------------------------//
    dqmStore_->setCurrentFolder(MEFolderName+"/GeneralProperties");

    histname = "NumberOfTracks_" + CatagoryName;
    NumberOfTracks = dqmStore_->book1D(histname, histname, TKNoBin, TKNoMin, TKNoMax);
    NumberOfTracks->setAxisTitle("Number of Tracks per Event", 1);
    NumberOfTracks->setAxisTitle("Number of Events", 2);

    histname = "NumberOfMeanRecHitsPerTrack_" + CatagoryName;
    NumberOfMeanRecHitsPerTrack = dqmStore_->book1D(histname, histname, MeanHitBin, MeanHitMin, MeanHitMax);
    NumberOfMeanRecHitsPerTrack->setAxisTitle("Mean number of RecHits per Track", 1);
    NumberOfMeanRecHitsPerTrack->setAxisTitle("Entries", 2);

    histname = "NumberOfMeanLayersPerTrack_" + CatagoryName;
    NumberOfMeanLayersPerTrack = dqmStore_->book1D(histname, histname, MeanLayBin, MeanLayMin, MeanLayMax);
    NumberOfMeanLayersPerTrack->setAxisTitle("Mean number of Layers per Track", 1);
    NumberOfMeanLayersPerTrack->setAxisTitle("Entries", 2);

    theTrackAnalyzer->beginJob(dqmStore_);

    // book the Seed Property histograms
    // ---------------------------------------------------------------------------------//
    if (conf_.getParameter<bool>("doSeedParameterHistos")) 
    {
        dqmStore_->setCurrentFolder(MEFolderName+"/TrackBuilding");

        histname = "NumberOfSeeds_" + CatagoryName;
        NumberOfSeeds = dqmStore_->book1D(histname, histname, TKNoSeedBin, TKNoSeedMin, TKNoSeedMax);
        NumberOfSeeds->setAxisTitle("Number of Seeds per Event", 1);
        NumberOfSeeds->setAxisTitle("Number of Events", 2);

        histname = "NumberOfTrackCandidates_" + CatagoryName;
        NumberOfTrackCandidates = dqmStore_->book1D(histname, histname, TCNoBin, TCNoMin, TCNoMax);
        NumberOfTrackCandidates->setAxisTitle("Number of Track Candidates per Event", 1);
        NumberOfTrackCandidates->setAxisTitle("Number of Event", 2);
    
        theTrackBuildingAnalyzer->beginJob(dqmStore_);
    }
}

// -- Analyse
// ---------------------------------------------------------------------------------//
void TrackingMonitor::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) 
{
    using namespace edm;

    // input tags for collections from the configuration
    InputTag trackProducer  = conf_.getParameter<edm::InputTag>("TrackProducer");
    InputTag seedProducer   = conf_.getParameter<edm::InputTag>("SeedProducer");
    InputTag tcProducer     = conf_.getParameter<edm::InputTag>("TCProducer");
    InputTag bsSrc          = conf_.getParameter<edm::InputTag>("beamSpot");
    std::string Quality     = conf_.getParameter<std::string>("Quality");
    std::string Algo        = conf_.getParameter<std::string>("AlgoName");

    //  Analyse the tracks
    //  if the collection is empty, do not fill anything
    // ---------------------------------------------------------------------------------//

    // get the track collection
    Handle<reco::TrackCollection> trackHandle;
    iEvent.getByLabel(trackProducer, trackHandle);

    if (trackHandle.isValid()) 
    {

       reco::TrackCollection trackCollection = *trackHandle;
        // calculate the mean # rechits and layers
        int totalNumTracks = 0, totalRecHits = 0, totalLayers = 0;

        for (reco::TrackCollection::const_iterator track = trackCollection.begin(); track!=trackCollection.end(); ++track) 
        {
            // kludge --> do better
            if( trackCollection.size() > 100) continue;

            if( Quality == "highPurity") 
            {
                if( !track->quality(reco::TrackBase::highPurity) ) continue;
            }
            else if( Quality == "tight") 
            {
                if( !track->quality(reco::TrackBase::tight) ) continue;
            }
            else if( Quality == "loose") 
            {
                if( !track->quality(reco::TrackBase::loose) ) continue;
            }
            
            totalNumTracks++;
            totalRecHits    += track->found();
            totalLayers     += track->hitPattern().trackerLayersWithMeasurement();

            // do analysis per track
            theTrackAnalyzer->analyze(iEvent, iSetup, *track);
        }

        NumberOfTracks->Fill(totalNumTracks);

        if( totalNumTracks > 0 )
        {
            double meanRecHits = static_cast<double>(totalRecHits) / static_cast<double>(totalNumTracks);
            double meanLayers  = static_cast<double>(totalLayers)  / static_cast<double>(totalNumTracks);
            NumberOfMeanRecHitsPerTrack->Fill(meanRecHits);
            NumberOfMeanLayersPerTrack->Fill(meanLayers);
        }


	//  Analyse the Track Building variables 
	//  if the collection is empty, do not fill anything
	// ---------------------------------------------------------------------------------//
	
	if (conf_.getParameter<bool>("doSeedParameterHistos")) 
	  {
	    
	    // magnetic field
	    edm::ESHandle<MagneticField> theMF;
	    iSetup.get<IdealMagneticFieldRecord>().get(theMF);  
	    
	    // get the beam spot
	    Handle<reco::BeamSpot> recoBeamSpotHandle;
	    iEvent.getByLabel(bsSrc,recoBeamSpotHandle);
	    const reco::BeamSpot& bs = *recoBeamSpotHandle;      
	    
	    // get the candidate collection
	    Handle<TrackCandidateCollection> theTCHandle;
	    iEvent.getByLabel(tcProducer, theTCHandle ); 
	    const TrackCandidateCollection& theTCCollection = *theTCHandle;
	    
	    // fill the TrackCandidate info
	    if (theTCHandle.isValid())
	      {
		NumberOfTrackCandidates->Fill(theTCCollection.size());
		iSetup.get<TransientRecHitRecord>().get(builderName,theTTRHBuilder);
		for( TrackCandidateCollection::const_iterator cand = theTCCollection.begin(); cand != theTCCollection.end(); ++cand)
		  {
		    theTrackBuildingAnalyzer->analyze(iEvent, iSetup, *cand, bs, theMF, theTTRHBuilder);
		  }
	      }
	    else
	      {
		LogWarning("TrackingMonitor") << "No Track Candidates in the event.  Not filling associated histograms";
	      }

	    // get the seed collection
	    Handle<edm::View<TrajectorySeed> > seedHandle;
	    iEvent.getByLabel(seedProducer, seedHandle);
	    const edm::View<TrajectorySeed>& seedCollection = *seedHandle;
	    
	    // fill the seed info
	    if (seedHandle.isValid()) 
	      {
		NumberOfSeeds->Fill(seedCollection.size());
		
		iSetup.get<TransientRecHitRecord>().get(builderName,theTTRHBuilder);
		for(size_t i=0; i < seedHandle->size(); ++i)
		  {
		    edm::RefToBase<TrajectorySeed> seed(seedHandle, i);
		    theTrackBuildingAnalyzer->analyze(iEvent, iSetup, *seed, bs, theMF, theTTRHBuilder);
		  }
	      }
	    else
	      {
		LogWarning("TrackingMonitor") << "No Trajectory seeds in the event.  Not filling associated histograms";
	      }
	  }
    }
    else
    {
        return;
    }
}


void TrackingMonitor::endJob(void) 
{
    bool outputMEsInRootFile   = conf_.getParameter<bool>("OutputMEsInRootFile");
    std::string outputFileName = conf_.getParameter<std::string>("OutputFileName");
    if(outputMEsInRootFile)
    {
        dqmStore_->showDirStructure();
        dqmStore_->save(outputFileName);
    }
}

DEFINE_FWK_MODULE(TrackingMonitor);
