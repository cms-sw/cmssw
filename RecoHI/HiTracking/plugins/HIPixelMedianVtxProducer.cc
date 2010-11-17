#include "HIPixelMedianVtxProducer.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>

#include "TROOT.h"
#include "TH1F.h"
#include "TF1.h"

/*****************************************************************************/
struct ComparePairs
{
	bool operator() (const reco::Track * t1,
					 const reco::Track * t2)
	{
		return (t1->vz() < t2->vz());
	};
};

/*****************************************************************************/
HIPixelMedianVtxProducer::HIPixelMedianVtxProducer
(const edm::ParameterSet& ps) : theConfig(ps)
{
	produces<reco::VertexCollection>();
}


/*****************************************************************************/
HIPixelMedianVtxProducer::~HIPixelMedianVtxProducer()
{ 
}

/*****************************************************************************/
void HIPixelMedianVtxProducer::beginJob()
{
}

/*****************************************************************************/
void HIPixelMedianVtxProducer::produce
(edm::Event& ev, const edm::EventSetup& es)
{
	// Get pixel tracks
	edm::Handle<reco::TrackCollection> trackCollection;
	std::string trackCollectionName =
    theConfig.getParameter<std::string>("TrackCollection");
	ev.getByLabel(trackCollectionName, trackCollection);
	const reco::TrackCollection tracks_ = *(trackCollection.product());
	
	thePtMin = theConfig.getParameter<double>("PtMin");
	
	// Select tracks 
	std::vector<const reco::Track *> tracks;
	for (unsigned int i=0; i<tracks_.size(); i++)
	{
		if (tracks_[i].pt() > thePtMin)
		{
			reco::TrackRef recTrack(trackCollection, i);
			tracks.push_back( &(*recTrack));
		}
	}
	
	LogTrace("MinBiasTracking")
	<< " [VertexProducer] selected tracks: "
	<< tracks.size() << " (out of " << tracks_.size()
	<< ")"; 
	
	std::auto_ptr<reco::VertexCollection> vertices(new reco::VertexCollection);
	
	if(tracks.size() > 0)
	{
		// Sort along vertex z position
		std::sort(tracks.begin(), tracks.end(), ComparePairs());
		
		// Median
		float med;
		if(tracks.size() % 2 == 0)
			med = (tracks[tracks.size()/2-1]->vz() + tracks[tracks.size()/2]->vz())/2;
		else
			med =  tracks[tracks.size()/2  ]->vz();
		
		LogTrace("MinBiasTracking")
		<< "  [vertex position] median    = " << med << " cm";
		
		if(tracks.size() > 10)
		{
			
			if(tracks.size() > 100)       // In high multiplicity events, use most probable position instead of median  
			{
				
			        // FIXME: the binning should be a configuration parameter
				// Find maximum bin
				TH1F hmax("hmax","hmax",20*30,-1*30,30);
				
				for(std::vector<const reco::Track *>::const_iterator
					track = tracks.begin(); track!= tracks.end(); track++)
					if(fabs((*track)->vz()) < 30)
						hmax.Fill((*track)->vz());
				
				int maxBin = hmax.GetMaximumBin();
				
				LogTrace("MinBiasTracking")
				<< "  [vertex position] most prob = "
				<< hmax.GetBinCenter(maxBin)
				<< " cm";
				
				// Find n-bin weighted average
				float num=0.0, denom=0.0;
				int nAvg=1;
				for(int i=-1*nAvg; i<=nAvg; i++) {
					num+=hmax.GetBinContent(maxBin+i)*hmax.GetBinCenter(maxBin+i);
					denom+=hmax.GetBinContent(maxBin+i); 
				}
				float nBinAvg = num/denom;
				med = nBinAvg; //use nBinAvg around most probably location instead of median
				
				LogTrace("MinBiasTracking")
				<< "  [vertex position] " << 2*nAvg+1 << "-bin weighted average = "
				<< nBinAvg
				<< " cm";
				
			}
			
			
			// Binning around med, halfWidth
			int nBin = 100;
			float halfWidth = 0.1; // cm
			
			// Most probable
			TH1F histo("histo","histo", nBin, -halfWidth,halfWidth);
			
			for(std::vector<const reco::Track *>::const_iterator
				track = tracks.begin(); track!= tracks.end(); track++)
				if(fabs((*track)->vz() - med) < halfWidth)
					histo.Fill((*track)->vz() - med);
			
			LogTrace("MinBiasTracking")
			<< "  [vertex position] most prob = "
			<< med + histo.GetBinCenter(histo.GetMaximumBin())
			<< " cm";
			
			// Fit above max/2
			histo.Sumw2();
			
			if(histo.GetEntries()>5) {
				
				
				TF1 f1("f1","[0]*exp(-0.5 * ((x-[1])/[2])^2) + [3]");
				//f1.SetParameters(10.,0.,0.01, 1.);
				f1.SetParameters(10.,0.,0.02,0.002*tracks.size());
				f1.SetParLimits(1,-0.1,0.1);
				f1.SetParLimits(2,0.001,0.05);
				f1.SetParLimits(3,0.0,0.005*tracks.size());
				
				histo.Fit("f1","QN");
				
				LogTrace("MinBiasTracking")
				<< "  [vertex position] fitted    = "
				<< med + f1.GetParameter(1) << " +- " << f1.GetParError(1)
				<< " cm";
				
				// Store
				reco::Vertex::Error err;
				err(2,2) = f1.GetParError(1) * f1.GetParError(1); 
				reco::Vertex ver(reco::Vertex::Point(0,0,med + f1.GetParameter(1)),
								 err, 0, 1, 1);
				vertices->push_back(ver);
				
			} else {
				
				LogTrace("MinBiasTracking")
				<< "  [vertex position] Fewer than six entries in fit histogram.  Using median.";
				
				// Store
				reco::Vertex::Error err;
				err(2,2) = 0.1 * 0.1;
				reco::Vertex ver(reco::Vertex::Point(0,0,med),
								 err, 0, 1, 1);
				vertices->push_back(ver);
			}
			
			
		}
		else
		{
			// Store
			reco::Vertex::Error err;
			err(2,2) = 0.1 * 0.1;
			reco::Vertex ver(reco::Vertex::Point(0,0,med),
							 err, 0, 1, 1);
			vertices->push_back(ver);
		}
	}
	ev.put(vertices);
}

