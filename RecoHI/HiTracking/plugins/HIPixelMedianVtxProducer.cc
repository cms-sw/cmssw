#include "HIPixelMedianVtxProducer.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"

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
HIPixelMedianVtxProducer::HIPixelMedianVtxProducer(const edm::ParameterSet& ps) : 
  theTrackCollection(ps.getParameter<edm::InputTag>("TrackCollection")),
  thePtMin(ps.getParameter<double>("PtMin")),
  thePeakFindThresh(ps.getParameter<unsigned int>("PeakFindThreshold")),
  thePeakFindMaxZ(ps.getParameter<double>("PeakFindMaxZ")),
  thePeakFindBinning(ps.getParameter<int>("PeakFindBinsPerCm")),
  theFitThreshold(ps.getParameter<int>("FitThreshold")),
  theFitMaxZ(ps.getParameter<double>("FitMaxZ")),
  theFitBinning(ps.getParameter<int>("FitBinsPerCm"))
{
  produces<reco::VertexCollection>();
}

/*****************************************************************************/
void HIPixelMedianVtxProducer::produce
(edm::Event& ev, const edm::EventSetup& es)
{
  // Get pixel tracks
  edm::Handle<reco::TrackCollection> trackCollection;
  ev.getByLabel(theTrackCollection, trackCollection);
  const reco::TrackCollection tracks_ = *(trackCollection.product());
  
  // Select tracks above minimum pt
  std::vector<const reco::Track *> tracks;
  for (unsigned int i=0; i<tracks_.size(); i++) {
    if (tracks_[i].pt() <  thePtMin && std::fabs(tracks_[i].vz()) < 100000.) continue;
    reco::TrackRef recTrack(trackCollection, i);
    tracks.push_back( &(*recTrack));
  }
	
  LogTrace("MinBiasTracking")
    << " [VertexProducer] selected tracks: "
    << tracks.size() << " (out of " << tracks_.size()
    << ") above pt = " << thePtMin; 
	
  // Output vertex collection
  std::auto_ptr<reco::VertexCollection> vertices(new reco::VertexCollection);

  // No tracks -> return empty collection
  if(tracks.size() == 0) {
      ev.put(vertices);
      return;
  }

  // Sort tracks according to vertex z position
  std::sort(tracks.begin(), tracks.end(), ComparePairs());
  
  // Calculate median vz
  float med;
  if(tracks.size() % 2 == 0)
    med = (tracks[tracks.size()/2-1]->vz() + tracks[tracks.size()/2]->vz())/2;
  else
    med =  tracks[tracks.size()/2  ]->vz();
  
  LogTrace("MinBiasTracking")
    << "  [vertex position] median    = " << med << " cm";
  
  // In high multiplicity events, fit around most probable position
  if(tracks.size() > thePeakFindThresh) { 
	  
    // Find maximum bin
    TH1F hmax("hmax","hmax",thePeakFindBinning*2.0*thePeakFindMaxZ,-1.0*thePeakFindMaxZ,thePeakFindMaxZ);
    
    for(std::vector<const reco::Track *>::const_iterator
	  track = tracks.begin(); track!= tracks.end(); track++)
      if(fabs((*track)->vz()) < thePeakFindMaxZ)
	hmax.Fill((*track)->vz());
    
    int maxBin = hmax.GetMaximumBin();
    
    LogTrace("MinBiasTracking")
      << "  [vertex position] most prob = "
      << hmax.GetBinCenter(maxBin) << " cm";
    
    // Find 3-bin weighted average
    float num=0.0, denom=0.0;
    for(int i=-1; i<=1; i++) {
      num += hmax.GetBinContent(maxBin+i)*hmax.GetBinCenter(maxBin+i);
      denom += hmax.GetBinContent(maxBin+i); 
    }

    if(denom==0) 
    {
      reco::Vertex::Error err;
      err(2,2) = 0.1 * 0.1;
      reco::Vertex ver(reco::Vertex::Point(0,0,99999.),
                       err, 0, 0, 1);
      vertices->push_back(ver);
      ev.put(vertices);
      return;
    }

    float nBinAvg = num/denom;

    // Center fit at 3-bin weighted average around max bin
    med = nBinAvg; 
    
    LogTrace("MinBiasTracking")
      << "  [vertex position] 3-bin weighted average = "
      << nBinAvg << " cm";
  }
  
  // Bin vz-values around most probable value (or median) for fitting
  TH1F histo("histo","histo", theFitBinning*2.0*theFitMaxZ,-1.0*theFitMaxZ,theFitMaxZ);
  histo.Sumw2();

  for(std::vector<const reco::Track *>::const_iterator
	track = tracks.begin(); track!= tracks.end(); track++)
    if(fabs((*track)->vz() - med) < theFitMaxZ)
      histo.Fill((*track)->vz() - med);
  
  LogTrace("MinBiasTracking")
    << "  [vertex position] most prob for fit = "
    << med + histo.GetBinCenter(histo.GetMaximumBin()) << " cm";

  // If there are very few entries, don't do the fit
  if(histo.GetEntries() <= theFitThreshold) {

    LogTrace("MinBiasTracking")
      << "  [vertex position] Fewer than" << theFitThreshold
      <<  " entries in fit histogram. Returning median.";
    
    reco::Vertex::Error err;
    err(2,2) = 0.1 * 0.1;
    reco::Vertex ver(reco::Vertex::Point(0,0,med),
		     err, 0, 1, 1);
    vertices->push_back(ver);
    ev.put(vertices);
    return;
  }

  // Otherwise, there are enough entries to refine the estimate with a fit
  TF1 f1("f1","[0]*exp(-0.5 * ((x-[1])/[2])^2) + [3]");
  f1.SetParameters(10.,0.,0.02,0.002*tracks.size());
  f1.SetParLimits(1,-0.1,0.1);
  f1.SetParLimits(2,0.001,0.05);
  f1.SetParLimits(3,0.0,0.005*tracks.size());
    
  histo.Fit("f1","QN");
    
  LogTrace("MinBiasTracking")
    << "  [vertex position] fitted    = "
    << med + f1.GetParameter(1) << " +- " << f1.GetParError(1) << " cm";
  
  reco::Vertex::Error err;
  err(2,2) = f1.GetParError(1) * f1.GetParError(1); 
  reco::Vertex ver(reco::Vertex::Point(0,0,med + f1.GetParameter(1)),
		   err, 0, 1, 1);
  vertices->push_back(ver);
  
  ev.put(vertices);
  return;

}

