//
// Package:   FilterScrapingPixelProbability
// Class:     FilterScrapingPixelProbability
//
// Original Author:  Gavril Giurgiu (JHU)
//
// This filter attempts to separate scraping events from collision 
// events. 
//
// It loops over all tracks in the event and 
// counts the number of barel pixel hits which have low probability. 
// The filter is applied on the fraction of hits with low 
// probability. The default "low probability" is 0 (less than 
// 10^{-15}). For this choice of low probability the optimal  
// cut is somewhere between 0.3 and 0.5. Default is 0.4. 
//


#ifndef FilterScrapingPixelProbability_H
#define FilterScrapingPixelProbability_H

// system include files
#include <memory>
#include <vector>
#include <map>
#include <set>

// user include files
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

class FilterScrapingPixelProbability : public edm::EDFilter 
{
public:
  explicit FilterScrapingPixelProbability( const edm::ParameterSet & );
  ~FilterScrapingPixelProbability();
  
private:
  virtual bool filter ( edm::Event &, const edm::EventSetup & );
  
  bool apply_filter;
  bool select_collision;
  bool select_pkam;
  bool select_other;
  double low_probability;              // Default is 0 which means less than ~10^{-15}.
  double low_probability_fraction_cut; // Default is 0.4.

  edm::InputTag tracks_;
};

#endif
