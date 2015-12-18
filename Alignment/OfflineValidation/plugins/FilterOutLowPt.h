// -*- C++ -*-
//
// Package:   FilterOutLowPt
// Class:     FilterOutLowPt
//
// Original Author:  Luca Malgeri

#ifndef FilterOutLowPt_H
#define FilterOutLowPt_H

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
#include "FWCore/Utilities/interface/EDGetToken.h"

//
// class declaration
//

class FilterOutLowPt : public edm::EDFilter {
public:
  explicit FilterOutLowPt( const edm::ParameterSet & );
  ~FilterOutLowPt();
  
private:
  virtual void beginJob() ;
  virtual bool filter ( edm::Event &, const edm::EventSetup&); 
  virtual void endJob() ;

  bool applyfilter;
  bool debugOn;
  double thresh;
  unsigned int numtrack;
  double  ptmin;
  edm::InputTag tracks_;
  double trials;
  double passes;
  bool runControl_;
  unsigned int runControlNumber_;
  std::map<unsigned int,std::pair<int,int>> eventsInRun_;

  reco::TrackBase::TrackQuality _trackQuality;
  edm::EDGetTokenT<reco::TrackCollection>  theTrackCollectionToken; 

};

#endif
