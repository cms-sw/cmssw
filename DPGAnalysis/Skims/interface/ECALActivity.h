// -*- C++ -*-
//
// Package:   ECALActivity
// Class:     ECALActivity
//
// Original Author:  Luca Malgeri

#ifndef ECALActivity_H
#define ECALActivity_H

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


//
// class declaration
//


class ECALActivity : public edm::EDFilter {
public:
  explicit ECALActivity( const edm::ParameterSet & );
  ~ECALActivity();
  
private:
  virtual bool filter ( edm::Event &, const edm::EventSetup&) override;
  
  edm::InputTag EBRecHitCollection_;
  edm::InputTag EERecHitCollection_;

  int EBnum;
  double EBthresh;
  int EEnum;
  double EEthresh;
  int ETOTnum;
  double ETOTthresh;
  bool applyfilter;

  
};

#endif
