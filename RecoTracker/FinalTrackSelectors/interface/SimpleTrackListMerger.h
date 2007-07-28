#ifndef SimpleTrackListMerger_h
#define SimpleTrackListMerger_h

//
// Package:         RecoTracker/FinalTrackSelectors
// Class:           SimpleTrackListMerger
// 
// Description:     Hit Dumper
//
// Original Author: Steve Wagner, stevew@pizero.colorado.edu
// Created:         Sat Jan 14 22:00:00 UTC 2006
//
// $Author: stevew $
// $Date: 2007/07/21 23:31:59 $
// $Revision: 1.2 $
//

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/EDProduct.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace cms
{
  class SimpleTrackListMerger : public edm::EDProducer
  {
  public:

    explicit SimpleTrackListMerger(const edm::ParameterSet& conf);

    virtual ~SimpleTrackListMerger();

    virtual void produce(edm::Event& e, const edm::EventSetup& c);

  private:
    edm::ParameterSet conf_;

  };
}


#endif
