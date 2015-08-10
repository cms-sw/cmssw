#ifndef __L1Analysis_L1AnalysisCSCTF_H__
#define __L1Analysis_L1AnalysisCSCTF_H__

//-------------------------------------------------------------------------------
// Created 08/03/2010 - A.-C. Le Bihan
//
//x
// Original code : L1Trigger/L1TNtuples/L1NtupleProducer -  Gian Piero Di Giovanni
//-------------------------------------------------------------------------------


#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCStatusDigiCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/CSCTriggerContainer.h"
#include "DataFormats/L1CSCTrackFinder/interface/TrackStub.h"
#include "L1Trigger/CSCTrackFinder/interface/CSCTFPtLUT.h" 

#include "L1Trigger/CSCTrackFinder/interface/CSCSectorReceiverLUT.h"
#include "L1Trigger/CSCTrackFinder/interface/CSCTFPtLUT.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerPtScale.h"

#include "L1AnalysisCSCTFDataFormat.h"
#include <TMath.h>

namespace L1Analysis
{
  class L1AnalysisCSCTF
  {
  public:
    L1AnalysisCSCTF();
    ~L1AnalysisCSCTF();
    
    void SetTracks(const edm::Handle<L1CSCTrackCollection> csctfTrks, const L1MuTriggerScales  *ts, const L1MuTriggerPtScale *tpts, 
        	   CSCSectorReceiverLUT* srLUTs_[5][2],
                   CSCTFPtLUT* ptLUTs_);  
    void SetStatus(const edm::Handle<L1CSCStatusDigiCollection> status);
    void SetLCTs(const edm::Handle<CSCCorrelatedLCTDigiCollection> corrlcts, CSCSectorReceiverLUT* srLUTs_[5][2]);
    void SetDTStubs(const edm::Handle<CSCTriggerContainer<csctf::TrackStub> > dtStubs);
    L1AnalysisCSCTFDataFormat * getData() {return &csctf_;}
    void Reset() {csctf_.Reset();}

  private : 
    L1AnalysisCSCTFDataFormat csctf_;

  }; 
} 
#endif


