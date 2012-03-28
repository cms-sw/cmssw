/*
 *  See header file for a description of this class.
 *
 *  $Date: 2012/02/19 12:17:47 $
 *  $Revision: 1.1 $
 *  \author:  Mia Tosi,40 3-B32,+41227671609 
 */

#include "DQM/TrackingMonitor/interface/GetLumi.h"

#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "DataFormats/Luminosity/interface/LumiDetails.h"

GetLumi::GetLumi(const edm::ParameterSet& iConfig)
  : lumiInputTag_ ( iConfig.getParameter<edm::InputTag>("lumi")  )
  , lumiScale_    ( iConfig.getParameter<double>("lumiScale")    )
{
}

GetLumi::GetLumi(edm::InputTag lumiInputTag, double lumiScale)
  : lumiInputTag_ ( lumiInputTag )
  , lumiScale_    ( lumiScale    )
{
}

GetLumi::~GetLumi()
{
}

double
GetLumi::getRawValue(const edm::Event& iEvent)
{

  // taken from 
  // DPGAnalysis/SiStripTools/src/DigiLumiCorrHistogramMaker.cc
  // the scale factor 6.37 should follow the lumi prescriptions
  edm::Handle<LumiDetails> lumi;
  iEvent.getLuminosityBlock().getByLabel(lumiInputTag_,lumi);

  double bxlumi = 0;
  if(lumi->isValid()) {
    bxlumi = lumi->lumiValue(LumiDetails::kOCC1,iEvent.bunchCrossing());
  }

  return bxlumi;

}


double
GetLumi::getValue(const edm::Event& iEvent)
{
  //    bxlumi = lumi->lumiValue(LumiDetails::kOCC1,iEvent.bunchCrossing())*6.37;
  return getRawValue(iEvent)*lumiScale_;
}
