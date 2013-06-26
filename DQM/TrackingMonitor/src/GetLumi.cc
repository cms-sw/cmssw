/*
 *  See header file for a description of this class.
 *
 *  $Date: 2013/05/30 22:09:25 $
 *  $Revision: 1.4 $
 *  \author:  Mia Tosi,40 3-B32,+41227671609 
 */

#include "DQM/TrackingMonitor/interface/GetLumi.h"

#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "DataFormats/Luminosity/interface/LumiDetails.h"
#include "DataFormats/Luminosity/interface/LumiSummary.h"

GetLumi::GetLumi(const edm::ParameterSet& iConfig)
  : lumiInputTag_ ( iConfig.getParameter<edm::InputTag>("lumi")  )
  , lumiScale_    ( iConfig.getParameter<double>("lumiScale")    )
{
}

GetLumi::GetLumi(const edm::InputTag& lumiInputTag, double lumiScale)
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

  double bxlumi = -1.;
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

double
GetLumi::getRawValue(edm::LuminosityBlock const& lumiBlock,
		     edm::EventSetup const&eSetup)
{
  
  double lumi = -1.;
  double intDelLumi = -1.;

  //  size_t LS = lumiBlock.luminosityBlockAuxiliary().luminosityBlock();
  // accumulate HF data at every LS as it is closed. 
  // note: lumi unit from DIPLumiSummary and Detail is microbarns
  edm::Handle<LumiSummary> lumiSummary_;
  lumiBlock.getByLabel(lumiInputTag_, lumiSummary_);
  if(lumiSummary_->isValid()){
    lumi = lumiSummary_->avgInsDelLumi();
    intDelLumi = lumiSummary_->intgDelLumi();
    std::cout << "Luminosity in this Lumi Section " << lumi << " --> " << intDelLumi << std::endl;
  } else {
    std::cout << "No valid data found!" << std::endl;
  }


  return lumi;

}


double
GetLumi::getValue(edm::LuminosityBlock const& lumiBlock,
		  edm::EventSetup const&eSetup)
{
  return getRawValue(lumiBlock,eSetup)*lumiScale_;
}


double
GetLumi::convert2PU(double instLumi, double inelastic_xSec = GetLumi::INELASTIC_XSEC_8TeV) // inelastic_xSec in mb
{

  // from https://cmswbm.web.cern.ch/cmswbm/images/pileup.png
  return instLumi*inelastic_xSec/FREQ_ORBIT;
}

double
GetLumi::convert2PU(double instLumi, int sqrt_s = GetLumi::SQRT_S_8TeV)
{
  
  double inelastic_xSec = 0.;
  
  switch(sqrt_s) {
  case GetLumi::SQRT_S_7TeV :
    inelastic_xSec = GetLumi::INELASTIC_XSEC_7TeV;
    break;
  case GetLumi::SQRT_S_8TeV :
    inelastic_xSec = GetLumi::INELASTIC_XSEC_8TeV;
    break;
  default :
    break;
  }
  
  return convert2PU(instLumi,inelastic_xSec);

}
