#include "DQM/SiPixelMonitorClient/interface/SiPixelDcsInfo.h"
//#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "CondFormats/RunInfo/interface/RunSummary.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"

using namespace std;
using namespace edm;
SiPixelDcsInfo::SiPixelDcsInfo(const edm::ParameterSet& ps) {
 
  firstRun = true;
}

SiPixelDcsInfo::~SiPixelDcsInfo(){}

void SiPixelDcsInfo::dqmEndLuminosityBlock(DQMStore::IBooker & iBooker, DQMStore::IGetter & iGetter, const edm::LuminosityBlock&  lumiBlock, const  edm::EventSetup& iSetup){
  
  if (firstRun){
    iBooker.setCurrentFolder("Pixel/EventInfo");
    Fraction_= iBooker.bookFloat("DCSSummary");  
    iBooker.setCurrentFolder("Pixel/EventInfo/DCSContents");
    FractionBarrel_= iBooker.bookFloat("PixelBarrelFraction");  
    FractionEndcap_= iBooker.bookFloat("PixelEndcapFraction");  
  }

  if(iSetup.tryToGet<RunInfoRcd>()) {
      //all Pixel:
      Fraction_->Fill(1.);
      //Barrel:
      FractionBarrel_->Fill(1.);
      //Endcap:
      FractionEndcap_->Fill(1.);
    return; 
  }
}

void SiPixelDcsInfo::dqmEndJob(DQMStore::IBooker & iBooker, DQMStore::IGetter & iGetter){
  //Nothing actually happened in the old endJob/endRun, so this is left empty.
}
