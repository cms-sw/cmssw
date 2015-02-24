#include <vector>
#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/HcalDigi/interface/HcalQIESample.h"
#include "DataFormats/HcalDigi/interface/HcalUpgradeQIESample.h"
#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "DataFormats/HcalDigi/interface/HODataFrame.h"
#include "DataFormats/HcalDigi/interface/HcalHistogramDigi.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalUnpackerReport.h"
#include "DataFormats/HcalDigi/interface/HcalLaserDigi.h"
#include "DataFormats/HcalDigi/interface/HcalTTPDigi.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace DataFormats_HcalDigi {
  struct dictionary {
    std::vector<HcalQIESample> vQIE_;
    std::vector<HcalUpgradeQIESample> vUQIE_;
    std::vector<HcalTriggerPrimitiveSample> vTPS_;
    
    edm::SortedCollection<HBHEDataFrame> vHBHE_;
    edm::SortedCollection<HODataFrame> vHO_;
    edm::SortedCollection<HFDataFrame> vHF_;
    edm::SortedCollection<HcalCalibDataFrame> vHC_;
    edm::SortedCollection<HcalTriggerPrimitiveDigi> vHTP_;
    edm::SortedCollection<HcalHistogramDigi> vHH_;
    edm::SortedCollection<HcalTTPDigi> vTTP_;
    edm::SortedCollection<HcalUpgradeDataFrame> vU_;

    HBHEDigiCollection theHBHE_;
    HODigiCollection theHO_;
    HFDigiCollection theHF_;
    HcalCalibDigiCollection theHC_;
    HcalTrigPrimDigiCollection theHTP_;
    HcalHistogramDigiCollection theHH_;
    ZDCDigiCollection theZDC_;
    HBHEUpgradeDigiCollection theUHBHE_;
    HFUpgradeDigiCollection theUHF_;
    CastorDigiCollection theCastor_;
    CastorTrigPrimDigiCollection theCastorTP_;
    HOTrigPrimDigiCollection theHOTP_;
    HcalTTPDigiCollection theTTP_;
    QIE10DigiCollection theqie10_;
    QIE11DigiCollection theqie11_;
      
    edm::Wrapper<edm::SortedCollection<HBHEDataFrame> > anotherHBHE_;
    edm::Wrapper<edm::SortedCollection<HODataFrame> > anotherHO_;
    edm::Wrapper<edm::SortedCollection<HFDataFrame> > anotherHF_;
    edm::Wrapper<edm::SortedCollection<HcalCalibDataFrame> > anotherHC_;
    edm::Wrapper<edm::SortedCollection<HcalTriggerPrimitiveDigi> > anotherHTP_;
    edm::Wrapper<edm::SortedCollection<HcalHistogramDigi> > anotherHH_;
    edm::Wrapper<edm::SortedCollection<ZDCDataFrame> > anotherZDC_;
    edm::Wrapper<edm::SortedCollection<CastorDataFrame> > anotherCastor_;
    edm::Wrapper<edm::SortedCollection<CastorTriggerPrimitiveDigi> > anotherCastorTP_;
    edm::Wrapper<edm::SortedCollection<HOTriggerPrimitiveDigi> > anotherHOTP_;
    edm::Wrapper<edm::SortedCollection<HcalTTPDigi> > anotherTTP_;
    edm::Wrapper<edm::SortedCollection<HcalUpgradeDataFrame> > anotherUG_;

    edm::Wrapper<HBHEDigiCollection> theHBHEw_;
    edm::Wrapper<HODigiCollection> theHOw_;
    edm::Wrapper<HFDigiCollection> theHFw_;
    edm::Wrapper<HcalCalibDigiCollection> theHCw_;
    edm::Wrapper<HcalTrigPrimDigiCollection> theHTPw_; 
    edm::Wrapper<HOTrigPrimDigiCollection> theHOTPw_; 
    edm::Wrapper<HcalHistogramDigiCollection> theHHw_; 
    edm::Wrapper<HcalUnpackerReport> theReport_;
    edm::Wrapper<HcalLaserDigi> theLaserw_;
    edm::Wrapper<HcalTTPDigiCollection> theTTPw_;
    edm::Wrapper<HBHEUpgradeDigiCollection> theUHBHEw_;
    edm::Wrapper<HFUpgradeDigiCollection> theUHFw_;
    edm::Wrapper<QIE10DigiCollection> theQIE10w_;
    edm::Wrapper<QIE11DigiCollection> theQIE11w_;
  };
}

