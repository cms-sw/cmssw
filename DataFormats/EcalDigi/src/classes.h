#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace {
  struct dictionary {
    std::vector<EcalMGPASample> vMGPA_;
    std::vector<EcalFEMSample> vFEM_;
    std::vector<ESSample> vESSample_;
    std::vector<float> vETS_;
    std::vector<EcalTriggerPrimitiveSample> vETPS_;
    std::vector<EcalPseudoStripInputSample> vEPSIS_;
    std::vector<EcalMatacqDigi> vMD_;

    edm::SortedCollection<ESDataFrame> vES_;
    edm::SortedCollection<EcalTimeDigi> vETDP_;
    edm::SortedCollection<EcalTriggerPrimitiveDigi> vETP_;
    edm::SortedCollection<EcalPseudoStripInputDigi> vEPSI_;
    edm::SortedCollection<EBSrFlag> vEBSRF_;
    edm::SortedCollection<EESrFlag> vEESRF_;
    edm::SortedCollection<EcalPnDiodeDigi> vEPN_;
    edm::SortedCollection<EcalMatacqDigi> vMDS_;
    EcalMatacqDigi Matacq_;

    EBDigiCollection theEB_;
    EEDigiCollection theEE_;
    ESDigiCollection theES_;
    EcalTimeDigiCollection theEBTime_;
    EcalTrigPrimDigiCollection theETP_;
    EcalTrigPrimCompactColl theETP2_;
    
    EBSrFlagCollection theEBSRF_;
    EESrFlagCollection theEESRF_;
    EcalPnDiodeDigiCollection theEPN_;
    EcalMatacqDigiCollection theMD_;

    edm::Wrapper<EcalDigiCollection> anotherECalw_;
    edm::Wrapper<EBDigiCollection> anotherEBw_;
    edm::Wrapper<EEDigiCollection> anotherEEw_;
    edm::Wrapper<ESDigiCollection> anotherESw_;
    edm::Wrapper<EcalTimeDigiCollection> anotherETDw_;
    edm::Wrapper<EcalTrigPrimDigiCollection> anotherETPw_;
    edm::Wrapper<EcalTrigPrimCompactColl> anotherETP2w_;
    edm::Wrapper<EBSrFlagCollection> anotherEBSRFw_;
    edm::Wrapper<EESrFlagCollection> anotherEESRFw_;
    edm::Wrapper<EcalPnDiodeDigiCollection> anotherEPNw_;
    edm::Wrapper<EcalMatacqDigiCollection> anotherMDw_;

    edm::Wrapper< edm::SortedCollection<ESDataFrame> > theESw_;
    edm::Wrapper< edm::SortedCollection<EcalTimeDigi> > theETDw_;
    edm::Wrapper< edm::SortedCollection<EcalTriggerPrimitiveDigi> > theETPw_;
    edm::Wrapper< edm::SortedCollection<EcalPseudoStripInputDigi> > theEPSIw_;
    edm::Wrapper< edm::SortedCollection<EBSrFlag> > theEBSRFw_;
    edm::Wrapper< edm::SortedCollection<EESrFlag> > theEESRFw_;
    edm::Wrapper< edm::SortedCollection<EcalPnDiodeDigi> > theEPNw_; 
    edm::Wrapper< edm::SortedCollection<EcalMatacqDigi> > theMDw_; 
 };
}

