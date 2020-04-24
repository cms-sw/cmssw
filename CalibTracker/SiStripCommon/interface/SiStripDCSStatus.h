#ifndef SiStripDCSStatus_H
#define SiStripDCSStatus_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

 
class SiStripDCSStatus {
 public:
  SiStripDCSStatus(edm::ConsumesCollector && iC) : SiStripDCSStatus( iC ) {};
  SiStripDCSStatus(edm::ConsumesCollector & iC);
 ~SiStripDCSStatus();

  bool getStatus(edm::Event const& e, edm::EventSetup const& eSetup);

 private: 

  void initialise(edm::Event const& e, edm::EventSetup const& eSetup);

  bool TIBTIDinDAQ, TOBinDAQ, TECFinDAQ, TECBinDAQ;
  bool statusTIBTID, statusTOB, statusTECF, statusTECB;
  bool trackerAbsent;
  bool rawdataAbsent;
  bool initialised;

  edm::EDGetTokenT<DcsStatusCollection>  dcsStatusToken_;
  edm::EDGetTokenT<FEDRawDataCollection> rawDataToken_;

};

#endif
