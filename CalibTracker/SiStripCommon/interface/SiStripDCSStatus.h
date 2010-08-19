#ifndef SiStripDCSStatus_H
#define SiStripDCSStatus_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
 
class SiStripDCSStatus {
 public:
  SiStripDCSStatus();
 ~SiStripDCSStatus();

  bool getStatus(edm::Event const& e, edm::EventSetup const& eSetup);

 private: 

  void initialise(edm::Event const& e, edm::EventSetup const& eSetup);

  bool TIBTIDinDAQ, TOBinDAQ, TECFinDAQ, TECBinDAQ;
  bool statusTIBTID, statusTOB, statusTECF, statusTECB;
  bool trackerAbsent;
  bool rawdataAbsent;
  bool initialised;
};

#endif
