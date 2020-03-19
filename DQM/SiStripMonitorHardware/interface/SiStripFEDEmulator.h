#ifndef DQM_SiStripMonitorHardware_SiStripFEDEmulator_H
#define DQM_SiStripMonitorHardware_SiStripFEDEmulator_H

// Created 2010-01-20 by A.-M. Magnan
// Class intended to mimic the data path in the FED firmware in software
// steps: pedestal subtraction, CM subtraction, clustering and zero-suppression.
#include <sstream>
#include <fstream>
#include <iostream>

#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripProcessedRawDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"

#include "DataFormats/Common/interface/Wrapper.h"

#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"

#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBuffer.h"

//for the zero suppression algorithm(s)
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripRawProcessingFactory.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripRawProcessingAlgorithms.h"

namespace sistrip {

  class FEDEmulator {
  public:
    FEDEmulator();
    ~FEDEmulator();

    void initialise(const bool byModule);

    void initialiseModule(const uint32_t aDetId, const uint32_t aNPairs, const uint32_t aPair);

    void retrievePedestals(const edm::ESHandle<SiStripPedestals>& aHandle);
    void retrieveNoises(const edm::ESHandle<SiStripNoises>& aHandle);

    void subtractPedestals(const edm::DetSetVector<SiStripRawDigi>::const_iterator& inputChannel,
                           std::vector<SiStripRawDigi>& pedsDetSetData,
                           std::vector<SiStripProcessedRawDigi>& noiseDetSetData,
                           std::vector<SiStripRawDigi>& pedSubtrDetSetData,
                           std::vector<uint32_t>& medsDetSetData,
                           const bool fillApvsForCM);

    void subtractCM(const std::vector<SiStripRawDigi>& pedSubtrDetSetData,
                    std::vector<SiStripRawDigi>& cmSubtrDetSetData);

    void zeroSuppress(const std::vector<SiStripRawDigi>& cmSubtrDetSetData,
                      edm::DetSet<SiStripDigi>& zsDetSetData,
                      const std::unique_ptr<SiStripRawProcessingAlgorithms>& algorithms);

    uint32_t fedIndex(const uint16_t aFedChannel);

    void fillPeds(const edm::DetSetVector<SiStripRawDigi>::const_iterator& peds);
    void fillNoises(const edm::DetSetVector<SiStripProcessedRawDigi>::const_iterator& noise);

    void fillMedians(const std::map<uint32_t, std::vector<uint32_t> >::const_iterator& meds);

    void print(std::ostream& aOs);
    void printPeds(std::ostream& aOs);
    void printNoises(std::ostream& aOs);
    void printMeds(std::ostream& aOs);

  private:
    static const char* messageLabel_;

    bool byModule_;

    uint32_t detId_;
    uint32_t nPairs_;
    uint32_t pair_;

    uint32_t minStrip_;
    uint32_t maxStrip_;

    std::vector<int> pedestals_;
    std::vector<float> noises_;
    std::vector<uint32_t> medians_;

  };  //class FEDEmulator

}  //namespace sistrip
#endif  //DQM_SiStripMonitorHardware_SiStripFEDEmulator_H
