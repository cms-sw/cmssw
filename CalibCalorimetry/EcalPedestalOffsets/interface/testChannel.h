#ifndef testChannel_H
#define testChannel_H

/**
 * \file testChannel.h
 * \class testChannel
 * \brief calculate the best DAC value to obtain a pedestal = 200
 * \author P. Govoni (testChannel.govoni@cernNOSPAM.ch)
 *
 */

#include "CalibCalorimetry/EcalPedestalOffsets/interface/TPedResult.h"
#include "CalibCalorimetry/EcalPedestalOffsets/interface/TPedValues.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include <map>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TROOT.h"
#include "TStyle.h"

#include "TH2.h"
#include "TProfile.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

class testChannel : public edm::one::EDAnalyzer<> {
public:
  //! Constructor
  testChannel(const edm::ParameterSet &ps);

  //! Destructor
  ~testChannel() override;

  //! Subscribe/Unsubscribe to Monitoring Elements
  void subscribe(void);
  void subscribeNew(void);
  void unsubscribe(void);

  ///! Analyze
  void analyze(edm::Event const &event, edm::EventSetup const &eventSetup) override;

  //! BeginJob
  void beginJob() override;

  //! EndJob
  void endJob(void) override;

private:
  int getHeaderSMId(const int headerId);

  const edm::EDGetTokenT<EBDigiCollection> m_digiProducerToken;         //! Token to access digis
  const edm::EDGetTokenT<EcalRawDataCollection> m_headerProducerToken;  //! Token to access headers

  const std::string m_xmlFile;  //! name of the xml file to be saved

  const int m_DACmin;
  const int m_DACmax;
  const double m_RMSmax;
  const int m_bestPed;

  const int m_xtal;

  TH2F m_pedVSDAC;
  TH2F m_singlePedVSDAC_1;
  TH2F m_singlePedVSDAC_2;
  TH2F m_singlePedVSDAC_3;
};

#endif
