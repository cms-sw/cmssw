#ifndef CalibCalorimetry_EcalPedestalOffsets_EcalPedOffset_H
#define CalibCalorimetry_EcalPedestalOffsets_EcalPedOffset_H

/**
 * \file EcalPedOffset.h
 * \class EcalPedOffset
 * \brief calculate the best DAC value to obtain a pedestal = 200
 * \author P. Govoni (pietro.govoni@cernNOSPAM.ch)
 *
 */

#include <map>
#include <string>

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>
#include <DataFormats/EcalRawData/interface/EcalRawDataCollections.h>
#include <Geometry/EcalMapping/interface/EcalElectronicsMapping.h>
#include <Geometry/EcalMapping/interface/EcalMappingRcd.h>

#include "CalibCalorimetry/EcalPedestalOffsets/interface/TPedResult.h"
#include "CalibCalorimetry/EcalPedestalOffsets/interface/TPedValues.h"

class EcalPedOffset : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  //! Constructor
  EcalPedOffset(const edm::ParameterSet &ps);

  //! Destructor
  ~EcalPedOffset() override;

  ///! Analyze
  void analyze(edm::Event const &event, edm::EventSetup const &eventSetup) override;

  //! BeginRun
  void beginRun(edm::Run const &, edm::EventSetup const &eventSetup) override;

  //! EndRun
  void endRun(edm::Run const &, edm::EventSetup const &) override;

  //! EndJob
  void endJob(void) override;

  //! write the results into xml format
  void writeXMLFiles(std::string fileName);

  //! WriteDB
  void writeDb();

  //! create the plots of the DAC pedestal trend
  void makePlots();

private:
  const EcalElectronicsMapping *ecalElectronicsMap_;

  std::string intToString(int num);
  void readDACs(const edm::Handle<EBDigiCollection> &pDigis, const std::map<int, int> &DACvalues);
  void readDACs(const edm::Handle<EEDigiCollection> &pDigis, const std::map<int, int> &DACvalues);

  const edm::InputTag m_barrelDigiCollection;  //!< secondary name given to collection of digis
  const edm::InputTag m_endcapDigiCollection;  //!< secondary name given to collection of digis
  const edm::InputTag m_headerCollection;      //!< name of module/plugin/producer making headers

  const edm::EDGetTokenT<EcalRawDataCollection> m_rawDataToken;
  const edm::EDGetTokenT<EBDigiCollection> m_ebDigiToken;
  const edm::EDGetTokenT<EEDigiCollection> m_eeDigiToken;
  const edm::ESGetToken<EcalElectronicsMapping, EcalMappingRcd> m_mappingToken;

  std::string m_xmlFile;  //!< name of the xml file to be saved

  std::map<int, TPedValues *> m_pedValues;
  std::map<int, TPedResult *> m_pedResult;

  int m_DACmin;
  int m_DACmax;
  double m_RMSmax;
  int m_bestPed;

  //! database host name
  std::string m_dbHostName;
  //! database name
  std::string m_dbName;
  //! database user name
  std::string m_dbUserName;
  //! database user password
  std::string m_dbPassword;
  //! database
  int m_dbHostPort;
  //! allow the creation of a new moniov if not existing in the DB
  //! by default it is false.
  bool m_create_moniov;
  // used to retrieve the run_iov
  std::string m_location;
  //! run number
  int m_run;

  //! the root file where to store the detail plots
  std::string m_plotting;
  //! max slope (in magnitude) allowed for linearity test
  double m_maxSlopeAllowed_;
  //! min slope (in magnitude) allowed for linearity test
  double m_minSlopeAllowed_;
  //! max chi2/ndf allowed for linearity test
  double m_maxChi2OverNDFAllowed_;
};

#endif
