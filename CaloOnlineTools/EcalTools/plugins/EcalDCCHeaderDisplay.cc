/**
 * module  for displaying unpacked DCCHeader information
 *   
 * \author A. Ghezzi
 * \author S. Cooper
 * \author G. Franzoni
 *
 */

#include "CaloOnlineTools/EcalTools/plugins/EcalDCCHeaderDisplay.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <sstream>

EcalDCCHeaderDisplay::EcalDCCHeaderDisplay(const edm::ParameterSet& iConfig)
    : EcalDCCHeaderCollection_(
          consumes<EcalRawDataCollection>(iConfig.getParameter<edm::InputTag>("EcalDCCHeaderCollection"))) {}

void EcalDCCHeaderDisplay::analyze(const edm::Event& e, const edm::EventSetup& c) {
  const edm::Handle<EcalRawDataCollection>& DCCHeaders = e.getHandle(EcalDCCHeaderCollection_);

  edm::LogVerbatim("EcalTools") << "\n\n ^^^^^^^^^^^^^^^^^^ [EcalDCCHeaderDisplay]  DCCHeaders collection size "
                                << DCCHeaders->size() << std::endl;
  edm::LogVerbatim("EcalTools") << "          [EcalDCCHeaderDisplay]  the Header(s)\n" << std::endl;
  //short dumpConter =0;

  for (EcalRawDataCollection::const_iterator headerItr = DCCHeaders->begin(); headerItr != DCCHeaders->end();
       ++headerItr) {
    //      int nevt =headerItr->getLV1();
    bool skip = false;

    if (skip) {
      continue;
    }
    edm::LogVerbatim("EcalTools") << "######################################################################";
    edm::LogVerbatim("EcalTools") << "FedId: " << headerItr->fedId();

    edm::LogVerbatim("EcalTools") << "DCCErrors: " << headerItr->getDCCErrors();
    edm::LogVerbatim("EcalTools") << "Run Number: " << headerItr->getRunNumber();
    edm::LogVerbatim("EcalTools") << "Event number (LV1): " << headerItr->getLV1();
    edm::LogVerbatim("EcalTools") << "Orbit: " << headerItr->getOrbit();
    edm::LogVerbatim("EcalTools") << "BX: " << headerItr->getBX();
    edm::LogVerbatim("EcalTools") << "TRIGGER TYPE: " << headerItr->getBasicTriggerType();

    edm::LogVerbatim("EcalTools") << "RUNTYPE: " << headerItr->getRunType();
    edm::LogVerbatim("EcalTools") << "Half: " << headerItr->getRtHalf();
    edm::LogVerbatim("EcalTools") << "DCCIdInTCCCommand: " << headerItr->getDccInTCCCommand();
    edm::LogVerbatim("EcalTools") << "MGPA gain: " << headerItr->getMgpaGain();
    edm::LogVerbatim("EcalTools") << "MEM gain: " << headerItr->getMemGain();
    EcalDCCHeaderBlock::EcalDCCEventSettings settings = headerItr->getEventSettings();
    edm::LogVerbatim("EcalTools") << "LaserPower: " << settings.LaserPower;
    edm::LogVerbatim("EcalTools") << "LAserFilter: " << settings.LaserFilter;
    edm::LogVerbatim("EcalTools") << "Wavelenght: " << settings.wavelength;
    edm::LogVerbatim("EcalTools") << "delay: " << settings.delay;
    edm::LogVerbatim("EcalTools") << "MEM Vinj: " << settings.MEMVinj;
    edm::LogVerbatim("EcalTools") << "MGPA content: " << settings.mgpa_content;
    edm::LogVerbatim("EcalTools") << "Ped offset dac: " << settings.ped_offset;

    edm::LogVerbatim("EcalTools") << "Selective Readout: " << headerItr->getSelectiveReadout();
    edm::LogVerbatim("EcalTools") << "ZS: " << headerItr->getZeroSuppression();
    edm::LogVerbatim("EcalTools") << "TZS: " << headerItr->getTestZeroSuppression();
    edm::LogVerbatim("EcalTools") << "SRStatus: " << headerItr->getSrpStatus();

    std::vector<short> TCCStatus = headerItr->getTccStatus();
    edm::LogVerbatim("EcalTools") << "TCC Status size: " << TCCStatus.size();
    std::ostringstream st0;
    st0 << "TCC Status: ";
    for (unsigned u = 0; u < TCCStatus.size(); u++) {
      st0 << TCCStatus[u] << " ";
    }
    edm::LogVerbatim("EcalTools") << st0.str();

    std::vector<short> TTStatus = headerItr->getFEStatus();
    edm::LogVerbatim("EcalTools") << "TT Status size: " << TTStatus.size() << std::endl;
    std::ostringstream st1[100];
    int k(0), kk(0);
    st1[k] << "TT Statuses: ";
    for (unsigned u = 0; u < TTStatus.size(); u++) {
      ++kk;
      if (!(u % 14)) {
        edm::LogVerbatim("EcalTools") << st1[k].str();  // TODO: add space after first six in a row
        ++k;
        kk = 0;
      }
      st1[k] << TTStatus[u] << " ";
    }
    if (kk > 0)
      edm::LogVerbatim("EcalTools") << st1[k].str();
    edm::LogVerbatim("EcalTools") << "\n######################################################################";
    ;
  }
}
