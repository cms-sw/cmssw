// -*- C++ -*-
//
// Package:    L1EmEtScaleOnlineProd
// Class:      L1EmEtScaleOnlineProd
//
/**\class L1EmEtScaleOnlineProd L1EmEtScaleOnlineProd.h L1Trigger/L1EmEtScaleProducers/src/L1EmEtScaleOnlineProd.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Werner Man-Li Sun
//         Created:  Tue Sep 16 22:43:22 CEST 2008
//
//

// system include files

// user include files
#include "CondTools/L1Trigger/interface/L1ConfigOnlineProdBase.h"
#include "CondFormats/L1TObjects/interface/L1CaloHcalScale.h"
#include "CondFormats/DataRecord/interface/L1CaloHcalScaleRcd.h"
#include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "CalibCalorimetry/CaloTPG/interface/CaloTPGTranscoderULUT.h"
#include "CondTools/L1Trigger/interface/OMDSReader.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <cmath>
#include <iostream>
#include <iomanip>

//
// class declaration
//

class L1CaloHcalScaleConfigOnlineProd : public L1ConfigOnlineProdBase<L1CaloHcalScaleRcd, L1CaloHcalScale> {
public:
  L1CaloHcalScaleConfigOnlineProd(const edm::ParameterSet& iConfig);
  ~L1CaloHcalScaleConfigOnlineProd() override;

  std::unique_ptr<L1CaloHcalScale> produce(const L1CaloHcalScaleRcd& iRecord) override;

  std::unique_ptr<L1CaloHcalScale> newObject(const std::string& objectKey) override;

private:
  edm::ESGetToken<HcalTrigTowerGeometry, CaloGeometryRecord> theTrigTowerGeometryToken;
  const HcalTrigTowerGeometry* theTrigTowerGeometry;
  CaloTPGTranscoderULUT* caloTPG;
  typedef std::vector<double> RCTdecompression;
  std::vector<RCTdecompression> hcaluncomp;

  //  HcaluLUTTPGCoder* tpgCoder;// = new	 HcaluLUTTPGCoder();

  HcalTrigTowerDetId* ttDetId;

  // ----------member data ---------------------------
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
L1CaloHcalScaleConfigOnlineProd::L1CaloHcalScaleConfigOnlineProd(const edm::ParameterSet& iConfig)
    : L1ConfigOnlineProdBase<L1CaloHcalScaleRcd, L1CaloHcalScale>(iConfig), theTrigTowerGeometry(nullptr) {
  theTrigTowerGeometryToken = m_consumesCollector->consumes();
  caloTPG = new CaloTPGTranscoderULUT();
}

L1CaloHcalScaleConfigOnlineProd::~L1CaloHcalScaleConfigOnlineProd() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

  if (caloTPG != nullptr)
    delete caloTPG;
}

std::unique_ptr<L1CaloHcalScale> L1CaloHcalScaleConfigOnlineProd::newObject(const std::string& objectKey) {
  assert(theTrigTowerGeometry != nullptr);

  edm::LogInfo("L1CaloHcalScaleConfigOnlineProd") << "object Key " << objectKey;

  if (objectKey == "NULL" || objectKey.empty()) {  // return default blank ecal scale
    return std::make_unique<L1CaloHcalScale>(0);
  }
  if (objectKey == "IDENTITY") {  // return identity ecal scale
    return std::make_unique<L1CaloHcalScale>(1);
  }

  std::vector<unsigned int> analyticalLUT(1024, 0);
  std::vector<unsigned int> identityLUT(1024, 0);

  // Compute compression LUT
  for (unsigned int i = 0; i < 1024; i++) {
    analyticalLUT[i] = (unsigned int)(sqrt(14.94 * log(1. + i / 14.94) * i) + 0.5);
    identityLUT[i] = std::min(i, 0xffu);
  }

  hcaluncomp.clear();
  for (int i = 0; i < 4176; i++) {
    RCTdecompression decompressionTable(256, 0);
    hcaluncomp.push_back(decompressionTable);
  }

  std::vector<std::string> mainStrings;
  mainStrings.push_back("HCAL_LUT_METADATA");
  mainStrings.push_back("HCAL_LUT_CHAN_DATA");

  // ~~~~~~~~~ Cut values ~~~~~~~~~

  std::vector<std::string> metaStrings;
  metaStrings.push_back("RCTLSB");
  metaStrings.push_back("NOMINAL_GAIN");

  l1t::OMDSReader::QueryResults paramResults =
      m_omdsReader.basicQueryView(metaStrings,
                                  "CMS_HCL_HCAL_COND",
                                  "V_HCAL_LUT_METADATA_V1",
                                  "V_HCAL_LUT_METADATA_V1.TAG_NAME",
                                  m_omdsReader.basicQuery("HCAL_LUT_METADATA",
                                                          "CMS_RCT",
                                                          "HCAL_SCALE_KEY",
                                                          "HCAL_SCALE_KEY.HCAL_TAG",
                                                          m_omdsReader.singleAttribute(objectKey)));

  if (paramResults.queryFailed() || (paramResults.numberRows() != 1))  // check query successful
  {
    edm::LogError("L1-O2O") << "Problem with L1CaloHcalScale key.  Unable to find lutparam dat table";
    return std::unique_ptr<L1CaloHcalScale>();
  }

  double hcalLSB, nominal_gain;
  paramResults.fillVariable("RCTLSB", hcalLSB);
  paramResults.fillVariable("NOMINAL_GAIN", nominal_gain);

  float rctlsb = hcalLSB;

  l1t::OMDSReader::QueryResults chanKey = m_omdsReader.basicQuery("HCAL_LUT_CHAN_DATA",
                                                                  "CMS_RCT",
                                                                  "HCAL_SCALE_KEY",
                                                                  "HCAL_SCALE_KEY.HCAL_TAG",
                                                                  m_omdsReader.singleAttribute(objectKey));

  //coral::AttributeList myresult;
  //    myresult.extend(

  std::string schemaName("CMS_HCL_HCAL_COND");
  coral::ISchema& schema = m_omdsReader.dbSession().coralSession().schema(schemaName);
  coral::IQuery* query = schema.newQuery();
  ;

  std::vector<std::string> channelStrings;
  channelStrings.push_back("IPHI");
  channelStrings.push_back("IETA");
  channelStrings.push_back("DEPTH");
  channelStrings.push_back("LUT_GRANULARITY");
  channelStrings.push_back("OUTPUT_LUT_THRESHOLD");
  channelStrings.push_back("OBJECTNAME");

  std::vector<std::string>::const_iterator it = channelStrings.begin();
  std::vector<std::string>::const_iterator end = channelStrings.end();
  for (; it != end; ++it) {
    query->addToOutputList(*it);
  }

  std::string ob = "OBJECTNAME";
  coral::AttributeList myresult;
  myresult.extend("IPHI", typeid(int));
  myresult.extend("IETA", typeid(int));
  myresult.extend("DEPTH", typeid(int));
  myresult.extend("LUT_GRANULARITY", typeid(int));
  myresult.extend("OUTPUT_LUT_THRESHOLD", typeid(int));
  myresult.extend(ob, typeid(std::string));  //, typeid(std::string));

  query->defineOutput(myresult);

  query->addToTableList("V_HCAL_LUT_CHAN_DATA_V1");

  query->setCondition("V_HCAL_LUT_CHAN_DATA_V1.TAG_NAME = :" + chanKey.columnNames().front(),
                      chanKey.attributeLists().front());

  coral::ICursor& cursor = query->execute();

  // when the query goes out of scope.
  std::vector<coral::AttributeList> atts;
  while (cursor.next()) {
    atts.push_back(cursor.currentRow());
  };

  delete query;

  l1t::OMDSReader::QueryResults chanResults(channelStrings, atts);
  if (chanResults.queryFailed() || (chanResults.numberRows() == 0))  // check query successful
  {
    edm::LogError("L1-O2O") << "Problem with L1CaloHcalScale key.  Unable to find lutparam dat table nrows"
                            << chanResults.numberRows();
    return std::unique_ptr<L1CaloHcalScale>();
  }

  chanResults.attributeLists();
  for (int i = 0; i < chanResults.numberRows(); ++i) {
    std::string objectName;
    chanResults.fillVariableFromRow("OBJECTNAME", i, objectName);
    //       int
    if (objectName == "HcalTrigTowerDetId") {  //trig tower
      int ieta, iphi, depth, lutGranularity, threshold;

      chanResults.fillVariableFromRow("LUT_GRANULARITY", i, lutGranularity);
      chanResults.fillVariableFromRow("IPHI", i, iphi);
      chanResults.fillVariableFromRow("IETA", i, ieta);
      chanResults.fillVariableFromRow("DEPTH", i, depth);
      chanResults.fillVariableFromRow("OUTPUT_LUT_THRESHOLD", i, threshold);

      unsigned int outputLut[1024];

      const int tp_version = depth / 10;
      uint32_t lutId = caloTPG->getOutputLUTId(ieta, iphi, tp_version);

      double eta_low = 0., eta_high = 0.;
      theTrigTowerGeometry->towerEtaBounds(ieta, tp_version, eta_low, eta_high);
      double cosh_ieta = fabs(cosh((eta_low + eta_high) / 2.));

      if (!caloTPG->HTvalid(ieta, iphi, tp_version))
        continue;
      double factor = 0.;
      if (abs(ieta) >= theTrigTowerGeometry->firstHFTower(tp_version))
        factor = rctlsb;
      else
        factor = nominal_gain / cosh_ieta * lutGranularity;
      for (int k = 0; k < threshold; ++k)
        outputLut[k] = 0;

      for (unsigned int k = threshold; k < 1024; ++k)
        outputLut[k] = (abs(ieta) < theTrigTowerGeometry->firstHFTower(tp_version)) ? analyticalLUT[k] : identityLUT[k];

      // tpg - compressed value
      unsigned int tpg = outputLut[0];

      int low = 0;

      for (unsigned int k = 0; k < 1024; ++k) {
        if (outputLut[k] != tpg) {
          unsigned int mid = (low + k) / 2;
          hcaluncomp[lutId][tpg] = (tpg == 0 ? low : factor * mid);
          low = k;
          tpg = outputLut[k];
        }
      }
      hcaluncomp[lutId][tpg] = factor * low;
    }
  }

  auto hcalScale = std::make_unique<L1CaloHcalScale>(0);

  // XXX L1CaloHcalScale is only setup for 2x3 TP
  const int tp_version = 0;
  for (unsigned short ieta = 1; ieta <= L1CaloHcalScale::nBinEta; ++ieta) {
    for (int pos = 0; pos <= 1; pos++) {
      for (unsigned short irank = 0; irank < L1CaloHcalScale::nBinRank; ++irank) {
        int zside = (int)pow(-1, pos);
        int nphi = 0;
        double etvalue = 0.;

        for (int iphi = 1; iphi <= 72; iphi++) {
          if (!caloTPG->HTvalid(ieta, iphi, tp_version))
            continue;
          uint32_t lutId = caloTPG->getOutputLUTId(ieta, iphi, tp_version);
          nphi++;
          etvalue += (double)hcaluncomp[lutId][irank];

        }  // phi
        if (nphi > 0)
          etvalue /= nphi;

        hcalScale->setBin(irank, ieta, zside, etvalue);

      }  // rank
    }    // zside
  }      // eta

  std::stringstream s;
  s << std::setprecision(10);
  hcalScale->print(s);
  edm::LogInfo("L1CaloHcalScaleConfigOnlineProd") << s.str();
  // ------------ method called to produce the data  ------------
  return hcalScale;
}

std::unique_ptr<L1CaloHcalScale> L1CaloHcalScaleConfigOnlineProd::produce(const L1CaloHcalScaleRcd& iRecord) {
  theTrigTowerGeometry = &iRecord.get(theTrigTowerGeometryToken);

  return (L1ConfigOnlineProdBase<L1CaloHcalScaleRcd, L1CaloHcalScale>::produce(iRecord));
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1CaloHcalScaleConfigOnlineProd);
