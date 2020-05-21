// -*- C++ -*-
//
// Package:    L1HtMissScaleOnlineProd
// Class:      L1HtMissScaleOnlineProd
//
/**\class L1HtMissScaleOnlineProd L1HtMissScaleOnlineProd.h L1TriggerConfig/L1ScalesProducers/src/L1HtMissScaleOnlineProd.cc

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

#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"
#include "CondFormats/DataRecord/interface/L1HtMissScaleRcd.h"

//
// class declaration
//

class L1HtMissScaleOnlineProd : public L1ConfigOnlineProdBase<L1HtMissScaleRcd, L1CaloEtScale> {
public:
  L1HtMissScaleOnlineProd(const edm::ParameterSet&);
  ~L1HtMissScaleOnlineProd() override;

  std::unique_ptr<L1CaloEtScale> newObject(const std::string& objectKey) override;

private:
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
L1HtMissScaleOnlineProd::L1HtMissScaleOnlineProd(const edm::ParameterSet& iConfig)
    : L1ConfigOnlineProdBase<L1HtMissScaleRcd, L1CaloEtScale>(iConfig) {
  //the following line is needed to tell the framework what
  // data is being produced

  //now do what ever other initialization is needed
}

L1HtMissScaleOnlineProd::~L1HtMissScaleOnlineProd() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

std::unique_ptr<L1CaloEtScale> L1HtMissScaleOnlineProd::newObject(const std::string& objectKey) {
  // get scales keys
  l1t::OMDSReader::QueryResults scalesKeyResults = m_omdsReader.basicQuery("GCT_SCALES_KEY",
                                                                           "CMS_GCT",
                                                                           "GCT_PHYS_PARAMS",
                                                                           "GCT_PHYS_PARAMS.CONFIG_KEY",
                                                                           m_omdsReader.singleAttribute(objectKey));

  std::string scalesKey;

  if (scalesKeyResults.queryFailed()) {
    edm::LogError("L1-O2O") << "Problem with key for L1HtMissScaleRcd : GCT scales key query failed ";
  } else if (scalesKeyResults.numberRows() != 1) {
    edm::LogError("L1-O2O") << "Problem with key for L1HtMissScaleRcd : " << (scalesKeyResults.numberRows())
                            << " rows were returned when getting GCT scales key";
  } else {
    scalesKeyResults.fillVariable(scalesKey);
  }

  edm::LogInfo("L1-O2O") << "L1 scales key : " << scalesKey << std::endl;

  // get jet scale key
  l1t::OMDSReader::QueryResults htmScaleKeyResults =
      m_omdsReader.basicQuery("SC_HTM_FK", "CMS_GT", "L1T_SCALES", "L1T_SCALES.ID", scalesKeyResults);

  std::string htmScaleKey;

  if (htmScaleKeyResults.queryFailed()) {
    edm::LogError("L1-O2O") << "Problem with key for L1HtMissEtScaleRcd : HtMiss scale key query failed ";
  } else if (htmScaleKeyResults.numberRows() != 1) {
    edm::LogError("L1-O2O") << "Problem with key for L1HtMissScaleRcd : " << (htmScaleKeyResults.numberRows())
                            << " rows were returned when getting HtMiss scale key";
  } else {
    htmScaleKeyResults.fillVariable(htmScaleKey);
  }

  edm::LogInfo("L1-O2O") << "L1HtMiss scale key : " << htmScaleKey << std::endl;

  // get thresholds
  std::vector<std::string> queryStrings;
  queryStrings.emplace_back("E_GEV_BIN_LOW_0");
  queryStrings.emplace_back("E_GEV_BIN_LOW_1");
  queryStrings.emplace_back("E_GEV_BIN_LOW_2");
  queryStrings.emplace_back("E_GEV_BIN_LOW_3");
  queryStrings.emplace_back("E_GEV_BIN_LOW_4");
  queryStrings.emplace_back("E_GEV_BIN_LOW_5");
  queryStrings.emplace_back("E_GEV_BIN_LOW_6");
  queryStrings.emplace_back("E_GEV_BIN_LOW_7");
  queryStrings.emplace_back("E_GEV_BIN_LOW_8");
  queryStrings.emplace_back("E_GEV_BIN_LOW_9");
  queryStrings.emplace_back("E_GEV_BIN_LOW_10");
  queryStrings.emplace_back("E_GEV_BIN_LOW_11");
  queryStrings.emplace_back("E_GEV_BIN_LOW_12");
  queryStrings.emplace_back("E_GEV_BIN_LOW_13");
  queryStrings.emplace_back("E_GEV_BIN_LOW_14");
  queryStrings.emplace_back("E_GEV_BIN_LOW_15");
  queryStrings.emplace_back("E_GEV_BIN_LOW_16");
  queryStrings.emplace_back("E_GEV_BIN_LOW_17");
  queryStrings.emplace_back("E_GEV_BIN_LOW_18");
  queryStrings.emplace_back("E_GEV_BIN_LOW_19");
  queryStrings.emplace_back("E_GEV_BIN_LOW_20");
  queryStrings.emplace_back("E_GEV_BIN_LOW_21");
  queryStrings.emplace_back("E_GEV_BIN_LOW_22");
  queryStrings.emplace_back("E_GEV_BIN_LOW_23");
  queryStrings.emplace_back("E_GEV_BIN_LOW_24");
  queryStrings.emplace_back("E_GEV_BIN_LOW_25");
  queryStrings.emplace_back("E_GEV_BIN_LOW_26");
  queryStrings.emplace_back("E_GEV_BIN_LOW_27");
  queryStrings.emplace_back("E_GEV_BIN_LOW_28");
  queryStrings.emplace_back("E_GEV_BIN_LOW_29");
  queryStrings.emplace_back("E_GEV_BIN_LOW_30");
  queryStrings.emplace_back("E_GEV_BIN_LOW_31");
  queryStrings.emplace_back("E_GEV_BIN_LOW_32");
  queryStrings.emplace_back("E_GEV_BIN_LOW_33");
  queryStrings.emplace_back("E_GEV_BIN_LOW_34");
  queryStrings.emplace_back("E_GEV_BIN_LOW_35");
  queryStrings.emplace_back("E_GEV_BIN_LOW_36");
  queryStrings.emplace_back("E_GEV_BIN_LOW_37");
  queryStrings.emplace_back("E_GEV_BIN_LOW_38");
  queryStrings.emplace_back("E_GEV_BIN_LOW_39");
  queryStrings.emplace_back("E_GEV_BIN_LOW_40");
  queryStrings.emplace_back("E_GEV_BIN_LOW_41");
  queryStrings.emplace_back("E_GEV_BIN_LOW_42");
  queryStrings.emplace_back("E_GEV_BIN_LOW_43");
  queryStrings.emplace_back("E_GEV_BIN_LOW_44");
  queryStrings.emplace_back("E_GEV_BIN_LOW_45");
  queryStrings.emplace_back("E_GEV_BIN_LOW_46");
  queryStrings.emplace_back("E_GEV_BIN_LOW_47");
  queryStrings.emplace_back("E_GEV_BIN_LOW_48");
  queryStrings.emplace_back("E_GEV_BIN_LOW_49");
  queryStrings.emplace_back("E_GEV_BIN_LOW_50");
  queryStrings.emplace_back("E_GEV_BIN_LOW_51");
  queryStrings.emplace_back("E_GEV_BIN_LOW_52");
  queryStrings.emplace_back("E_GEV_BIN_LOW_53");
  queryStrings.emplace_back("E_GEV_BIN_LOW_54");
  queryStrings.emplace_back("E_GEV_BIN_LOW_55");
  queryStrings.emplace_back("E_GEV_BIN_LOW_56");
  queryStrings.emplace_back("E_GEV_BIN_LOW_57");
  queryStrings.emplace_back("E_GEV_BIN_LOW_58");
  queryStrings.emplace_back("E_GEV_BIN_LOW_59");
  queryStrings.emplace_back("E_GEV_BIN_LOW_60");
  queryStrings.emplace_back("E_GEV_BIN_LOW_61");
  queryStrings.emplace_back("E_GEV_BIN_LOW_62");
  queryStrings.emplace_back("E_GEV_BIN_LOW_63");
  queryStrings.emplace_back("E_GEV_BIN_LOW_64");
  queryStrings.emplace_back("E_GEV_BIN_LOW_65");
  queryStrings.emplace_back("E_GEV_BIN_LOW_66");
  queryStrings.emplace_back("E_GEV_BIN_LOW_67");
  queryStrings.emplace_back("E_GEV_BIN_LOW_68");
  queryStrings.emplace_back("E_GEV_BIN_LOW_69");
  queryStrings.emplace_back("E_GEV_BIN_LOW_70");
  queryStrings.emplace_back("E_GEV_BIN_LOW_71");
  queryStrings.emplace_back("E_GEV_BIN_LOW_72");
  queryStrings.emplace_back("E_GEV_BIN_LOW_73");
  queryStrings.emplace_back("E_GEV_BIN_LOW_74");
  queryStrings.emplace_back("E_GEV_BIN_LOW_75");
  queryStrings.emplace_back("E_GEV_BIN_LOW_76");
  queryStrings.emplace_back("E_GEV_BIN_LOW_77");
  queryStrings.emplace_back("E_GEV_BIN_LOW_78");
  queryStrings.emplace_back("E_GEV_BIN_LOW_79");
  queryStrings.emplace_back("E_GEV_BIN_LOW_80");
  queryStrings.emplace_back("E_GEV_BIN_LOW_81");
  queryStrings.emplace_back("E_GEV_BIN_LOW_82");
  queryStrings.emplace_back("E_GEV_BIN_LOW_83");
  queryStrings.emplace_back("E_GEV_BIN_LOW_84");
  queryStrings.emplace_back("E_GEV_BIN_LOW_85");
  queryStrings.emplace_back("E_GEV_BIN_LOW_86");
  queryStrings.emplace_back("E_GEV_BIN_LOW_87");
  queryStrings.emplace_back("E_GEV_BIN_LOW_88");
  queryStrings.emplace_back("E_GEV_BIN_LOW_89");
  queryStrings.emplace_back("E_GEV_BIN_LOW_90");
  queryStrings.emplace_back("E_GEV_BIN_LOW_91");
  queryStrings.emplace_back("E_GEV_BIN_LOW_92");
  queryStrings.emplace_back("E_GEV_BIN_LOW_93");
  queryStrings.emplace_back("E_GEV_BIN_LOW_94");
  queryStrings.emplace_back("E_GEV_BIN_LOW_95");
  queryStrings.emplace_back("E_GEV_BIN_LOW_96");
  queryStrings.emplace_back("E_GEV_BIN_LOW_97");
  queryStrings.emplace_back("E_GEV_BIN_LOW_98");
  queryStrings.emplace_back("E_GEV_BIN_LOW_99");
  queryStrings.emplace_back("E_GEV_BIN_LOW_100");
  queryStrings.emplace_back("E_GEV_BIN_LOW_101");
  queryStrings.emplace_back("E_GEV_BIN_LOW_102");
  queryStrings.emplace_back("E_GEV_BIN_LOW_103");
  queryStrings.emplace_back("E_GEV_BIN_LOW_104");
  queryStrings.emplace_back("E_GEV_BIN_LOW_105");
  queryStrings.emplace_back("E_GEV_BIN_LOW_106");
  queryStrings.emplace_back("E_GEV_BIN_LOW_107");
  queryStrings.emplace_back("E_GEV_BIN_LOW_108");
  queryStrings.emplace_back("E_GEV_BIN_LOW_109");
  queryStrings.emplace_back("E_GEV_BIN_LOW_110");
  queryStrings.emplace_back("E_GEV_BIN_LOW_111");
  queryStrings.emplace_back("E_GEV_BIN_LOW_112");
  queryStrings.emplace_back("E_GEV_BIN_LOW_113");
  queryStrings.emplace_back("E_GEV_BIN_LOW_114");
  queryStrings.emplace_back("E_GEV_BIN_LOW_115");
  queryStrings.emplace_back("E_GEV_BIN_LOW_116");
  queryStrings.emplace_back("E_GEV_BIN_LOW_117");
  queryStrings.emplace_back("E_GEV_BIN_LOW_118");
  queryStrings.emplace_back("E_GEV_BIN_LOW_119");
  queryStrings.emplace_back("E_GEV_BIN_LOW_120");
  queryStrings.emplace_back("E_GEV_BIN_LOW_121");
  queryStrings.emplace_back("E_GEV_BIN_LOW_122");
  queryStrings.emplace_back("E_GEV_BIN_LOW_123");
  queryStrings.emplace_back("E_GEV_BIN_LOW_124");
  queryStrings.emplace_back("E_GEV_BIN_LOW_125");
  queryStrings.emplace_back("E_GEV_BIN_LOW_126");
  queryStrings.emplace_back("E_GEV_BIN_LOW_127");

  l1t::OMDSReader::QueryResults scaleResults = m_omdsReader.basicQuery(
      queryStrings, "CMS_GT", "L1T_SCALE_HTM_ENERGY", "L1T_SCALE_HTM_ENERGY.ID", htmScaleKeyResults);

  // L1T_SCALE_HTM_ENERGY

  std::vector<double> thresholds;

  if (scaleResults.queryFailed() || scaleResults.numberRows() != 1)  // check query successful
  {
    edm::LogError("L1-O2O") << "Problem with L1HtMissScale key : when reading scale.";
  } else {
    for (std::vector<std::string>::iterator thresh = queryStrings.begin(); thresh != queryStrings.end(); ++thresh) {
      float tempScale = 0.;
      scaleResults.fillVariable(*thresh, tempScale);
      thresholds.push_back(tempScale);
    }
  }

  // get region LSB
  double rgnEtLsb = 0.;

  l1t::OMDSReader::QueryResults lsbResults = m_omdsReader.basicQuery("GCT_RGN_ET_LSB",
                                                                     "CMS_GCT",
                                                                     "GCT_PHYS_PARAMS",
                                                                     "GCT_PHYS_PARAMS.CONFIG_KEY",
                                                                     m_omdsReader.singleAttribute(objectKey));

  if (lsbResults.queryFailed()) {
    edm::LogError("L1-O2O") << "Problem with L1HtMissScale key.";
  } else {
    lsbResults.fillVariable("GCT_RGN_ET_LSB", rgnEtLsb);
  }

  // return object
  return std::make_unique<L1CaloEtScale>(0, 0x7f, rgnEtLsb, thresholds);
}

// ------------ method called to produce the data  ------------

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1HtMissScaleOnlineProd);
