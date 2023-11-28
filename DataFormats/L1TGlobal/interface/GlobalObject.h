#ifndef DataFormats_L1TGlobal_GlobalObject_h
#define DataFormats_L1TGlobal_GlobalObject_h

#include <string>
#include <utility>
#include <vector>

namespace l1t {

  /* Enum of L1T GlobalObjects

     IMPORTANT
       The elements of the enum l1t::GlobalObject are used in the data format GlobalObjectMapRecord.
       One instance of GlobalObjectMapRecord is produced online by the HLT,
       and it is written to disk as part of the RAW data tier.
       Said instance of GlobalObjectMapRecord is produced at HLT by the plugin L1TGlobalProducer,
       which implements the emulator of the Stage-2 Level-1 Global Trigger.

       In order not to change the meaning of existing data when adding entries to the enum l1t::GlobalObject,
       it is necessary to add such new entries with an explicit integer value which has not been used before in the enum.

       When adding new elements to the enum l1t::GlobalObject, make sure to also update accordingly
       (a) the vector l1t::kGlobalObjectEnumStringPairs in this file, and
       (b) the unit test implemented in test/test_catch2_l1tGlobalObject.cc in this package.

       For further information on the subject, please see
        https://github.com/cms-sw/cmssw/pull/42634#discussion_r1302636113
        https://github.com/cms-sw/cmssw/issues/42719
  */
  enum GlobalObject {
    gtMu = 0,
    gtMuShower = 1,
    gtEG = 2,
    gtJet = 3,
    gtTau = 4,
    gtETM = 5,
    gtETT = 6,
    gtHTT = 7,
    gtHTM = 8,
    gtETMHF = 9,
    gtTowerCount = 10,
    gtMinBiasHFP0 = 11,
    gtMinBiasHFM0 = 12,
    gtMinBiasHFP1 = 13,
    gtMinBiasHFM1 = 14,
    gtETTem = 15,
    gtAsymmetryEt = 16,
    gtAsymmetryHt = 17,
    gtAsymmetryEtHF = 18,
    gtAsymmetryHtHF = 19,
    gtCentrality0 = 20,
    gtCentrality1 = 21,
    gtCentrality2 = 22,
    gtCentrality3 = 23,
    gtCentrality4 = 24,
    gtCentrality5 = 25,
    gtCentrality6 = 26,
    gtCentrality7 = 27,
    gtExternal = 28,
    gtZDCP = 29,
    gtZDCM = 30,
    ObjNull = 31,
  };

  const std::vector<std::pair<GlobalObject, std::string>> kGlobalObjectEnumStringPairs = {
      {gtMu, "Mu"},                    //  0
      {gtMuShower, "MuShower"},        //  1
      {gtEG, "EG"},                    //  2
      {gtJet, "Jet"},                  //  3
      {gtTau, "Tau"},                  //  4
      {gtETM, "ETM"},                  //  5
      {gtETT, "ETT"},                  //  6
      {gtHTT, "HTT"},                  //  7
      {gtHTM, "HTM"},                  //  8
      {gtETMHF, "ETMHF"},              //  9
      {gtTowerCount, "TowerCount"},    // 10
      {gtMinBiasHFP0, "MinBiasHFP0"},  // 11
      {gtMinBiasHFM0, "MinBiasHFM0"},  // 12
      {gtMinBiasHFP1, "MinBiasHFP1"},  // 13
      {gtMinBiasHFM1, "MinBiasHFM1"},  // 14
      {gtETTem, "ETTem"},              // 15
      {gtAsymmetryEt, "AsymEt"},       // 16
      {gtAsymmetryHt, "AsymHt"},       // 17
      {gtAsymmetryEtHF, "AsymEtHF"},   // 18
      {gtAsymmetryHtHF, "AsymHtHF"},   // 19
      {gtCentrality0, "CENT0"},        // 20
      {gtCentrality1, "CENT1"},        // 21
      {gtCentrality2, "CENT2"},        // 22
      {gtCentrality3, "CENT3"},        // 23
      {gtCentrality4, "CENT4"},        // 24
      {gtCentrality5, "CENT5"},        // 25
      {gtCentrality6, "CENT6"},        // 26
      {gtCentrality7, "CENT7"},        // 27
      {gtExternal, "External"},        // 28
      {gtZDCP, "ZDCP"},                // 29
      {gtZDCM, "ZDCM"},                // 30
      {ObjNull, "ObjNull"},            // 31
  };

  // utility functions to convert GlobalObject enum to std::string and viceversa
  l1t::GlobalObject GlobalObjectStringToEnum(const std::string&);
  std::string GlobalObjectEnumToString(const GlobalObject&);

}  // namespace l1t

#endif
