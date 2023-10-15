#ifndef DataFormats_L1TGlobal_GlobalObject_h
#define DataFormats_L1TGlobal_GlobalObject_h

#include <string>

namespace l1t {

  /* Enum of L1T GlobalObjects

     IMPORTANT
       The values of the l1t::GlobalObject enum used in the data format GlobalObjectMapRecord,
       which is produced by the L1TGlobalProducer plugin (emulator of the Stage-2 Level-1 Global Trigger).
       One instance of this product is created online by the HLT, and is written to disk as part of the RAW data tier.
       In order not to change the meaning of existing data when adding entries to the enum below,
       it is necessary to add such new entries with an explicit integer value which has not been used before in the enum.
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
    ObjNull = 31
  };

  // utility functions to convert GlobalObject enum to std::string and viceversa
  l1t::GlobalObject l1TGtObjectStringToEnum(const std::string&);
  std::string l1TGtObjectEnumToString(const GlobalObject&);

  struct L1TGtObjectStringToEnum {
    const char* label;
    GlobalObject value;
  };

}  // namespace l1t

#endif
