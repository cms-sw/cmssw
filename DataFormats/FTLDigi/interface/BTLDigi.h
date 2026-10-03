#ifndef DataFormats_FTLDigi_interface_BTLDigi_h
#define DataFormats_FTLDigi_interface_BTLDigi_h

#include <cstdint>
#include <ostream>

namespace btldigi {

  class BTLDigi {
  public:
    /**
     @short key to sort the collection
  */
    typedef uint32_t key_type;

    BTLDigi()
        : krawId_(0),
          kBC0count_(0),
          kstatus_(false),
          kBCcount_(0),
          kchIDPlus_(0),
          kT1coarsePlus_(0),
          kT2coarsePlus_(0),
          kEOIcoarsePlus_(0),
          kChargePlus_(0),
          kT1finePlus_(0),
          kT2finePlus_(0),
          kIdleTimePlus_(0),
          kPrevTrigFPlus_(0),
          kTACIDPlus_(0),
          kchIDMinus_(0),
          kT1coarseMinus_(0),
          kT2coarseMinus_(0),
          kEOIcoarseMinus_(0),
          kChargeMinus_(0),
          kT1fineMinus_(0),
          kT2fineMinus_(0),
          kIdleTimeMinus_(0),
          kPrevTrigFMinus_(0),
          kTACIDMinus_(0) {}

    BTLDigi(uint32_t rawId,
            uint16_t BC0count,
            bool status,
            uint32_t BCcount,
            uint8_t chIDPlus,
            uint16_t T1coarsePlus,
            uint16_t T2coarsePlus,
            uint16_t EOIcoarsePlus,
            uint16_t ChargePlus,
            uint16_t T1finePlus,
            uint16_t T2finePlus,
            uint16_t IdleTimePlus,
            uint8_t PrevTrigFPlus,
            uint8_t TACIDPlus,
            uint8_t chIDMinus,
            uint16_t T1coarseMinus,
            uint16_t T2coarseMinus,
            uint16_t EOIcoarseMinus,
            uint16_t ChargeMinus,
            uint16_t T1fineMinus,
            uint16_t T2fineMinus,
            uint16_t IdleTimeMinus,
            uint8_t PrevTrigFMinus,
            uint8_t TACIDMinus)
        : krawId_(rawId),
          kBC0count_(BC0count),
          kstatus_(status),
          kBCcount_(BCcount),
          kchIDPlus_(chIDPlus),
          kT1coarsePlus_(T1coarsePlus),
          kT2coarsePlus_(T2coarsePlus),
          kEOIcoarsePlus_(EOIcoarsePlus),
          kChargePlus_(ChargePlus),
          kT1finePlus_(T1finePlus),
          kT2finePlus_(T2finePlus),
          kIdleTimePlus_(IdleTimePlus),
          kPrevTrigFPlus_(PrevTrigFPlus),
          kTACIDPlus_(TACIDPlus),
          kchIDMinus_(chIDMinus),
          kT1coarseMinus_(T1coarseMinus),
          kT2coarseMinus_(T2coarseMinus),
          kEOIcoarseMinus_(EOIcoarseMinus),
          kChargeMinus_(ChargeMinus),
          kT1fineMinus_(T1fineMinus),
          kT2fineMinus_(T2fineMinus),
          kIdleTimeMinus_(IdleTimeMinus),
          kPrevTrigFMinus_(PrevTrigFMinus),
          kTACIDMinus_(TACIDMinus) {}

    uint32_t krawId() const { return krawId_; }
    uint16_t kBC0count() const { return kBC0count_; }
    bool kstatus() const { return kstatus_; }
    uint32_t kBCcount() const { return kBCcount_; }
    uint8_t kchIDPlus() const { return kchIDPlus_; }
    uint16_t kT1coarsePlus() const { return kT1coarsePlus_; }
    uint16_t kT2coarsePlus() const { return kT2coarsePlus_; }
    uint16_t kEOIcoarsePlus() const { return kEOIcoarsePlus_; }
    uint16_t kChargePlus() const { return kChargePlus_; }
    uint16_t kT1finePlus() const { return kT1finePlus_; }
    uint16_t kT2finePlus() const { return kT2finePlus_; }
    uint16_t kIdleTimePlus() const { return kIdleTimePlus_; }
    uint8_t kPrevTrigFPlus() const { return kPrevTrigFPlus_; }
    uint8_t kTACIDPlus() const { return kTACIDPlus_; }
    uint8_t kchIDMinus() const { return kchIDMinus_; }
    uint16_t kT1coarseMinus() const { return kT1coarseMinus_; }
    uint16_t kT2coarseMinus() const { return kT2coarseMinus_; }
    uint16_t kEOIcoarseMinus() const { return kEOIcoarseMinus_; }
    uint16_t kChargeMinus() const { return kChargeMinus_; }
    uint16_t kT1fineMinus() const { return kT1fineMinus_; }
    uint16_t kT2fineMinus() const { return kT2fineMinus_; }
    uint16_t kIdleTimeMinus() const { return kIdleTimeMinus_; }
    uint8_t kPrevTrigFMinus() const { return kPrevTrigFMinus_; }
    uint8_t kTACIDMinus() const { return kTACIDMinus_; }

    // needed for sorting in SortedCollection
    uint32_t id() const { return krawId_; }

  private:
    uint32_t krawId_;     // Raw ID of the module/TOFHIR
    uint16_t kBC0count_;  // BC0 count (reserved)
    bool kstatus_;        // status of the TOFHIR
    uint32_t kBCcount_;
    uint8_t kchIDPlus_;       // TOFHIR channel ID, plus side of crystal
    uint16_t kT1coarsePlus_;  // data from crystal plus side
    uint16_t kT2coarsePlus_;
    uint16_t kEOIcoarsePlus_;
    uint16_t kChargePlus_;
    uint16_t kT1finePlus_;
    uint16_t kT2finePlus_;
    uint16_t kIdleTimePlus_;
    uint8_t kPrevTrigFPlus_;
    uint8_t kTACIDPlus_;
    uint8_t kchIDMinus_;       // TOFHIR channel ID, minus side of crystal
    uint16_t kT1coarseMinus_;  // data from crystal minus side
    uint16_t kT2coarseMinus_;
    uint16_t kEOIcoarseMinus_;
    uint16_t kChargeMinus_;
    uint16_t kT1fineMinus_;
    uint16_t kT2fineMinus_;
    uint16_t kIdleTimeMinus_;
    uint8_t kPrevTrigFMinus_;
    uint8_t kTACIDMinus_;
  };

  std::ostream& operator<<(std::ostream&, const BTLDigi&);

}  // namespace btldigi
#endif  // DataFormats_FTLDigi_interface_BTLDigi_h
