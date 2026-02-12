#ifndef DataFormats_FTLDigiSoA_interface_BTLDigiSoA_h
#define DataFormats_FTLDigiSoA_interface_BTLDigiSoA_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace btldigi {

  GENERATE_SOA_LAYOUT(BTLDigiSoALayout,
                      SOA_COLUMN(uint32_t, rawId),     // Raw ID of the module/TOFHIR
                      SOA_COLUMN(uint16_t, BC0count),  // BC0 count (reserved)
                      SOA_COLUMN(bool, status),        // status of the TOFHIR
                      SOA_COLUMN(uint32_t, BCcount),
                      SOA_COLUMN(uint8_t, chIDR),       // TOFHIR channel ID, right side of crystal
                      SOA_COLUMN(uint16_t, T1coarseR),  // data from crystal right side
                      SOA_COLUMN(uint16_t, T2coarseR),
                      SOA_COLUMN(uint16_t, EOIcoarseR),
                      SOA_COLUMN(uint16_t, ChargeR),
                      SOA_COLUMN(uint16_t, T1fineR),
                      SOA_COLUMN(uint16_t, T2fineR),
                      SOA_COLUMN(uint16_t, IdleTimeR),
                      SOA_COLUMN(uint8_t, PrevTrigFR),
                      SOA_COLUMN(uint8_t, TACIDR),
                      SOA_COLUMN(uint8_t, chIDL),       // TOFHIR channel ID, left side of crystal
                      SOA_COLUMN(uint16_t, T1coarseL),  // data from crystal left side
                      SOA_COLUMN(uint16_t, T2coarseL),
                      SOA_COLUMN(uint16_t, EOIcoarseL),
                      SOA_COLUMN(uint16_t, ChargeL),
                      SOA_COLUMN(uint16_t, T1fineL),
                      SOA_COLUMN(uint16_t, T2fineL),
                      SOA_COLUMN(uint16_t, IdleTimeL),
                      SOA_COLUMN(uint8_t, PrevTrigFL),
                      SOA_COLUMN(uint8_t, TACIDL))

  using BTLDigiSoA = BTLDigiSoALayout<>;
  using BTLDigiSoAView = BTLDigiSoA::View;
  using BTLDigiSoAConstView = BTLDigiSoA::ConstView;

  std::ostream &operator<<(std::ostream &out, BTLDigiSoA::View::const_element const &digi);

  // Getters
  ALPAKA_FN_HOST_ACC inline uint32_t rawId(const BTLDigiSoAConstView &btlDigi, int32_t i) {
    return (btlDigi[i].rawId());
  }
  ALPAKA_FN_HOST_ACC inline uint16_t BC0count(const BTLDigiSoAConstView &btlDigi, int32_t i) {
    return (btlDigi[i].BC0count());
  }
  ALPAKA_FN_HOST_ACC inline bool status(const BTLDigiSoAConstView &btlDigi, int32_t i) { return (btlDigi[i].status()); }
  ALPAKA_FN_HOST_ACC inline uint32_t BCcount(const BTLDigiSoAConstView &btlDigi, int32_t i) {
    return (btlDigi[i].BCcount());
  }
  ALPAKA_FN_HOST_ACC inline uint8_t chIDR(const BTLDigiSoAConstView &btlDigi, int32_t i) {
    return (btlDigi[i].chIDR());
  }
  ALPAKA_FN_HOST_ACC inline uint16_t T1coarseR(const BTLDigiSoAConstView &btlDigi, int32_t i) {
    return (btlDigi[i].T1coarseR());
  }
  ALPAKA_FN_HOST_ACC inline uint16_t T2coarseR(const BTLDigiSoAConstView &btlDigi, int32_t i) {
    return (btlDigi[i].T2coarseR());
  }
  ALPAKA_FN_HOST_ACC inline uint16_t EOIcoarseR(const BTLDigiSoAConstView &btlDigi, int32_t i) {
    return (btlDigi[i].EOIcoarseR());
  }
  ALPAKA_FN_HOST_ACC inline uint16_t ChargeR(const BTLDigiSoAConstView &btlDigi, int32_t i) {
    return (btlDigi[i].ChargeR());
  }
  ALPAKA_FN_HOST_ACC inline uint16_t T1fineR(const BTLDigiSoAConstView &btlDigi, int32_t i) {
    return (btlDigi[i].T1fineR());
  }
  ALPAKA_FN_HOST_ACC inline uint16_t T2fineR(const BTLDigiSoAConstView &btlDigi, int32_t i) {
    return (btlDigi[i].T2fineR());
  }
  ALPAKA_FN_HOST_ACC inline uint16_t IdleTimeR(const BTLDigiSoAConstView &btlDigi, int32_t i) {
    return (btlDigi[i].IdleTimeR());
  }
  ALPAKA_FN_HOST_ACC inline uint8_t PrevTrigFR(const BTLDigiSoAConstView &btlDigi, int32_t i) {
    return (btlDigi[i].PrevTrigFR());
  }
  ALPAKA_FN_HOST_ACC inline uint8_t TACIDR(const BTLDigiSoAConstView &btlDigi, int32_t i) {
    return (btlDigi[i].TACIDR());
  }
  ALPAKA_FN_HOST_ACC inline uint8_t chIDL(const BTLDigiSoAConstView &btlDigi, int32_t i) {
    return (btlDigi[i].chIDL());
  }
  ALPAKA_FN_HOST_ACC inline uint16_t T1coarseL(const BTLDigiSoAConstView &btlDigi, int32_t i) {
    return (btlDigi[i].T1coarseL());
  }
  ALPAKA_FN_HOST_ACC inline uint16_t T2coarseL(const BTLDigiSoAConstView &btlDigi, int32_t i) {
    return (btlDigi[i].T2coarseL());
  }
  ALPAKA_FN_HOST_ACC inline uint16_t EOIcoarseL(const BTLDigiSoAConstView &btlDigi, int32_t i) {
    return (btlDigi[i].EOIcoarseL());
  }
  ALPAKA_FN_HOST_ACC inline uint16_t ChargeL(const BTLDigiSoAConstView &btlDigi, int32_t i) {
    return (btlDigi[i].ChargeL());
  }
  ALPAKA_FN_HOST_ACC inline uint16_t T1fineL(const BTLDigiSoAConstView &btlDigi, int32_t i) {
    return (btlDigi[i].T1fineL());
  }
  ALPAKA_FN_HOST_ACC inline uint16_t T2fineL(const BTLDigiSoAConstView &btlDigi, int32_t i) {
    return (btlDigi[i].T2fineL());
  }
  ALPAKA_FN_HOST_ACC inline uint16_t IdleTimeL(const BTLDigiSoAConstView &btlDigi, int32_t i) {
    return (btlDigi[i].IdleTimeL());
  }
  ALPAKA_FN_HOST_ACC inline uint8_t PrevTrigFL(const BTLDigiSoAConstView &btlDigi, int32_t i) {
    return (btlDigi[i].PrevTrigFL());
  }
  ALPAKA_FN_HOST_ACC inline uint8_t TACIDL(const BTLDigiSoAConstView &btlDigi, int32_t i) {
    return (btlDigi[i].TACIDL());
  }

  // Setters
  ALPAKA_FN_HOST_ACC inline void rawId(BTLDigiSoA::View &btlDigi, int32_t i, uint32_t value) {
    btlDigi[i].rawId() = value;
  }
  ALPAKA_FN_HOST_ACC inline void BC0count(BTLDigiSoA::View &btlDigi, int32_t i, uint16_t value) {
    btlDigi[i].BC0count() = value;
  }
  ALPAKA_FN_HOST_ACC inline void status(BTLDigiSoA::View &btlDigi, int32_t i, bool value) {
    btlDigi[i].status() = value;
  }
  ALPAKA_FN_HOST_ACC inline void BCcount(BTLDigiSoA::View &btlDigi, int32_t i, uint32_t value) {
    btlDigi[i].BCcount() = value;
  }
  ALPAKA_FN_HOST_ACC inline void chIDR(BTLDigiSoA::View &btlDigi, int32_t i, uint8_t value) {
    btlDigi[i].chIDR() = value;
  }
  ALPAKA_FN_HOST_ACC inline void T1coarseR(BTLDigiSoA::View &btlDigi, int32_t i, uint16_t value) {
    btlDigi[i].T1coarseR() = value;
  }
  ALPAKA_FN_HOST_ACC inline void T2coarseR(BTLDigiSoA::View &btlDigi, int32_t i, uint16_t value) {
    btlDigi[i].T2coarseR() = value;
  }
  ALPAKA_FN_HOST_ACC inline void EOIcoarseR(BTLDigiSoA::View &btlDigi, int32_t i, uint16_t value) {
    btlDigi[i].EOIcoarseR() = value;
  }
  ALPAKA_FN_HOST_ACC inline void ChargeR(BTLDigiSoA::View &btlDigi, int32_t i, uint16_t value) {
    btlDigi[i].ChargeR() = value;
  }
  ALPAKA_FN_HOST_ACC inline void T1fineR(BTLDigiSoA::View &btlDigi, int32_t i, uint16_t value) {
    btlDigi[i].T1fineR() = value;
  }
  ALPAKA_FN_HOST_ACC inline void T2fineR(BTLDigiSoA::View &btlDigi, int32_t i, uint16_t value) {
    btlDigi[i].T2fineR() = value;
  }
  ALPAKA_FN_HOST_ACC inline void IdleTimeR(BTLDigiSoA::View &btlDigi, int32_t i, uint16_t value) {
    btlDigi[i].IdleTimeR() = value;
  }
  ALPAKA_FN_HOST_ACC inline void PrevTrigFR(BTLDigiSoA::View &btlDigi, int32_t i, uint8_t value) {
    btlDigi[i].PrevTrigFR() = value;
  }
  ALPAKA_FN_HOST_ACC inline void TACIDR(BTLDigiSoA::View &btlDigi, int32_t i, uint8_t value) {
    btlDigi[i].TACIDR() = value;
  }
  ALPAKA_FN_HOST_ACC inline void chIDL(BTLDigiSoA::View &btlDigi, int32_t i, uint8_t value) {
    btlDigi[i].chIDL() = value;
  }
  ALPAKA_FN_HOST_ACC inline void T1coarseL(BTLDigiSoA::View &btlDigi, int32_t i, uint16_t value) {
    btlDigi[i].T1coarseL() = value;
  }
  ALPAKA_FN_HOST_ACC inline void T2coarseL(BTLDigiSoA::View &btlDigi, int32_t i, uint16_t value) {
    btlDigi[i].T2coarseL() = value;
  }
  ALPAKA_FN_HOST_ACC inline void EOIcoarseL(BTLDigiSoA::View &btlDigi, int32_t i, uint16_t value) {
    btlDigi[i].EOIcoarseL() = value;
  }
  ALPAKA_FN_HOST_ACC inline void ChargeL(BTLDigiSoA::View &btlDigi, int32_t i, uint16_t value) {
    btlDigi[i].ChargeL() = value;
  }
  ALPAKA_FN_HOST_ACC inline void T1fineL(BTLDigiSoA::View &btlDigi, int32_t i, uint16_t value) {
    btlDigi[i].T1fineL() = value;
  }
  ALPAKA_FN_HOST_ACC inline void T2fineL(BTLDigiSoA::View &btlDigi, int32_t i, uint16_t value) {
    btlDigi[i].T2fineL() = value;
  }
  ALPAKA_FN_HOST_ACC inline void IdleTimeL(BTLDigiSoA::View &btlDigi, int32_t i, uint16_t value) {
    btlDigi[i].IdleTimeL() = value;
  }
  ALPAKA_FN_HOST_ACC inline void PrevTrigFL(BTLDigiSoA::View &btlDigi, int32_t i, uint8_t value) {
    btlDigi[i].PrevTrigFL() = value;
  }
  ALPAKA_FN_HOST_ACC inline void TACIDL(BTLDigiSoA::View &btlDigi, int32_t i, uint8_t value) {
    btlDigi[i].TACIDL() = value;
  }
  ALPAKA_FN_HOST_ACC inline void setDigi(BTLDigiSoA::View &btlDigi,
                                         int32_t i,
                                         uint32_t rawId_val,
                                         uint16_t BC0count_val,
                                         bool status_val,
                                         uint32_t BCcount_val,
                                         uint8_t chIDR_val,
                                         uint16_t T1coarseR_val,
                                         uint16_t T2coarseR_val,
                                         uint16_t EOIcoarseR_val,
                                         uint16_t ChargeR_val,
                                         uint16_t T1fineR_val,
                                         uint16_t T2fineR_val,
                                         uint16_t IdleTimeR_val,
                                         uint8_t PrevTrigFR_val,
                                         uint8_t TACIDR_val,
                                         uint8_t chIDL_val,
                                         uint16_t T1coarseL_val,
                                         uint16_t T2coarseL_val,
                                         uint16_t EOIcoarseL_val,
                                         uint16_t ChargeL_val,
                                         uint16_t T1fineL_val,
                                         uint16_t T2fineL_val,
                                         uint16_t IdleTimeL_val,
                                         uint8_t PrevTrigFL_val,
                                         uint8_t TACIDL_val) {
    btlDigi[i].rawId() = rawId_val;
    btlDigi[i].BC0count() = BC0count_val;
    btlDigi[i].status() = status_val;
    btlDigi[i].BCcount() = BCcount_val;
    btlDigi[i].chIDR() = chIDR_val;
    btlDigi[i].T1coarseR() = T1coarseR_val;
    btlDigi[i].T2coarseR() = T2coarseR_val;
    btlDigi[i].EOIcoarseR() = EOIcoarseR_val;
    btlDigi[i].ChargeR() = ChargeR_val;
    btlDigi[i].T1fineR() = T1fineR_val;
    btlDigi[i].T2fineR() = T2fineR_val;
    btlDigi[i].IdleTimeR() = IdleTimeR_val;
    btlDigi[i].PrevTrigFR() = PrevTrigFR_val;
    btlDigi[i].TACIDR() = TACIDR_val;
    btlDigi[i].chIDL() = chIDL_val;
    btlDigi[i].T1coarseL() = T1coarseL_val;
    btlDigi[i].T2coarseL() = T2coarseL_val;
    btlDigi[i].EOIcoarseL() = EOIcoarseL_val;
    btlDigi[i].ChargeL() = ChargeL_val;
    btlDigi[i].T1fineL() = T1fineL_val;
    btlDigi[i].T2fineL() = T2fineL_val;
    btlDigi[i].IdleTimeL() = IdleTimeL_val;
    btlDigi[i].PrevTrigFL() = PrevTrigFL_val;
    btlDigi[i].TACIDL() = TACIDL_val;
  }

}  // namespace btldigi
#endif  // DataFormats_FTLDigi_interface_BTLDigiSoA_h
