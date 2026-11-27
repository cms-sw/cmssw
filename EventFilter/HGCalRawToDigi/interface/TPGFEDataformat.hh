#ifndef TPGFEDataformat_h
#define TPGFEDataformat_h

#include "DataFormats/HGCalDigi/interface/HGCalTriggerDefinitions.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <bit>
#include "TMath.h"

namespace TPGFEDataformat {

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////// Data Formats by Paul Dauncey /////////////////////////////////
  ///provision for tctp is added by Indra (without altering the existing functionalitites)



  class TcRawData {
  public:
    TcRawData() : _data(0), _rawE(0), _istctp1(false), _istctp2(false), _istctp3(false) {}

    TcRawData(hgcal::econt::Type t,
              uint8_t a,
              uint16_t e,
              uint64_t rawE = 0,
              bool istctp1 = false,
              bool istctp2 = false,
              bool istctp3 = false) {
      setTriggerCell(t, a, e, rawE, istctp1, istctp2, istctp3);
    }

    uint16_t address() const { return uint16_t(_data & 0x3f); }

    uint16_t energy() const { return ((_data >> 6) & 0x1ff); }

    uint64_t rawE() const { return _rawE; }

    uint16_t data() const { return _data; }

    bool isTcTp1() const { return _istctp1; }

    bool isTcTp2() const { return _istctp2; }

    bool isTcTp3() const { return _istctp3; }

    void setTcTp1(bool tctp1) { _istctp1 = tctp1; }
    void setTcTp2(bool tctp2) { _istctp2 = tctp2; }
    void setTcTp3(bool tctp3) { _istctp3 = tctp3; }

    friend void swap(TcRawData& lhs, TcRawData& rhs) { std::swap(lhs._data, rhs._data); }
    // friend bool operator<(const TcRawData& lhs, const TcRawData& rhs) {
    //   return lhs.energy() < rhs.energy();
    // }
    friend bool operator<(const TcRawData& lhs, const TcRawData& rhs) { return lhs.rawE() < rhs.rawE(); }
    // friend bool operator<=(const TcRawData& lhs, const TcRawData& rhs) {
    //   return lhs.energy() <= rhs.energy();
    // }
    friend bool operator<=(const TcRawData& lhs, const TcRawData& rhs) { return lhs.rawE() <= rhs.rawE(); }
    friend bool operator>(const TcRawData& lhs, const TcRawData& rhs) { return lhs.energy() > rhs.energy(); }
    friend bool operator>=(const TcRawData& lhs, const TcRawData& rhs) { return lhs.energy() > rhs.energy(); }
    friend bool operator!=(const TcRawData& lhs, const TcRawData& rhs) {
      return (lhs.energy() != rhs.energy() or lhs.address() != rhs.address());
    }
    friend std::ostream& operator<<(std::ostream& os, TcRawData const& atc) {
      return os << "TPGFEDataformat::TcRawData(" << atc << ")::print(): Data = 0x" << std::hex << ::std::setfill('0')
                << std::setw(4) << atc.data() << std::dec << ::std::setfill(' ') << ", address = " << std::setw(2)
                << unsigned(atc.address()) << ", energy = " << std::setw(3) << atc.energy()
                << ", rawE = " << std::setw(8) << atc.rawE() << ", istctp1 = " << std::setw(2) << atc.isTcTp1()
                << ", istctp2 = " << std::setw(2) << atc.isTcTp2() << ", istctp3 = " << std::setw(2) << atc.isTcTp3()
                << std::endl;
    }

    void setTriggerCell(
        hgcal::econt::Type t, uint8_t a, uint16_t e, uint64_t rawE, bool istctp1, bool istctp2, bool istctp3) {
      check(t, a, e);
      _data = (e << 6 | a);
      _rawE = rawE;
      setTcTp1(istctp1);
      setTcTp2(istctp2);
      setTcTp3(istctp3);
    }

    void print() const {
      std::cout << "TPGFEDataformat::TcRawData(" << this << ")::print(): Data = 0x" << std::hex << ::std::setfill('0')
                << std::setw(4) << _data << std::dec << ::std::setfill(' ') << ", address = " << std::setw(2)
                << unsigned(address()) << ", energy = " << std::setw(3) << energy() << ", rawE = " << std::setw(10)
                << rawE() << ", istctp1 = " << std::setw(2) << isTcTp1() << ", istctp2 = " << std::setw(2) << isTcTp2()
                << ", isTot = " << std::setw(2) << isTcTp3() << std::endl;
    }

    template <unsigned E, unsigned M>
    static constexpr uint16_t EncodeEM(uint64_t value){
      // special case corresponding with exponent in {0,1} -- just return the input, it is small enough
      if(value < (uint64_t(2) << M)) return (uint16_t)value;

      const unsigned leadingBit = std::bit_width(value) - 1; // std::bit_width(x) == ceil(log_2(x+1)), see: https://en.cppreference.com/cpp/numeric/bit_width
      const unsigned expo = leadingBit - M + 1;

      // special case: value is too large, so saturate to the max code
      if (expo >= (1u << E)) return (uint16_t)((1u << (E + M)) - 1);

      const uint64_t mant = (value >> (leadingBit - M)) & ((1u << M) - 1);
      return (uint16_t)((expo << M) | mant);
    }

    template <unsigned E, unsigned M>
    static constexpr uint64_t DecodeEM(uint16_t compressed){
      const uint64_t mant = compressed & ((1u << M) - 1);
      const uint32_t expo = (compressed >> M) & ((1u << E) - 1);

      // special cases -- no rounding/compression
      if(expo == 0) return mant;
      if(expo == 1) return (uint64_t(1) << M) | mant;
      const uint32_t shift = expo + M - 1;
      return (uint64_t(1) << shift) | (mant << (shift - M)) | (uint64_t(1) << (shift - M - 1));
    }
    static uint32_t Decode4E3M(uint16_t c) { return static_cast<uint32_t>(DecodeEM<4, 3>(c)); } // TODO: Remove the cast? Not sure why this one was a different type. - Jan
    static uint64_t Decode5E3M(uint16_t c) { return DecodeEM<5, 3>(c); }
    static uint64_t Decode5E4M(uint16_t c) { return DecodeEM<5, 4>(c); }

    uint64_t decodedE(hgcal::econt::Type t) const {
      uint64_t decompressed = 0;
      switch (t) {
        case hgcal::econt::BestC:
          decompressed = Decode4E3M(energy());
          break;
        case hgcal::econt::STC4A:
          decompressed = Decode4E3M(energy());
          break;
        case hgcal::econt::STC4B:
          decompressed = Decode5E4M(energy());
          break;
        case hgcal::econt::STC16:
          decompressed = Decode5E4M(energy());
          break;
        case hgcal::econt::CTC4A:
          decompressed = Decode4E3M(energy());
          break;
        case hgcal::econt::CTC4B:
          decompressed = Decode5E4M(energy());
          break;
        default:  //to allow unknown type
            ;
      }
      return decompressed;
    }

    void print(hgcal::econt::Type t) const {
      std::cout << "TPGFEDataformat::TcRawData(" << this << ")::print(): Data = 0x" << std::hex << ::std::setfill('0')
                << std::setw(4) << _data << std::dec << ::std::setfill(' ') << ", address = " << std::setw(2)
                << unsigned(address()) << ", raw = " << std::setw(10) << rawE() << ", energy = " << std::setw(3)
                << energy() << ", unpacked = " << std::setw(10) << decodedE(t) << ", istctp1 = " << std::setw(2)
                << isTcTp1() << ", istctp2 = " << std::setw(2) << isTcTp2() << ", istctp3 = " << std::setw(2)
                << isTcTp3() << std::endl;
    }

  private:
    uint16_t _data;
    uint64_t _rawE;
    bool _istctp1;
    bool _istctp2;
    bool _istctp3;

    // TODO: Should check the values in here; not sure they are correct.
    //       These can ultimately be moved to HGCalTriggerDefinitions.h - Jan
    void check(hgcal::econt::Type t, uint8_t a, uint16_t e) {

      uint16_t eMax1 = 0x7f;
      uint16_t eMax2 = 0x1ff;

      uint16_t aDefaultUnknown = 0x3f;

      uint16_t eMax = 0;
      uint16_t aMax = 0;

      switch (t) {
        case hgcal::econt::BestC:
          aMax = hgcal::ECONT_FRAME::BC_MAX_TCADDR;
          eMax = eMax1;
          break;
        case hgcal::econt::STC4A:
          aMax = 3;
          eMax = eMax1;
          break;
        case hgcal::econt::STC4B:
          aMax = 3;
          eMax = eMax2;
          break;
        case hgcal::econt::STC16:
          aMax = 15;
          eMax = eMax2;
          break;
        case hgcal::econt::CTC4A:
          eMax = eMax1;
          break;
        case hgcal::econt::CTC4B:
          eMax = eMax2;
          break;
        default:  //to allow unknown type

          if(a != aDefaultUnknown){
            cms::Exception ex("ValueError");
            ex << "a != "
            << aDefaultUnknown
            << " for hgcal::econt::Type "
            << t;
            ex.addContext("Calling TcRawData::check()");
            throw ex;
          }

          if(e != 0){
            cms::Exception ex("ValueError");
            ex << "e != "
            << 0
            << " for hgcal::econt::Type "
            << t;
            ex.addContext("Calling TcRawData::check()");
            throw ex;
          }
          return;
          // ;
      }

      if(a > aMax && aMax > 0){
        cms::Exception ex("ValueError");
        ex << "a > "
        << aMax
        << " for hgcal::econt::Type "
        << t;
        ex.addContext("Calling TcRawData::check()");
        throw ex;
      }

      if(e > eMax && eMax > 0){
        cms::Exception ex("ValueError");
        ex << "e > "
        << eMax
        << " for hgcal::econt::Type "
        << t;
        ex.addContext("Calling TcRawData::check()");
        throw ex;
      }
      return;
    }
  };

  static std::string tctypeName[10] = {
      "BestC", "STC4A", "STC4B", "STC16", "CTC4A", "CTC4B", "TS", "RA", "AE", "Unknown"};

  class TcRawDataPacket {
  public:
    TcRawDataPacket() : _t(hgcal::econt::Unknown), _bx(0), _ms(0), _rawEms(0) { _tcdata.resize(0); }
    TcRawDataPacket(hgcal::econt::Type t, uint8_t bx, uint8_t e) : _t(t), _bx(bx), _ms(e) { _tcdata.resize(0); }
    TcRawDataPacket(hgcal::econt::Type t, uint8_t bx, uint8_t e, uint64_t upe)
        : _t(t), _bx(bx), _ms(e), _rawEms(upe) {
      _tcdata.resize(0);
    }
    const std::string& typeName() const { return TPGFEDataformat::tctypeName[type()]; }
    hgcal::econt::Type type() const { return _t; }
    size_t size() const { return _tcdata.size(); }
    uint16_t moduleSum() const { return uint16_t(_ms & 0xff); }
    uint64_t rawEMS() const { return _rawEms; }
    uint16_t bx() const { return uint16_t(_bx); }
    const std::vector<TPGFEDataformat::TcRawData>& getTcData() const { return _tcdata; }
    TPGFEDataformat::TcRawData& getTc(uint32_t i) { return _tcdata.at(i); }
    bool is4E3M() const { return (_t == hgcal::econt::BestC or _t == hgcal::econt::STC4A or _t == hgcal::econt::CTC4A or _t == hgcal::econt::TS or _t == hgcal::econt::RA) ? true : false; }
    void reset() {
      _t = hgcal::econt::Type::Unknown;
      _bx = 0;
      _ms = 0;
      _rawEms = 0;
      _tcdata.resize(0);
    }
    bool isTcTp1() const {
      for (const auto& itc : getTcData())
        if (itc.isTcTp1())
          return true;
      return false;
    }
    bool isTcTp2() const {
      for (const auto& itc : getTcData())
        if (itc.isTcTp2())
          return true;
      return false;
    }
    bool isTcTp3() const {
      for (const auto& itc : getTcData())
        if (itc.isTcTp3())
          return true;
      return false;
    }

    void setTBM(hgcal::econt::Type t, uint8_t bx, uint8_t e) {
      _t = t;
      _bx = bx;
      _ms = e;
    }
    void setTBM(hgcal::econt::Type t, uint8_t bx, uint8_t e, uint64_t upe) {
      _t = t;
      _bx = bx;
      _ms = e;
      _rawEms = upe;
    }
    void setType(hgcal::econt::Type t) { _t = t; }
    void setModuleSum(uint8_t e) { _ms = e; }
    void setModuleSum(uint8_t e, uint64_t upe) {
      _ms = e;
      _rawEms = upe;
    }
    void setBX(uint8_t bx) { _bx = bx; }
    std::vector<TPGFEDataformat::TcRawData>& setTcData() { return _tcdata; }
    void setTcData(
        hgcal::econt::Type t, uint8_t a, uint16_t e, uint64_t upe, bool isTcTp1, bool isTcTp2, bool isTcTp3) {
      TPGFEDataformat::TcRawData tc;
      tc.setTriggerCell(t, a, e, upe, isTcTp1, isTcTp2, isTcTp3);
      _tcdata.push_back(tc);
    }
    TPGFEDataformat::TcRawData& operator[](int index) {
      if (uint32_t(index) >= _tcdata.size()) {
        std::cerr << "TPGFEDataformat::TcRawDataPacket Array index out of bound, exiting" << std::endl;
        exit(0);
      }
      return _tcdata[index];
    }
    static struct {
      bool operator()(TPGFEDataformat::TcRawData& a, TPGFEDataformat::TcRawData& b) const {
        return a.address() < b.address();
      }
    } customLTA;
    static struct {
      bool operator()(TPGFEDataformat::TcRawData& a, TPGFEDataformat::TcRawData& b) const {
        return a.address() > b.address();
      }
    } customGTA;
    static struct {
      bool operator()(TPGFEDataformat::TcRawData& a, TPGFEDataformat::TcRawData& b) const {
        return a.energy() < b.energy();
      }
    } customLTE;
    static struct {
      bool operator()(TPGFEDataformat::TcRawData& a, TPGFEDataformat::TcRawData& b) const {
        return a.energy() > b.energy();
      }
    } customGTE;

    void sortCh() { std::sort(setTcData().begin(), setTcData().end(), customLTA); }
    friend std::ostream& operator<<(std::ostream& os, TcRawDataPacket const& atcp) {
      return os << "TPGFEDataformat::TcRawDataPacket(" << atcp << ")::print(): "
                << "type = " << atcp.typeName() << ", bx = " << atcp.bx() << ", rawms = " << atcp.rawEMS()
                << ", ms = " << atcp.moduleSum()
                << ", unpacked = " << TPGFEDataformat::TcRawData::Decode5E3M(atcp.moduleSum())
                << ", isMtctp1 = " << std::setw(2) << atcp.isTcTp1() << ", isMtctp2 = " << std::setw(2)
                << atcp.isTcTp2() << ", isMtctp3 = " << std::setw(2) << atcp.isTcTp3() << std::endl;
      for (const auto& itc : atcp.getTcData())
        itc.print(atcp.type());
    }
    friend bool operator==(TcRawDataPacket& lhs, TcRawDataPacket& rhs) {
      if (lhs.size() != rhs.size())
        return false;
      for (uint32_t itc = 0; itc < lhs.size(); itc++)
        if (lhs.getTc(itc) != rhs.getTc(itc))
          return false;
      return true;
    }

    void print() const {
      std::cout << "TPGFEDataformat::TcRawDataPacket(" << this << ")::print(): "
                << "type = " << type() << ", typename = " << typeName() << ", bx = " << bx() << ", rawms = " << rawEMS()
                << ", ms = " << moduleSum() << ", unpacked = " << TPGFEDataformat::TcRawData::Decode5E3M(moduleSum())
                << ", isMtctp1 = " << std::setw(2) << isTcTp1() << ", isMtctp2 = " << std::setw(2) << isTcTp2()
                << ", isMtctp3 = " << std::setw(2) << isTcTp3() << std::endl;
      for (const auto& itc : getTcData())
        itc.print(type());
    }

  private:
    hgcal::econt::Type _t;
    uint8_t _bx;
    uint8_t _ms;
    uint64_t _rawEms;
    std::vector<TPGFEDataformat::TcRawData> _tcdata;
  };

}
#endif
