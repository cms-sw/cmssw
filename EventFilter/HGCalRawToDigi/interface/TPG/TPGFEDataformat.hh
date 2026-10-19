#ifndef TPGFEDataformat_h
#define TPGFEDataformat_h

#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <vector>
#include <algorithm>
#include "TMath.h"

namespace TPGFEDataformat {

  typedef std::array<uint32_t, 14> OrderedElinkPacket;

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////// Data Formats by Paul Dauncey /////////////////////////////////
  ///provision for tctp is added by Indra (without altering the existing functionalitites)

  class HalfHgcrocChannelData {
  public:
    HalfHgcrocChannelData() { setZero(); }

    void setZero() { _data = 0; }

    bool isTot() const { return _data >= 0x8000; }

    uint16_t getTcTp() const { return (_data >> 12) & 0x3; }

    uint16_t getAdc() const {
      if (isTot())
        return 0;
      return _data & 0x3ff;
    }

    uint16_t getTot() const {
      if (!isTot())
        return 0;
      return _data & 0xfff;
    }

    void setAdc(uint16_t a, uint16_t tctp) {
      assert(a < 0x400 and tctp < 0x4);
      _data = tctp << 12 | a;
    }

    void setTot(uint16_t a, uint16_t tctp) {
      assert(a < 0x1000 and tctp < 0x4);
      _data = tctp << 12 | a | 0x8000;
    }

    uint16_t getData() const { return _data; }
    void setData(uint16_t data) { _data = data; }

    void print() const {
      std::cout << "HalfHgcrocChannelData(" << this << ")::print()" << std::endl;

      std::cout << std::dec << ::std::setfill(' ') << " data: 0x" << std::hex << ::std::setfill('0') << std::setw(4)
                << _data << std::dec << ::std::setfill(' ') << ", "
                << "TcTp: " << getTcTp() << ", " << (isTot() ? "TOT = " : "ADC = ") << std::setw(4)
                << (isTot() ? getTot() : getAdc())
                // << "ADC = " << std::setw(4) << getAdc() <<", "
                // << ((getTcTp()!=0) ? Form("TOT = %u",getTot()) : "")
                << std::endl;
    }

  private:
    uint16_t _data;
  };

  class HalfHgcrocData {
  public:
    enum { NumberOfChannels = 36 };

    HalfHgcrocData() : _bx(0xFFFF) { setZero(); }

    void setZero() { std::memset((void*)_data, 0, sizeof(HalfHgcrocChannelData) * NumberOfChannels); }

    void setBx(uint16_t bx) { _bx = bx; }
    const uint32_t getBx() const { return uint32_t(_bx); }
    void setSlinkBx(uint16_t bx) { _bxId = bx; }
    const uint32_t getSlinkBx() const { return uint32_t(_bxId); }

    const HalfHgcrocChannelData* getChannels() const { return _data; }

    bool hasTOT() const { return hasTcTp(3); }
    bool hasTcTp1() const { return hasTcTp(1); }
    bool hasTcTp2() const { return hasTcTp(2); }
    bool hasTcTp3() const { return hasTOT(); }

    bool hasTcTp(uint16_t tctpval) const {
      for (unsigned i(0); i < NumberOfChannels; i++)
        if (_data[i].getTcTp() == tctpval) {
          // std::cout << "TOT/TOA noted for ich : " << i << std::endl;
          // _data[i].print();
          return true;
        }
      return false;
    }

    /////////////// Following are the modification/addition for emulation ///////////////////
    // HalfHgcrocChannelData* setChannels() {
    //   return _data;
    // }
    void setChannel(int ch, HalfHgcrocChannelData chdata) {
      assert(ch >= 0 and ch <= 35);
      _data[ch] = chdata;
    }

    void setChannels(const HalfHgcrocChannelData* data) {
      for (unsigned i(0); i <= NumberOfChannels; i++)
        _data[i] = data[i];
    }
    HalfHgcrocChannelData& getChannelData(uint32_t i) { return _data[i]; }
    ///////////////////////////////////////////////////////////////////////////////////////

    void print() const {
      std::cout << "HalfHgcrocData(" << this << ")::print()" << std::endl;

      const uint16_t* p((const uint16_t*)_data);

      for (unsigned i(0); i < NumberOfChannels; i++) {
        std::cout << " Channel " << std::setw(2) << i << " = 0x" << std::hex << ::std::setfill('0') << std::setw(4)
                  << p[i] << std::dec << ::std::setfill(' ') << ", "
                  << " bx: " << getBx() << ", "
                  << " TcTp: " << _data[i].getTcTp() << ", " << (_data[i].isTot() ? "TOT = " : "ADC = ") << std::setw(4)
                  << (_data[i].isTot() ? _data[i].getTot() : _data[i].getAdc())
                  // << "ADC = " << std::setw(4) << _data[i].getAdc() <<", "
                  // << ((_data[i].getTcTp()!=0) ? Form("TOT = %u",_data[i].getTot()) : "")
                  << std::endl;
      }
    }

  private:
    uint16_t _bx;    //from ECOND header
    uint16_t _bxId;  //from Slink trailer
    HalfHgcrocChannelData _data[36];
  };

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////// Data Formats by Paul and Indra ///////////////////////////////////////////////
  enum Type {
    //select : 0 = Threshold Sum (TS), 1 = Super Trigger Cell (STC), 2 = Best Choice (BC), 3 = Repeater, 4 = Autoencoder (AE) Settings of [5,7] revert to the Repeater algorithm
    //stc_type : 0 = STC4B(5E+4M), 1 = STC16(5E+4M), 2 = CTC4A(4E+3M), 3 = STC4A(4E+3M), 4 = CTC4B(5E+3M)
    BestC,  // TC energy format = 4E3M
    STC4A,  // 4E3M
    STC4B,  // 5E4M
    STC16,  // 5E4M
    CTC4A,  // 4E3M
    CTC4B,  // 5E4M
    TS,     // 4E3M, placeholder
    RA,     // 4E3M, Repeater algorithm is intended for testing and debugging of ECONT
    AE,     // blocks of 16b, uses convolutional neural network
    Unknown
  };

  class TcRawData {
  public:
    TcRawData() : _data(0), _rawE(0), _istctp1(false), _istctp2(false), _istctp3(false) {}

    TcRawData(TPGFEDataformat::Type t,
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
        TPGFEDataformat::Type t, uint8_t a, uint16_t e, uint64_t rawE, bool istctp1, bool istctp2, bool istctp3) {
      setTriggerCell(t, a, e);
      _rawE = rawE;
      setTcTp1(istctp1);
      setTcTp2(istctp2);
      setTcTp3(istctp3);
    }

    void setTriggerCell(TPGFEDataformat::Type t, uint8_t a, uint16_t e) {
      switch (t) {
        case TPGFEDataformat::BestC:
          //assert(a<=47); // moved as exc before filling in STAGEIO
          assert(e <= 0x7f);
          break;
        case TPGFEDataformat::STC4A:
          assert(a <= 3);
          assert(e <= 0x7f);
          break;
        case TPGFEDataformat::STC4B:
          assert(a <= 3);
          assert(e <= 0x1ff);
          break;
        case TPGFEDataformat::STC16:
          assert(a <= 15);
          assert(e <= 0x1ff);
          break;
        case TPGFEDataformat::CTC4A:
          assert(e <= 0x7f);
          //a = 0;
          break;
        case TPGFEDataformat::CTC4B:
          assert(e <= 0x1ff);
          //a = 0;
          break;
        default:  //to allow unknown type
          assert(a == 0x3f);
          assert(e == 0);
          ;
      }
      _data = (e << 6 | a);
    }

    void print() const {
      std::cout << "TPGFEDataformat::TcRawData(" << this << ")::print(): Data = 0x" << std::hex << ::std::setfill('0')
                << std::setw(4) << _data << std::dec << ::std::setfill(' ') << ", address = " << std::setw(2)
                << unsigned(address()) << ", energy = " << std::setw(3) << energy() << ", rawE = " << std::setw(10)
                << rawE() << ", istctp1 = " << std::setw(2) << isTcTp1() << ", istctp2 = " << std::setw(2) << isTcTp2()
                << ", isTot = " << std::setw(2) << isTcTp3() << std::endl;
    }

    static uint32_t Decode4E3M(uint16_t compressed) {
      uint32_t mant = compressed & 0x7;
      uint32_t expo = (compressed >> 3) & 0xF;

      if (expo == 0)
        return mant;
      if (expo == 1)
        return 8 + mant;

      uint32_t shift = expo + 2;
      uint32_t decomp = 1 << shift;
      uint32_t mpdeco = 1 << (shift - 4);
      decomp = decomp | (mant << (shift - 3));
      decomp = decomp | mpdeco;

      return decomp;
    }

    static uint64_t Decode5E3M(uint16_t compressed) {
      uint32_t mant = compressed & 0x7;
      uint32_t expo = (compressed >> 3) & 0x1F;

      if (expo == 0)
        return mant;
      if (expo == 1)
        return 8 + mant;

      uint32_t shift = expo + 2;
      uint64_t decomp = 1 << shift;
      uint32_t mpdeco = 1 << (shift - 4);
      decomp = decomp | (mant << (shift - 3));
      decomp = decomp | mpdeco;

      return decomp;
    }

    static uint64_t Decode5E4M(uint16_t compressed) {
      uint32_t mant = compressed & 0xF;
      uint32_t expo = (compressed >> 4) & 0x1F;

      if (expo == 0)
        return mant;
      if (expo == 1)
        return 8 + mant;

      uint32_t shift = expo + 3;
      uint64_t decomp = 1 << shift;
      uint32_t mpdeco = 1 << (shift - 4);
      decomp = decomp | (mant << (shift - 4));
      decomp = decomp | mpdeco;

      return decomp;
    }

    uint64_t decodedE(TPGFEDataformat::Type t) const {
      uint64_t decompressed = 0;
      switch (t) {
        case TPGFEDataformat::BestC:
          decompressed = Decode4E3M(energy());
          break;
        case TPGFEDataformat::STC4A:
          decompressed = Decode4E3M(energy());
          break;
        case TPGFEDataformat::STC4B:
          decompressed = Decode5E4M(energy());
          break;
        case TPGFEDataformat::STC16:
          decompressed = Decode5E4M(energy());
          break;
        case TPGFEDataformat::CTC4A:
          decompressed = Decode4E3M(energy());
          break;
        case TPGFEDataformat::CTC4B:
          decompressed = Decode5E4M(energy());
          break;
        default:  //to allow unknown type
            ;
      }
      return decompressed;
    }

    void print(TPGFEDataformat::Type t) const {
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
  };

  static std::string tctypeName[10] = {
      "BestC", "STC4A", "STC4B", "STC16", "CTC4A", "CTC4B", "TS", "RA", "AE", "Unknown"};

  class TcRawDataPacket {
  public:
    TcRawDataPacket() : _t(Unknown), _bx(0), _ms(0), _rawEms(0) { _tcdata.resize(0); }
    TcRawDataPacket(TPGFEDataformat::Type t, uint8_t bx, uint8_t e) : _t(t), _bx(bx), _ms(e) { _tcdata.resize(0); }
    TcRawDataPacket(TPGFEDataformat::Type t, uint8_t bx, uint8_t e, uint64_t upe)
        : _t(t), _bx(bx), _ms(e), _rawEms(upe) {
      _tcdata.resize(0);
    }
    const std::string& typeName() const { return TPGFEDataformat::tctypeName[type()]; }
    TPGFEDataformat::Type type() const { return _t; }
    size_t size() const { return _tcdata.size(); }
    uint16_t moduleSum() const { return uint16_t(_ms & 0xff); }
    uint64_t rawEMS() const { return _rawEms; }
    uint16_t bx() const { return uint16_t(_bx); }
    const std::vector<TPGFEDataformat::TcRawData>& getTcData() const { return _tcdata; }
    TPGFEDataformat::TcRawData& getTc(uint32_t i) { return _tcdata.at(i); }
    bool is4E3M() const { return (_t == BestC or _t == STC4A or _t == CTC4A or _t == TS or _t == RA) ? true : false; }
    void reset() {
      _t = TPGFEDataformat::Type::Unknown;
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

    void setTBM(TPGFEDataformat::Type t, uint8_t bx, uint8_t e) {
      _t = t;
      _bx = bx;
      _ms = e;
    }
    void setTBM(TPGFEDataformat::Type t, uint8_t bx, uint8_t e, uint64_t upe) {
      _t = t;
      _bx = bx;
      _ms = e;
      _rawEms = upe;
    }
    void setType(TPGFEDataformat::Type t) { _t = t; }
    void setModuleSum(uint8_t e) { _ms = e; }
    void setModuleSum(uint8_t e, uint64_t upe) {
      _ms = e;
      _rawEms = upe;
    }
    void setBX(uint8_t bx) { _bx = bx; }
    std::vector<TPGFEDataformat::TcRawData>& setTcData() { return _tcdata; }
    void setTcData(
        TPGFEDataformat::Type t, uint8_t a, uint16_t e, uint64_t upe, bool isTcTp1, bool isTcTp2, bool isTcTp3) {
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
    TPGFEDataformat::Type _t;
    uint8_t _bx;
    uint8_t _ms;
    uint64_t _rawEms;
    std::vector<TPGFEDataformat::TcRawData> _tcdata;
  };

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  typedef std::vector<std::pair<uint32_t, TPGFEDataformat::HalfHgcrocData>> HRocarray;
  //typedef std::pair<uint32_t,std::vector<TPGFEDataformat::TcRawData>> TcRawDataPacket;
  typedef std::pair<uint32_t, TPGFEDataformat::TcRawDataPacket> TcModulePacket;
  typedef std::pair<uint32_t, std::vector<TPGFEDataformat::TcRawDataPacket>> TcModuleBxPackets;
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  class HgcrocTcData {
  public:
    HgcrocTcData() { setZero(); }

    void setZero() {
      _data = 0;
      _cdata = 0;
      _isTot = false;
      _istctp1 = false;
      _istctp2 = false;
      _istctp3 = false;
    }

    bool isTot() const { return _isTot; }

    uint16_t getCdata() const { return _cdata; }

    uint32_t getCharge() const { return _data; }

    bool isTcTp1() const { return _istctp1; }

    bool isTcTp2() const { return _istctp2; }

    bool isTcTp3() const { return _istctp3; }

    void setCdata(uint16_t a) {
      assert(a < 0x80);  //compressed HGCROC TC data is packed into7bits
      _cdata = a;
    }

    void setCharge(uint32_t a) {
      //assert(a<0x1000000); //assuming 21bit for HD module, then bit shift of 3, then +1
      // if(a<0x1000000)
      // 	_data=a;
      // else
      _data = (a < 0x1000000) ? a : 0xffffff;
    }

    void setTot(bool a) { _isTot = a; }

    void setTcTp1(bool tctp1) { _istctp1 = tctp1; }

    void setTcTp2(bool tctp2) { _istctp2 = tctp2; }

    void setTcTp3(bool tctp3) { _istctp3 = tctp3; }

  private:
    uint32_t _data;   //raw-uncompressed data
    uint16_t _cdata;  //compressed 4E+3M ()
    bool _isTot;      //isTOT or ADC
    bool _istctp1;
    bool _istctp2;
    bool _istctp3;
  };

  class ModuleTcData {
  public:
    enum { MaxNumberOfTCs = 48 };

    ModuleTcData() { setZero(); }

    void setZero() {
      std::memset((void*)_data, 0, sizeof(HgcrocTcData) * MaxNumberOfTCs);
      NumberOfTCs = 0;
    }

    void setBx(uint16_t bx) { _bx = bx; }
    const uint32_t getBx() const { return uint32_t(_bx); }

    const uint32_t getNofTCs() const { return uint32_t(NumberOfTCs); }
    const HgcrocTcData* getTCs() const { return _data; }
    const HgcrocTcData& getTC(uint32_t i) const { return _data[i]; }

    bool isTcTp1() const {
      for (uint16_t i(0); i < NumberOfTCs; i++)
        if (_data[i].isTcTp1())
          return true;
      return false;
    }

    bool isTcTp2() const {
      for (uint16_t i(0); i < NumberOfTCs; i++)
        if (_data[i].isTcTp2())
          return true;
      return false;
    }

    bool isTcTp3() const {
      for (uint16_t i(0); i < NumberOfTCs; i++)
        if (_data[i].isTot())
          return true;
      return false;
    }

    void setNofTCs(const unsigned nofTCs) { NumberOfTCs = nofTCs; }
    void setTCs(const HgcrocTcData* data) {
      for (uint16_t i(0); i < NumberOfTCs; i++)
        _data[i] = data[i];
    }

    void print() const {
      std::cout << "ModuleTriggerCellData(" << this << ")::print() : NumberOfTCs :" << NumberOfTCs
                << ", bx: " << getBx() << ", isTcTp1 : " << isTcTp1() << ", isTcTp2 : " << isTcTp2()
                << ", isTcTp3 : " << isTcTp3() << std::endl;

      for (uint16_t i(0); i < NumberOfTCs; i++) {
        std::cout << " TC " << std::setw(2) << i << ": compressed = " << std::dec << ::std::setfill(' ')
                  << std::setw(10) << _data[i].getCdata() << std::dec << ::std::setfill(' ') << ", raw-uncompressed "
                  << std::setw(10) << _data[i].getCharge() << ", istot : " << std::setw(2) << _data[i].isTot()
                  << ", istctp1 : " << std::setw(2) << _data[i].isTcTp1() << ", istctp2 : " << std::setw(2)
                  << _data[i].isTcTp2() << std::endl;
      }
    }

  private:
    HgcrocTcData _data[48];
    uint16_t _bx;
    uint16_t NumberOfTCs;
  };

}  // namespace TPGFEDataformat

namespace TPGFEConfiguration {
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////The configuration of half of ROC based on HGCROC3a [doc. no. v2.0] (See Table@Page-43)
  //////EDMS ROCv3a: https://edms.cern.ch/ui/#!master/navigator/document?D:100570166:100570166:subDocs
  //////EDMS ROCv3b(recent): https://edms.cern.ch/ui/#!master/navigator/document?D:101362066:101362066:subDocs
  class ConfigHfROC {
  public:
    ConfigHfROC() {}
    uint32_t getAdcTH() const { return uint32_t(Adc_TH); }
    uint64_t getClrAdcTottrig() const { return ClrAdcTot_trig; }
    bool isChMasked(uint32_t ich) const {
      int chnl = ich % 36;
      return (getClrAdcTottrig() >> chnl) & 0x1;
    }
    uint32_t getTotTH(uint32_t ich) const {
      uint32_t chnl = ich % 36;
      uint32_t tot_idx = TMath::FloorNint(chnl / 9);
      return uint32_t(Tot_TH[tot_idx]);
    }
    uint32_t getTotP(uint32_t ich) const {
      uint32_t chnl = ich % 36;
      uint32_t tot_idx = TMath::FloorNint(chnl / 9);
      return uint32_t(Tot_P[tot_idx]);
    }
    uint32_t getMultFactor() const { return uint32_t(MultFactor); }
    void setAdcTH(uint32_t adcth) { Adc_TH = adcth & 0x1F; }
    void setClrAdcTottrig(uint64_t clradctottrig) { ClrAdcTot_trig = clradctottrig & 0xFFFFFFFF; }
    void setMultFactor(uint32_t multfactor) { MultFactor = multfactor & 0x1F; }
    void setTotTH(uint32_t tot_idx, uint32_t tot_th) { Tot_TH[tot_idx] = tot_th & 0xFF; }
    void setTotP(uint32_t tot_idx, uint32_t tot_p) { Tot_P[tot_idx] = tot_p & 0x7F; }
    void print() const {
      std::cout << std::dec << ::std::setfill(' ') << "ConfigHfROC(" << this << ")::print(): "
                << "Adc_TH = " << std::setw(4) << getAdcTH() << ", MultFactor = " << std::setw(3) << getMultFactor()
                << std::endl;

      std::cout << std::dec << ::std::setfill(' ') << "ConfigHfROC(" << this << ")::print(): "
                << "ClrAdcTot_trig = ";
      for (uint32_t ich = 0; ich < 36; ich++)
        std::cout << std::setw(2) << "(" << ich << ": " << isChMasked(ich) << ") ";
      std::cout << std::endl;

      std::cout << std::dec << ::std::setfill(' ') << "ConfigHfROC(" << this << ")::print(): "
                << "Tot_P = ";
      for (uint32_t itotch = 0; itotch < 4; itotch++)
        std::cout << std::setw(4) << "(" << itotch << ": " << uint32_t(Tot_P[itotch]) << ") ";
      std::cout << std::endl;

      std::cout << std::dec << ::std::setfill(' ') << "ConfigHfROC(" << this << ")::print(): "
                << "Tot_TH = ";
      for (uint32_t itotch = 0; itotch < 4; itotch++)
        std::cout << std::setw(4) << "(" << itotch << ": " << uint32_t(Tot_TH[itotch]) << ") ";
      std::cout << std::endl;
    }

  private:
    //Digital Info
    uint8_t Adc_TH;           //5-bits
    uint64_t ClrAdcTot_trig;  //36-bits
    uint8_t MultFactor;       //5-bits
    uint8_t Tot_P[4];         //one per 9 channel (each with 7-bits):not present in 2023 beam test
    uint8_t Tot_TH[4];        //one per 9 channel (each with 8 bits):not present in 2023 beam test
  };
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //The configuration of channel corresponding to ADC per module
  class ConfigCh {
  public:
    ConfigCh() {}
    uint32_t getAdcpedestal() const { return uint32_t(Adc_pedestal); }
    void setAdcpedestal(uint32_t ped) { Adc_pedestal = ped & 0xFF; }
    void print() {
      std::cout << std::dec << ::std::setfill(' ') << "ConfigCh(" << this << ")::print(): "
                << "Adc_pedestal = " << std::setw(4) << getAdcpedestal() << std::endl;
    }
    void print(uint32_t ich) {
      std::cout << std::dec << ::std::setfill(' ') << "ConfigCh(" << this << ")::print(): "
                << "ich: " << ich << ", Adc_pedestal = " << std::setw(4) << getAdcpedestal() << std::endl;
    }

  private:
    uint8_t Adc_pedestal;  //8-bits
  };
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////ECON-D [doc. no: v1.1]
  //////https://edms.cern.ch/ui/#!master/navigator/document?P:100053490:100904542:subDocs
  class ConfigEconD {
  public:
    ConfigEconD() : isPassThrough(false), neRx(0) {}
    bool passThrough() const { return isPassThrough; }
    uint32_t getNeRx() const {
      assert(neRx != 0);
      return uint32_t(neRx);
    }
    void setPassThrough(bool isPT) { isPassThrough = isPT; }
    void setNeRx(uint32_t nofeRx) {
      assert(nofeRx != 0);
      neRx = nofeRx;
    }
    void print() {
      std::cout << std::dec << ::std::setfill(' ') << "ConfigEconD(" << this << ")::print(): "
                << "isPassThrough mode = " << std::setw(2) << passThrough() << ", \tnof eRx = " << std::setw(2)
                << getNeRx() << std::endl;
    }

  private:
    bool isPassThrough;  //1-bit
    uint8_t neRx;
  };
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////ECON-T [doc. no: v10]
  //////https://edms.cern.ch/ui/#!master/navigator/document?P:100053490:100430098:subDocs
  class ConfigEconT {
  public:
    ConfigEconT() : density(0), dropLSB(0), select(0), stc_type(0), usesum(true) {
      for (uint32_t itc = 0; itc < 48; itc++) {
        calv[itc] = 0;
        mux[itc] = 0;
      }
    }
    bool getDensity() const {
      assert(density == 0 or density == 1);
      return (density == 1) ? true : false;
    }
    uint32_t getDropLSB() const {
      assert(dropLSB >= 0 and dropLSB <= 4);
      return uint32_t(dropLSB);
    }
    uint32_t getSelect() const {
      assert(select == 1 or select == 2);
      return uint32_t(select);
    }
    uint32_t getSTCType() const {
      assert(stc_type >= 0 and stc_type <= 4);
      return uint32_t(stc_type);
    }
    uint32_t getNElinks() const {
      assert(eporttx_numen != 0);
      return uint32_t(eporttx_numen);
    }
    bool getMSSumType() const { return usesum; }
    uint32_t getNofTCs() const { return getBCType(); }
    uint32_t getBCType() const {
      uint32_t maxTcs = 0;
      if (getOutType() == TPGFEDataformat::BestC) {
        switch (getNElinks()) {
          case 1:
            maxTcs = 1;
            break;
          case 2:
            maxTcs = 4;
            break;
          case 3:
            maxTcs = 6;
            break;
          case 4:
            maxTcs = 9;
            break;
          case 5:
            maxTcs = 14;
            break;
          case 6:
            maxTcs = 18;
            break;
          case 7:
            maxTcs = 23;
            break;
          case 8:
            maxTcs = 28;
            break;
          case 9:
            maxTcs = 32;
            break;
          case 10:
            maxTcs = 37;
            break;
          case 11:
            maxTcs = 41;
            break;
          case 12:
            maxTcs = 46;
            break;
          default:  //same as case13 or above
            maxTcs = 48;
            break;
        }
      }
      return maxTcs;
    }
    uint32_t getNofSTCs() const {
      uint32_t maxSTCs = 0;
      switch (getSTCType()) {
        case 0:  //0 = STC4B(5E+4M)
          switch (getNElinks()) {
            case 1:
              maxSTCs = 2;
              break;
            case 2:
              maxSTCs = 5;
              break;
            case 3:
              maxSTCs = 8;
              break;
            case 4:
              maxSTCs = 11;
              break;
            default:  //case 5 or higher
              maxSTCs = 12;
              break;
          }
          break;
        case 1:  //1 = STC16(5E+4M)
          switch (getNElinks()) {
            case 1:
              maxSTCs = 2;
              break;
            default:  //case 2 or higher
              maxSTCs = 3;
              break;
          }
          break;
        case 2:  //2 = CTC4A(4E+3M)
          switch (getNElinks()) {
            case 1:
              maxSTCs = 4;
              break;
            case 2:
              maxSTCs = 8;
              break;
            default:  //case 3 or higher
              maxSTCs = 12;
              break;
          }
          break;
        default:  //3 = STC4A(4E+3M) //4 = CTC4B(5E+3M)
          switch (getNElinks()) {
            case 1:
              maxSTCs = 3;
              break;
            case 2:
              maxSTCs = 6;
              break;
            case 3:
              maxSTCs = 10;
              break;
            default:  //case 4 or higher
              maxSTCs = 12;
              break;
          }
      }  //stctype
      return maxSTCs;
    }
    uint32_t getCalibration(uint32_t itc) const {
      assert(itc <= 47);
      return uint32_t(calv[itc]);
    }
    uint32_t getInputMux(uint32_t itc) const {
      assert(itc <= 47);
      return (isConnectedMux(itc)) ? uint32_t(mux[itc]) : 0x80;
    }
    bool isConnectedMux(uint32_t itc) const {
      assert(itc <= 47);
      return (mux[itc] >> 7) ? false : true;
    }
    TPGFEDataformat::Type getOutType() const {
      TPGFEDataformat::Type type;
      switch (select) {
        case 0:
          type = TPGFEDataformat::TS;
          break;
        case 1:
          switch (stc_type) {
            case 0:
              type = TPGFEDataformat::STC4B;
              break;
            case 1:
              type = TPGFEDataformat::STC16;
              break;
            case 2:
              type = TPGFEDataformat::CTC4A;
              break;
            case 3:
              type = TPGFEDataformat::STC4A;
              break;
            case 4:
              type = TPGFEDataformat::CTC4B;
              break;
            default:
              type = TPGFEDataformat::Unknown;
              break;
          }
          break;
        case 2:
          type = TPGFEDataformat::BestC;
          break;
        case 3:
          type = TPGFEDataformat::RA;
          break;
        case 4:
          type = TPGFEDataformat::AE;
          break;
        default:
          type = TPGFEDataformat::Unknown;
          break;
      }
      return type;
    }
    void setDensity(uint32_t den) {
      assert(density == 1);
      density = den;
    }
    void setDropLSB(uint32_t dLSB) {
      assert(dLSB <= 4);
      dropLSB = dLSB;
    }
    void setSelect(uint32_t sel) {
      assert(sel == 1 or sel == 2);
      select = sel;
    }
    void setSTCType(uint32_t stctype) {
      assert(stctype <= 4);
      stc_type = stctype;
    }
    void setNElinks(uint32_t nlinks) {
      assert(nlinks != 0);
      eporttx_numen = nlinks;
    }
    void setMSSumType(bool sumtype) { usesum = sumtype; }
    void setCalibration(uint32_t itc, uint32_t calib) {
      assert(itc <= 47);
      calv[itc] = (calib & 0xFFF);
    }
    void setInputMux(uint32_t itc, uint32_t muxval) {
      assert((muxval <= 47 or muxval == 0x80) and
             itc <= 47);  //since default value is not known, set to 0x80 for unconnected TC
      mux[itc] = muxval & 0xff;
    }
    void print() {
      std::cout << std::dec << ::std::setfill(' ') << "ConfigEconT(" << this
                << ")::print(): "
                // <<", \tECON-T mode = "
                // << std::setw(8) << getOutType()
                << ", \tLSB used for TC input = " << std::setw(2) << getDensity()
                << ", \tLSB in ECON-T output = " << std::setw(2) << getDropLSB()
                << ", \tECON-T select = " << std::setw(2) << getSelect() << ", \tECON-T stc_type = " << std::setw(2)
                << getSTCType() << ", \tECON-T maxTCs = " << std::setw(2)
                << ((getSelect() == 2) ? getBCType() : getNofSTCs()) << std::endl;
      for (uint32_t itc = 0; itc < 48; itc++)
        std::cout << "ConfigEconT(" << this << ")::print(): "
                  << " itc: " << itc << ", \tCALV = " << std::setw(5) << getCalibration(itc)
                  << ", \tMUX = " << std::setw(2) << getInputMux(itc) << std::endl;
    }

  private:
    uint8_t density;  //lsb at the input TC from ROC
    uint8_t dropLSB;  //lsb at the output during the packing
    uint8_t
        select;  //0 = Threshold Sum (TS), 1 = Super Trigger Cell (STC), 2 = Best Choice (BC), 3 = Repeater, 4=Autoencoder (AE).
    uint8_t stc_type;       //0 = STC4B(5E+4M), 1 = STC16(5E+4M), 2 = CTC4A(4E+3M), 3 = STC4A(4E+3M), 4 = CTC4B(5E+3M)
    uint8_t eporttx_numen;  //number of elinks
    uint16_t calv[48];      //12-bit calibration for 48 TCs
    uint8_t mux[48];        //multiplexer between HGCROC and TC to ECONT
    bool usesum;            //true: total of all TCs, false: (total-sumofselectedTcs)
  };
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  uint16_t getExpEcontHeader(uint16_t BXSLink,
                             TPGFEDataformat::Type type) {  //expected ECONT header from Slink BX counter
    uint16_t expEcontHeader = 0;
    uint16_t maxMSB = 0;
    uint16_t modEcontHeader = 1;

    if (type == TPGFEDataformat::TS) {  // 5 bits ECONT header
      maxMSB = 31;
      modEcontHeader = 16;
    } else {  // 4 bits ECONT header
      maxMSB = 15;
      modEcontHeader = 8;
    }

    if (BXSLink == 1) {
      expEcontHeader = maxMSB;
    } else
      expEcontHeader = ((BXSLink - 1) % modEcontHeader);
    return expEcontHeader;
  }
}  // namespace TPGFEConfiguration
#endif
