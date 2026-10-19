#ifndef TPGBEDataformat_h
#define TPGBEDataformat_h

#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cstring>
#include <cassert>

namespace TPGBEDataformat {

  class UnpackerOutputStreamPair {
  public:
    enum { NumberOfTCs = 7 };

    UnpackerOutputStreamPair() : _moduleId(0xdeadbeef) { setZero(); }

    static uint32_t unpack5E3MToUnsigned(uint16_t flt) {
      assert(flt < 0x100);

      uint32_t e((flt >> 3) & 0x1f);
      uint32_t m((flt) & 0x07);

      //return (e==0?m:(8+m)<<(e-1));
      if (e == 0)
        return m;
      else if (e == 1)
        return 8 + m;
      else
        return (16 + 2 * m + 1) << (e - 2);
    }

    static uint32_t unpack4E3MToUnsigned(uint16_t flt) {
      assert(flt < 0x80);

      uint32_t e((flt >> 3) & 0x0f);
      uint32_t m((flt) & 0x07);

      if (e == 0)
        return m;
      else if (e == 1)
        return 8 + m;
      //else return (16+2*m+1)<<(e-2);
      else
        return (16 + 2 * m) << (e - 2);
    }

    static uint32_t unpack5E4MToUnsigned(uint16_t flt) {
      assert(flt < 0x200);

      uint32_t e((flt >> 4) & 0x1f);
      uint32_t m((flt) & 0x0f);

      //return (e==0?m:(16+m)<<(e-1));
      if (e == 0)
        return m;
      else if (e == 1)
        return 16 + m;
      //else return (32+2*m+1)<<(e-2);
      else
        return (32 + 2 * m) << (e - 2);
    }

    static uint16_t pack5E4MFromUnsigned(uint32_t erg) {
      if (erg < 16)
        return erg;

      unsigned e(1);
      while (erg >= 32) {
        e++;
        erg >>= 1;
      }
      return 16 * (e - 1) + erg;
    }

    static uint16_t pack5E4MFrom4E3M(uint8_t flt) {
      /*
    assert(flt<0x80);
    uint32_t e((flt>>3)&0xf);
    uint32_t m((flt   )&0x7);

    uint32_t u;
    if(e==0) u=m;
    else if(e==1) u=8+m;
    //else u=(16+2*m+1)<<(e-2);
    else u=(16+2*m  )<<(e-2);

    return pack5E4MFromUnsigned(u);
    */
      return pack5E4MFromUnsigned(unpack4E3MToUnsigned(flt));

      // if(flt<16) return flt;
      // uint16_t e54((flt>>3)-1);
      // uint16_t m54(2*(flt&0x7));
      // return e54<<4|m54;
    }

    void setZero() {
      _msData[0] = 0;
      _msData[1] = 0;
      std::memset(_tcData[0], 0, sizeof(uint16_t) * 2 * NumberOfTCs);
    }

    bool isDoubleStream() const { return isValidModuleSum(1); }

    bool checkFormat() const {
      for (unsigned s(0); s < 2; s++) {
        // Check unused bits
        if (isValidModuleSum(s)) {
          if ((_msData[s] & 0x4030) != 0x0030)
            return false;
        } else {
          if ((_msData[s] & 0x4030) != 0x0000)
            return false;
        }

        // Check channel number is in the right range
        for (unsigned i(0); i < NumberOfTCs; i++) {
          if (isValidChannel(s, i)) {
            if (channelNumber(s, i) >= 48)
              return false;
          }
        }
      }

      uint16_t cOld;
      bool validCh(true);

      unsigned sHi(isDoubleStream() ? 2 : 1);

      for (unsigned i(0); i < NumberOfTCs; i++) {
        for (unsigned s(0); s < sHi; s++) {
          //Discuss with Paul how to deal for CTC and STC
          if (isValidChannel(s, i)) {
            // Check channel number is in the right order
            //std::cout <<"cOld: "<<cOld << ", channelNumber(s,i): " << channelNumber(s,i) << std::endl;
            if (i == 0 && s == 0)
              cOld = channelNumber(s, i);
            else if (cOld >= channelNumber(s, i))
              return false;
            cOld = channelNumber(s, i);
          }
          // Check all valid channels are before all invalid channels
          if (validCh && !isValidChannel(s, i))
            validCh = false;
          if (!validCh && isValidChannel(s, i))
            return false;
        }
      }

      return true;
    }

    uint32_t getModuleId() const { return _moduleId; }

    void setModuleId(uint32_t m) { _moduleId = m; }

    uint16_t getMsData(unsigned s) const {
      assert(s < 2);
      return _msData[s];
    }

    uint16_t getTcData(unsigned s, unsigned n) const {
      assert(s < 2);
      return (n < NumberOfTCs ? _tcData[s][n] : 0x7fff);
    }

    const uint16_t* getTcData(unsigned s) const {
      assert(s < 2);
      return _tcData[s];
    }

    const uint16_t* getData(unsigned s) const {
      assert(s < 2);
      return &_msData[s];
    }

    void setTcData(unsigned s, unsigned n, uint16_t d) {
      assert(s < 2);
      if (n < NumberOfTCs)
        _tcData[s][n] = d;
    }

    uint16_t* setTcData(unsigned s) {
      assert(s < 2);
      return _tcData[s];
    }

    uint16_t* setMsData(unsigned s) {
      assert(s < 2);
      return &_msData[s];
    }

    void setMsData(unsigned s, uint16_t d) {
      assert(s < 2);
      _msData[s] = d;
    }

    // For use with STCs when module sum is invalid
    void setBx(uint8_t bx) {
      assert(bx < 0x10);
      _msData[0] = 0x8030 | bx;
      _msData[1] = 0x8030 | bx;
    }

    void setBx(unsigned s, uint8_t bx) {
      assert(s < 2);
      assert(bx < 0x10);
      _msData[s] = 0x8030 | bx;
    }

    void setModuleSum(uint8_t bx, uint8_t ms) {
      assert(bx < 0x10);
      _msData[0] = (0x8030 | ms << 6 | bx);
      _msData[1] = (0x8030 | ms << 6 | bx);
    }

    void setModuleSum(unsigned s, uint8_t bx, uint8_t ms) {
      assert(s < 2);
      assert(bx < 0x10);
      _msData[s] = (0x8030 | ms << 6 | bx);
    }

    void setTriggerCell(unsigned s, unsigned n, uint8_t c, uint16_t e) {
      assert(s < 2);
      assert(c < 48);
      assert(e < 0x200);
      _tcData[s][n] = (0x8000 | e << 6 | c);
    }

    bool isValidModuleSum(unsigned s) const {
      assert(s < 2);
      return (_msData[s] & 0x8000) > 0;
    }

    uint16_t moduleSum(unsigned s) const {
      assert(s < 2);
      return (_msData[s] & 0x3fc0) >> 6;
    }

    uint32_t unpackedModuleSum(unsigned s) const {
      assert(s < 2);
      return unpack5E3MToUnsigned(moduleSum(s));
    }

    uint8_t bx(unsigned s) const {
      assert(s < 2);
      return _msData[s] & 0x000f;
    }

    bool isValidChannel(unsigned s, unsigned n) const {
      assert(s < 2);
      return (n < NumberOfTCs ? (_tcData[s][n] & 0x8000) > 0 : false);
    }

    uint16_t channelEnergy(unsigned s, unsigned n) const {
      assert(s < 2);
      return (n < NumberOfTCs ? (_tcData[s][n] & 0x7fc0) >> 6 : 0);
    }

    uint32_t unpackedChannelEnergy(unsigned s, unsigned n) const {
      assert(s < 2);
      return unpack5E4MToUnsigned(channelEnergy(s, n));
    }

    uint16_t channelNumber(unsigned s, unsigned n) const {
      assert(s < 2);
      return (n < NumberOfTCs ? _tcData[s][n] & 0x003f : 63);
    }

    unsigned numberOfValidChannels() const {
      unsigned n(0);
      for (unsigned i(0); i < NumberOfTCs; i++) {
        if (isValidChannel(0, i))
          n++;
        if (isValidChannel(1, i))
          n++;
      }
      return n;
    }

    bool operator==(const UnpackerOutputStreamPair& that) const {
      if (_msData[0] != that._msData[0])
        return false;
      if (_msData[1] != that._msData[1])
        return false;

      for (unsigned i(0); i <= NumberOfTCs; i++) {
        if (_tcData[0][i] != that._tcData[0][i])
          return false;
        if (_tcData[1][i] != that._tcData[1][i])
          return false;
      }
      return true;
    }

    bool isEqualSTC(const UnpackerOutputStreamPair& that) const {
      if (bx(0) != that.bx(0) and isValidModuleSum(0) and that.isValidModuleSum(0))
        return false;
      if (bx(1) != that.bx(1) and isValidModuleSum(1) and that.isValidModuleSum(1))
        return false;
      for (unsigned i(0); i <= NumberOfTCs; i++) {
        if (_tcData[0][i] != that._tcData[0][i] and isValidChannel(0, i) and that.isValidChannel(0, i))
          return false;
        if (_tcData[1][i] != that._tcData[1][i] and isValidChannel(1, i) and that.isValidChannel(1, i))
          return false;
      }
      return true;
    }

    bool isEqualBC(const UnpackerOutputStreamPair& that) const {
      if (_msData[0] != that._msData[0] and isValidModuleSum(0) and that.isValidModuleSum(0))
        return false;
      if (_msData[1] != that._msData[1] and isValidModuleSum(1) and that.isValidModuleSum(1))
        return false;
      for (unsigned i(0); i <= NumberOfTCs; i++) {
        if (_tcData[0][i] != that._tcData[0][i] and isValidChannel(0, i) and that.isValidChannel(0, i))
          return false;
        if (_tcData[1][i] != that._tcData[1][i] and isValidChannel(1, i) and that.isValidChannel(1, i))
          return false;
      }
      return true;
    }

    void print() {
      std::cout << "UnpackerOutputStreamPair(" << this
                << ")::print(), format = " << (checkFormat() ? "  valid" : "invalid")
                << ", module ID = " << getModuleId() << ", number of valid channels = " << numberOfValidChannels()
                << std::endl;

      for (unsigned s(0); s < 2; s++) {
        std::cout << " Stream " << s << std::endl
                  << "  MS  " << " = 0x" << std::hex << ::std::setfill('0') << std::setw(4) << _msData[s] << std::dec
                  << ::std::setfill(' ') << ", " << (isValidModuleSum(s) ? "  valid" : "invalid")
                  << "  module,    sum packed = " << std::setw(5) << moduleSum(s) << ", unpacked = " << std::setw(10)
                  << unpackedModuleSum(s) << ",     BX = " << std::setw(2) << unsigned(bx(s)) << std::endl;

        for (unsigned i(0); i < NumberOfTCs; i++) {
          std::cout << "  TC " << i << " = 0x" << std::hex << ::std::setfill('0') << std::setw(4) << _tcData[s][i]
                    << std::dec << ::std::setfill(' ') << ", " << (isValidChannel(s, i) ? "  valid" : "invalid")
                    << " channel, energy packed = " << std::setw(5) << channelEnergy(s, i)
                    << ", unpacked = " << std::setw(10) << unpackedChannelEnergy(s, i) << ", number = " << std::setw(2)
                    << channelNumber(s, i) << std::endl;
        }
      }
    }

  private:
    uint32_t _moduleId;
    uint16_t _msData[2];
    uint16_t _tcData[2][NumberOfTCs];
  };

  class Stage1ToStage2Data {
  private:
    std::array<uint64_t, 108> data;

  public:
    // Constructor
    Stage1ToStage2Data() {
      // Initialize data array with zeros
      data.fill(0);
    }

    // Methods to get and set TC and pTT values of each word
    uint16_t getTC(int wordIndex, int tcIndex) const {
      int bitOffset = tcIndex * 15;
      return static_cast<uint16_t>((data[wordIndex] >> bitOffset) & 0x7FFF);
    }

    void setTC(int wordIndex, int tcIndex, uint16_t value) {
      int bitOffset = tcIndex * 15;
      uint64_t mask = 0x7FFFULL << bitOffset;
      data[wordIndex] &= ~mask;                                               // Clear existing bits
      data[wordIndex] |= (static_cast<uint64_t>(value) << bitOffset) & mask;  // Set new value
    }

    uint8_t getPTT(int wordIndex, int pttIndex) const {
      int bitOffset = 45 + pttIndex * 8;
      return static_cast<uint8_t>((data[wordIndex] >> bitOffset) & 0xFF);
    }

    void setPTT(int wordIndex, int pttIndex, uint8_t value) {
      int bitOffset = 45 + pttIndex * 8;
      uint64_t mask = 0xFFULL << bitOffset;
      data[wordIndex] &= ~mask;                                               // Clear existing bits
      data[wordIndex] |= (static_cast<uint64_t>(value) << bitOffset) & mask;  // Set new value
    }

    std::array<uint64_t, 108>& accessData() { return data; }
  };

  class Stage1ToStage2DataArray {
  public:
    Stage1ToStage2DataArray() {
      for (unsigned sector(0); sector < 3; sector++) {
        for (unsigned s1Board(0); s1Board < 14; s1Board++) {
          for (unsigned link(0); link < 6; link++) {
            _s1Vector[sector][s1Board].push_back(&(_dataArray[sector][s1Board][link]));

            if (link < 4)
              _s2Vector[sector].push_back(&(_dataArray[sector][s1Board][link]));
            else
              _s2Vector[sector].push_back(&(_dataArray[(sector + 1) % 3][s1Board][link]));
          }
        }
      }
    }

    std::vector<Stage1ToStage2Data*>& s1Vector(unsigned s, unsigned b) {
      assert(s < 3);
      assert(b < 14);
      return _s1Vector[s][b];
    }

    std::vector<Stage1ToStage2Data*>& s2Vector(unsigned s) {
      assert(s < 3);
      return _s2Vector[s];
    }

  private:
    Stage1ToStage2Data _dataArray[3][14][6];
    std::vector<Stage1ToStage2Data*> _s1Vector[3][14];
    std::vector<Stage1ToStage2Data*> _s2Vector[3];
  };

  class Trig24Data {
  public:
    Trig24Data() : nofElinks(0) {
      nofUnpkdWords[0] = 0;
      nofUnpkdWords[1] = 0;
    }
    void setNofElinks(uint32_t nelinks) {
      assert(nelinks <= 7);
      nofElinks = uint8_t(nelinks);
    }
    void setNofUnpkWords(uint32_t nwords) {
      assert(nwords <= 8);
      nofUnpkdWords[0] = uint8_t(nwords);
    }
    void setNofUnpkWords(uint32_t istrm, uint32_t nwords) {
      assert(nwords <= 8);
      nofUnpkdWords[istrm] = uint8_t(nwords);
    }
    void setElink(uint32_t ib, uint32_t iw, uint32_t val) {
      assert(ib < 7);
      assert(iw <= 7);
      elinks[ib][iw] = val;
    }
    void setUnpkWord(uint32_t ib, uint32_t iw, uint32_t val) {
      assert(ib < 7);
      assert(iw < 8);
      unpackedWords[ib][0][iw] = val;
    }
    void setUnpkWord(uint32_t ib, uint32_t istrm, uint32_t iw, uint32_t val) {
      assert(ib < 7);
      assert(istrm <= 2);
      assert(iw < 8);
      unpackedWords[ib][istrm][iw] = val;
    }
    void setSlinkBx(uint16_t bx) { bxId = bx; }

    const uint32_t getSlinkBx() const { return uint32_t(bxId); }
    uint32_t getNofElinks() const { return uint32_t(nofElinks); }
    uint32_t getNofUnpkWords(uint32_t istrm = 0) const { return uint32_t(nofUnpkdWords[istrm]); }
    uint32_t getElink(uint32_t ib, uint32_t iw) const { return elinks[ib][iw]; }
    uint32_t getUnpkWord(uint32_t ib, uint32_t iw, uint32_t istrm = 0) const { return unpackedWords[ib][istrm][iw]; }
    const uint32_t* getElinks(uint32_t ib) const { return elinks[ib]; }
    const uint32_t* getUnpkWords(uint32_t ib, uint32_t istrm = 0) const { return unpackedWords[ib][istrm]; }
    void getUnpkStream(uint32_t ib, TPGBEDataformat::UnpackerOutputStreamPair& up) {
      uint16_t* tc = up.setTcData(0);
      for (unsigned iw(0); iw < getNofUnpkWords(); iw++) {
        if (iw == 0) {
          uint16_t* ms = up.setMsData(0);
          *ms = getUnpkWord(ib, iw);
        } else {
          *(tc + iw - 1) = getUnpkWord(ib, iw);
        }
      }  //iw loop
      if (getNofUnpkWords(1) > 0) {
        tc = up.setTcData(1);
        for (unsigned iw(0); iw < getNofUnpkWords(1); iw++) {
          if (iw == 0) {
            uint16_t* ms = up.setMsData(1);
            *ms = getUnpkWord(ib, iw, 1);
          } else {
            *(tc + iw - 1) = getUnpkWord(ib, iw, 1);
          }
        }  //iw loop
      }  //if second stream exists
    }
    void print(uint32_t bxindex = 0) {
      for (unsigned ib(0); ib < 7; ib++) {
        if (ib != bxindex)
          continue;
        for (unsigned iel(0); iel < getNofElinks(); iel++)
          std::cout << " ib " << ib << ", iel  " << iel << ", elinks = 0x" << std::hex << ::std::setfill('0')
                    << std::setw(8) << getElink(ib, iel) << std::dec << std::endl;
      }

      std::cout << "NofUnpkdWords[0]: " << getNofUnpkWords(0) << ", NofUnpkdWords[1]: " << getNofUnpkWords(1)
                << std::endl;
      for (unsigned ib(0); ib < 7; ib++) {
        if (ib != bxindex)
          continue;
        TPGBEDataformat::UnpackerOutputStreamPair up;
        uint16_t* tc = up.setTcData(0);

        for (unsigned iw(0); iw < getNofUnpkWords(); iw++) {
          if (iw == 0) {
            uint16_t* ms = up.setMsData(0);
            *ms = getUnpkWord(ib, iw);
          } else {
            *(tc + iw - 1) = getUnpkWord(ib, iw);
          }
          std::cout << " ib " << ib << ", iw  " << iw << ", unpackedWords = 0x" << std::hex << ::std::setfill('0')
                    << std::setw(4) << getUnpkWord(ib, iw) << std::dec << std::endl;
        }  //iw loop
        if (getNofUnpkWords(1) == 0)
          up.print();

        if (getNofUnpkWords(1) > 0) {
          tc = up.setTcData(1);
          for (unsigned iw(0); iw < getNofUnpkWords(1); iw++) {
            if (iw == 0) {
              uint16_t* ms = up.setMsData(1);
              *ms = getUnpkWord(ib, iw, 1);
            } else {
              *(tc + iw - 1) = getUnpkWord(ib, iw, 1);
            }
            std::cout << " ib " << ib << ", iw  " << iw << ", unpackedWords = 0x" << std::hex << ::std::setfill('0')
                      << std::setw(4) << getUnpkWord(ib, iw, 1) << std::dec << std::endl;
          }  //iw loop
          up.print();
        }
      }  //ib loop
    }

  private:
    uint8_t nofElinks, nofUnpkdWords[2];
    uint16_t bxId;                    //from Slink trailer
    uint32_t elinks[7][7];            //the first 7 is for bx and second one for number of elinks
    uint32_t unpackedWords[7][2][8];  //7:bxs,2:stream,8:words per half
  };

  class TrigTCProcData {  //Todo this could do with a print method!
  public:
    TrigTCProcData() {
      for (uint32_t ib = 0; ib < 7; ib++)
        for (uint32_t ibin = 0; ibin < 9; ibin++)
          for (uint32_t iinst = 0; iinst < 2; iinst++)
            unpackedWords[ib][ibin][iinst] = 0;
    };
    //void initUnpkWords(){for (uint32_t ib=0;ib<7;ib++) for (uint32_t ibin=0;ibin<9;ibin++) for (uint32_t iinst=0;iinst<2;iinst++) unpackedWords[ib][ibin][iinst]=0;}
    void setUnpkWord(uint32_t ib, uint32_t ibin, uint32_t iinst, uint32_t val) {
      assert(ib < 7);
      assert(ibin < 9);
      assert(iinst < 2);
      unpackedWords[ib][ibin][iinst] = val;
    }
    int getUnpkWord(uint32_t ib, uint32_t ibin, uint32_t iinst) const {
      return unpkWordIsValid(ib, ibin, iinst) ? unpackedWords[ib][ibin][iinst] : -1;
    }
    bool unpkWordIsValid(uint32_t ib, uint32_t ibin, uint32_t iinst) const {
      if ((unpackedWords[ib][ibin][iinst] & 0x7FFF) != 0)
        return true;
      else if (((unpackedWords[ib][ibin][iinst] & 0x7FFF) == 0) && ibin == 0 && iinst == 0 &&
               ((unpackedWords[ib][ibin + 1][iinst] & 0x7FFF) != 0 ||
                (unpackedWords[ib][ibin][iinst + 1] & 0x7FFF) != 0))
        return true;
      else
        return false;
    }
    int getTDAQEntry(int ilp, int iecon) {
      if (ilp == 0)
        return 0;  //first lpGBT link: all 3 ECON-Ts in first TDAQ block
      else if (ilp == 1 && iecon < 2)
        return iecon;  //second lpGBT link: first ECON-T is in first TDAQ block, second ECON-T in 2nd TDAQ block; third ECON-T not read out
      else if (ilp == 2 && iecon < 2)
        return 2;  //third lpGBT link: has two ECON-Ts, both in the third TDAQ block
      else if (ilp == 3 && iecon < 2)
        return 2;  //fourth lpGBT link: two ECON-T, both in the third TDAQ block
      else
        return -1;
    }
    std::map<int, std::vector<std::pair<int, int>>> getWordAndColPerBin(int ilp, int iecon) {
      std::map<int, std::vector<std::pair<int, int>>> theTmpMap;
      std::vector<std::pair<int, int>> theTmpVec;
      if ((ilp == 0 || ilp == 2) &&
          iecon ==
              0) {  //first, third lpGBT link, first ECON-T: start at 1st word of the TDAQ block, has 9 bins (2 TC/bin in the first 2), 3 words in total
        theTmpMap.clear();
        for (int ibin = 0; ibin < 9; ibin++) {
          theTmpVec.resize(0);
          if (ibin == 0) {
            theTmpVec.push_back(std::make_pair(0, 0));
            theTmpVec.push_back(std::make_pair(0, 1));
          } else if (ibin == 1) {
            theTmpVec.push_back(std::make_pair(0, 2));
            theTmpVec.push_back(std::make_pair(0, 3));
          } else {
            theTmpVec.push_back(std::make_pair(std::floor((ibin - 2) / 4) + 1, (ibin - 2) % 4));
          }
          theTmpMap[ibin] = theTmpVec;
        }
      } else if (ilp == 0 &&
                 iecon <
                     3) {  //first lpGBT link, second and third ECON-T: read out in the next 2 words, 4 bins (so 1 word per module)
        theTmpMap.clear();
        for (int ibin = 0; ibin < 4; ibin++) {
          theTmpVec.resize(0);
          theTmpVec.push_back(std::make_pair(iecon + 2, ibin));  //2nd and 3d module have 4 bins, one entry each
          theTmpMap[ibin] = theTmpVec;
        }
      } else if (ilp == 2 &&
                 iecon < 2) {  //third lpGBT link only has two econTs. Second one: read out 1 word per module
        theTmpMap.clear();
        for (int ibin = 0; ibin < 4; ibin++) {
          theTmpVec.resize(0);
          theTmpVec.push_back(std::make_pair(iecon + 2, ibin));  //2nd module has 4 bins, one entry each
          theTmpMap[ibin] = theTmpVec;
        }
      } else if ((ilp == 1) &&
                 iecon ==
                     0) {  //second lpGBT link, first ECON-T: last words of the TDAQ block, 9 bins (2TC/bin in the first 2), 3 words in total
        theTmpMap.clear();
        for (int ibin = 0; ibin < 9; ibin++) {
          theTmpVec.resize(0);
          if (ibin == 0) {
            theTmpVec.push_back(std::make_pair(5, 0));
            theTmpVec.push_back(std::make_pair(5, 1));
          } else if (ibin == 1) {
            theTmpVec.push_back(std::make_pair(5, 2));
            theTmpVec.push_back(std::make_pair(5, 3));
          } else {
            theTmpVec.push_back(std::make_pair(std::floor((ibin - 2) / 4) + 6, (ibin - 2) % 4));
          }
          theTmpMap[ibin] = theTmpVec;
        }
      } else if ((ilp == 3) && iecon == 0) {  //fourth lpGBT link, first ECON-T: 1 word per module (4 bins read out)
        theTmpMap.clear();
        for (int ibin = 0; ibin < 4; ibin++) {
          theTmpVec.resize(0);
          theTmpVec.push_back(std::make_pair(4, ibin));  //3d module has 4 bins, one entry each
          theTmpMap[ibin] = theTmpVec;
        }
      } else if (ilp == 3 &&
                 iecon ==
                     1) {  //fourth lpGBT link, second ECON-T: last words of the TDAQ block, 9 bins (2TC/bin in the first 2), 3 words in total
        theTmpMap.clear();
        for (int ibin = 0; ibin < 9; ibin++) {
          theTmpVec.resize(0);
          if (ibin == 0) {
            theTmpVec.push_back(std::make_pair(5, 0));
            theTmpVec.push_back(std::make_pair(5, 1));
          } else if (ibin == 1) {
            theTmpVec.push_back(std::make_pair(5, 2));
            theTmpVec.push_back(std::make_pair(5, 3));
          } else {
            theTmpVec.push_back(std::make_pair(std::floor((ibin - 2) / 4) + 6, (ibin - 2) % 4));
          }
          theTmpMap[ibin] = theTmpVec;
        }
      } else if ((ilp == 1) &&
                 iecon ==
                     1) {  //second lpGBT link, second ECON-T: first words of the next TDAQ block, 9 bins, 3 words in total.
        theTmpMap.clear();
        for (int ibin = 0; ibin < 9; ibin++) {
          theTmpVec.resize(0);
          theTmpVec.push_back(std::make_pair(std::floor(ibin / 4), ibin % 4));
          theTmpMap[ibin] = theTmpVec;
        }
      }
      return theTmpMap;
    }

    void print(uint32_t bxindex = 0) {
      for (unsigned ib(0); ib < 7; ib++) {
        if (ib != bxindex)
          continue;
        for (unsigned ibin = 0; ibin < 9; ibin++) {
          for (unsigned iinst = 0; iinst < 2; iinst++) {
            if (unpkWordIsValid(ib, ibin, iinst))
              std::cout << " ib " << ib << ", bin number " << ibin << ", entry in bin" << iinst << ", unpackedWord = 0x"
                        << std::hex << ::std::setfill('0') << std::setw(4) << getUnpkWord(ib, ibin, iinst) << std::dec
                        << std::endl;
          }
        }
      }
    }

  private:
    uint32_t unpackedWords[7][9][2];  //7: bxs, 9: bins, 2: max entries per bin
  };

}  // namespace TPGBEDataformat

#endif
