#ifndef Stage1IOFwCfg_h
#define Stage1IOFwCfg_h

#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cstring>
#include <cassert>

#include "TPGFEDataformat.hh"

namespace TPGStage1Emulation {

  class Stage1IOFwCfg {
  public:
    enum { MaximumLpgbtPairs = 64, MaximumUnpackersPerLpgbtPair = 6 };

    Stage1IOFwCfg() {
      // Defaults here temporarily
      for (unsigned lp(0); lp < MaximumLpgbtPairs; lp++) {
        if (lp == 10 || lp == 11 || lp == 14 || lp == 15 || lp == 16 || lp == 17) {
          for (unsigned j(0); j < MaximumUnpackersPerLpgbtPair; j++) {
            _unpackerType[lp][j] = TPGFEDataformat::Unknown;
            _unpackerElink0[lp][j] = 14;
            _unpackerElinks[lp][j] = 0;
            _unpackerTCs[lp][j] = 0;
            //_unpackerStreams[lp][j]=0;
          }

        } else {
          _unpackerType[lp][0] = TPGFEDataformat::BestC;
          _unpackerElink0[lp][0] = 0;
          _unpackerElinks[lp][0] = 4;
          _unpackerTCs[lp][0] = 9;
          //_unpackerStreams[lp][0]=2;

          _unpackerType[lp][1] = TPGFEDataformat::BestC;
          _unpackerElink0[lp][1] = 4;
          _unpackerElinks[lp][1] = 2;
          _unpackerTCs[lp][1] = 4;
          //_unpackerStreams[lp][1]=1;

          _unpackerType[lp][2] = TPGFEDataformat::STC4A;
          _unpackerElink0[lp][2] = 6;
          _unpackerElinks[lp][2] = 4;
          _unpackerTCs[lp][2] = 12;
          //_unpackerStreams[lp][2]=2;

          _unpackerType[lp][3] = TPGFEDataformat::STC16;
          _unpackerElink0[lp][3] = 10;
          _unpackerElinks[lp][3] = 2;
          _unpackerTCs[lp][3] = 3;
          //_unpackerStreams[lp][3]=1;

          _unpackerType[lp][4] = TPGFEDataformat::STC16;
          _unpackerElink0[lp][4] = 12;
          _unpackerElinks[lp][4] = 2;
          _unpackerTCs[lp][4] = 3;
          //_unpackerStreams[lp][4]=1;

          _unpackerType[lp][5] = TPGFEDataformat::Unknown;
          _unpackerElink0[lp][5] = 14;
          _unpackerElinks[lp][5] = 0;
          _unpackerTCs[lp][5] = 0;
          //_unpackerStreams[lp][5]=0;
        }
      }
    }

    TPGFEDataformat::Type type(unsigned lp, unsigned up) const {
      assert(lp < MaximumLpgbtPairs);
      assert(up < MaximumUnpackersPerLpgbtPair);

      return _unpackerType[lp][up];
    }

    unsigned numberOfTCs(unsigned lp, unsigned up) const {
      assert(lp < MaximumLpgbtPairs);
      assert(up < MaximumUnpackersPerLpgbtPair);

      return _unpackerTCs[lp][up];
    }

    unsigned firstElink(unsigned lp, unsigned up) const {
      assert(lp < MaximumLpgbtPairs);
      assert(up < MaximumUnpackersPerLpgbtPair);

      return _unpackerElink0[lp][up];
    }

    unsigned numberOfElinks(unsigned lp, unsigned up) const {
      assert(lp < MaximumLpgbtPairs);
      assert(up < MaximumUnpackersPerLpgbtPair);

      return _unpackerElinks[lp][up];
    }

    bool connected(unsigned lp, unsigned up) const {
      assert(lp < MaximumLpgbtPairs);
      assert(up < MaximumUnpackersPerLpgbtPair);

      return _unpackerType[lp][up] != TPGFEDataformat::Unknown;
    }

    unsigned unpackerStreams(unsigned lp, unsigned up) const {
      assert(lp < MaximumLpgbtPairs);
      assert(up < MaximumUnpackersPerLpgbtPair);

      if (_unpackerType[lp][up] == TPGFEDataformat::Unknown)
        return 0;
      return _unpackerTCs[lp][up] < 8 ? 1 : 2;
    }

    void print() const {
      std::cout << "Stage1IOFwCfg(" << this << ")::print()" << std::endl;
      for (unsigned lp(0); lp < MaximumLpgbtPairs; lp++) {
        std::cout << "LpGBT pair " << std::setw(2) << lp << std::endl;
        for (unsigned up(0); up < MaximumUnpackersPerLpgbtPair; up++) {
          std::cout << " Type " << TPGFEDataformat::tctypeName[_unpackerType[lp][up]] << ", number of elinks "
                    << _unpackerElinks[lp][up] << ", first elink " << std::setw(2) << _unpackerElink0[lp][up]
                    << ", last elink " << std::setw(2) << _unpackerElink0[lp][up] + _unpackerElinks[lp][up] - 1
                    << ", number of TCs " << std::setw(2) << _unpackerTCs[lp][up] << ", number of streams "
                    << unpackerStreams(lp, up) << std::endl;
        }
      }
    }

  protected:
    //unsigned _unpackersPerLpgbtPair[MaximumLpgbtPairs];
    TPGFEDataformat::Type _unpackerType[MaximumLpgbtPairs][MaximumUnpackersPerLpgbtPair];
    unsigned _unpackerElink0[MaximumLpgbtPairs][MaximumUnpackersPerLpgbtPair];
    unsigned _unpackerElinks[MaximumLpgbtPairs][MaximumUnpackersPerLpgbtPair];
    unsigned _unpackerTCs[MaximumLpgbtPairs][MaximumUnpackersPerLpgbtPair];
    //unsigned _unpackerStreams[MaximumLpgbtPairs][MaximumUnpackersPerLpgbtPair];
  };

}  // namespace TPGStage1Emulation
#endif
