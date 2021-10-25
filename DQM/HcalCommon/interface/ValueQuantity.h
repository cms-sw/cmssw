#ifndef ValueQuantity_h
#define ValueQuantity_h

#include "DQM/HcalCommon/interface/Flag.h"
#include "DQM/HcalCommon/interface/Quantity.h"
#include <unordered_map>

namespace hcaldqm {
  namespace quantity {
    enum ValueQuantityType {
      fN,
      fEvents,
      fEnergy,
      fTiming_ns,
      fADC_128,
      fADC_5,
      fADC_15,
      ffC_10000,
      ffC_1000,
      ffC_3000,
      fTiming_TS,
      fTiming_TS200,
      fLS,
      fEt_256,
      fEt_128,
      fFG,
      fRatio,
      fDigiSize,
      fAroundZero,
      fRatio2,
      fdEtRatio,
      fSumdEt,
      fTiming_100TS,
      fQIE10ADC_256,
      fQIE10TDC_64,
      fQIE10TDC_16,
      fQIE10TDC_4,
      fDiffAbs,
      fRatio_0to2,
      fN_to8000,
      fEnergyTotal,
      fN_m0to10000,
      fEtCorr_256,
      fEtCorr_data,
      fEtCorr_emul,
      fADCCorr_128,
      fBX,
      fEnergy_1TeV,
      fState,
      fQIE10fC_400000,
      fDataSize,
      fQIE10fC_2000,
      fQIE10fC_10000,
      fQIE8fC_1000_50,
      fTime_ns_250,
      fADC_256,
      ffC_generic_10000,
      ffC_generic_400000,
      fQIE10ADC_16,
      fDualAnodeAsymmetry,
      fTimingRatio,
      fQIE10fC_100000Coarse,
      fBadTDC,
      fRBX,
      fTimingDiff_ns,
      ffC_1000000,
      fTime_ns_250_coarse,
      fCapidMinusBXmod4,
      fBX_36,
      fADC_256_4,  // ADC from 0-255, with 4 ADC granularity
    };
    const std::map<ValueQuantityType, std::string> name_value = {
        {fN, "N"},
        {fEvents, "Events"},
        {fEnergy, "Energy"},
        {fTiming_ns, "Timing"},
        {fADC_5, "ADC (QIE8)"},
        {fADC_15, "ADC (QIE8)"},
        {fADC_128, "ADC (QIE8)"},
        {fADC_256, "ADC"},
        {fQIE10ADC_256, "ADC (QIE10/11)"},
        {fQIE10ADC_16, "ADC (QIE10/11)"},
        {ffC_1000, "fC (QIE8)"},
        {ffC_3000, "fC (QIE8)"},
        {ffC_10000, "fC (QIE8)"},
        {fQIE8fC_1000_50, "fC"},
        {fQIE10fC_2000, "fC"},
        {fQIE10fC_10000, "fC"},
        {fQIE10fC_400000, "fC"},
        {ffC_generic_10000, "fC (QIE8/10/11)"},
        {ffC_generic_400000, "fC (QIE8/10/11)"},
        {fTiming_TS, "Timing"},
        {fTiming_TS200, "Timing"},
        {fLS, "LS"},
        {fEt_256, "Et"},
        {fEt_128, "Et"},
        {fFG, "FG"},
        {fRatio, "Ratio"},
        {fDigiSize, "DigiSize"},
        {fAroundZero, "Q"},
        {fRatio2, "Ratio"},
        {fdEtRatio, "dEtRatio"},
        {fSumdEt, "SumdEt"},
        {fTiming_100TS, "Timing"},
        {fQIE10TDC_64, "TDC"},
        {fQIE10TDC_16, "TDC"},
        {fQIE10TDC_4, "TDC"},
        {fDiffAbs, "Q"},
        {fRatio_0to2, "Ratio"},
        {fN_to8000, "N"},
        {fEnergyTotal, "Energy"},
        {fN_m0to10000, "N"},
        {fEtCorr_256, "Et"},
        {fEtCorr_data, "256*TS + Et (Data TP)"},
        {fEtCorr_emul, "256*TS + Et (Emul TP)"},
        {fADCCorr_128, "ADC"},
        {fBX, "BX"},
        {fEnergy_1TeV, "Energy"},
        {fState, "State"},
        {fDataSize, "FED Data Size (kB)"},
        {fTime_ns_250, "Time (ns)"},
        {fDualAnodeAsymmetry, "(q_{1}-q_{2})/(q_{1}+q_{2})"},
        {fTimingRatio, "q_{SOI+1}/q_{SOI}"},
        {fQIE10fC_100000Coarse, "fC"},
        {fBadTDC, "TDC"},
        {fRBX, "RBX"},
        {fTimingDiff_ns, "#Delta timing [ns]"},
        {ffC_1000000, "fC"},
        {fTime_ns_250_coarse, "Time (ns)"},
        {fCapidMinusBXmod4, "(CapId - BX) % 4"},
        {fBX_36, "BX"},
        {fADC_256_4, "ADC"},
    };
    const std::map<ValueQuantityType, double> min_value = {
        {fN, -0.05},
        {fEvents, 0},
        {fEnergy, 0},
        {fTiming_ns, -50},
        {fADC_128, -0.5},
        {fADC_5, -0.5},
        {fADC_15, -0.5},
        {ffC_10000, 0},
        {ffC_1000, 0},
        {ffC_3000, 0},
        {fTiming_TS, -0.5},
        {fTiming_TS200, -0.5},
        {fLS, 0.5},
        {fEt_256, 0},
        {fEt_128, 0},
        {fFG, 0},
        {fRatio, 0},
        {fDigiSize, -0.5},
        {fAroundZero, -1},
        {fRatio2, 0.5},
        {fdEtRatio, 0},
        {fSumdEt, 0},
        {fTiming_100TS, -0.5},
        {fQIE10ADC_256, -0.5},
        {fQIE10ADC_16, -0.5},
        {fQIE10TDC_64, -0.5},
        {fQIE10TDC_16, -0.5},
        {fQIE10TDC_4, -0.5},
        {fDiffAbs, 0},
        {fRatio_0to2, 0},
        {fN_to8000, 0},
        {fEnergyTotal, 0},
        {fN_m0to10000, -0.05},
        {fEtCorr_256, -1},
        {fEtCorr_data, -4},
        {fEtCorr_emul, -4},
        {fADCCorr_128, -2},
        {fBX, -0.5},
        {fEnergy_1TeV, 0},
        {fState, flag::fNA},
        {fQIE10fC_400000, 0},
        {fDataSize, 0},
        {fQIE10fC_2000, 0},
        {fQIE10fC_10000, 0},
        {fQIE8fC_1000_50, 0},
        {fTime_ns_250, -0.5},
        {fADC_256, -0.5},
        {ffC_generic_10000, 0.},
        {ffC_generic_400000, 0.},
        {fDualAnodeAsymmetry, -1.},
        {fTimingRatio, 0.},
        {fQIE10fC_100000Coarse, 0},
        {fBadTDC, 49.5},
        {fRBX, 0.5},
        {fTimingDiff_ns, -125.},
        {ffC_1000000, 0.},
        {fTime_ns_250_coarse, -0.5},
        {fCapidMinusBXmod4, -0.5},
        {fBX_36, -0.5},
        {fADC_256_4, -0.5},
    };
    const std::map<ValueQuantityType, double> max_value = {
        {fN, 1000},
        {fEvents, 1000},
        {fEnergy, 200},
        {fTiming_ns, 50},
        {fADC_128, 127.5},
        {fADC_5, 5},
        {fADC_15, 15},
        {ffC_10000, 10000},
        {ffC_1000, 1000},
        {ffC_3000, 3000},
        {fTiming_TS, 9.5},
        {fTiming_TS200, 9.5},
        {fLS, 4000.5},
        {fEt_256, 255.5},
        {fEt_128, 255.5},
        {fFG, 2},
        {fRatio, 1},
        {fDigiSize, 20.5},
        {fAroundZero, 1},
        {fRatio2, 1.5},
        {fdEtRatio, 1},
        {fSumdEt, 1000},
        {fTiming_100TS, 99.5},
        {fQIE10ADC_256, 255.5},
        {fQIE10ADC_16, 15.5},
        {fQIE10TDC_64, 63.5},
        {fQIE10TDC_16, 15.5},
        {fQIE10TDC_4, 3.5},
        {fDiffAbs, 1},
        {fRatio_0to2, 2},
        {fN_to8000, 8000},
        {fEnergyTotal, 100000},
        {fN_m0to10000, 10000},
        {fEtCorr_256, 257},
        {fEtCorr_data, 1028},
        {fEtCorr_emul, 1028},
        {fADCCorr_128, 128},
        {fBX, 3600.5},
        {fEnergy_1TeV, 1000},
        {fState, flag::nState},
        {fQIE10fC_400000, 400000},
        {fDataSize, 100},
        {fQIE10fC_2000, 2000},
        {fQIE10fC_10000, 10000},
        {fQIE8fC_1000_50, 1000},
        {fTime_ns_250, 249.5},
        {fADC_256, 255.5},
        {ffC_generic_10000, 10000},
        {ffC_generic_400000, 400000},
        {fDualAnodeAsymmetry, 1.},
        {fTimingRatio, 2.5},
        {fQIE10fC_100000Coarse, 100000},
        {fBadTDC, 61.5},
        {fRBX, 18.5},
        {fTimingDiff_ns, 125.},
        {ffC_1000000, 1.e6},
        {fTime_ns_250_coarse, 249.5},
        {fCapidMinusBXmod4, 3.5},
        {fBX_36, 3564. - 0.5},
        {fADC_256_4, 255},
    };
    const std::map<ValueQuantityType, int> nbins_value = {
        {fN, 200},
        {fEvents, 200},
        {fEnergy, 100},
        {fTiming_ns, 200},
        {fADC_128, 128},
        {fADC_5, 100},
        {fADC_15, 300},
        {ffC_10000, 1000},
        {ffC_1000, 200},
        {ffC_3000, 600},
        {fTiming_TS, 10},
        {fTiming_TS200, 200},
        {fLS, 4000},
        {fEt_256, 256},
        {fEt_128, 128},
        {fFG, 2},
        {fRatio, 100},
        {fDigiSize, 21},
        {fAroundZero, 100},
        {fRatio2, 100},
        {fdEtRatio, 100},
        {fSumdEt, 100},
        {fTiming_100TS, 100},
        {fQIE10ADC_256, 256},
        {fQIE10TDC_64, 64},
        {fQIE10TDC_16, 16},
        {fQIE10TDC_4, 4},
        {fDiffAbs, 200},
        {fRatio_0to2, 100},
        {fN_to8000, 8000},
        {fEnergyTotal, 500},
        {fN_m0to10000, 100},
        {fEtCorr_256, 258},
        {fEtCorr_data, 258},
        {fEtCorr_emul, 258},
        {fADCCorr_128, 130},
        {fBX, 3601},
        {fEnergy_1TeV, 200},
        {fState, flag::nState},
        {fQIE10fC_400000, 1000},
        {fDataSize, 100},
        {fQIE10fC_2000, 100},
        {fQIE10fC_10000, 500},
        {fQIE8fC_1000_50, 50},
        {fTime_ns_250, 250},
        {fADC_256, 256},
        {ffC_generic_10000, 10000},
        {ffC_generic_400000, 10000},
        {fDualAnodeAsymmetry, 40},
        {fTimingRatio, 50},
        {fQIE10fC_100000Coarse, 1000},
        {fBadTDC, 12},
        {fRBX, 18},
        {fTimingDiff_ns, 40},
        {ffC_1000000, 1000},
        {fTime_ns_250_coarse, 100},
        {fCapidMinusBXmod4, 4},
        {fBX_36, 99},
        {fADC_256_4, 64},
    };
    class ValueQuantity : public Quantity {
    public:
      ValueQuantity() : _type() {}
      ValueQuantity(ValueQuantityType type, bool isLog = false) : Quantity(name_value.at(type), isLog), _type(type) {}
      ~ValueQuantity() override {}

      ValueQuantity *makeCopy() override { return new ValueQuantity(_type, _isLog); }

      //	get Value to be overriden
      int getValue(int x) override {
        int ret_x = x;
        if (_showOverflow) {
          if (x < min()) {
            ret_x = min();
          } else if (x > max()) {
            ret_x = max();
          }
        }
        return ret_x;
      }
      double getValue(double x) override {
        double ret_x = x;
        if (_showOverflow) {
          if (x < min()) {
            ret_x = min();
          } else if (x > max()) {
            ret_x = max();
          }
        }
        return ret_x;
      }

      //	standard properties
      QuantityType type() override { return fValueQuantity; }
      int nbins() override { return nbins_value.at(_type); }
      double min() override { return min_value.at(_type); }
      double max() override { return max_value.at(_type); }

      void setBits(TH1 *o) override {
        Quantity::setBits(o);
        setLS(o);
      }
      virtual void setLS(TH1 *o) {
        if (_type == fLS) {
          //	for LS axis - set the bit
          //	set extendable axes.
          o->SetBit(BIT(constants::BIT_OFFSET + constants::BIT_AXIS_LS));
          //				o->SetCanExtend(TH1::kXaxis);
        }
      }

    protected:
      ValueQuantityType _type;
    };

    class FlagQuantity : public ValueQuantity {
    public:
      FlagQuantity() {}
      FlagQuantity(std::vector<flag::Flag> const &flags) : _flags(flags) {}
      ~FlagQuantity() override {}

      FlagQuantity *makeCopy() override { return new FlagQuantity(_flags); }

      std::string name() override { return "Flag"; }
      int nbins() override { return _flags.size(); }
      double min() override { return 0; }
      double max() override { return _flags.size(); }
      int getValue(int f) override { return f; }
      uint32_t getBin(int f) override { return f + 1; }
      std::vector<std::string> getLabels() override {
        std::vector<std::string> vnames;
        for (std::vector<flag::Flag>::const_iterator it = _flags.begin(); it != _flags.end(); ++it)
          vnames.push_back(it->_name);

        return vnames;
      }

    protected:
      std::vector<flag::Flag> _flags;
    };

    class LumiSection : public ValueQuantity {
    public:
      LumiSection() : ValueQuantity(fLS), _n(4000) {}
      LumiSection(int n) : ValueQuantity(fLS), _n(n) {}
      ~LumiSection() override {}

      LumiSection *makeCopy() override { return new LumiSection(_n); }

      std::string name() override { return "LS"; }
      int nbins() override { return _n; }
      double min() override { return 1; }
      double max() override { return _n + 1; }
      int getValue(int l) override { return l; }
      uint32_t getBin(int l) override { return getValue(l); }
      void setMax(double x) override { _n = x; }

    protected:
      int _n;
    };

    /**
 * Coarse LumiSection axis. Specify binning (default=10 LS)
 */
    class LumiSectionCoarse : public ValueQuantity {
    public:
      LumiSectionCoarse() : ValueQuantity(fLS), _n(4000), _binning(10) {}
      LumiSectionCoarse(int n, int binning) : ValueQuantity(fLS), _n(n), _binning(binning) {}
      ~LumiSectionCoarse() override {}

      LumiSectionCoarse *makeCopy() override { return new LumiSectionCoarse(_n, _binning); }

      std::string name() override { return "LS"; }
      int nbins() override { return (_n + _binning - 1) / _binning; }
      double min() override { return 1; }
      double max() override { return _n + 1; }
      int getValue(int l) override { return l; }
      uint32_t getBin(int l) override { return (l + _binning - 1) / _binning; }
      void setMax(double x) override { _n = x; }

    protected:
      int _n;
      int _binning;
    };

    class RunNumber : public ValueQuantity {
    public:
      RunNumber() {}
      RunNumber(std::vector<int> runs) : _runs(runs) {}
      ~RunNumber() override {}

      std::string name() override { return "Run"; }
      int nbins() override { return _runs.size(); }
      double min() override { return 0; }
      double max() override { return _runs.size(); }
      std::vector<std::string> getLabels() override {
        char name[10];
        std::vector<std::string> labels;
        for (uint32_t i = 0; i < _runs.size(); i++) {
          sprintf(name, "%d", _runs[i]);
          labels.push_back(name);
        }
        return labels;
      }
      int getValue(int run) override {
        int ir = -1;
        for (uint32_t i = 0; i < _runs.size(); i++)
          if (_runs[i] == run) {
            ir = (int)i;
            break;
          }

        if (ir == -1)
          throw cms::Exception("HCALDQM") << "run number doens't exist " << run;

        return ir;
      }

      uint32_t getBin(int run) override { return (this->getValue(run) + 1); }

    protected:
      std::vector<int> _runs;
    };

    class EventNumber : public ValueQuantity {
    public:
      EventNumber() {}
      EventNumber(int nevents) : ValueQuantity(fN), _nevents(nevents) {}
      ~EventNumber() override {}

      std::string name() override { return "Event"; }
      int nbins() override { return _nevents; }
      double min() override { return 0.5; }
      double max() override { return _nevents + 0.5; }

    protected:
      int _nevents;
    };

    class EventType : public ValueQuantity {
    public:
      EventType() {}
      EventType(std::vector<uint32_t> const &vtypes) : ValueQuantity(fN) { this->setup(vtypes); }
      ~EventType() override {}

      virtual void setup(std::vector<uint32_t> const &vtypes) {
        std::cout << "SIZE = " << vtypes.size() << std::endl;
        for (uint32_t i = 0; i < vtypes.size(); i++)
          _types.insert(std::make_pair((uint32_t)vtypes[i], i));
      }
      int getValue(int v) override { return _types[(uint32_t)v]; }
      uint32_t getBin(int v) override { return getValue(v) + 1; }

      int nbins() override { return _types.size(); }
      double min() override { return 0; }
      double max() override { return _types.size(); }
      std::string name() override { return "Event Type"; }

    protected:
      typedef std::unordered_map<uint32_t, int> TypeMap;
      TypeMap _types;

    public:
      std::vector<std::string> getLabels() override {
        std::vector<std::string> labels(_types.size());
        std::cout << "SIZE = " << _types.size() << std::endl;
        for (auto const &v : _types) {
          labels[v.second] = utilities::ogtype2string((constants::OrbitGapType)v.first);
        }
        return labels;
      }
      EventType *makeCopy() override {
        std::vector<uint32_t> vtypes;
        for (auto const &p : _types) {
          vtypes.push_back(p.first);
        }

        std::sort(vtypes.begin(), vtypes.end());
        return new EventType(vtypes);
      }
    };
  }  // namespace quantity
}  // namespace hcaldqm

#endif
