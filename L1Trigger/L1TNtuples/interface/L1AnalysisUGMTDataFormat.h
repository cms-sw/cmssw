#ifndef __L1Analysis_L1AnalysisUGMTDataFormat_H__
#define __L1Analysis_L1AnalysisUGMTDataFormat_H__

#include <vector>
#include <utility>
#include <map>

namespace L1Analysis
{
  enum tftype {
    bmtf = 0, omtf, emtf
  };

  class L1AnalysisRegMuonDataFormat {
  public:
    L1AnalysisRegMuonDataFormat() {};
    explicit L1AnalysisRegMuonDataFormat(size_t nMax) {
      pt.reserve(nMax);
      eta.reserve(nMax);
      phi.reserve(nMax);
      qual.reserve(nMax);
      ch.reserve(nMax);
      bx.reserve(nMax);
      trAddress.reserve(nMax);
      processor.reserve(nMax);

      packedPt.reserve(nMax);
      packedEta.reserve(nMax);
      packedPhi.reserve(nMax);
    };
    virtual ~L1AnalysisRegMuonDataFormat() {};

    void ResetRegional() {
      pt.clear();
      eta.clear();
      phi.clear();
      qual.clear();
      ch.clear();
      bx.clear();
      trAddress.clear();
      processor.clear();

      packedPt.clear();
      packedEta.clear();
      packedPhi.clear();

      n = 0;
    }

    // member data (public to be compatible with legacy)
    int n;

    std::vector<float> pt;
    std::vector<float> eta;
    std::vector<float> phi;
    std::vector<float> qual;
    std::vector<float> ch;
    std::vector<int> bx;
    std::vector<int> trAddress;
    std::vector<int> processor;

    std::vector<int> packedPt;
    std::vector<int> packedEta;
    std::vector<int> packedPhi;
  };

  class L1AnalysisMuTwrDataFormat {
  public:
    L1AnalysisMuTwrDataFormat() {};
    // explicit L1AnalysisMuTwrDataFormat(size_t nMax) {
    //   pt.reserve(nMax);
    //   eta.reserve(nMax);
    //   phi.reserve(nMax);

    //   packedPt.reserve(nMax);
    //   packedEta.reserve(nMax);
    //   packedPhi.reserve(nMax);
    // };
    virtual ~L1AnalysisMuTwrDataFormat() {};

    void Reset() {
      // pt.clear();
      // eta.clear();
      // phi.clear();

      packedPt.clear();
      packedEta.clear();
      packedPhi.clear();
      bx.clear();

      n = 0;
    }

    // member data (public to be compatible with legacy)
    int n;

    // std::vector<float> pt;
    // std::vector<float> eta;
    // std::vector<float> phi;

    std::vector<int> packedPt;
    std::vector<int> packedEta;
    std::vector<int> packedPhi;
    std::vector<int> bx;
  };

  struct TFLink {
    TFLink(tftype tft, int index) : tf(tft), idx(index) {};
    TFLink() : tf(tftype::bmtf), idx(-1) {};
    tftype tf;
    int idx;
  };

  class L1AnalysisUGMTDataFormat
  {
  public:
    L1AnalysisUGMTDataFormat() {};
    explicit L1AnalysisUGMTDataFormat(size_t nMax) {
      relIso.reserve(nMax);
      absIso.reserve(nMax);
      rank.reserve(nMax);
      isoEnergy.reserve(nMax);
      packedIso.reserve(nMax);
      tfLink.reserve(nMax);
      pt.reserve(nMax);
      eta.reserve(nMax);
      phi.reserve(nMax);
      qual.reserve(nMax);
      ch.reserve(nMax);
      bx.reserve(nMax);

      packedPt.reserve(nMax);
      packedEta.reserve(nMax);
      packedPhi.reserve(nMax);

      tfInfo.emplace(tftype::bmtf, L1AnalysisRegMuonDataFormat(108));
      tfInfo.emplace(tftype::omtf, L1AnalysisRegMuonDataFormat(108));
      tfInfo.emplace(tftype::emtf, L1AnalysisRegMuonDataFormat(108));;
    };
    ~L1AnalysisUGMTDataFormat() {};

    void Reset()
    {
      pt.clear();
      eta.clear();
      phi.clear();
      qual.clear();
      ch.clear();
      bx.clear();

      packedPt.clear();
      packedEta.clear();
      packedPhi.clear();

      n = 0;

      nBmtf = 0;
      nEmtf = 0;
      nOmtf = 0;

      relIso.clear();
      absIso.clear();
      rank.clear();
      isoEnergy.clear();
      packedIso.clear();
      tfLink.clear();

      tfInfo.at(bmtf).ResetRegional();
      tfInfo.at(emtf).ResetRegional();
      tfInfo.at(omtf).ResetRegional();
    }

    // member data (public to be compatible with legacy)
    int n;

    std::vector<float> pt;
    std::vector<float> eta;
    std::vector<float> phi;
    std::vector<float> qual;
    std::vector<float> ch;
    std::vector<int> bx;

    std::vector<int> packedPt;
    std::vector<int> packedEta;
    std::vector<int> packedPhi;
    int nBmtf;
    int nEmtf;
    int nOmtf;

    std::vector<bool> isFinal;

    std::vector<int> relIso;
    std::vector<int> absIso;
    std::vector<int> rank;
    std::vector<int> isoEnergy;

    std::vector<int> packedIso;

    std::vector<TFLink> tfLink;
    std::map<L1Analysis::tftype, L1AnalysisRegMuonDataFormat> tfInfo;

  };
}
#endif


