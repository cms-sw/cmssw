#ifndef L1T_OmtfP1_AlgoMuon_H
#define L1T_OmtfP1_AlgoMuon_H

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/GoldenPatternBase.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/GoldenPatternResult.h"
#include <ostream>

class AlgoMuon {
public:
  AlgoMuon() {}

  AlgoMuon(const GoldenPatternResult& gpResult, GoldenPatternBase* gp, unsigned int refHitNumber, int bx = 0)
      : gpResultConstr(gpResult),
        goldenPaternConstr(gp),
        //m_q(gpResult.getFiredLayerCnt()),  //initial value of quality, can be altered later
        m_bx(bx),
        m_rhitNumb(refHitNumber) {}

  GoldenPatternBase* getGoldenPatern() const { return goldenPaternConstr; }

  ~AlgoMuon() {}

  //vertex-constrained golden pattern result
  const GoldenPatternResult& getGpResultConstr() const { return gpResultConstr; }

  const GoldenPatternResult& getGpResultUnconstr() const { return gpResultUnconstr; }

  void setGpResultUnconstr(const GoldenPatternResult& gpResultUnconstr) { this->gpResultUnconstr = gpResultUnconstr; }

  void setEta(int eta) { gpResultConstr.setEta(eta); }

  int getEtaHw() const { return gpResultConstr.getEta(); }

  unsigned int getRefHitNumber() const { return m_rhitNumb; }

  void setRefHitNumber(unsigned int aRefHitNum) { m_rhitNumb = aRefHitNum; }

  int getRefLayer() const { return gpResultConstr.getRefLayer(); }

  int getBx() const { return m_bx; }

  //hardware pt
  int getPtConstr() const { return goldenPaternConstr == nullptr ? 0 : goldenPaternConstr->key().thePt; }

  //hardware upt, in the phase1 the upt scale unit is 1 GeV, while for the pt the unit is 0.5GeV
  int getPtUnconstr() const {
    return goldenPaternUnconstr == nullptr ? 0 : (goldenPaternUnconstr->key().thePt - 1) / 2 + 1;
  }

  int getChargeConstr() const { return goldenPaternConstr == nullptr ? -1 : goldenPaternConstr->key().theCharge; }

  int getPhiRHit() const { return gpResultConstr.getRefHitPhi(); }

  unsigned int getPatternNum() const;

  unsigned int getPatternNumConstr() const {
    return goldenPaternConstr == nullptr ? 0 : goldenPaternConstr->key().theNumber;
  }

  unsigned int getPatternNumUnconstr() const {
    return goldenPaternUnconstr == nullptr ? 0 : goldenPaternUnconstr->key().theNumber;
  }

  unsigned int getHwPatternNumConstr() const {
    return goldenPaternConstr == nullptr ? 0 : goldenPaternConstr->key().getHwPatternNumber();
  }

  unsigned int getHwPatternNumUnconstr() const {
    return goldenPaternUnconstr == nullptr ? 0 : goldenPaternUnconstr->key().getHwPatternNumber();
  }

  bool isValid() const {
    return (getPtConstr() > 0) || (getPtUnconstr() > 0);  //PtConstr == 0 denotes empty candidate
  }

  double getPdfSumConstr() const { return gpResultConstr.getPdfSum(); }

  double getPdfSum() const {
    return (gpResultUnconstr.getPdfSumUnconstr() > gpResultConstr.getPdfSum() ? gpResultUnconstr.getPdfSumUnconstr()
                                                                              : gpResultConstr.getPdfSum());
  }

  PdfValueType getDisc() const {
    return (gpResultUnconstr.getPdfSumUnconstr() > gpResultConstr.getPdfSum() ? gpResultUnconstr.getPdfSumUnconstr()
                                                                              : gpResultConstr.getPdfSum());
  }

  int getPhi() const {
    return (gpResultUnconstr.getPdfSumUnconstr() > gpResultConstr.getPdfSum() ? gpResultUnconstr.getPhi()
                                                                              : gpResultConstr.getPhi());
  }

  unsigned int getFiredLayerCnt() const {
    return (gpResultUnconstr.getPdfSumUnconstr() > gpResultConstr.getPdfSum() ? gpResultUnconstr.getFiredLayerCnt()
                                                                              : gpResultConstr.getFiredLayerCnt());
  }

  unsigned int getFiredLayerCntConstr() const { return gpResultConstr.getFiredLayerCnt(); }

  unsigned int getFiredLayerBits() const {
    return (gpResultUnconstr.getPdfSumUnconstr() > gpResultConstr.getPdfSum() ? gpResultUnconstr.getFiredLayerBits()
                                                                              : gpResultConstr.getFiredLayerBits());
  }

  int getQ() const {
    return (gpResultUnconstr.getPdfSumUnconstr() > gpResultConstr.getPdfSum() ? gpResultUnconstr.getFiredLayerCnt()
                                                                              : gpResultConstr.getFiredLayerCnt());
  }

  const StubResult& getStubResult(unsigned int iLayer) const { return gpResultConstr.getStubResults().at(iLayer); }

  const StubResults& getStubResultsConstr() const { return gpResultConstr.getStubResults(); }

  const bool isKilled() const { return killed; }

  void kill() { killed = true; }

  friend std::ostream& operator<<(std::ostream& out, const AlgoMuon& o);

  std::vector<std::shared_ptr<AlgoMuon>>& getKilledMuons() { return killedMuons; }

  GoldenPatternBase* getGoldenPaternUnconstr() const { return goldenPaternUnconstr; }

  void setGoldenPaternUnconstr(GoldenPatternBase* goldenPaternUnconstr) {
    this->goldenPaternUnconstr = goldenPaternUnconstr;
  }

  int getChargeNNConstr() const { return chargeNNConstr; }

  void setChargeNNConstr(int chargeNn = 0) { chargeNNConstr = chargeNn; }

  int getPtNNConstr() const { return ptNNConstr; }

  void setPtNNConstr(int ptNn = 0) { ptNNConstr = ptNn; }

  int getChargeNNUnconstr() const { return chargeNNUnconstr; }

  void setChargeNNUnconstr(int chargeNnUnconstr = 0) { chargeNNUnconstr = chargeNnUnconstr; }

  int getPtNNUnconstr() const { return ptNNUnconstr; }

  void setPtNNUnconstr(int ptNnUnconstr = 0) { ptNNUnconstr = ptNnUnconstr; }

private:
  ///FIXME maybe the gpResult cannot be a reference or pointer, ad not a copy
  GoldenPatternResult gpResultConstr;

  //GoldenPatternResult without vertex constraint (unconstrained pt)
  //TODO make it pointer
  GoldenPatternResult gpResultUnconstr;

  GoldenPatternBase* goldenPaternConstr = nullptr;

  //GoldenPattern without vertex constraint (unconstrained pt)
  GoldenPatternBase* goldenPaternUnconstr = nullptr;

  //int m_q = -1;
  int m_bx = 0;

  unsigned int m_rhitNumb = 0;

  bool killed = false;

  unsigned int index = 0;

  std::vector<std::shared_ptr<AlgoMuon>> killedMuons;

  int ptNNConstr = 0;
  int chargeNNConstr = 0;

  int ptNNUnconstr = 0;
  int chargeNNUnconstr = 0;
};

typedef std::shared_ptr<AlgoMuon> AlgoMuonPtr;
typedef std::vector<AlgoMuonPtr> AlgoMuons;

#endif  //L1T_OmtfP1_AlgoMuon_H
