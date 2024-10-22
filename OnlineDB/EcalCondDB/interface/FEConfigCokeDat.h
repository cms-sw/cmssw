#ifndef ONLINEDB_ECALCONDDB_FECONFIGCOKEDAT_H
#define ONLINEDB_ECALCONDDB_FECONFIGCOKEDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigCokeInfo.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class FEConfigCokeDat : public IDataItem {
public:
  friend class EcalCondDBInterface;  // XXX temp should not need
  FEConfigCokeDat();
  ~FEConfigCokeDat() override;

  // User data methods
  inline std::string getTable() override { return "FE_CONFIG_COKE_DAT"; }

  inline void setThreshold(int x) { m_thre = x; }
  inline int getThreshold() const { return m_thre; }
  inline void setSuccEventLimit(int x) { m_succ_ev_lim = x; }
  inline int getSuccEventLimit() const { return m_succ_ev_lim; }
  inline void setCumulEventLimit(int x) { m_cum_ev_lim = x; }
  inline int getCumulEventLimit() const { return m_cum_ev_lim; }
  inline void setSuccDetectEnable(int x) { m_succ_det_en = x; }
  inline int getSuccDetectEnable() const { return m_succ_det_en; }
  inline void setCumDetectEnable(int x) { m_cum_det_en = x; }
  inline int getCumDetectEnable() const { return m_cum_det_en; }
  inline void setThreshold1(int x) { m_thre1 = x; }
  inline int getThreshold1() const { return m_thre1; }
  inline void setSucc1EventLimit(int x) { m_succ1_ev_lim = x; }
  inline int getSucc1EventLimit() const { return m_succ1_ev_lim; }
  inline void setCumul1EventLimit(int x) { m_cum1_ev_lim = x; }
  inline int getCumul1EventLimit() const { return m_cum1_ev_lim; }
  inline void setCombiMode(int x) { m_combi_mode = x; }
  inline int getCombiMode() const { return m_combi_mode; }
  inline void setOccMode(int x) { m_occ_mode = x; }
  inline int getOccMode() const { return m_occ_mode; }
  inline void setCombSuccDetect(int x) { m_comb_succ_det = x; }
  inline int getCombSuccDetect() const { return m_comb_succ_det; }
  inline void setCombCumulDetect(int x) { m_comb_cumul_det = x; }
  inline int getCombCumulDetect() const { return m_comb_cumul_det; }
  inline void setOccDetect(int x) { m_occ_det = x; }
  inline int getOccDetect() const { return m_occ_det; }
  inline void setCumul1Detect(int x) { m_cum1_det = x; }
  inline int getCumul1Detect() const { return m_cum1_det; }
  inline void setThreshold2(int x) { m_thre2 = x; }
  inline int getThreshold2() const { return m_thre2; }
  inline void setOccLimit(int x) { m_occ_lim = x; }
  inline int getOccLimit() const { return m_occ_lim; }
  inline void setThreshold3(int x) { m_thre3 = x; }
  inline int getThreshold3() const { return m_thre3; }
  inline void setCumul2Limit(int x) { m_cum2_lim = x; }
  inline int getCumul2Limit() const { return m_cum2_lim; }
  inline void setStopBufW(int x) { m_stop_bufw = x; }
  inline int getStopBufW() const { return m_stop_bufw; }
  // this simplifies the code
  inline void setPar1(int x) { m_thre = x; }
  inline int getPar1() const { return m_thre; }
  inline void setPar2(int x) { m_succ_ev_lim = x; }
  inline int getPar2() const { return m_succ_ev_lim; }
  inline void setPar3(int x) { m_cum_ev_lim = x; }
  inline int getPar3() const { return m_cum_ev_lim; }
  inline void setPar4(int x) { m_succ_det_en = x; }
  inline int getPar4() const { return m_succ_det_en; }
  inline void setPar5(int x) { m_cum_det_en = x; }
  inline int getPar5() const { return m_cum_det_en; }
  inline void setPar6(int x) { m_thre1 = x; }
  inline int getPar6() const { return m_thre1; }
  inline void setPar7(int x) { m_succ1_ev_lim = x; }
  inline int getPar7() const { return m_succ1_ev_lim; }
  inline void setPar8(int x) { m_cum1_ev_lim = x; }
  inline int getPar8() const { return m_cum1_ev_lim; }
  inline void setPar9(int x) { m_combi_mode = x; }
  inline int getPar9() const { return m_combi_mode; }
  inline void setPar10(int x) { m_occ_mode = x; }
  inline int getPar10() const { return m_occ_mode; }
  inline void setPar11(int x) { m_comb_succ_det = x; }
  inline int getPar11() const { return m_comb_succ_det; }
  inline void setPar12(int x) { m_comb_cumul_det = x; }
  inline int getPar12() const { return m_comb_cumul_det; }
  inline void setPar13(int x) { m_occ_det = x; }
  inline int getPar13() const { return m_occ_det; }
  inline void setPar14(int x) { m_cum1_det = x; }
  inline int getPar14() const { return m_cum1_det; }
  inline void setPar15(int x) { m_thre2 = x; }
  inline int getPar15() const { return m_thre2; }
  inline void setPar16(int x) { m_occ_lim = x; }
  inline int getPar16() const { return m_occ_lim; }
  inline void setPar17(int x) { m_thre3 = x; }
  inline int getPar17() const { return m_thre3; }
  inline void setPar18(int x) { m_cum2_lim = x; }
  inline int getPar18() const { return m_cum2_lim; }
  inline void setPar19(int x) { m_stop_bufw = x; }
  inline int getPar19() const { return m_stop_bufw; }

  void clear() {
    m_thre = 0;
    m_succ_ev_lim = 0;
    m_cum_ev_lim = 0;
    m_succ_det_en = 0;
    m_cum_det_en = 0;
    m_thre1 = 0;
    m_succ1_ev_lim = 0;
    m_cum1_ev_lim = 0;
    m_combi_mode = 0;
    m_occ_mode = 0;
    m_comb_succ_det = 0;
    m_comb_cumul_det = 0;
    m_occ_det = 0;
    m_cum1_det = 0;
    m_thre2 = 0;
    m_occ_lim = 0;
    m_thre3 = 0;
    m_cum2_lim = 0;
    m_stop_bufw = 0;
  }

private:
  void prepareWrite() noexcept(false) override;
  void writeDB(const EcalLogicID* ecid, const FEConfigCokeDat* item, FEConfigCokeInfo* iconf) noexcept(false);
  void writeArrayDB(const std::map<EcalLogicID, FEConfigCokeDat>* data, FEConfigCokeInfo* iconf) noexcept(false);
  void fetchData(std::map<EcalLogicID, FEConfigCokeDat>* fillMap, FEConfigCokeInfo* iconf) noexcept(false);
  // User data
  int m_thre;
  int m_succ_ev_lim;
  int m_cum_ev_lim;
  int m_succ_det_en;
  int m_cum_det_en;
  int m_thre1;
  int m_succ1_ev_lim;
  int m_cum1_ev_lim;
  int m_combi_mode;
  int m_occ_mode;
  int m_comb_succ_det;
  int m_comb_cumul_det;
  int m_occ_det;
  int m_cum1_det;
  int m_thre2;
  int m_occ_lim;
  int m_thre3;
  int m_cum2_lim;
  int m_stop_bufw;
};

#endif
