#ifndef ONLINEDB_ECALCONDDB_FECONFIGODDWEIGHTMODEDAT_H
#define ONLINEDB_ECALCONDDB_FECONFIGODDWEIGHTMODEDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigOddWeightInfo.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class FEConfigOddWeightModeDat : public IDataItem {
public:
  friend class EcalCondDBInterface;  // XXX temp should not need
  FEConfigOddWeightModeDat();
  ~FEConfigOddWeightModeDat() override;

  // User data methods
  inline std::string getTable() override { return "FE_WEIGHT2_MODE_DAT"; }

  inline void setEnableEBOddFilter(int x) { m_en_EB_flt = x; }
  inline float getEnableEBOddFilter() const { return m_en_EB_flt; }
  inline void setEnableEEOddFilter(int x) { m_en_EE_flt = x; }
  inline float getEnableEEOddFilter() const { return m_en_EE_flt; }

  inline void setEnableEBOddPeakFinder(int x) { m_en_EB_pf = x; }
  inline float getEnableEBOddPeakFinder() const { return m_en_EB_pf; }
  inline void setEnableEEOddPeakFinder(int x) { m_en_EE_pf = x; }
  inline float getEnableEEOddPeakFinder() const { return m_en_EE_pf; }

  inline void setDisableEBEvenPeakFinder(int x) { m_dis_EB_even_pf = x; }
  inline float getDisableEBEvenPeakFinder() const { return m_dis_EB_even_pf; }
  inline void setDisableEEEvenPeakFinder(int x) { m_dis_EE_even_pf = x; }
  inline float getDisableEEEvenPeakFinder() const { return m_dis_EE_even_pf; }

  inline void setFenixEBStripOutput(int x) { m_fe_EB_strout = x; }
  inline float getFenixEBStripOutput() const { return m_fe_EB_strout; }
  inline void setFenixEEStripOutput(int x) { m_fe_EE_strout = x; }
  inline float getFenixEEStripOutput() const { return m_fe_EE_strout; }

  inline void setFenixEBStripInfobit2(int x) { m_fe_EB_strib2 = x; }
  inline float getFenixEBStripInfobit2() const { return m_fe_EB_strib2; }
  inline void setFenixEEStripInfobit2(int x) { m_fe_EE_strib2 = x; }
  inline float getFenixEEStripInfobit2() const { return m_fe_EE_strib2; }

  inline void setFenixEBTcpOutput(int x) { m_fe_EB_tcpout = x; }
  inline float getFenixEBTcpOutput() const { return m_fe_EB_tcpout; }
  inline void setFenixEBTcpInfoBit1(int x) { m_fe_EB_tcpib1 = x; }
  inline float getFenixEBTcpInfobit1() const { return m_fe_EB_tcpib1; }

  inline void setFenixEETcpOutput(int x) { m_fe_EE_tcpout = x; }
  inline float getFenixEETcpOutput() const { return m_fe_EE_tcpout; }
  inline void setFenixEETcpInfoBit1(int x) { m_fe_EE_tcpib1 = x; }
  inline float getFenixEETcpInfobit1() const { return m_fe_EE_tcpib1; }

  // redundant methods for simplification of the code

  inline void setFenixPar1(int x) { m_en_EB_flt = x; }
  inline float getFenixPar1() const { return m_en_EB_flt; }
  inline void setFenixPar2(int x) { m_en_EE_flt = x; }
  inline float getFenixPar2() const { return m_en_EE_flt; }

  inline void setFenixPar3(int x) { m_en_EB_pf = x; }
  inline float getFenixPar3() const { return m_en_EB_pf; }
  inline void setFenixPar4(int x) { m_en_EE_pf = x; }
  inline float getFenixPar4() const { return m_en_EE_pf; }

  inline void setFenixPar5(int x) { m_dis_EB_even_pf = x; }
  inline float getFenixPar5() const { return m_dis_EB_even_pf; }

  inline void setFenixPar6(int x) { m_dis_EE_even_pf = x; }
  inline float getFenixPar6() const { return m_dis_EE_even_pf; }

  inline void setFenixPar7(int x) { m_fe_EB_strout = x; }
  inline float getFenixPar7() const { return m_fe_EB_strout; }
  inline void setFenixPar8(int x) { m_fe_EE_strout = x; }
  inline float getFenixPar8() const { return m_fe_EE_strout; }

  inline void setFenixPar9(int x) { m_fe_EB_strib2 = x; }
  inline float getFenixPar9() const { return m_fe_EB_strib2; }
  inline void setFenixPar10(int x) { m_fe_EE_strib2 = x; }
  inline float getFenixPar10() const { return m_fe_EE_strib2; }

  inline void setFenixPar11(int x) { m_fe_EB_tcpout = x; }
  inline float getFenixPar11() const { return m_fe_EB_tcpout; }
  inline void setFenixPar12(int x) { m_fe_EB_tcpib1 = x; }
  inline float getFenixPar12() const { return m_fe_EB_tcpib1; }

  inline void setFenixPar13(int x) { m_fe_EE_tcpout = x; }
  inline float getFenixPar13() const { return m_fe_EE_tcpout; }
  inline void setFenixPar14(int x) { m_fe_EE_tcpib1 = x; }
  inline float getFenixPar14() const { return m_fe_EE_tcpib1; }

  // extra parameters

  inline void setFenixPar15(int x) { m_fe_par15 = x; }
  inline float getFenixPar15() const { return m_fe_par15; }
  inline void setFenixPar16(int x) { m_fe_par16 = x; }
  inline float getFenixPar16() const { return m_fe_par16; }
  inline void setFenixPar17(int x) { m_fe_par17 = x; }
  inline float getFenixPar17() const { return m_fe_par17; }
  inline void setFenixPar18(int x) { m_fe_par18 = x; }
  inline float getFenixPar18() const { return m_fe_par18; }

private:
  void prepareWrite() noexcept(false) override;

  void writeDB(const EcalLogicID* ecid,
               const FEConfigOddWeightModeDat* item,
               FEConfigOddWeightInfo* iconf) noexcept(false);

  void writeArrayDB(const std::map<EcalLogicID, FEConfigOddWeightModeDat>* data,
                    FEConfigOddWeightInfo* iconf) noexcept(false);

  void fetchData(std::map<EcalLogicID, FEConfigOddWeightModeDat>* fillMap,
                 FEConfigOddWeightInfo* iconf) noexcept(false);

  // User data

  int m_en_EB_flt;
  int m_en_EE_flt;
  int m_en_EB_pf;
  int m_en_EE_pf;
  int m_dis_EB_even_pf;
  int m_dis_EE_even_pf;
  int m_fe_EB_strout;
  int m_fe_EE_strout;
  int m_fe_EB_strib2;
  int m_fe_EE_strib2;
  int m_fe_EB_tcpout;
  int m_fe_EB_tcpib1;
  int m_fe_EE_tcpout;
  int m_fe_EE_tcpib1;
  int m_fe_par15;
  int m_fe_par16;
  int m_fe_par17;
  int m_fe_par18;
};

#endif
