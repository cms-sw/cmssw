//-------------------------------------------------
//
//   \class L1MuGMTParameters
//
/**   Description:  Parameters of the GMT
 *                  
*/
//
//   $Date$
//   $Revision$
//
//
//   Author :
//   Ivan Mikulec      HEPHY / Vienna
//
//
//--------------------------------------------------
#ifndef CondFormatsL1TObjects_L1MuGMTParameters_h
#define CondFormatsL1TObjects_L1MuGMTParameters_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <string>

class L1MuGMTParameters {
public:
  L1MuGMTParameters() : m_VersionLUTs(0) {}
  ~L1MuGMTParameters() {}

  void setEtaWeight_barrel(const double EtaWeight_barrel) { m_EtaWeight_barrel = EtaWeight_barrel; }
  double getEtaWeight_barrel() const { return m_EtaWeight_barrel; }

  void setPhiWeight_barrel(const double PhiWeight_barrel) { m_PhiWeight_barrel = PhiWeight_barrel; }
  double getPhiWeight_barrel() const { return m_PhiWeight_barrel; }

  void setEtaPhiThreshold_barrel(const double EtaPhiThreshold_barrel) {
    m_EtaPhiThreshold_barrel = EtaPhiThreshold_barrel;
  }
  double getEtaPhiThreshold_barrel() const { return m_EtaPhiThreshold_barrel; }

  void setEtaWeight_endcap(const double EtaWeight_endcap) { m_EtaWeight_endcap = EtaWeight_endcap; }
  double getEtaWeight_endcap() const { return m_EtaWeight_endcap; }

  void setPhiWeight_endcap(const double PhiWeight_endcap) { m_PhiWeight_endcap = PhiWeight_endcap; }
  double getPhiWeight_endcap() const { return m_PhiWeight_endcap; }

  void setEtaPhiThreshold_endcap(const double EtaPhiThreshold_endcap) {
    m_EtaPhiThreshold_endcap = EtaPhiThreshold_endcap;
  }
  double getEtaPhiThreshold_endcap() const { return m_EtaPhiThreshold_endcap; }

  void setEtaWeight_COU(const double EtaWeight_COU) { m_EtaWeight_COU = EtaWeight_COU; }
  double getEtaWeight_COU() const { return m_EtaWeight_COU; }

  void setPhiWeight_COU(const double PhiWeight_COU) { m_PhiWeight_COU = PhiWeight_COU; }
  double getPhiWeight_COU() const { return m_PhiWeight_COU; }

  void setEtaPhiThreshold_COU(const double EtaPhiThreshold_COU) { m_EtaPhiThreshold_COU = EtaPhiThreshold_COU; }
  double getEtaPhiThreshold_COU() const { return m_EtaPhiThreshold_COU; }

  void setCaloTrigger(const bool CaloTrigger) { m_CaloTrigger = CaloTrigger; }
  bool getCaloTrigger() const { return m_CaloTrigger; }

  void setIsolationCellSizeEta(const int IsolationCellSizeEta) { m_IsolationCellSizeEta = IsolationCellSizeEta; }
  int getIsolationCellSizeEta() const { return m_IsolationCellSizeEta; }

  void setIsolationCellSizePhi(const int IsolationCellSizePhi) { m_IsolationCellSizePhi = IsolationCellSizePhi; }
  int getIsolationCellSizePhi() const { return m_IsolationCellSizePhi; }

  void setDoOvlRpcAnd(const bool DoOvlRpcAnd) { m_DoOvlRpcAnd = DoOvlRpcAnd; }
  bool getDoOvlRpcAnd() const { return m_DoOvlRpcAnd; }

  void setPropagatePhi(const bool PropagatePhi) { m_PropagatePhi = PropagatePhi; }
  bool getPropagatePhi() const { return m_PropagatePhi; }

  void setMergeMethodPhiBrl(const std::string& MergeMethodPhiBrl) { m_MergeMethodPhiBrl = MergeMethodPhiBrl; }
  const std::string& getMergeMethodPhiBrl() const { return m_MergeMethodPhiBrl; }

  void setMergeMethodPhiFwd(const std::string& MergeMethodPhiFwd) { m_MergeMethodPhiFwd = MergeMethodPhiFwd; }
  const std::string& getMergeMethodPhiFwd() const { return m_MergeMethodPhiFwd; }

  void setMergeMethodEtaBrl(const std::string& MergeMethodEtaBrl) { m_MergeMethodEtaBrl = MergeMethodEtaBrl; }
  const std::string& getMergeMethodEtaBrl() const { return m_MergeMethodEtaBrl; }

  void setMergeMethodEtaFwd(const std::string& MergeMethodEtaFwd) { m_MergeMethodEtaFwd = MergeMethodEtaFwd; }
  const std::string& getMergeMethodEtaFwd() const { return m_MergeMethodEtaFwd; }

  void setMergeMethodPtBrl(const std::string& MergeMethodPtBrl) { m_MergeMethodPtBrl = MergeMethodPtBrl; }
  const std::string& getMergeMethodPtBrl() const { return m_MergeMethodPtBrl; }

  void setMergeMethodPtFwd(const std::string& MergeMethodPtFwd) { m_MergeMethodPtFwd = MergeMethodPtFwd; }
  const std::string& getMergeMethodPtFwd() const { return m_MergeMethodPtFwd; }

  void setMergeMethodChargeBrl(const std::string& MergeMethodChargeBrl) {
    m_MergeMethodChargeBrl = MergeMethodChargeBrl;
  }
  const std::string& getMergeMethodChargeBrl() const { return m_MergeMethodChargeBrl; }

  void setMergeMethodChargeFwd(const std::string& MergeMethodChargeFwd) {
    m_MergeMethodChargeFwd = MergeMethodChargeFwd;
  }
  const std::string& getMergeMethodChargeFwd() const { return m_MergeMethodChargeFwd; }

  void setMergeMethodMIPBrl(const std::string& MergeMethodMIPBrl) { m_MergeMethodMIPBrl = MergeMethodMIPBrl; }
  const std::string& getMergeMethodMIPBrl() const { return m_MergeMethodMIPBrl; }

  void setMergeMethodMIPFwd(const std::string& MergeMethodMIPFwd) { m_MergeMethodMIPFwd = MergeMethodMIPFwd; }
  const std::string& getMergeMethodMIPFwd() const { return m_MergeMethodMIPFwd; }

  void setMergeMethodMIPSpecialUseANDBrl(const bool MergeMethodMIPSpecialUseANDBrl) {
    m_MergeMethodMIPSpecialUseANDBrl = MergeMethodMIPSpecialUseANDBrl;
  }
  bool getMergeMethodMIPSpecialUseANDBrl() const { return m_MergeMethodMIPSpecialUseANDBrl; }

  void setMergeMethodMIPSpecialUseANDFwd(const bool MergeMethodMIPSpecialUseANDFwd) {
    m_MergeMethodMIPSpecialUseANDFwd = MergeMethodMIPSpecialUseANDFwd;
  }
  bool getMergeMethodMIPSpecialUseANDFwd() const { return m_MergeMethodMIPSpecialUseANDFwd; }

  void setMergeMethodISOBrl(const std::string& MergeMethodISOBrl) { m_MergeMethodISOBrl = MergeMethodISOBrl; }
  const std::string& getMergeMethodISOBrl() const { return m_MergeMethodISOBrl; }

  void setMergeMethodISOFwd(const std::string& MergeMethodISOFwd) { m_MergeMethodISOFwd = MergeMethodISOFwd; }
  const std::string& getMergeMethodISOFwd() const { return m_MergeMethodISOFwd; }

  void setMergeMethodISOSpecialUseANDBrl(const bool MergeMethodISOSpecialUseANDBrl) {
    m_MergeMethodISOSpecialUseANDBrl = MergeMethodISOSpecialUseANDBrl;
  }
  bool getMergeMethodISOSpecialUseANDBrl() const { return m_MergeMethodISOSpecialUseANDBrl; }

  void setMergeMethodISOSpecialUseANDFwd(const bool MergeMethodISOSpecialUseANDFwd) {
    m_MergeMethodISOSpecialUseANDFwd = MergeMethodISOSpecialUseANDFwd;
  }
  bool getMergeMethodISOSpecialUseANDFwd() const { return m_MergeMethodISOSpecialUseANDFwd; }

  void setMergeMethodSRKBrl(const std::string& MergeMethodSRKBrl) { m_MergeMethodSRKBrl = MergeMethodSRKBrl; }
  const std::string& getMergeMethodSRKBrl() const { return m_MergeMethodSRKBrl; }

  void setMergeMethodSRKFwd(const std::string& MergeMethodSRKFwd) { m_MergeMethodSRKFwd = MergeMethodSRKFwd; }
  const std::string& getMergeMethodSRKFwd() const { return m_MergeMethodSRKFwd; }

  void setHaloOverwritesMatchedBrl(const bool HaloOverwritesMatchedBrl) {
    m_HaloOverwritesMatchedBrl = HaloOverwritesMatchedBrl;
  }
  bool getHaloOverwritesMatchedBrl() const { return m_HaloOverwritesMatchedBrl; }

  void setHaloOverwritesMatchedFwd(const bool HaloOverwritesMatchedFwd) {
    m_HaloOverwritesMatchedFwd = HaloOverwritesMatchedFwd;
  }
  bool getHaloOverwritesMatchedFwd() const { return m_HaloOverwritesMatchedFwd; }

  void setSortRankOffsetBrl(const unsigned SortRankOffsetBrl) { m_SortRankOffsetBrl = SortRankOffsetBrl; }
  unsigned getSortRankOffsetBrl() const { return m_SortRankOffsetBrl; }

  void setSortRankOffsetFwd(const unsigned SortRankOffsetFwd) { m_SortRankOffsetFwd = SortRankOffsetFwd; }
  unsigned getSortRankOffsetFwd() const { return m_SortRankOffsetFwd; }

  void setCDLConfigWordDTCSC(const unsigned CDLConfigWordDTCSC) { m_CDLConfigWordDTCSC = CDLConfigWordDTCSC; }
  unsigned getCDLConfigWordDTCSC() const { return m_CDLConfigWordDTCSC; }

  void setCDLConfigWordCSCDT(const unsigned CDLConfigWordCSCDT) { m_CDLConfigWordCSCDT = CDLConfigWordCSCDT; }
  unsigned getCDLConfigWordCSCDT() const { return m_CDLConfigWordCSCDT; }

  void setCDLConfigWordbRPCCSC(const unsigned CDLConfigWordbRPCCSC) { m_CDLConfigWordbRPCCSC = CDLConfigWordbRPCCSC; }
  unsigned getCDLConfigWordbRPCCSC() const { return m_CDLConfigWordbRPCCSC; }

  void setCDLConfigWordfRPCDT(const unsigned CDLConfigWordfRPCDT) { m_CDLConfigWordfRPCDT = CDLConfigWordfRPCDT; }
  unsigned getCDLConfigWordfRPCDT() const { return m_CDLConfigWordfRPCDT; }

  void setVersionSortRankEtaQLUT(const unsigned VersionSortRankEtaQLUT) {
    m_VersionSortRankEtaQLUT = VersionSortRankEtaQLUT;
  }
  unsigned getVersionSortRankEtaQLUT() const { return m_VersionSortRankEtaQLUT; }

  void setVersionLUTs(const unsigned VersionLUTs) { m_VersionLUTs = VersionLUTs; }
  unsigned getVersionLUTs() const { return m_VersionLUTs; }

private:
  double m_EtaWeight_barrel;
  double m_PhiWeight_barrel;
  double m_EtaPhiThreshold_barrel;
  double m_EtaWeight_endcap;
  double m_PhiWeight_endcap;
  double m_EtaPhiThreshold_endcap;
  double m_EtaWeight_COU;
  double m_PhiWeight_COU;
  double m_EtaPhiThreshold_COU;
  bool m_CaloTrigger;
  int m_IsolationCellSizeEta;
  int m_IsolationCellSizePhi;
  bool m_DoOvlRpcAnd;
  bool m_PropagatePhi;
  std::string m_MergeMethodPhiBrl;
  std::string m_MergeMethodPhiFwd;
  std::string m_MergeMethodEtaBrl;
  std::string m_MergeMethodEtaFwd;
  std::string m_MergeMethodPtBrl;
  std::string m_MergeMethodPtFwd;
  std::string m_MergeMethodChargeBrl;
  std::string m_MergeMethodChargeFwd;
  std::string m_MergeMethodMIPBrl;
  std::string m_MergeMethodMIPFwd;
  bool m_MergeMethodMIPSpecialUseANDBrl;
  bool m_MergeMethodMIPSpecialUseANDFwd;
  std::string m_MergeMethodISOBrl;
  std::string m_MergeMethodISOFwd;
  bool m_MergeMethodISOSpecialUseANDBrl;
  bool m_MergeMethodISOSpecialUseANDFwd;
  std::string m_MergeMethodSRKBrl;
  std::string m_MergeMethodSRKFwd;
  bool m_HaloOverwritesMatchedBrl;
  bool m_HaloOverwritesMatchedFwd;
  unsigned m_SortRankOffsetBrl;
  unsigned m_SortRankOffsetFwd;
  unsigned m_CDLConfigWordDTCSC;
  unsigned m_CDLConfigWordCSCDT;
  unsigned m_CDLConfigWordbRPCCSC;
  unsigned m_CDLConfigWordfRPCDT;
  unsigned m_VersionSortRankEtaQLUT;
  unsigned m_VersionLUTs;

  COND_SERIALIZABLE;
};

#endif
