//-------------------------------------------------
//
//   \class L1MuGMTParametersProducer
//
//   Description:  A class to produce the L1 GMT emulator Parameters record in the event setup
//
//   $Date$
//   $Revision$
//
//   Author :
//   I. Mikulec
//
//--------------------------------------------------
#include "L1TriggerConfig/GMTConfigProducers/interface/L1MuGMTParametersProducer.h"

L1MuGMTParametersProducer::L1MuGMTParametersProducer(const edm::ParameterSet& ps)
{
 
  m_ps = new edm::ParameterSet(ps);
  setWhatProduced(this, &L1MuGMTParametersProducer::produceL1MuGMTParameters);
  setWhatProduced(this, &L1MuGMTParametersProducer::produceL1MuGMTChannelMask);
  
}


L1MuGMTParametersProducer::~L1MuGMTParametersProducer() {
  delete m_ps;
}


//
// member functions
//

// ------------ methods called to produce the data  ------------
std::auto_ptr<L1MuGMTParameters> 
L1MuGMTParametersProducer::produceL1MuGMTParameters(const L1MuGMTParametersRcd& iRecord)
{
  using namespace edm::es;

  std::auto_ptr<L1MuGMTParameters> gmtparams = std::auto_ptr<L1MuGMTParameters>( new L1MuGMTParameters() );

  gmtparams->setEtaWeight_barrel(m_ps->getParameter<double>("EtaWeight_barrel"));
  gmtparams->setPhiWeight_barrel(m_ps->getParameter<double>("PhiWeight_barrel"));
  gmtparams->setEtaPhiThreshold_barrel(m_ps->getParameter<double>("EtaPhiThreshold_barrel"));
  gmtparams->setEtaWeight_endcap(m_ps->getParameter<double>("EtaWeight_endcap"));
  gmtparams->setPhiWeight_endcap(m_ps->getParameter<double>("PhiWeight_endcap"));
  gmtparams->setEtaPhiThreshold_endcap(m_ps->getParameter<double>("EtaPhiThreshold_endcap"));
  gmtparams->setEtaWeight_COU(m_ps->getParameter<double>("EtaWeight_COU"));
  gmtparams->setPhiWeight_COU(m_ps->getParameter<double>("PhiWeight_COU"));
  gmtparams->setEtaPhiThreshold_COU(m_ps->getParameter<double>("EtaPhiThreshold_COU"));
  gmtparams->setCaloTrigger(m_ps->getParameter<bool>("CaloTrigger"));
  gmtparams->setIsolationCellSizeEta(m_ps->getParameter<int>("IsolationCellSizeEta"));
  gmtparams->setIsolationCellSizePhi(m_ps->getParameter<int>("IsolationCellSizePhi"));
  gmtparams->setDoOvlRpcAnd(m_ps->getParameter<bool>("DoOvlRpcAnd"));
  gmtparams->setPropagatePhi(m_ps->getParameter<bool>("PropagatePhi"));
  gmtparams->setMergeMethodPhiBrl(m_ps->getParameter<std::string>("MergeMethodPhiBrl"));
  gmtparams->setMergeMethodPhiFwd(m_ps->getParameter<std::string>("MergeMethodPhiFwd"));
  gmtparams->setMergeMethodEtaBrl(m_ps->getParameter<std::string>("MergeMethodEtaBrl"));
  gmtparams->setMergeMethodEtaFwd(m_ps->getParameter<std::string>("MergeMethodEtaFwd"));
  gmtparams->setMergeMethodPtBrl(m_ps->getParameter<std::string>("MergeMethodPtBrl"));
  gmtparams->setMergeMethodPtFwd(m_ps->getParameter<std::string>("MergeMethodPtFwd"));
  gmtparams->setMergeMethodChargeBrl(m_ps->getParameter<std::string>("MergeMethodChargeBrl"));
  gmtparams->setMergeMethodChargeFwd(m_ps->getParameter<std::string>("MergeMethodChargeFwd"));
  gmtparams->setMergeMethodMIPBrl(m_ps->getParameter<std::string>("MergeMethodMIPBrl"));
  gmtparams->setMergeMethodMIPFwd(m_ps->getParameter<std::string>("MergeMethodMIPFwd"));
  gmtparams->setMergeMethodMIPSpecialUseANDBrl(m_ps->getParameter<bool>("MergeMethodMIPSpecialUseANDBrl"));
  gmtparams->setMergeMethodMIPSpecialUseANDFwd(m_ps->getParameter<bool>("MergeMethodMIPSpecialUseANDFwd"));
  gmtparams->setMergeMethodISOBrl(m_ps->getParameter<std::string>("MergeMethodISOBrl"));
  gmtparams->setMergeMethodISOFwd(m_ps->getParameter<std::string>("MergeMethodISOFwd"));
  gmtparams->setMergeMethodISOSpecialUseANDBrl(m_ps->getParameter<bool>("MergeMethodISOSpecialUseANDBrl"));
  gmtparams->setMergeMethodISOSpecialUseANDFwd(m_ps->getParameter<bool>("MergeMethodISOSpecialUseANDFwd"));
  gmtparams->setMergeMethodSRKBrl(m_ps->getParameter<std::string>("MergeMethodSRKBrl"));
  gmtparams->setMergeMethodSRKFwd(m_ps->getParameter<std::string>("MergeMethodSRKFwd"));
  gmtparams->setHaloOverwritesMatchedBrl(m_ps->getParameter<bool>("HaloOverwritesMatchedBrl"));
  gmtparams->setHaloOverwritesMatchedFwd(m_ps->getParameter<bool>("HaloOverwritesMatchedFwd"));
  gmtparams->setSortRankOffsetBrl(m_ps->getParameter<unsigned>("SortRankOffsetBrl"));
  gmtparams->setSortRankOffsetFwd(m_ps->getParameter<unsigned>("SortRankOffsetFwd"));
  gmtparams->setCDLConfigWordDTCSC(m_ps->getParameter<unsigned>("CDLConfigWordDTCSC"));
  gmtparams->setCDLConfigWordCSCDT(m_ps->getParameter<unsigned>("CDLConfigWordCSCDT"));
  gmtparams->setCDLConfigWordbRPCCSC(m_ps->getParameter<unsigned>("CDLConfigWordbRPCCSC"));
  gmtparams->setCDLConfigWordfRPCDT(m_ps->getParameter<unsigned>("CDLConfigWordfRPCDT"));
  gmtparams->setVersionSortRankEtaQLUT(m_ps->getParameter<unsigned>("VersionSortRankEtaQLUT"));
  gmtparams->setVersionLUTs(m_ps->getParameter<unsigned>("VersionLUTs"));

  return gmtparams ;
}

std::auto_ptr<L1MuGMTChannelMask> 
L1MuGMTParametersProducer::produceL1MuGMTChannelMask(const L1MuGMTChannelMaskRcd& iRecord)
{
  using namespace edm::es;

  std::auto_ptr<L1MuGMTChannelMask> gmtchanmask = std::auto_ptr<L1MuGMTChannelMask>( new L1MuGMTChannelMask() );

  gmtchanmask->setSubsystemMask(m_ps->getParameter<unsigned>("SubsystemMask"));

  return gmtchanmask ;
}


