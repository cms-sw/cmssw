#include <DataFormats/CSCDigi/interface/CSCWireDigi.h>
#include <DataFormats/CSCDigi/interface/CSCWireDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCRPCDigi.h>
#include <DataFormats/CSCDigi/interface/CSCRPCDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCStripDigi.h>
#include <DataFormats/CSCDigi/interface/CSCStripDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCComparatorDigi.h>
#include <DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCCLCTDigi.h>
#include <DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCALCTDigi.h>
#include <DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h>
#include <DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h>
#include <DataFormats/CSCDigi/interface/GEMCSCLCTDigi.h>
#include <DataFormats/CSCDigi/interface/GEMCSCLCTDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCCFEBStatusDigi.h>
#include <DataFormats/CSCDigi/interface/CSCCFEBStatusDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCDMBStatusDigi.h>
#include <DataFormats/CSCDigi/interface/CSCDMBStatusDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCTMBStatusDigi.h>
#include <DataFormats/CSCDigi/interface/CSCTMBStatusDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCDCCFormatStatusDigi.h>
#include <DataFormats/CSCDigi/interface/CSCDCCFormatStatusDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCALCTStatusDigi.h>
#include <DataFormats/CSCDigi/interface/CSCALCTStatusDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCDDUStatusDigi.h>
#include <DataFormats/CSCDigi/interface/CSCDDUStatusDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCDCCStatusDigi.h>
#include <DataFormats/CSCDigi/interface/CSCDCCStatusDigiCollection.h>
#include "DataFormats/CSCDigi/interface/CSCCLCTPreTriggerCollection.h"



#include <DataFormats/Common/interface/Wrapper.h>
#include <vector>
#include <map>

namespace DataFormats_CSCDigi {
  struct dictionary {

  CSCWireDigi cWD_;
  CSCRPCDigi  cRD_;
  CSCStripDigi cSD_;
  CSCComparatorDigi cCD_;
  CSCCLCTDigi cCLCTD_;
  CSCALCTDigi cALCTD_;
  CSCCorrelatedLCTDigi cCorLCTD_;
  GEMCSCLCTDigi gcLCTD_;
  CSCCFEBStatusDigi cCSD_;
  CSCTMBStatusDigi cTMBSD_;
  CSCDCCFormatStatusDigi cDFSD_;
  CSCDMBStatusDigi cDMBSD_;
  CSCDDUStatusDigi cDDUSD_;
  CSCDCCStatusDigi cDCCSD_;
  CSCALCTStatusDigi cALCTSD_;



  std::vector<CSCWireDigi>  vWD_;
  std::vector<CSCRPCDigi>   vRD_;
  std::vector<CSCStripDigi>  vSD_;
  std::vector<CSCComparatorDigi>  vCD_;
  std::vector<CSCCLCTDigi> vCLCTD_;
  std::vector<CSCALCTDigi> vALCTD_;
  std::vector<CSCCorrelatedLCTDigi> vCorLCTD_;
  std::vector<GEMCSCLCTDigi> vgcLCTD_;
  std::vector<CSCCFEBStatusDigi>  vCSD_;
  std::vector<CSCTMBStatusDigi>  vTMBSD_;
  std::vector<CSCDCCFormatStatusDigi>  vDFSD_;
  std::vector<CSCDMBStatusDigi>  vDMBSD_;
  std::vector<CSCDDUStatusDigi>  vDDUSD_;
  std::vector<CSCDCCStatusDigi>  vDCCSD_;
  std::vector<CSCALCTStatusDigi>  vALCTSD_;
  std::vector<CSCCLCTPreTrigger> vPreTriggerBX_;

  std::vector<std::vector<CSCWireDigi> >  vvWD_;
  std::vector<std::vector<CSCRPCDigi>  >  vvRD_;
  std::vector<std::vector<CSCStripDigi> >  vvSD_;
  std::vector<std::vector<CSCComparatorDigi> >  vvCD_;
  std::vector<std::vector<CSCCLCTDigi> > vvCLCTD_;
  std::vector<std::vector<CSCALCTDigi> > vvALCTD_;
  std::vector<std::vector<CSCCorrelatedLCTDigi> > vvCorLCTD_;
  std::vector<std::vector<GEMCSCLCTDigi> > vvgcLCTD_;
  std::vector<std::vector<CSCCFEBStatusDigi> >  vvCSD_;
  std::vector<std::vector<CSCTMBStatusDigi> >  vvTMBSD_;
  std::vector<std::vector<CSCDMBStatusDigi> >  vvDMBSD_;
  std::vector<std::vector<CSCDCCFormatStatusDigi> >  vvDFSD_;
  std::vector<std::vector<CSCDDUStatusDigi> >  vvDDUSD_;
  std::vector<std::vector<CSCDCCStatusDigi> >  vvDCCSD_;
  std::vector<std::vector<CSCALCTStatusDigi> >  vvALCTSD_;
  std::vector<std::vector<CSCCLCTPreTrigger> > vvPreTrigger_;

  CSCWireDigiCollection clWD_;
  CSCRPCDigiCollection  clRD_;
  CSCStripDigiCollection clSD_;
  CSCComparatorDigiCollection clCD_;
  CSCCLCTDigiCollection clCLCTD_;
  CSCALCTDigiCollection clALCTD_;
  CSCCorrelatedLCTDigiCollection clCorLCTD_;
  GEMCSCLCTDigiCollection clgcLCTD_;  
  CSCCFEBStatusDigiCollection clCSD_;
  CSCTMBStatusDigiCollection clTMBSD_;
  CSCDCCFormatStatusDigiCollection clDFSD_;
  CSCDMBStatusDigiCollection clDMBSD_;
  CSCDMBStatusDigiCollection clDDUSD_;
  CSCDMBStatusDigiCollection clDCCSD_;
  CSCDMBStatusDigiCollection clALCTSD_;
  CSCCLCTPreTriggerCollection clPreTrigger_;

  edm::Wrapper<CSCWireDigiCollection> wWD_;
  edm::Wrapper<CSCRPCDigiCollection> wRD_;
  edm::Wrapper<CSCStripDigiCollection> wSD_;
  edm::Wrapper<CSCComparatorDigiCollection> wCD_;
  edm::Wrapper<CSCCLCTDigiCollection> wCLCTD_;
  edm::Wrapper<CSCALCTDigiCollection> wALCTD_;
  edm::Wrapper<CSCCorrelatedLCTDigiCollection> wCorLCTD_;
  edm::Wrapper<GEMCSCLCTDigiCollection> wgcLCTD_;
  edm::Wrapper<CSCCFEBStatusDigiCollection> wCSD_;
  edm::Wrapper<CSCTMBStatusDigiCollection> wTMBSD_;
  edm::Wrapper<CSCDCCFormatStatusDigiCollection> wDFSD_;
  edm::Wrapper<CSCDMBStatusDigiCollection> wDMBSD_;
  edm::Wrapper<CSCDDUStatusDigiCollection> wDDUSD_;
  edm::Wrapper<CSCDCCStatusDigiCollection> wDCCSD_;
  edm::Wrapper<CSCALCTStatusDigiCollection> wALCTSD_;
  edm::Wrapper<CSCCLCTPreTriggerCollection> wPreTrigger_;
  };
}
