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
#include <DataFormats/CSCDigi/interface/CSCCFEBStatusDigi.h>
#include <DataFormats/CSCDigi/interface/CSCCFEBStatusDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCDCCFormatStatusDigi.h>
#include <DataFormats/CSCDigi/interface/CSCDCCFormatStatusDigiCollection.h>

#include <DataFormats/Common/interface/Wrapper.h>
#include <vector>
#include <map>

namespace{ 
  namespace {

  CSCWireDigi cWD_;
  CSCRPCDigi  cRD_;
  CSCStripDigi cSD_;
  CSCComparatorDigi cCD_;
  CSCCLCTDigi cCLCTD_;
  CSCALCTDigi cALCTD_;
  CSCCorrelatedLCTDigi cCorLCTD_;
  CSCCFEBStatusDigi cCSD_;
  CSCDCCFormatStatusDigi cDFSD_; 

  std::vector<CSCWireDigi>  vWD_;
  std::vector<CSCRPCDigi>   vRD_;
  std::vector<CSCStripDigi>  vSD_;
  std::vector<CSCComparatorDigi>  vCD_;
  std::vector<CSCCLCTDigi> vCLCTD_;
  std::vector<CSCALCTDigi> vALCTD_;
  std::vector<CSCCorrelatedLCTDigi> vCorLCTD_;
  std::vector<CSCCFEBStatusDigi>  vCSD_;
  std::vector<CSCDCCFormatStatusDigi>  vDFSD_;

  std::vector<std::vector<CSCWireDigi> >  vvWD_; 
  std::vector<std::vector<CSCRPCDigi>  >  vvRD_;
  std::vector<std::vector<CSCStripDigi> >  vvSD_; 
  std::vector<std::vector<CSCComparatorDigi> >  vvCD_;
  std::vector<std::vector<CSCCLCTDigi> > vvCLCTD_;
  std::vector<std::vector<CSCALCTDigi> > vvALCTD_;
  std::vector<std::vector<CSCCorrelatedLCTDigi> > vvCorLCTD_;
  std::vector<std::vector<CSCCFEBStatusDigi> >  vvCSD_;
  std::vector<std::vector<CSCDCCFormatStatusDigi> >  vvDFSD_;

  CSCWireDigiCollection clWD_;
  CSCRPCDigiCollection  clRD_;
  CSCStripDigiCollection clSD_;
  CSCComparatorDigiCollection clCD_;
  CSCCLCTDigiCollection clCLCTD_;
  CSCALCTDigiCollection clALCTD_;
  CSCCorrelatedLCTDigiCollection clCorLCTD_;
  CSCCFEBStatusDigiCollection clCSD_;
  CSCDCCFormatStatusDigiCollection clDFSD_;

  edm::Wrapper<CSCWireDigiCollection> wWD_;
  edm::Wrapper<CSCRPCDigiCollection> wRD_;
  edm::Wrapper<CSCStripDigiCollection> wSD_;
  edm::Wrapper<CSCComparatorDigiCollection> wCD_;
  edm::Wrapper<CSCCLCTDigiCollection> wCLCTD_;
  edm::Wrapper<CSCALCTDigiCollection> wALCTD_;
  edm::Wrapper<CSCCorrelatedLCTDigiCollection> wCorLCTD_;
  edm::Wrapper<CSCCFEBStatusDigiCollection> wCSD_;
  edm::Wrapper<CSCDCCFormatStatusDigiCollection> wDFSD_;

  }
}
