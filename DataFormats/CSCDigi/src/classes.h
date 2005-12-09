#include <DataFormats/CSCDigi/interface/CSCWireDigi.h>
#include <DataFormats/CSCDigi/interface/CSCWireDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCRPCDigi.h>
#include <DataFormats/CSCDigi/interface/CSCRPCDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCStripDigi.h>
#include <DataFormats/CSCDigi/interface/CSCStripDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCComparatorDigi.h>
#include <DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCALCTDigi.h>
#include <DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCCLCTDigi.h>
#include <DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h>

#include <FWCore/EDProduct/interface/Wrapper.h>
#include <vector>
#include <map>

namespace{ 
  namespace {

  CSCWireDigi       cWD_;
  CSCRPCDigi        cRD_;
  CSCStripDigi      cSD_;
  CSCComparatorDigi cCD_;
  CSCALCTDigi       cALCTD_;
  CSCCLCTDigi       cCLCTD_;

  CSCWireDigi::PersistentPacking ppWD_;
  CSCRPCDigi::PersistentPacking ppRD_;
  CSCALCTDigi::PersistentPacking ppALCTD_;
  CSCCLCTDigi::PersistentPacking ppCLCTD_;

  std::vector<CSCWireDigi>        vWD_;
  std::vector<CSCRPCDigi>         vRD_;
  std::vector<CSCStripDigi>       vSD_;
  std::vector<CSCComparatorDigi>  vCD_;
  std::vector<CSCALCTDigi>        vALCTD_;
  std::vector<CSCCLCTDigi>        vCLCTD_;

  std::vector<std::vector<CSCWireDigi> >        vvWD_; 
  std::vector<std::vector<CSCRPCDigi>  >        vvRD_;
  std::vector<std::vector<CSCStripDigi> >       vvSD_; 
  std::vector<std::vector<CSCComparatorDigi> >  vvCD_; 
  std::vector<std::vector<CSCALCTDigi> >        vvALCTD_;
  std::vector<std::vector<CSCCLCTDigi> >        vvCLCTD_;

  CSCWireDigiCollection       clWD_;
  CSCRPCDigiCollection        clRD_;
  CSCStripDigiCollection      clSD_;
  CSCComparatorDigiCollection clCD_;
  CSCALCTDigiCollection       clALCTD_;
  CSCCLCTDigiCollection       clCLCTD_;
    
  edm::Wrapper<CSCWireDigiCollection>      wWD_;
  edm::Wrapper<CSCRPCDigiCollection>       wRD_;
  edm::Wrapper<CSCStripDigiCollection>     wSD_;
  edm::Wrapper<CSCComparatorDigiCollection> wCD_;
  edm::Wrapper<CSCALCTDigiCollection>       wALCTD_;
  edm::Wrapper<CSCCLCTDigiCollection>       wCLCTD_;
  }
}
