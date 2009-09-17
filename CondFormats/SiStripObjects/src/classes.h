#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/SiStripObjects/interface/SiStripThreshold.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "CondFormats/SiStripObjects/interface/SiStripModuleHV.h"
#include "CondFormats/SiStripObjects/interface/SiStripDetVOff.h"
#include "CondFormats/SiStripObjects/interface/SiStripRunSummary.h"
#include "CondFormats/SiStripObjects/interface/SiStripPerformanceSummary.h"
#include "CondFormats/SiStripObjects/interface/SiStripSummary.h"

namespace {
  struct dictionary {
    std::vector<FedChannelConnection>::iterator tmp0;
    std::vector<FedChannelConnection>::const_iterator tmp1;
    std::vector< std::vector<FedChannelConnection> >::iterator tmp2;
    std::vector< std::vector<FedChannelConnection> >::const_iterator tmp3;
  
#ifdef SISTRIPCABLING_USING_NEW_STRUCTURE
  
    SiStripFedCabling::Feds                temp1;
    SiStripFedCabling::FedsIter            temp2;
    SiStripFedCabling::FedsIterRange       temp3;
    SiStripFedCabling::FedsConstIter       temp4;
    SiStripFedCabling::FedsConstIterRange  temp5;
    SiStripFedCabling::Conns               temp6;
    SiStripFedCabling::ConnsPair           temp7;
    SiStripFedCabling::ConnsIter           temp8;
    SiStripFedCabling::ConnsIterRange      temp9;
    SiStripFedCabling::ConnsConstIter      temp10;
    SiStripFedCabling::ConnsConstIterRange temp11;
    SiStripFedCabling::Registry            temp12;

#endif
  
    std::vector< SiStripPedestals::DetRegistry >::iterator tmp6;
    std::vector< SiStripPedestals::DetRegistry >::const_iterator tmp7;
    
    std::vector< SiStripNoises::DetRegistry >::iterator tmp10;
    std::vector< SiStripNoises::DetRegistry >::const_iterator tmp11;
 
    std::vector< SiStripBadStrip::DetRegistry >::iterator tmp14;
    std::vector< SiStripBadStrip::DetRegistry >::const_iterator tmp15;
 
    std::vector< SiStripPerformanceSummary::DetSummary >::iterator tmp20;
    std::vector< SiStripPerformanceSummary::DetSummary >::const_iterator tmp21;
 
    std::vector<SiStripThreshold::Container>::iterator tmp22;
    std::vector<SiStripThreshold::Container>::const_iterator tmp23;
    std::vector< SiStripThreshold::DetRegistry >::iterator tmp24;
    std::vector< SiStripThreshold::DetRegistry >::const_iterator tmp25;
 
    std::vector< SiStripSummary::DetRegistry >::iterator tmp32;
    std::vector< SiStripSummary::DetRegistry >::const_iterator tmp33;
  };
}  
  
