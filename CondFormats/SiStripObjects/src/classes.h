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
  
    std::vector<char>::iterator tmp4;
    std::vector<char>::const_iterator tmp5;
    std::vector< SiStripPedestals::DetRegistry >::iterator tmp6;
    std::vector< SiStripPedestals::DetRegistry >::const_iterator tmp7;
    
    std::vector<unsigned char>::iterator tmp8;
    std::vector<unsigned char>::const_iterator tmp9;
    std::vector< SiStripNoises::DetRegistry >::iterator tmp10;
    std::vector< SiStripNoises::DetRegistry >::const_iterator tmp11;
 
    std::vector<unsigned int>::iterator tmp12;
    std::vector<unsigned int>::const_iterator tmp13;
 
    std::vector<int>::iterator tmp12a;
    std::vector<int>::const_iterator tmp13a;
    std::vector<uint32_t>::const_iterator tmp16;
    std::vector<uint32_t>::iterator tmp17;
    std::vector< SiStripBadStrip::DetRegistry >::iterator tmp14;
    std::vector< SiStripBadStrip::DetRegistry >::const_iterator tmp15;
 
    std::vector<float>::iterator tmp18;
    std::vector<float>::const_iterator tmp19;
    std::vector< SiStripPerformanceSummary::DetSummary >::iterator tmp20;
    std::vector< SiStripPerformanceSummary::DetSummary >::const_iterator tmp21;
 
    std::vector<SiStripThreshold::Container>::iterator tmp22;
    std::vector<SiStripThreshold::Container>::const_iterator tmp23;
    std::vector< SiStripThreshold::DetRegistry >::iterator tmp24;
    std::vector< SiStripThreshold::DetRegistry >::const_iterator tmp25;
 
    std::vector<std::string>::iterator tmp30;
    std::vector<std::string>::const_iterator tmp31;
    std::vector< SiStripSummary::DetRegistry >::iterator tmp32;
    std::vector< SiStripSummary::DetRegistry >::const_iterator tmp33;
  };
}  
  
