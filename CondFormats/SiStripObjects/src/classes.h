#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"

namespace {

  std::vector<FedChannelConnection>::iterator tmp0;
  std::vector<FedChannelConnection>::const_iterator tmp1;
  std::vector< std::vector<FedChannelConnection> >::iterator tmp2;
  std::vector< std::vector<FedChannelConnection> >::const_iterator tmp3;

  std::vector<char>::iterator tmp4;
  std::vector<char>::const_iterator tmp5;
  std::vector< SiStripPedestals::DetRegistry >::iterator tmp6;
  std::vector< SiStripPedestals::DetRegistry >::const_iterator tmp7;
  
  std::vector<short>::iterator tmp8;
  std::vector<short>::const_iterator tmp9;
  std::vector< SiStripNoises::DetRegistry >::iterator tmp10;
  std::vector< SiStripNoises::DetRegistry >::const_iterator tmp11;

}  
  
/* 

#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
template std::vector<char>::iterator;
template std::vector<char>::const_iterator;
template std::vector< SiStripPedestals::DetRegistry >::iterator;
template std::vector< SiStripPedestals::DetRegistry >::const_iterator;

#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
//template std::vector<SiStripNoises::SiStripData>::iterator;
//template std::vector<SiStripNoises::SiStripData>::const_iterator;
template std::vector<short>::iterator;
template std::vector<short>::const_iterator;
//template std::map< uint32_t, SiStripNoiseVector>::iterator;
//template std::map< uint32_t, SiStripNoiseVector>::const_iterator;
template std::vector< SiStripNoises::DetRegistry >::iterator;
template std::vector< SiStripNoises::DetRegistry >::const_iterator;

#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
template std::vector< std::vector<FedChannelConnection> >::iterator;
template std::vector< std::vector<FedChannelConnection> >::const_iterator;
template std::vector<FedChannelConnection>::iterator;
template std::vector<FedChannelConnection>::const_iterator;
//template std::vector<uint16_t>::iterator;
//template std::vector<uint16_t>::const_iterator;
*/

