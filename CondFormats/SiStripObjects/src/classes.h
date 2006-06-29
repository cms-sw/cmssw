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

