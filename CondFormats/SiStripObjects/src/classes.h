#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
template std::vector<SiStripPedestals::SiStripData>::iterator;
template std::vector<SiStripPedestals::SiStripData>::const_iterator;
template std::map< uint32_t, SiStripPedestalsVector>::iterator;
template std::map< uint32_t, SiStripPedestalsVector>::const_iterator;

#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
template std::vector<SiStripNoises::SiStripData>::iterator;
template std::vector<SiStripNoises::SiStripData>::const_iterator;
template std::map< uint32_t, SiStripNoiseVector>::iterator;
template std::map< uint32_t, SiStripNoiseVector>::const_iterator;

#include "CondFormats/SiStripObjects/interface/SiStripReadoutCabling.h"
template std::vector<std::pair<uint32_t, unsigned short> >::iterator;
template std::vector<std::pair<uint32_t, unsigned short> >::const_iterator;

#include "CondFormats/SiStripObjects/interface/SiStripControlCabling.h"
template std::vector< std::vector<std::pair<uint32_t, unsigned short> > >::iterator;
template std::vector< std::vector<std::pair<uint32_t, unsigned short> > >::const_iterator;
template std::vector<unsigned short>::iterator;
template std::vector<unsigned short>::const_iterator;

#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
template std::vector< std::vector<FedChannelConnection> >::iterator;
template std::vector< std::vector<FedChannelConnection> >::const_iterator;
template std::vector<FedChannelConnection>::iterator;
template std::vector<FedChannelConnection>::const_iterator;


