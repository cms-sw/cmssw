// Declaration of the iterator (necessary for the generation of the dictionary)

#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
template std::vector<SiStripPedestals::SiStripData>::iterator;
template std::vector<SiStripPedestals::SiStripData>::const_iterator;
template std::map< uint32_t, SiStripPedestalsVector>::iterator;
template std::map< uint32_t, SiStripPedestalsVector>::const_iterator;

#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
template SiStripNoises::SiStripNoiseVector::iterator;
template SiStripNoises::SiStripNoiseVector::const_iterator;
template SiStripNoises::Registry::iterator;                 
template SiStripNoises::Registry::const_iterator; 

#include "CondFormats/SiStripObjects/interface/SiStripReadoutCabling.h"
#include "CondFormats/SiStripObjects/interface/SiStripControlCabling.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
template std::vector<std::pair<uint32_t, unsigned short> >::iterator;
template std::vector<std::pair<uint32_t, unsigned short> >::const_iterator;

template std::vector< std::vector<std::pair<uint32_t, unsigned short> > >::iterator;
template std::vector< std::vector<std::pair<uint32_t, unsigned short> > >::const_iterator;
template std::vector<unsigned short>::iterator;
template std::vector<unsigned short>::const_iterator;

template std::vector< std::vector<FedChannelConnection> >::iterator;
template std::vector< std::vector<FedChannelConnection> >::const_iterator;
template std::vector<FedChannelConnection>::iterator;
template std::vector<FedChannelConnection>::const_iterator;

