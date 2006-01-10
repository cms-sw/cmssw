#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CondFormats/SiStripObjects/interface/SiStripReadoutCabling.h"
#include "CondFormats/SiStripObjects/interface/SiStripControlCabling.h"
// Declaration of the iterator (necessary for the generation of the dictionary)

template std::vector<SiStripPedestals::SiStripData>::iterator;
template std::vector<SiStripPedestals::SiStripData>::const_iterator;

template std::map< uint32_t, SiStripPedestalsVector>::iterator;
template std::map< uint32_t, SiStripPedestalsVector>::const_iterator;

template std::vector<std::pair<uint32_t, unsigned short> >::iterator;
template std::vector<std::pair<uint32_t, unsigned short> >::const_iterator;

template std::vector< std::vector<std::pair<uint32_t, unsigned short> > >::iterator;
template std::vector< std::vector<std::pair<uint32_t, unsigned short> > >::const_iterator;
template std::vector<unsigned short>::iterator;
template std::vector<unsigned short>::const_iterator;
