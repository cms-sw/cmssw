#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CondFormats/TrackerMapping/interface/SiStripReadOutCabling.h"
#include "CondFormats/TrackerMapping/interface/SiStripControlCabling.h"
// Declaration of the iterator (necessary for the generation of the dictionary)

template std::vector<SiStripPedestals::Item>::iterator;
template std::vector<SiStripPedestals::Item>::const_iterator;

template std::map< int, SiStripPedestals::SiStripPedestalsVector>::iterator;
template std::map< int, SiStripPedestals::SiStripPedestalsVector>::const_iterator;

