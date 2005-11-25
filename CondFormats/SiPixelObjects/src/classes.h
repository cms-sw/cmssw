#include "CondFormats/SiPixelObjects/interface/SiPixelPedestals.h"

// Declaration of the iterator (necessary for the generation of the dictionary)

template std::vector<SiPixelDbItem>::iterator;
template std::vector<SiPixelDbItem>::const_iterator;

template std::map< unsigned int, SiPixelPedestals::SiPixelPedestalsVector>::iterator;
template std::map< unsigned int, SiPixelPedestals::SiPixelPedestalsVector>::const_iterator;
