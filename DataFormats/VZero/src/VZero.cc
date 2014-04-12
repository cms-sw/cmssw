#include "DataFormats/VZero/interface/VZero.h"

using namespace reco;

VZero::VZero(const Vertex& vertex, const VZeroData& data) :
             vertex_(vertex), data_(data)
{ }
