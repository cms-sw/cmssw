#ifndef SiStripQualityHistos_H
#define SiStripQualityHistos_H

#include <ext/hash_map>
#include "TH1F.h"

namespace SiStrip {
  typedef __gnu_cxx::hash_map<unsigned int, std::shared_ptr<TH1F> > QualityHistosMap;
}
#endif
