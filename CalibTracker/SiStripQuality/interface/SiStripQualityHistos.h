#ifndef SiStripQualityHistos_H
#define SiStripQualityHistos_H

#include <unordered_map>
#include "TH1F.h"

namespace SiStrip {
  typedef std::unordered_map<unsigned int, std::shared_ptr<TH1F> > QualityHistosMap;
}
#endif
