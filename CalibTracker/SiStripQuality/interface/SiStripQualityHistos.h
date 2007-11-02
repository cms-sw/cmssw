#ifndef SiStripQualityHistos_H
#define  SiStripQualityHistos_H

#include "boost/shared_ptr.hpp"
#include <ext/hash_map>
#include "TH1F.h"

namespace SiStrip{
  typedef __gnu_cxx::hash_map< unsigned int, boost::shared_ptr<TH1F> > QualityHistosMap;  
}
#endif 
