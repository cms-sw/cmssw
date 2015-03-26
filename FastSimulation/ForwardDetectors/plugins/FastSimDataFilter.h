#ifndef FastSimDataFilter_h
#define FastSimDataFilter_h


#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "DataFormats/Math/interface/Vector3D.h"

#include <vector>
#include <utility>
#include <ostream>
#include <string>
#include <algorithm>
#include <cmath>

namespace edm {
  class Event;
  class EventSetup;
  class ParameterSet;
}


namespace cms
{

  class  FastSimDataFilter : public edm::stream::EDFilter <>
{
public:  

  FastSimDataFilter(const edm::ParameterSet& pset);

  virtual ~FastSimDataFilter();

  virtual bool filter(edm::Event&, const edm::EventSetup&);
  virtual void beginJob();
  virtual void endJob();
   
private:

  typedef math::RhoEtaPhiVector Vector;

  double towercut;
  int ntotal, npassed;
 
};
}
#endif
