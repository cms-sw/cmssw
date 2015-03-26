#ifndef ImpactParameter_TemplatedNegativeTrackCountingComputer_h
#define ImpactParameter_TemplatedNegativeTrackCountingComputer_h

#include "RecoBTag/ImpactParameter/interface/TemplatedTrackCountingComputer.h"

template <class Container, class Base>
class TemplatedNegativeTrackCountingComputer : public TemplatedTrackCountingComputer<Container,Base>
{
 public:
  TemplatedNegativeTrackCountingComputer(const edm::ParameterSet & parameters ) : TemplatedTrackCountingComputer<Container,Base>(parameters)
  {
  }
 
  float discriminator(const JetTagComputer::TagInfoHelper & ti) const 
  {
    const typename TemplatedTrackCountingComputer<Container,Base>::TagInfo & tkip = ti.get<typename TemplatedTrackCountingComputer<Container,Base>::TagInfo>();
    std::multiset<float> significances = this->orderedSignificances(tkip);
    std::multiset<float>::iterator nth=significances.begin();
    for(int i=0;i<this->m_nthTrack-1 && nth!=significances.end();i++) nth++;  
    if(nth!=significances.end()) return -(*nth); else return -100.;
  }

};

#endif // ImpactParameter_TemplatedNegativeTrackCountingComputer_h