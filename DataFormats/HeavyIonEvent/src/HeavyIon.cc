
#include "DataFormats/HeavyIonEvent/interface/HeavyIon.h"

pat::HeavyIon::HeavyIon(){;}

pat::HeavyIon::HeavyIon(const reco::CentralityCollection & c, const reco::EvtPlaneCollection & e) :
   cents_(c),
   planes_(e),
   isMC_(false),
   b_(-99),
   npart_(-99),
   ncoll_(-99),
   nhard_(-99),
   phi_(-99) {;}

pat::HeavyIon::HeavyIon(const reco::CentralityCollection & c, const reco::EvtPlaneCollection & e, double b, int npart, int ncoll, int nhard, double phi) : 
   cents_(c),
   planes_(e),
   isMC_(true), 
   b_(b), 
   npart_(npart), 
   ncoll_(ncoll), 
   nhard_(nhard), 
   phi_(phi) {;}



