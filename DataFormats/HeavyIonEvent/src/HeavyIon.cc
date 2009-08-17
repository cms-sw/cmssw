
#include "DataFormats/HeavyIonEvent/interface/HeavyIon.h"

pat::HeavyIon::HeavyIon() : reco::Centrality(), reco::EvtPlane() {;}

pat::HeavyIon::HeavyIon(const reco::Centrality & c, const reco::EvtPlane & e) :
   reco::Centrality(c),
   reco::EvtPlane(e),
   isMC_(false),
   b_(-99),
   npart_(-99),
   ncoll_(-99),
   nhard_(-99),
   phi_(-99) {;}

pat::HeavyIon::HeavyIon(const reco::Centrality & c, const reco::EvtPlane & e, double b, int npart, int ncoll, int nhard, double phi) : 
   reco::Centrality(c), 
   reco::EvtPlane(e), 
   isMC_(true), 
   b_(b), 
   npart_(npart), 
   ncoll_(ncoll), 
   nhard_(nhard_), 
   phi_(phi) {;}


bool pat::HeavyIon::initializeCentrality(){
   // Get Bin Values from "DataBase"
   return false;
}

int pat::HeavyIon::centralityBin(){
   return 0;
}

