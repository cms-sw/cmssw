#include "DataFormats/CaloRecHit/interface/CaloCluster.h"


#include <sstream>
#include <iostream>

using namespace std;
using namespace reco;


void CaloCluster::reset() {
  position_ = math::XYZPoint();
  energy_ = 0;
  hitsAndFractions_.clear();
}

string CaloCluster::printHitAndFraction(unsigned i) const {
  
  ostringstream out; 
  if( i>=hitsAndFractions().size() ) // i >= 0, since i is unsigned
    out<<"out of range "<<i; 
  else
    out<<"( "<<hitsAndFractions()[i].first
       <<", "<<hitsAndFractions()[i].second
       <<" )";
  return out.str();
}


std::ostream& reco::operator<<(std::ostream& out, 
                               const CaloCluster& cluster) {
  
  if(!out) return out;

  const math::XYZPoint&  pos = cluster.position();

  out<<"CaloCluster , algoID="<<cluster.algoID()
     <<", "<<cluster.caloID()    
     <<", E="<<cluster.energy();
  if( cluster.correctedEnergy() != -1.0 ) {
    out << ", E_corr="<<cluster.correctedEnergy();
  }
  out<<", eta,phi="<<pos.eta()<<","<<pos.phi()
     <<", nhits="<<cluster.hitsAndFractions().size()<<endl;
  for(unsigned i=0; i<cluster.hitsAndFractions().size(); i++ ) {
    out<<""<<cluster.printHitAndFraction(i)<<", ";
  }

  return out;
}
