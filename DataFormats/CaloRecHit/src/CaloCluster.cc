#include "DataFormats/CaloRecHit/interface/CaloCluster.h"


#include <sstream>
#include <iostream>

using namespace std;
using namespace reco;

string CaloCluster::printHitAndFraction(unsigned i) const {
  
  ostringstream out; 
  if( i<0 || i>=hitsAndFractions().size() ) 
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
     <<", E="<<cluster.energy()
     <<", eta,phi="<<pos.eta()<<","<<pos.phi()
     <<", nhits="<<cluster.hitsAndFractions().size()<<endl;
  for(unsigned i=0; i<cluster.hitsAndFractions().size(); i++ ) {
    out<<""<<cluster.printHitAndFraction(i)<<", ";
  }

  return out;
}
