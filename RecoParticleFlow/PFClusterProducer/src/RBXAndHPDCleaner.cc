#include "RBXAndHPDCleaner.h"

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include <cmath>

namespace {
   bool greaterByEnergy(const std::pair<unsigned,double>& a,
		       const std::pair<unsigned,double>& b) {
    return a.second > b.second;
  }
}


void RBXAndHPDCleaner::
clean(const edm::Handle<reco::PFRecHitCollection>& input,
      std::vector<bool>& mask ) {
  _hpds.clear(); _rbxs.clear();
  //need to run over energy sorted rechits
  std::vector<std::pair<unsigned,double> > ordered_hits;
  ordered_hits.reserve(input->size());
  for( unsigned i = 0; i < input->size(); ++i ) {
    std::pair<unsigned,double> val = std::make_pair(i,input->at(i).energy());
    auto pos = std::upper_bound(ordered_hits.begin(),ordered_hits.end(),
				val, greaterByEnergy);
    ordered_hits.insert(pos,val);
  }   
 
  for( const auto& idx_e : ordered_hits ) {
    if( !mask[idx_e.first] ) continue;
    const unsigned idx = idx_e.first;
    const reco::PFRecHit& rechit = input->at(idx);
    int layer = rechit.layer();
    if ( layer != PFLayer::HCAL_BARREL1 && layer != PFLayer::HCAL_ENDCAP ) {
      break; 
    }
    HcalDetId theHcalDetId(rechit.detId());
    int ieta = theHcalDetId.ieta();
    int iphi = theHcalDetId.iphi();
    int ihpd = 0, irbx = 0;
    switch( layer ) {
    case PFLayer::HCAL_BARREL1:
      ihpd = ( ieta < 0 ? -iphi : iphi );
      irbx = ( ieta < 0 ? -(iphi+5)/4 : (iphi+5)/4 );
      break;
    case PFLayer::HCAL_ENDCAP:
      ihpd = ( ieta < 0 ? -(iphi+1)/2-100 : (iphi+1)/2+100 );
      irbx = ( ieta < 0 ? -(iphi+5)/4-20 : (iphi+5)/4+20 );
      break;
    default:
      break;
    }
    switch( std::abs(irbx) ) {
    case 19:      
      irbx = ( irbx < 0 ? -1 : 1 );
      break;
    case 39:
      irbx = ( irbx < 0 ? -21 : 21 );
      break;
    default:
      break;
    }
    _hpds[ihpd].push_back(idx);
    _rbxs[irbx].push_back(idx);
  }
  // loop on the rbx's we found and clean RBX's with tons of rechits
  // and lots of energy
  double totalEta = 0., totalEtaW = 0., totalPhi = 0., totalPhiW = 0., 
    totalEnergy = 0.;
  double totalEta2 = 1E-9, totalEta2W = 1E-9, totalPhi2 = 1E-9, 
    totalPhi2W = 1E-9, totalEnergy2 = 1E-9;
  unsigned nSeeds = 0, nSeeds0 = 0;
  std::unordered_map<int, std::vector<unsigned> > theHPDs;
  std::unordered_multimap<double, unsigned> theEnergies;
  for( const auto& itrbx : _rbxs ) {
    if( ( std::abs(itrbx.first)<20 && itrbx.second.size() > 30 ) || 
	( std::abs(itrbx.first)>20 && itrbx.second.size() > 30 ) ) {
      const std::vector<unsigned>& rechits = itrbx.second;
      theHPDs.clear();
      theEnergies.clear();
      totalEta = totalEtaW = totalPhi = totalPhiW = totalEnergy = 0.;
      totalEta2 = totalEta2W = totalPhi2 = totalPhi2W = totalEnergy2 = 1e-9;
      nSeeds = nSeeds0 = rechits.size();
      for( unsigned jh = 0; jh < rechits.size(); ++jh ) {
	const reco::PFRecHit&  rechit = input->at(jh);
	// check if rechit is a seed
	unsigned nN = 0 ; // neighbours over threshold
	bool isASeed = true;
	const reco::PFRecHitRefVector& neighbours4 = rechit.neighbours4();
	for( const reco::PFRecHitRef& neighbour : neighbours4 ) {
	  if( neighbour->energy() > rechit.energy() ) {	    
	    --nSeeds; --nSeeds0;
	    isASeed = false;
	    break;
	  } else {
	    if( neighbour->energy() > 0.4 ) ++nN;
	  }
	}
	if ( isASeed && !nN ) --nSeeds0;

	HcalDetId theHcalDetId(rechit.detId());
	int iphi = theHcalDetId.iphi();
	switch( rechit.layer() ) {
	case PFLayer::HCAL_BARREL1:
	  theHPDs[iphi].push_back(rechits[jh]);
	  break;
	case PFLayer::HCAL_ENDCAP:
	  theHPDs[(iphi-1)/2].push_back(rechits[jh]);
	  break;
	default:
	  break;
	}
	const double rhenergy = rechit.energy();
	const double rhphi = rechit.position().phi();
	const double rhphi2 = rhphi*rhphi;
	const double rheta = rechit.position().eta();
	const double rheta2 = rheta*rheta;
	theEnergies.emplace(rhenergy,rechits[jh]);
	totalEnergy += rhenergy;
	totalPhi += std::abs(rhphi);
	totalPhiW += std::abs(rhphi)*rhenergy;
	totalEta += rheta;
	totalEtaW += rheta*rhenergy;
	totalEnergy2 += rhenergy*rhenergy;
	totalPhi2 += rhphi2;
	totalPhi2W += rhphi2*rhenergy;
	totalEta2 += rheta2;
	totalEta2W += rheta2*rhenergy;
      }
      totalPhi /= rechits.size();
      totalEta /= rechits.size();
      totalPhiW /= totalEnergy;
      totalEtaW /= totalEnergy;
      totalPhi2 /= rechits.size();
      totalEta2 /= rechits.size();
      totalPhi2W /= totalEnergy;
      totalEta2W /= totalEnergy;
      totalPhi2 = std::sqrt(totalPhi2 - totalPhi*totalPhi);
      totalEta2 = std::sqrt(totalEta2 - totalEta*totalEta);
      totalPhi2W = std::sqrt(totalPhi2W - totalPhiW*totalPhi2);
      totalEta2W = std::sqrt(totalEta2W - totalEtaW*totalEtaW);
      totalEnergy /= rechits.size();
      totalEnergy2 /= rechits.size();
      totalEnergy2 = std::sqrt(totalEnergy2 - totalEnergy*totalEnergy);
      if( nSeeds0 > 6 ) {
	unsigned nHPD15 = 0;
	for( const auto& itHPD : theHPDs ) {
	  int hpdN = itHPD.first;
	  const std::vector<unsigned>& hpdHits = itHPD.second;
	  if( ( std::abs(hpdN) < 100 && hpdHits.size() > 14 ) || 
	      ( std::abs(hpdN) > 100 && hpdHits.size() > 14 ) ) ++nHPD15;
	}
	if( nHPD15 > 1 ) {
	  unsigned nn = 0;
	  double threshold = 1.0;
	  for( const auto& itEn : theEnergies ) {
	    ++nn;
	    if( nn< 5 ) {
	      mask[itEn.second] = false;
	    } else if ( nn == 5 ) {
	      threshold = itEn.first*5;
	      mask[itEn.second] = false;	      
	    } else {
	      if( itEn.first < threshold ) mask[itEn.second] = false;
	    }
	  }
	}
      }
    }
  }

  // loop on the HPDs
  std::unordered_map<int,std::vector<unsigned> >::iterator neighbour1;
  std::unordered_map<int,std::vector<unsigned> >::iterator neighbour2;
  std::unordered_map<int,std::vector<unsigned> >::iterator neighbour0;
  std::unordered_map<int,std::vector<unsigned> >::iterator neighbour3;
  unsigned size1 = 0, size2 = 0;  
  for( const auto& ithpd : _hpds ) {
    const std::vector<unsigned>& rechits = ithpd.second;
    theEnergies.clear();
    totalEnergy = 0;
    totalEnergy2 = 1e-9;
    for( const unsigned rhidx : rechits ) {
      const reco::PFRecHit & rechit = input->at(rhidx);
      const double e = rechit.energy();
      totalEnergy += e;
      totalEnergy2 += e*e;
      theEnergies.emplace(rechit.energy(),rhidx); 
    }
    totalEnergy /= rechits.size();
    totalEnergy2 /= rechits.size();
    totalEnergy2 = std::sqrt(totalEnergy2 - totalEnergy*totalEnergy);
   
    const int thehpd = ithpd.first;
    switch( std::abs(thehpd) ) {
    case 1:
      neighbour1 = ( thehpd > 0 ? _hpds.find(72) : _hpds.find(-72) );
      break;
    case 72:
      neighbour2 = ( thehpd > 0 ? _hpds.find(1) : _hpds.find(-1) );
      break;
    case 101:
      neighbour1 = ( thehpd > 0 ? _hpds.find(136) : _hpds.find(-136) );
      break;
    case 136:
      neighbour2 = ( thehpd > 0 ? _hpds.find(101) : _hpds.find(-101) );
      break;
    default:
      neighbour1 = ( thehpd > 0 ? _hpds.find(thehpd-1) : _hpds.find(thehpd+1) );
      neighbour2 = ( thehpd > 0 ? _hpds.find(thehpd+1) : _hpds.find(thehpd-1) );
      break;
    }
    if( neighbour1 != _hpds.end() ) {
      const int nb1 = neighbour1->first;
      switch( std::abs(nb1) ) {
      case 1:
	neighbour0 = ( nb1 > 0 ? _hpds.find(72) : _hpds.find(-72) );
	break;
      case 101:
	neighbour0 = ( nb1 > 0 ? _hpds.find(136) : _hpds.find(-136) );
	break;
      default:
	neighbour0 = ( nb1 > 0 ? _hpds.find(nb1-1) : _hpds.find(nb1+1) );
	break;
      }
    } else {
      neighbour0 = _hpds.end();
    }
    
    if( neighbour2 != _hpds.end() ) {
      const int nb2 = neighbour2->first;
      switch( std::abs(nb2) ) {
      case 72:
	neighbour3 = ( nb2 > 0 ? _hpds.find(1) : _hpds.find(-1) );
	break;
      case 136:
	neighbour3 = ( nb2 > 0 ? _hpds.find(101) : _hpds.find(-101) );
	break;
      default:
	neighbour3 = ( nb2 > 0 ? _hpds.find(nb2+1) : _hpds.find(nb2-1) );
	break;
      }
    } else {
      neighbour3 = _hpds.end();
    }
    
    size1 = neighbour1 != _hpds.end() ? neighbour1->second.size() : 0;
    size2 = neighbour2 != _hpds.end() ? neighbour2->second.size() : 0;
    if( size1 > 10 ) {
      if ( ( abs(neighbour1->first) > 100 && neighbour1->second.size() > 15 ) || 
	   ( abs(neighbour1->first) < 100 && neighbour1->second.size() > 12 ) ) 
	size1 = neighbour0 != _hpds.end() ? neighbour0->second.size() : 0;
    }
    if( size2 > 10 ) {
      if ( ( abs(neighbour2->first) > 100 && neighbour2->second.size() > 15 ) || 
	   ( abs(neighbour2->first) < 100 && neighbour2->second.size() > 12 ) ) 
	size2 = neighbour3 != _hpds.end() ? neighbour3->second.size() : 0;
    }
    if( ( std::abs(ithpd.first) > 100 && ithpd.second.size() > 15 ) ||
	( std::abs(ithpd.first) < 100 && ithpd.second.size() > 12 ) ) {
      if( (double)(size1+size2)/(float)ithpd.second.size() < 1.0 ) {
	unsigned nn = 0;
	double threshold = 1.0;
	for( const auto& itEn : theEnergies ) {
	  if( nn < 5 ) {
	    mask[itEn.second] = false;
	  } else if ( nn == 5 ) {
	    threshold = itEn.first*2.5;
	    mask[itEn.second] = false;
	  } else {
	    if( itEn.first < threshold ) mask[itEn.second] = false;
	  }
	}
      }
    }
  }
}
