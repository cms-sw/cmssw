#include "SpikeAndDoubleSpikeCleaner.h"
#include <cmath>

namespace {
  std::pair<double,double> dCrack(double phi, double eta) {
    constexpr double oneOverCrystalSize=1.0/0.0175;
    constexpr double pi=M_PI;
    constexpr double twopi=2*pi;
    // the result below is from unrolling 
    // the lazy-eval loop in PFClusterAlgo::dCrack
    constexpr double cPhi[18] = {2.97025, 2.621184149601134, 2.272118299202268,
				 1.9230524488034024, 1.5739865984045365, 
				 1.2249207480056705, 0.8758548976068048, 
				 0.5267890472079388, 0.1777231968090729, 
				 -0.17134265358979306, -0.520408503988659, 
				 -0.8694743543875245, -1.2185402047863905, 
				 -1.5676060551852569, -1.9166719055841224, 
				 -2.265737755982988, -2.6148036063818543, 
				 -2.9638694567807207};
    constexpr double cEta[9] = {0.0,
				4.44747e-01, -4.44747e-01,
				7.92824e-01, -7.92824e-01,
				1.14090e+00, -1.14090e+00,
				1.47464e+00, -1.47464e+00};
    // shift for eta < 0
    constexpr double delta_cPhi = 0.00638;
    // let's calculate dphi
    double defi = 0;
    if( eta < 0 ) phi+=delta_cPhi;
    if( phi >= -pi && phi <= pi ) {
      //the problem of the extrema
      if( phi < cPhi[17] || phi >= cPhi[0] ) {
	if( phi < 0 ) phi += 2*pi;
	defi = std::min(std::abs(phi-cPhi[0]),std::abs(phi-cPhi[17]-twopi));
      } else { // between these extrema
	bool OK = false;
	unsigned i = 16;
	while(!OK) {
	  if( phi < cPhi[i] ) {
	    defi = std::min(std::abs(phi-cPhi[i+1]),std::abs(phi-cPhi[i]));
	    OK=true;
	  } else {
	    i -= 1;
	  }
	}// end while
      }
    } else { // if there's a problem assume we're in a crack
      defi = 0;      
    }
    // let's calculate deta
    double deta = 999.0;
    for( const double etaGap : cEta ) {
      deta = std::min(deta,std::abs(eta-etaGap));
    }
    defi *= oneOverCrystalSize;
    deta *= oneOverCrystalSize;
    return std::make_pair(defi,deta);
  }
}

void SpikeAndDoubleSpikeCleaner::
clean(const edm::Handle<reco::PFRecHitCollection>& input,
      std::vector<bool>& mask ) {
  for( unsigned i = 0; i < input->size(); ++i ) {
    if( !mask[i] ) continue; // don't need to re-mask things :-)
    const reco::PFRecHit& rechit = input->at(i);
    if( rechit.energy() > _cleaningThreshold ) continue;
    const double rhenergy = rechit.energy();
    // single spike cleaning
    const std::vector<unsigned>& neighbours4 = rechit.neighbours4();
    double surroundingEnergy = rechit.energyUp();
    double neighbourEnergy = 0.0;
    double layerEnergy = 0.0;
    for( const unsigned neighbour : neighbours4 ) {
      if( !mask[neighbour] ) continue;
      const reco::PFRecHit& nrechit = input->at(neighbour);
      const double sum = nrechit.energy() + nrechit.energyUp();
      surroundingEnergy += sum;
      neighbourEnergy   += sum;
      layerEnergy       += nrechit.energy();
    }    
    //   wannaBeSeed.energyUp()/wannaBeSeed.energy() : 1.;
    // Fraction 1 is the balance between the hit and its neighbours 
    // from both layers
    const double fraction1 = surroundingEnergy/rhenergy;    
    // removed spurious comments from old pfcluster algo... 
    // look there if you want more history
    const double f1Cut = _minS4S1_a*std::log10(rechit.energy()) + _minS4S1_b;
    if( fraction1 < f1Cut ) { 
      const double eta = rechit.positionREP().eta();
      const double aeta = std::abs(eta);
      const double phi = rechit.positionREP().phi();
      std::pair<double,double> dcr = dCrack(phi,eta);
      const double dcrmin = ( rechit.layer() ==  PFLayer::ECAL_BARREL ? 
			      std::min(dcr.first,dcr.second):
			      dcr.second );
      if( aeta < 5.0 && 
	  ( (aeta < 2.85 && dcrmin > 1.0) || 
	    (rhenergy > _eneThreshMod*_cleaningThreshold && 
	     fraction1 < f1Cut/_fracThreshMod ) ) ) {
	mask[i] = false;
      }
    }//if initial fraction cut (single spike)
    // double spike removal
    if( mask[i] && rhenergy > _doubleSpikeThresh ) {
      //Determine energy surrounding the seed and the most energetic neighbour
      double surroundingEnergyi = 0.0;      
      double enmax = -999.0;
      unsigned mostEnergeticNeighbour = 0;
      const std::vector<unsigned>& neighbours4i = rechit.neighbours4();
      for( const unsigned neighbour : neighbours4i ) {
	if( !mask[neighbour] ) continue;
	const reco::PFRecHit& nrechit = input->at(neighbour);
	const double nenergy = nrechit.energy();
	surroundingEnergyi += nenergy;
	if( nenergy > enmax ) {
	  enmax = nenergy;
	  mostEnergeticNeighbour = neighbour;
	}
      }
      // is there an energetic neighbour
      if( enmax > 0.0 ) {
	double surroundingEnergyj = 0.0;
	const reco::PFRecHit& neighbourj = input->at(mostEnergeticNeighbour);
	const std::vector<unsigned>& neighbours4j = neighbourj.neighbours4();
	for( const unsigned neighbour : neighbours4j ) {
	  //if( !mask[neighbour] &&  neighbour != i) continue; // leave out?
	  const reco::PFRecHit& nrechit = input->at(neighbour);
	  surroundingEnergyj += nrechit.energy();
	}
	// The energy surrounding the double spike candidate 
	const double surroundingEnergyFraction = 
	  (surroundingEnergyi+surroundingEnergyj) / 
	  (rechit.energy()+neighbourj.energy()) - 1.;
	if ( surroundingEnergyFraction < _doubleSpikeS6S2 ) { 
	  const double eta = rechit.positionREP().eta();
	  const double aeta = std::abs(eta);
	  const double phi = rechit.positionREP().phi();
	  std::pair<double,double> dcr = dCrack(phi,eta);
	  const double dcrmin = ( rechit.layer() == PFLayer::ECAL_BARREL ? 
				  std::min(dcr.first,dcr.second):
				  dcr.second );
	  if( aeta < 5.0 && 
	      ( (aeta < 2.85 && dcrmin > 1.0) || 
		(rhenergy > _eneThreshMod*_doubleSpikeThresh && 
		 surroundingEnergyFraction < _doubleSpikeS6S2/_fracThreshMod ) 
		) ) {
	    mask[i] = false;
	    mask[mostEnergeticNeighbour] = false;
	  }
	}
      } // was there an energetic neighbour ? 
    }// if double spike thresh
  } // rechit loop
}
