#include "SpikeAndDoubleSpikeCleaner.h"
#include <cmath>

namespace {
  bool greaterByEnergy(const std::pair<unsigned,double>& a,
		       const std::pair<unsigned,double>& b) {
    return a.second > b.second;
  }
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

SpikeAndDoubleSpikeCleaner::
SpikeAndDoubleSpikeCleaner(const edm::ParameterSet& conf) :
    RecHitTopologicalCleanerBase(conf),
    _layerMap({ {"PS2",(int)PFLayer::PS2},
	      {"PS1",(int)PFLayer::PS1},
	      {"ECAL_ENDCAP",(int)PFLayer::ECAL_ENDCAP},
	      {"ECAL_BARREL",(int)PFLayer::ECAL_BARREL},
	      {"NONE",(int)PFLayer::NONE},
	      {"HCAL_BARREL1",(int)PFLayer::HCAL_BARREL1},
	      {"HCAL_BARREL2_RING0",(int)PFLayer::HCAL_BARREL2},
              // hack to deal with ring1 in HO 
	      {"HCAL_BARREL2_RING1",100*(int)PFLayer::HCAL_BARREL2}, 
	      {"HCAL_ENDCAP",(int)PFLayer::HCAL_ENDCAP},
	      {"HF_EM",(int)PFLayer::HF_EM},
	      {"HF_HAD",(int)PFLayer::HF_HAD} }) {
  const std::vector<edm::ParameterSet>& thresholds =
    conf.getParameterSetVector("cleaningByDetector");
  for( const auto& pset : thresholds ) {
    spike_cleaning info;
    const std::string& det = pset.getParameter<std::string>("detector");
    info._minS4S1_a = pset.getParameter<double>("minS4S1_a");    
    info._minS4S1_b = pset.getParameter<double>("minS4S1_b");    
    info._doubleSpikeS6S2 = pset.getParameter<double>("doubleSpikeS6S2");
    info._eneThreshMod = pset.getParameter<double>("energyThresholdModifier");
    info._fracThreshMod = 
      pset.getParameter<double>("fractionThresholdModifier");
    info._doubleSpikeThresh = pset.getParameter<double>("doubleSpikeThresh");
    info._singleSpikeThresh = pset.getParameter<double>("singleSpikeThresh");
    auto entry = _layerMap.find(det);
    if( entry == _layerMap.end() ) {
      throw cms::Exception("InvalidDetectorLayer")
	<< "Detector layer : " << det << " is not in the list of recognized"
	<< " detector layers!";
    }
    _thresholds.emplace(_layerMap.find(det)->second,info);
  }  
}


void SpikeAndDoubleSpikeCleaner::
clean(const edm::Handle<reco::PFRecHitCollection>& input,
      std::vector<bool>& mask ) {
  //need to run over energy sorted rechits
  std::vector<std::pair<unsigned,double> > ordered_hits;
  for( unsigned i = 0; i < input->size(); ++i ) {
    std::pair<unsigned,double> val = std::make_pair(i,input->at(i).energy());
    auto pos = std::upper_bound(ordered_hits.begin(),ordered_hits.end(),
				val, greaterByEnergy);
    ordered_hits.insert(pos,val);
  }  

  for( const auto& idx_e : ordered_hits ) {
    const unsigned i = idx_e.first;
    if( !mask[i] ) continue; // don't need to re-mask things :-)
    const reco::PFRecHit& rechit = input->at(i);
    int hitlayer = (int)rechit.layer();
    if( hitlayer == PFLayer::HCAL_BARREL2 && 
	std::abs(rechit.positionREP().eta()) > 0.34 ) {
      hitlayer *= 100;
    }    
    const spike_cleaning& clean = _thresholds.find(hitlayer)->second;    
    if( rechit.energy() < clean._singleSpikeThresh ) continue;
    const double rhenergy = rechit.energy();
    // single spike cleaning
    const reco::PFRecHitRefVector& neighbours4 = rechit.neighbours4();
    double surroundingEnergy = rechit.energy();
    double neighbourEnergy = 0.0;
    double layerEnergy = 0.0;
    for( const reco::PFRecHitRef& neighbour : neighbours4 ) {
      if( !mask[neighbour.key()] ) continue;
      const double sum = neighbour->energy(); //energyUp is just rechit energy?
      surroundingEnergy += sum;
      neighbourEnergy   += sum;
      layerEnergy       += neighbour->energy();
    }    
    //   wannaBeSeed.energyUp()/wannaBeSeed.energy() : 1.;
    // Fraction 1 is the balance between the hit and its neighbours 
    // from both layers
    const double fraction1 = surroundingEnergy/rhenergy;    
    // removed spurious comments from old pfcluster algo... 
    // look there if you want more history
    const double f1Cut = ( clean._minS4S1_a*std::log10(rechit.energy()) + 
			   clean._minS4S1_b );
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
	    (rhenergy > clean._eneThreshMod*clean._singleSpikeThresh && 
	     fraction1 < f1Cut/clean._fracThreshMod ) ) ) {	
	mask[i] = false;
      }
    }//if initial fraction cut (single spike)
    // double spike removal
    if( mask[i] && rhenergy > clean._doubleSpikeThresh ) {
      //Determine energy surrounding the seed and the most energetic neighbour
      double surroundingEnergyi = 0.0;      
      double enmax = -999.0;
      reco::PFRecHitRef mostEnergeticNeighbour;
      const reco::PFRecHitRefVector& neighbours4i = rechit.neighbours4();
      for( reco::PFRecHitRef neighbour : neighbours4i ) {
	if( !mask[neighbour.key()] ) continue;
	const double nenergy = neighbour->energy();
	surroundingEnergyi += nenergy;
	if( nenergy > enmax ) {
	  enmax = nenergy;
	  mostEnergeticNeighbour = neighbour;
	}
      }
      // is there an energetic neighbour
      if( enmax > 0.0 ) {
	double surroundingEnergyj = 0.0;
	const reco::PFRecHitRefVector& neighbours4j = 
	  mostEnergeticNeighbour->neighbours4();
	for( const reco::PFRecHitRef& neighbour : neighbours4j ) {
	  //if( !mask[neighbour] &&  neighbour != i) continue; // leave out?
	  surroundingEnergyj += neighbour->energy();
	}
	// The energy surrounding the double spike candidate 
	const double surroundingEnergyFraction = 
	  (surroundingEnergyi+surroundingEnergyj) / 
	  (rechit.energy()+mostEnergeticNeighbour->energy()) - 1.;
	if ( surroundingEnergyFraction < clean._doubleSpikeS6S2 ) { 
	  const double eta = rechit.positionREP().eta();
	  const double aeta = std::abs(eta);
	  const double phi = rechit.positionREP().phi();
	  std::pair<double,double> dcr = dCrack(phi,eta);
	  const double dcrmin = ( rechit.layer() == PFLayer::ECAL_BARREL ? 
				  std::min(dcr.first,dcr.second):
				  dcr.second );
	  if( aeta < 5.0 && 
	      ( (aeta < 2.85 && dcrmin > 1.0) || 
		(rhenergy > clean._eneThreshMod*clean._doubleSpikeThresh && 
		 surroundingEnergyFraction < clean._doubleSpikeS6S2/clean._fracThreshMod ) 
		) ) {	    
	    mask[i] = false;
	    mask[mostEnergeticNeighbour.key()] = false;
	  }
	}
      } // was there an energetic neighbour ? 
    }// if double spike thresh
  } // rechit loop
}
