#include "FastSimulation/Event/interface/KineParticleFilter.h"
#include "FastSimulation/Particle/interface/RawParticle.h"
//Framework Headers
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>
#include <iterator>

KineParticleFilter::KineParticleFilter(const edm::ParameterSet& kine) 
{

  // Set the kinematic cuts
  // Upper abs(eta) bound
  etaMax = kine.getParameter<double>("etaMax"); 
  // Lower pT  bound (charged, in GeV/c)
  pTMin  = kine.getParameter<double>("pTMin");
  // Lower E  bound - reject (all, in GeV)
  EMin   = kine.getParameter<double>("EMin");
  // Lower E  bound - accept (all, in GeV)
  EMax   = kine.getParameter<double>("EProton");

  // pdg codes of the particles to be removed from the events
  // ParameterSet cannot handle sets, only vectors
  std::vector<int> tmpcodes 
    = kine.getUntrackedParameter< std::vector<int> >
    ("forbiddenPdgCodes", std::vector<int>() );
  
  std::copy(tmpcodes.begin(), 
	    tmpcodes.end(),  
	    std::insert_iterator< std::set<int> >(forbiddenPdgCodes,
						  forbiddenPdgCodes.begin() ));
  
  if( !forbiddenPdgCodes.empty() ) {
    std::cout<<"KineParticleFilter : Forbidden PDG codes : ";
    copy(forbiddenPdgCodes.begin(), forbiddenPdgCodes.end(), 
	 std::ostream_iterator<int>(std::cout, " "));
  }  

  // Change eta cuts to cos**2(theta) cuts (less CPU consuming)
  if ( etaMax > 20. ) etaMax = 20.; // Protection against paranoid people.
  double cosMax = (std::exp(2.*etaMax)-1.) / (std::exp(2.*etaMax)+1.);
  cos2Max = cosMax*cosMax;

  double etaPreshMin = 1.479;
  double etaPreshMax = 1.594;
  double cosPreshMin = (std::exp(2.*etaPreshMin)-1.) / (std::exp(2.*etaPreshMin)+1.);
  double cosPreshMax = (std::exp(2.*etaPreshMax)-1.) / (std::exp(2.*etaPreshMax)+1.);
  cos2PreshMin = cosPreshMin*cosPreshMin;
  cos2PreshMax = cosPreshMax*cosPreshMax;

  // Change pt cut to pt**2 cut (less CPU consuming)
  pTMin *= pTMin;

}

bool KineParticleFilter::accept(const RawParticle & particle) const
{

  // Do not consider quarks, gluons, Z, W, strings, diquarks
  // ... and supesymmetric particles
  int pId = abs(particle.pid());

  // Vertices are coming with pId = 0
  if ( pId != 0 ) { 
    bool particleCut = ( pId > 10  && pId != 12 && pId != 14 && 
			 pId != 16 && pId != 18 && pId != 21 &&
			 (pId < 23 || pId > 40  ) &&
			 (pId < 81 || pId > 100 ) && pId != 2101 &&
			 pId != 3101 && pId != 3201 && pId != 1103 &&
			 pId != 2103 && pId != 2203 && pId != 3103 &&
			 pId != 3203 && pId != 3303 );
    //    particleCut = particleCut || pId == 0;


    if ( !particleCut ) return false;

    // Keep protons with energy in excess of 5 TeV
    bool protonTaggers =  (pId == 2212 && particle.E() >= EMax) ;
    if ( protonTaggers ) return true;

    std::set<int>::iterator is = forbiddenPdgCodes.find(pId);
    if( is != forbiddenPdgCodes.end() ) return false;

  //  bool kineCut = pId == 0;
  // Cut on kinematic properties
    // Cut on the energy of all particles
    bool eneCut = particle.E() >= EMin;
    if (!eneCut) return false;

    // Cut on the transverse momentum of charged particles
    bool pTCut = particle.charge()==0 || particle.Perp2()>=pTMin;
    if (!pTCut) return false;

    // Cut on eta if the origin vertex is close to the beam
    //    bool etaCut = (particle.vertex()-mainVertex).perp()>5. || fabs(particle.eta())<=etaMax;
    bool etaCut = (particle.vertex()-mainVertex).Perp2()>25. || particle.cos2Theta()<= cos2Max;

    /*
    if ( etaCut != etaCut2 ) 
      cout << "WANRNING ! etaCut != etaCut2 " 
	   << etaCut << " " 
	   << etaCut2 << " "
	   << (particle.eta()) << " " << etaMax << " " 
	   << particle.vect().cos2Theta() << " " << cos2Max << endl; 
    */
    if (!etaCut) return false;

    // Cut on the origin vertex position (prior to the ECAL for all 
    // particles, except for muons  ! Just modified: Muons included as well !
    double radius2 = particle.R2();
    double zed = fabs(particle.Z());
    double cos2Tet = particle.cos2ThetaV();
    // Ecal entrance
    bool ecalAcc = ( (radius2<129.01*129.01 && zed<317.01) ||
		     (cos2Tet>cos2PreshMin && cos2Tet<cos2PreshMax 
		      && radius2<171.11*171.11 && zed<317.01) );

    return ecalAcc;

  } else { 
    // Cut for vertices
    double radius2 = particle.Perp2();
    double zed = fabs(particle.Pz());
    double cos2Tet = particle.cos2Theta();

    // Vertices must be before the Ecal entrance
    bool ecalAcc = ( (radius2<129.01*129.01 && zed<317.01) ||
		     (cos2Tet>cos2PreshMin && cos2Tet<cos2PreshMax 
		      && radius2<171.11*171.11 && zed<317.01) );

    return ecalAcc;

  }

}
