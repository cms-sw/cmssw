#include "HeavyIonsAnalysis/PhotonAnalysis/interface/GenParticleParentage.h"
#include <iostream>

using namespace std;
using namespace genpartparentage;

GenParticleParentage::
GenParticleParentage(reco::GenParticleRef& _match) {
  if( _match.isNonnull() && _match.isAvailable() ) {
    getParentageRecursive(_match, 0);
    resolveParentage();
  }
}

void GenParticleParentage::getParentageRecursive(const reco::GenParticleRef& p, int daughterId) {
  // stopping condition
  if( p->numberOfMothers() == 0 ) return; // no mothers
  if( p->pt() < 0.1 ) return;  // reached initial state quarks/proton

  // do not count a copy of the particle, and do not start with the particle itself :)
  if ( std::abs(daughterId) != std::abs(p->pdgId()) && daughterId != 0 ) {    
    switch(std::abs(p->pdgId()))  {
    case 12:
    case 14:
    case 16:
    case 22:
      break; // disregard neutrinos, photons
    case 11:
    case 13:
    case 15:
      _leptonParents.push_back(p);
      break;
    case 21:
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
    case 2212:
      _qcdParents.push_back(p);
      break;
    case 23:
    case 24:
    case 25:
      _ewkBosonParents.push_back(p);
      break;
      // excited leptons
    case 4000001:
    case 4000002:
    case 4000011:
    case 4000012:
      // technicolor
    case 3000111:
    case 3000211:
    case 3000221:
    case 3100221:
    case 3000113:
    case 3000213:
    case 3000223:
    case 3100021:
    case 3060111:
    case 3160111:
    case 3130113:
    case 3140113:
    case 3150113:
    case 3160113:
      // Special
    case 39:
    case 41:
    case 42:
      _exoticParents.push_back(p);
      break;
    default:
      _nonPromptParents.push_back(p);
    }
  }

  if ( daughterId != 0 && 
       !_realParent.isNonnull() && 
       p->pdgId() != daughterId ) {
    _realParent = p;
  }

  const int nmom = p->numberOfMothers();
  for( int i = 0; i < nmom; ++i ) {
    reco::GenParticleRef next = p->motherRef(i);
    if( next.isAvailable() && next.isNonnull() ) {
      getParentageRecursive(next, p->pdgId());
    }
  }
}

void GenParticleParentage::resolveParentage() {
  auto lp = _leptonParents.cbegin();
  auto lpend = _leptonParents.cend();  
  for( ; lp != lpend; ++lp ) {
    if( lp == _leptonParents.cbegin() ) {
      _leptonParent = *lp;
    } else if( hasAsParent(_leptonParent,*lp) ) {
      _leptonParent = *lp;
    }
  }

  auto qp = _qcdParents.cbegin();
  auto qpend = _qcdParents.cend(); 
  for( ; qp != qpend; ++qp ) {    
    if( qp == _qcdParents.cbegin() ) {
      _qcdParent = *qp;
    } else if( hasAsParent(_qcdParent,*qp) ) {
      _qcdParent = *qp;
    }    
  }

  auto ep = _ewkBosonParents.cbegin();
  auto epend = _ewkBosonParents.cend();  
  for( ; ep != epend; ++ep ) {     
    if( ep == _ewkBosonParents.cbegin() ) {
      _ewkBosonParent = *ep;
    } else if( hasAsParent(_ewkBosonParent,*ep) ) {
      _ewkBosonParent = *ep;
    }   
  }
  
  auto np = _nonPromptParents.cbegin();
  auto npend = _nonPromptParents.cend();
  for( ; np != npend; ++np ) {    
    if( np == _nonPromptParents.cbegin() ) {
      _nonPromptParent = *np;
    } else if( hasAsParent(_nonPromptParent,*np) ) {
      _nonPromptParent = *np;
    }   
  }

  auto exo = _exoticParents.cbegin();
  auto exoend = _exoticParents.cend();
  for( ; exo != exoend; ++exo ) {    
    if( exo == _exoticParents.cbegin() ) {
      _exoticParent = *exo;
    } else if( hasAsParent(_exoticParent,*exo) ) {
      _exoticParent = *exo;
    }   
  }
  /*
  std::cout << "Best parents:" << std::endl;
  if( _leptonParent.isNonnull() && _leptonParent.isAvailable() ) {
    std::cout << "Lepton Parent: " << _leptonParent->pdgId() << ' ' 
	      << _leptonParent->status() << std::endl;
  }
  if( _qcdParent.isNonnull() && _qcdParent.isAvailable() ) {
    std::cout << "QCD Parent: " << _qcdParent->pdgId() << ' ' 
	      << _qcdParent->status() << std::endl;
  }
  if( _ewkBosonParent.isNonnull() && _ewkBosonParent.isAvailable() ) {
    std::cout << "EWK Boson Parent: " << _ewkBosonParent->pdgId() << ' ' 
	      << _ewkBosonParent->status() << std::endl;
  }
  if( _nonPromptParent.isNonnull() && _nonPromptParent.isAvailable() ) {
    std::cout << "NonPrompt Parent: " << _nonPromptParent->pdgId() << ' ' 
	      << _nonPromptParent->status() << std::endl;
  }
  */

}

bool GenParticleParentage::hasAsParent(const reco::GenParticleRef& d,
				  const reco::GenParticleRef& pc) const {
  if( d->numberOfMothers() == 0 ) return false;
  const int nmom = d->numberOfMothers();
  bool result = false;
  for( int i = 0; i < nmom; ++i ) {
    if( pc == d->motherRef(i) ) return true;
    else {
      result += hasAsParent(d->motherRef(i),pc);
    }
  }
  return result;
}
