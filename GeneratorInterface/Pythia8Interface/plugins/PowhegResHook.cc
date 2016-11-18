// Hook for setting shower scale in top and W resonances
// for Powheg ttb_NLO_dec and b_bbar_4l processes
// C++ port of algorithm by Jezo et. al. (arXiv:1607.04538, Appendix B.2)

#include "Pythia8/Pythia.h"

using namespace Pythia8;

#include "GeneratorInterface/Pythia8Interface/plugins/PowhegResHook.h"

double PowhegResHook::scaleResonance( const int iRes, const Event& event) {
  calcScales_ = settingsPtr->flag("POWHEGres:calcScales");
  
  double scale = 0.;
  
  int nDau = event[iRes].daughterList().size();
  
  if (!calcScales_ or nDau == 0) {
    // No resonance found, set scale to high value
    // Pythia will shower any MC generated resonance unrestricted
    scale = 1e30;
  }
  
  else if (nDau < 3) {
    // No radiating resonance found
    scale = 0.8;
  }
  
  else if (abs(event[iRes].id()) == 6) {
    // Find top daughters
    int idw = -1, idb = -1, idg = -1;
    
    for (int i = 0; i < nDau; i++) {
      int iDau = event[iRes].daughterList()[i];
      if (abs(event[iDau].id()) == 24) idw = iDau;
      if (abs(event[iDau].id()) ==  5) idb = iDau;
      if (abs(event[iDau].id()) == 21) idg = iDau;
    }
    
    // Get daughter 4-vectors in resonance frame
    Vec4 pw(event[idw].p());
    pw.bstback(event[iRes].p());
    
    Vec4 pb(event[idb].p());
    pb.bstback(event[iRes].p());
    
    Vec4 pg(event[idg].p());
    pg.bstback(event[iRes].p());
    
    // Calculate scale
    scale = sqrt(2*pg*pb*pg.e()/pb.e());
  }
  
  else if (abs(event[iRes].id()) == 24) {
    // Find W daughters
    int idq = -1, ida = -1, idg = -1;
    
    for (int i = 0; i < nDau; i++) {
      int iDau = event[iRes].daughterList()[i];
      if      (event[iDau].id() == 21) idg = iDau;
      else if (event[iDau].id()  >  0) idq = iDau;
      else if (event[iDau].id()  <  0) ida = iDau;
    }
    
    // Get daughter 4-vectors in resonance frame
    Vec4 pq(event[idq].p());
    pq.bstback(event[iRes].p());
    
    Vec4 pa(event[ida].p());
    pa.bstback(event[iRes].p());
    
    Vec4 pg(event[idg].p());
    pg.bstback(event[iRes].p());
    
    // Calculate scale
    Vec4 pw = pq + pa + pg;
    double q2 = pw*pw;
    double csi = 2*pg.e()/sqrt(q2);
    double yq = 1 - pg*pq/(pg.e()*pq.e());
    double ya = 1 - pg*pa/(pg.e()*pa.e());
    
    scale = sqrt(min(1-yq,1-ya)*pow2(csi)*q2/2);
  }
  
  return scale;
}

