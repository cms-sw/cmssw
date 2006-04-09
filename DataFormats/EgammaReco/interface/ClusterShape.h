#ifndef EgammaReco_ClusterShape_h
#define EgammaReco_ClusterShape_h
/** \class reco::ClusterShape
 *  
 * shape variables for an Ecal cluster
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: ClusterShape.h,v 1.8 2006/03/01 15:03:32 llista Exp $
 *
 */
#include <Rtypes.h>
#include "DataFormats/EgammaReco/interface/ClusterShapeFwd.h"

namespace reco {

  class ClusterShape {
  public:
    ClusterShape() { }
    ClusterShape( double cEE, double cEP, double cPP, 
                  double eMax, double e2x2, double e3x3, double e5x5,
                  double hadOverEcal );
    double eMax() const { return eMax_; }
    double e2x2() const { return e2x2_; }
    double e3x3() const { return e3x3_; }
    double e5x5() const { return e5x5_; }
    double covEtaEta() const { return covEtaEta_; }
    double covEtaPhi() const { return covEtaPhi_; }
    double covPhiPhi() const { return covPhiPhi_; }
    double hadOverEcal() const { return hadOverEcal_; }

  private:
    Double32_t covEtaEta_, covEtaPhi_, covPhiPhi_;
    Double32_t eMax_, e2x2_, e3x3_, e5x5_;
    Double32_t hadOverEcal_;
  };

}

#endif
