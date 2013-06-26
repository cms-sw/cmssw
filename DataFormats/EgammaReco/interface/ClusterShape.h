#ifndef EgammaReco_ClusterShape_h
#define EgammaReco_ClusterShape_h

/** \class reco::ClusterShape
 *  
 * shape vars dataholder for an Ecal cluster
 *
 * \author Michael A. Balazs, UVa
 * \author Luca Lista, INFN
 *
 * \version $Id: ClusterShape.h,v 1.11 2013/04/22 22:53:02 wmtan Exp $
 *
 */

#include <Rtypes.h>

#include "DataFormats/EgammaReco/interface/ClusterShapeFwd.h"
#include "DataFormats/DetId/interface/DetId.h"

namespace reco {

  class ClusterShape {
  public:
    ClusterShape() { }
    ClusterShape( double cEE, double cEP, double cPP, 
                  double eMax, DetId eMaxId, double e2nd, DetId e2ndId,
		  double e2x2, double e3x2, double e3x3, double e4x4,
		  double e5x5, double E10_Right_, double E10_Left_,
		  double E10_Top_, double E10_Bottom_, double e3x2Ratio,
		  double LAT, double etaLAT, double phiLAT, double A20, double A42,
                  const std::vector<double>& energyBasketFractionEta_,
                  const std::vector<double>& energyBasketFractionPhi_);
    double eMax() const { return eMax_; }
    double e2nd() const { return e2nd_; }
    double e2x2() const { return e2x2_; }
    double e3x2() const { return e3x2_; }
    double e3x3() const { return e3x3_; }
    double e4x4() const { return e4x4_; }
    double e5x5() const { return e5x5_; }
    double e2x5Right() const { return e2x5Right_; }
    double e2x5Left() const { return e2x5Left_; }
    double e2x5Top() const { return e2x5Top_; }
    double e2x5Bottom() const { return e2x5Bottom_; }
    double e3x2Ratio() const { return e3x2Ratio_; }
    double covEtaEta() const { return covEtaEta_; }
    double covEtaPhi() const { return covEtaPhi_; }
    double covPhiPhi() const { return covPhiPhi_; }
    double lat() const { return LAT_; }
    double etaLat() const { return etaLAT_; }
    double phiLat() const { return phiLAT_; }
    double zernike20() const { return A20_; }
    double zernike42() const { return A42_; }

    std::vector<double> energyBasketFractionEta() const { return energyBasketFractionEta_;}
    std::vector<double> energyBasketFractionPhi() const { return energyBasketFractionPhi_;}
    DetId eMaxId() const { return eMaxId_;}
    DetId e2ndId() const { return e2ndId_;}

  private:
    Double32_t covEtaEta_, covEtaPhi_, covPhiPhi_;
    Double32_t eMax_, e2nd_, e2x2_, e3x2_, e3x3_, e4x4_, e5x5_;
    Double32_t e2x5Right_, e2x5Left_, e2x5Top_, e2x5Bottom_;
    Double32_t e3x2Ratio_;
    Double32_t LAT_;
    Double32_t etaLAT_;
    Double32_t phiLAT_;
    Double32_t A20_, A42_;
    std::vector<double> energyBasketFractionEta_;
    std::vector<double> energyBasketFractionPhi_;
    DetId eMaxId_, e2ndId_;
  };

}

#endif
