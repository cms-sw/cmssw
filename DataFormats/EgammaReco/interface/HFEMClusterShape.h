
#ifndef HFEMClusterShape_h
#define HFEMClusterShape_h

#include <Rtypes.h>
#include "DataFormats/EgammaReco/interface/HFEMClusterShapeFwd.h"
#include "DataFormats/DetId/interface/DetId.h"
/*class reco::HFEMClusterShape HFEMClusterShape.h DataFormats/EgammaReco/interface/HFEMClusterShape.h
 *  
 * A Cluster Shape of a Possible EM cluster in the HF detector
 * Contains the DetId of its Seed
 *
 * \author Kevin Klapoetke, University of Minnesota
 *
 * \version $Id: HFEMClusterShape.h,v 1.3 2007/10/08 18:52:13 futyand Exp $
 *
 */

namespace reco {

  class HFEMClusterShape {
  public:
    HFEMClusterShape() { }
   
    HFEMClusterShape(double eLong1x1,double eShort1x1,
		     double eLong3x3,double eShort3x3,double eLong5x5,
		     double eShort5x5,double eLongCore,
		     double CellEta,double CellPhi,
		     DetId seed);
    
    
    //energy in long or short fibers various cluster sizes
    double eLong1x1() const {return eLong1x1_;}
    double eShort1x1() const {return eShort1x1_;}
    double eLong3x3() const {return eLong3x3_;}
    double eShort3x3() const {return eShort3x3_;}
    double eLong5x5() const {return eLong5x5_;}
    double eShort5x5() const {return eShort5x5_;}
    
    //total energy in various clusters
    double e1x1() const;
    double e3x3() const;
    double e5x5() const;

    //Identification Variables
    //Longetudinal variable: E(3x3,short fibers)/E(3x3,long fibers)
    double eSeL() const;
    //Transverse Variable: E(Core of cluster)/E(3x3)
    double eCOREe9() const;
    //Shower Exclusion Variable: E(3x3)/E(5x5)
    double e9e25() const;

    //energy in central highest energy cells (at least 50% energy of previous total energy startign with seed cell)
    double eCore() const {return eLongCore_;}
  
    double CellEta() const {return CellEta_;}
    double CellPhi() const {return CellPhi_;}
    
    //seed cell of cluster DetId
    DetId seed() const {return seed_;}

  private:
    double eLong1x1_, eShort1x1_, eLong3x3_,eShort3x3_, eLong5x5_, eShort5x5_,  eLongCore_,CellEta_,CellPhi_;
    DetId seed_;
 
  };
  
}


#endif
