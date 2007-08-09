
#ifndef HFEMClusterShape_h
#define HFEMClusterShape_h

#include <Rtypes.h>
#include "DataFormats/EgammaReco/interface/HFEMClusterShapeFwd.h"
#include "DataFormats/DetId/interface/DetId.h"


namespace reco {

  class HFEMClusterShape {
  public:
    HFEMClusterShape() { }
   
    HFEMClusterShape(double energy, double eLong1x1,double eShort1x1,
		     double eLong3x3,double eShort3x3,double eLong5x5,
		     double eShort5x5,double eLongCore,
		     double CellEta,double CellPhi);
    //"total" energy in cluster 
    double energy() const {return energy_;}
    //    //transverse energy of cluster  
    //double et() const; 
    
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

    //energy in central highest energy cells (at least 50% energy of previous total energy startign with seed cell)
    double eCore() const {return eLongCore_;}
  
    double CellEta() const {return CellEta_;}
    double CellPhi() const {return CellPhi_;}
    

  private:
    double energy_,eLong1x1_, eShort1x1_, eLong3x3_,eShort3x3_, eLong5x5_, eShort5x5_,  eLongCore_,CellEta_,CellPhi_;
 


  };
  
}


#endif
