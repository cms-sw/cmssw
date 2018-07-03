#include "Geometry/EcalTestBeam/interface/EcalTBHodoscopeGeometry.h"
#include "SimDataFormats/EcalTestBeam/interface/HodoscopeDetId.h"

#include <vector>
#include <iostream>

int main() {

  EcalTBHodoscopeGeometry theTestGeom;
  
  for ( int j = 0 ; j < theTestGeom.getNPlanes() ; ++j ) 
    {
      for ( int i = 0 ; i < 1000 ; ++i ) 
        {
          std::cout << "Position " << -17.+ 34./1000.*i << " Plane " << j << std::endl;
          std::vector<int> firedFibres=theTestGeom.getFiredFibresInPlane(-17.+ 34./1000.*i,j);
          for (int firedFibre : firedFibres) {
            std::cout << firedFibre << std::endl;
         
            HodoscopeDetId myDetId = HodoscopeDetId( j , (int)firedFibre );
            std::cout << myDetId << std::endl;
   
          }
          
        }
    }


  return 0;

}
