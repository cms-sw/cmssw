#include "PhysicsTools/IsolationUtils/interface/auxiliaryFunctions.h"

// ROOT include files
#include <TMath.h>

double getNormPhi(double phi)
{
//--- map azimuth angle into interval [-pi,+pi]
  double normPhi = phi;

  while ( normPhi < (-TMath::Pi()) ) normPhi += 2*TMath::Pi();
  while ( normPhi >   TMath::Pi()  ) normPhi -= 2*TMath::Pi();  

  return normPhi;
}
