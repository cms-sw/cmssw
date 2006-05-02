#include "RecoEcal/EgammaClusterAlgos/interface/LogPositionCalc.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"

Point LogPositionCalc::getECALposition(std::vector<reco::EcalRecHitData> recHits, const CaloSubdetectorGeometry geometry)
{
  // Calculates position of cluster using the algorithm presented in
  // Awes et al., NIM A311, p130-138.  See also CMS Note 2001/034 p11
  
  // Specify constants for log calculation
  double w_zero = 4.2;
  double w_min = 0.;

  // Specify constants for radius correction
  double x_zero = 0.89;

  // Initialize position variables
  double clusterPhi=0;
  double clusterEta=0;
  double eTot=0;
  double wRadius=0;
  double wSinPhi=0;
  double wCosPhi=0;
  double wTheta=0;
  double weightSum = 0;
  
  // Sum cluster energy
  for (int i=0;i<int(recHits.size());++i){
    eTot+= recHits[i].energy();
  }
  
  // Calculate log weighted positions
  if(eTot>0.){
    for (int j=0; j<int(recHits.size()); ++j){
       
      // Find out what the physical location of the cell is
      DetId blerg = recHits[j].detId();
      const CaloCellGeometry *this_cell = geometry.getGeometry(blerg);
      GlobalPoint posi = this_cell->getPosition();
      
      // Get energy of the single hit
      double e_j = recHits[j].energy();
      
      if (e_j > 0.) {
	// Put position of cell in spherical coordinates
	double r=sqrt(posi.x()*posi.x()+posi.y()*posi.y()+posi.z()*posi.z());
	double pphi =posi.phi();
	double ttheta =posi.theta();
	
	// Do the log weighting
	double weight = std::max(w_min, w_zero + log(e_j/eTot));
	
	// Increment the coordinates 
	wRadius+=weight*r;
	wSinPhi+=weight*sin(pphi);
	wCosPhi+=weight*cos(pphi);
	wTheta+=weight*ttheta;
	weightSum += weight;
      }
    }

    // Divide everything by the sum of the weights to calculate a position
    if (weightSum > 0) {
      wRadius /= weightSum;
      wSinPhi /= weightSum;
      wCosPhi /= weightSum;
      wTheta /= weightSum;
    }
    
    // Solving 2*PI problem and stay most linear
    //    double M_SQRT1_2 = 0.70710678118654752440084436210; /* sqrt(1/2) */
    if(wCosPhi>=M_SQRT1_2)
      {
	if(wSinPhi>0)
	  clusterPhi=asin(wSinPhi);
	else
	  clusterPhi=2*M_PI+asin(wSinPhi);
      }
    else if(wCosPhi<-M_SQRT1_2)
      clusterPhi=M_PI-asin(wSinPhi);
    else if(wSinPhi>0)
      clusterPhi=acos(wCosPhi);
    else
      clusterPhi=2*M_PI-acos(wCosPhi);
    
    clusterEta = -log(tan(wTheta*0.5));
    
    
    double t_zero = 0.;
    
    // Where the barrel ends
    double barrelEta = 1.479;
    
    // Need to have a way of getting info about preshower
    bool preshower = false;
    
    // t_zero values for various scenarios
    double bar_t_zero = 5.7;
    double end_t_zero = 4.0;
    double pre_t_zero = 0.4;
    
    // Decide which t_zero to use
    if (clusterEta <= barrelEta) {
      t_zero = bar_t_zero;
    }
    else {
      if (preshower) {
	t_zero = pre_t_zero;
      }
      else {
	t_zero = end_t_zero;
      }
    }
    
    // Correct the radius for shower depth
    // See CMS Note 2001/034 p10
    wRadius += x_zero * (t_zero + log(eTot));
    
    std::cout << "Log weighted position:" << std::endl;
    std::cout << "Cluster eta, phi = " << clusterEta << ", " << clusterPhi << std::endl;
    
    // Calculate (x, y, z) and return it
    double xpos = wRadius * cos(clusterPhi) * sin(wTheta);
    double ypos = wRadius * sin(clusterPhi) * sin(wTheta);
    double zpos = wRadius * cos(wTheta);
    return Point(xpos, ypos, zpos);
  }

  // If there was no energy in the cluster, return (0, 0, 0).
  // This is arguably the wrong thing to do, but it should
  // never happen.
  return Point(0., 0., 0.);
  
}
