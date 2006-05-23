#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"


//Set Default Values 
bool        PositionCalc::param_LogWeighted_;
Double32_t  PositionCalc::param_X0_;
Double32_t  PositionCalc::param_T0_; 
Double32_t  PositionCalc::param_W0_;
std::string PositionCalc::param_CollectionType_ = ""; 
const std::map<EBDetId,EcalRecHit> *PositionCalc::storedRecHitsMap_ = NULL;
const CaloSubdetectorGeometry *PositionCalc::storedSubdetectorGeometry_ = NULL;

void PositionCalc::Initialize(std::map<std::string,double> providedParameters, 
                                       const std::map<EBDetId,EcalRecHit> *passedRecHitsMap,
                                       std::string passedCollectionType,
                                       const CaloSubdetectorGeometry *passedGeometry) 
{
  param_LogWeighted_ = providedParameters.find("LogWeighted")->second;
  param_X0_ = providedParameters.find("X0")->second;
  param_T0_ =  providedParameters.find("T0")->second; 
  param_W0_ =  providedParameters.find("W0")->second;

  storedRecHitsMap_ = passedRecHitsMap;
  param_CollectionType_ = passedCollectionType;
  storedSubdetectorGeometry_ = passedGeometry;
}



math::XYZPoint PositionCalc::Calculate_Location(std::vector<DetId> passedDetIds)
{
  
  // Throw an error if the cluster was not initialized properly

  if(storedRecHitsMap_ == NULL || param_CollectionType_ == "" || storedSubdetectorGeometry_ == NULL)
    throw(std::runtime_error("\n\nPositionCalc::Calculate_Location called uninitialized or wrong initialization.\n\n"));
   
 
  // Calculates position of cluster using the algorithm presented in
  // Awes et al., NIM A311, p130-138.  See also CMS Note 2001/034 p11
  
  // Specify constants for log calculation
  double w_min = 0.;

  // Specify constants for radius correction

  // Initialize position variables
  double clusterPhi=0;
  double clusterEta=0;
  double eTot=0;
  double wRadius=0;
  double wSinPhi=0;
  double wCosPhi=0;
  double wTheta=0;
  double weightSum = 0;
  
  // Sum cluster energy for weighting

  std::vector<DetId>::iterator i;
  for (i = passedDetIds.begin(); i != passedDetIds.end(); i++) {
    DetId id_ = (*i);
    eTot += ((storedRecHitsMap_->find(id_))->second).energy();
  }
  
  // Calculate positions
  
  if(eTot>0.){

    // Main loop through given DetIds

    std::vector<DetId>::iterator j;
    for (j = passedDetIds.begin(); j != passedDetIds.end(); j++){
      
      // Find out what the physical location of the cell is
      
      DetId id_ = (*i);
      const CaloCellGeometry *this_cell = storedSubdetectorGeometry_->getGeometry(id_);
      GlobalPoint posi = this_cell->getPosition();
      
      // Get energy of the single hit
      
      double e_j = ((storedRecHitsMap_->find(id_))->second).energy();
      double weight = 0;

      if (e_j > 0.) {
	
	// Put position of cell in spherical coordinates
	
	double r=sqrt(posi.x()*posi.x()+posi.y()*posi.y()+posi.z()*posi.z());
	double pphi =posi.phi();
	double ttheta =posi.theta();
	
	// Do the log weighting
	
	if (param_LogWeighted_) {
	  weight = std::max(w_min, param_W0_ + log(e_j/eTot));
	}

	// Or the arithmetic weighting
	
	else {
	  weight = e_j/eTot;
	}

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

    // Calculate eta
    
    clusterEta = -log(tan(wTheta*0.5));
    

    // Set passed value of T0

    double t_zero = param_T0_;
    
    
    // t_zero values for various scenarios
    const double bar_t_zero = 5.7;
    const double end_t_zero = 4.0;
    const double pre_t_zero = 0.4;
    
    // Decide which t_zero to use from location
    if (param_CollectionType_ == "Barrel") {
      t_zero = bar_t_zero;
    }

    if (param_CollectionType_ == "EndCap") {
      t_zero = end_t_zero;
    }

    if (param_CollectionType_ == "PreShower") {
      t_zero = pre_t_zero;
    }

    

    
    
    // Correct the radius for shower depth
    // See CMS Note 2001/034 p10
    
    wRadius += param_X0_ * (t_zero + log(eTot));
    
    // Display positions for debugging

    std::cout << "Log weighted position:" << std::endl;
    std::cout << "Cluster eta, phi = " << clusterEta << ", " << clusterPhi << std::endl;
    
    // Calculate (x, y, z) and return it

    double xpos = wRadius * cos(clusterPhi) * sin(wTheta);
    double ypos = wRadius * sin(clusterPhi) * sin(wTheta);
    double zpos = wRadius * cos(wTheta);
    return math::XYZPoint(xpos, ypos, zpos);
  }

  // If there was no energy in the cluster, return (0, 0, 0).
  // Give a warning to the user that this is so.

  std::cout << "\nPositionCalc::Calculate_Position:  no energy in supplied cells.\n";

  return math::XYZPoint(0., 0., 0.);
  
}

std::map<std::string,double> PositionCalc::Calculate_Covariances(math::XYZPoint passedPoint,
                                                                    std::vector<DetId> passedDetIds)
{
  
  // Check to see that PositionCalc was initialized.  Throw an error if not.

  if(storedRecHitsMap_ == NULL || param_CollectionType_ == "" || storedSubdetectorGeometry_ == NULL)
    throw(std::runtime_error("\n\nPositionCalc::Calculate_Covariance called uninitialized or wrong initialization.\n\n"));

  // Init cov variable

  double covEtaEta=0, covEtaPhi=0, covPhiPhi=0;

  // Find eta, phi of passedPoint 

  double pX = passedPoint.x();
  double pY = passedPoint.y();
  double pZ = passedPoint.z();
  
  double pEta = -log(tan(atan(sqrt(pX*pX+pY*pY)/pZ)*0.5));
  double pPhi = atan2(pY,pX);

  // Init variables for kth cell

  double kX = 0., kY = 0., kZ = 0., kEta = 0., kPhi = 0., weight = 0., wTot = 0., w_min = 0.;

  double eTot = 0.;

  // Sum total energy for weighting

  std::vector<DetId>::iterator n;
  
  for (n = passedDetIds.begin(); n != passedDetIds.end(); n++) {
    eTot += ((storedRecHitsMap_->find((*n)))->second).energy();
  }

  // Main loop to calculate covariances
    
  if (eTot > 0.) {  
    
    std::vector<DetId>::iterator k;
  
    for (k = passedDetIds.begin(); k != passedDetIds.end(); k++) {
          
      // Find out what the physical location of the kth cell is
      DetId id_ = *k;
      const CaloCellGeometry *this_cell = storedSubdetectorGeometry_->getGeometry(id_);
      GlobalPoint posi = this_cell->getPosition();

      kX = posi.x();
      kY = posi.y();
      kZ = posi.z();

      kEta = -log(tan(atan(sqrt(kX*kX+kY*kY)/kZ)*0.5));
      kPhi = atan2(kY,kX);

      double e_k = ((storedRecHitsMap_->find((*k)))->second).energy();
    
      // Do the log weighting

      if (param_LogWeighted_) {
	weight = std::max(w_min, param_W0_ + log(e_k/eTot));
      }

      // Or else arithmetic weighting

      else {
	weight = e_k;
      }

      // Increment covariances
      
      covEtaEta += (kEta - pEta)*(kEta - pEta);
      covEtaPhi += (kEta - pEta)*(kPhi - pPhi);
      covPhiPhi += (kPhi - pPhi)*(kPhi - pPhi);

      wTot += weight;

    }

    covEtaEta /= wTot;
    covEtaPhi /= wTot;
    covPhiPhi /= wTot;
  }
  else {

    // Warn the user if there was no energy in the cells and return zeroes.

    std::cout << "\nPositionCalc::Calculate_Covariances:  no energy in supplied cells.\n";

    covEtaEta = 0;
    covEtaPhi = 0;
    covPhiPhi = 0;
  }

  // Build a map of the covariances.

  std::map<std::string, double> covMap;

  covMap.insert(std::make_pair("covEtaEta",covEtaEta));
  covMap.insert(std::make_pair("covEtaPhi",covEtaPhi));
  covMap.insert(std::make_pair("covPhiPhi",covPhiPhi));

  return covMap;
}

