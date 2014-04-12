#include "RecoTBCalo/EcalTBAnalysisCoreTools/interface/TBPositionCalc.h"

using namespace std;

TBPositionCalc::TBPositionCalc(const std::map<std::string,double>& providedParameters, std::string const & fullMapName, const CaloSubdetectorGeometry *passedGeometry ) 
{
  // barrel geometry initialization
  if(passedGeometry == NULL)
    throw(std::runtime_error("\n\n TBPositionCalc: wrong initialization.\n\n"));
  theGeometry_ = passedGeometry;
  
  // parameters initialization
  param_LogWeighted_ = providedParameters.find("LogWeighted")->second;
  param_X0_ = providedParameters.find("X0")->second;
  param_T0_ = providedParameters.find("T0")->second; 
  param_W0_ = providedParameters.find("W0")->second;

  theTestMap = new EcalTBCrystalMap(fullMapName);
}

TBPositionCalc::~TBPositionCalc()
{
  if (theTestMap) delete theTestMap;
}

CLHEP::Hep3Vector TBPositionCalc::CalculateTBPos(const std::vector<EBDetId>& upassedDetIds, int myCrystal, EcalRecHitCollection const *passedRecHitsMap) {
  
  std::vector<EBDetId> passedDetIds = upassedDetIds;
  // throw an error if the cluster was not initialized properly  
  if(passedRecHitsMap == NULL)
    throw(std::runtime_error("\n\n TBPositionCalc::CalculateTBPos called uninitialized.\n\n"));
  
  // check DetIds are nonzero
  std::vector<EBDetId> validDetIds;
  std::vector<EBDetId>::const_iterator iter;
  for (iter = passedDetIds.begin(); iter != passedDetIds.end(); iter++) {
    if (((*iter) != DetId(0)) 
	&& (passedRecHitsMap->find(*iter) != passedRecHitsMap->end()))
      validDetIds.push_back(*iter);
  }
  passedDetIds.clear();
  passedDetIds = validDetIds;
  
  // computing the position in the cms frame
  CLHEP::Hep3Vector cmsPos = CalculateCMSPos(passedDetIds, myCrystal, passedRecHitsMap);

  // computing the rotation matrix (from CMS to TB)
  CLHEP::HepRotation *CMStoTB = new CLHEP::HepRotation();
  computeRotation(myCrystal, (*CMStoTB));

  // moving to testbeam frame
  CLHEP::Hep3Vector tbPos = (*CMStoTB)*cmsPos;
  delete CMStoTB;

  return tbPos;
} 


CLHEP::Hep3Vector TBPositionCalc::CalculateCMSPos(const std::vector<EBDetId>& passedDetIds, int myCrystal, EcalRecHitCollection const *passedRecHitsMap) {
  
  // Calculate the total energy
  double thisEne = 0;
  double eTot = 0;
  EBDetId myId;
  std::vector<EBDetId>::const_iterator myIt;
  for (myIt = passedDetIds.begin(); myIt !=  passedDetIds.end(); myIt++) {
    myId = (*myIt);
    EcalRecHitCollection::const_iterator itt = passedRecHitsMap->find(myId);
    thisEne = itt->energy();
    eTot += thisEne;
  }

  // Calculate shower depth
  float depth = 0.;
  if(eTot<=0.) {
    edm::LogError("NegativeClusterEnergy") << "cluster with negative energy: " << eTot << ", setting depth to 0.";
  } else {
    depth = param_X0_ * (param_T0_ + log(eTot));
  }

  // Get position of the central crystal from shower depth 
  EBDetId maxId_ = EBDetId(1, myCrystal, EBDetId::SMCRYSTALMODE);
  const CaloCellGeometry* center_cell = theGeometry_ -> getGeometry(maxId_);
  GlobalPoint center_pos = 
    (dynamic_cast<const TruncatedPyramid*>(center_cell))->getPosition(depth);

  // Loop over the hits collection
  double weight        = 0;
  double total_weight  = 0;
  double cluster_theta = 0;
  double cluster_phi   = 0;
  std::vector<EBDetId>::const_iterator myIt2;
  for (myIt2 = passedDetIds.begin(); myIt2 != passedDetIds.end(); myIt2++) {

    // getting weights
    myId = (*myIt2);
    EcalRecHitCollection::const_iterator itj = passedRecHitsMap->find(myId);
    double ener = itj->energy();

    if (param_LogWeighted_) {
      if(eTot<=0.) { weight = 0.; } 
      else { weight = std::max(0., param_W0_ + log( fabs(ener)/eTot) ); }
    } else {
      weight = ener/eTot;
    }
    total_weight += weight;

    // weighted position of this detId
    const CaloCellGeometry* jth_cell = theGeometry_->getGeometry(myId);
    GlobalPoint jth_pos = dynamic_cast<const TruncatedPyramid*>(jth_cell)->getPosition(depth);    
    cluster_theta += weight*jth_pos.theta();
    cluster_phi   += weight*jth_pos.phi();
  }
  
  // normalizing
  cluster_theta /= total_weight;
  cluster_phi /= total_weight;
  if (cluster_phi > M_PI) { cluster_phi -= 2.*M_PI; }
  if (cluster_phi < -M_PI){ cluster_phi += 2.*M_PI; }

  // position in the cms frame
  double radius = sqrt(center_pos.x()*center_pos.x() + center_pos.y()*center_pos.y() + center_pos.z()*center_pos.z());
  double xpos = radius*cos(cluster_phi)*sin(cluster_theta);
  double ypos = radius*sin(cluster_phi)*sin(cluster_theta); 
  double zpos = radius*cos(cluster_theta);

  return CLHEP::Hep3Vector(xpos, ypos, zpos);
}

// rotation matrix to move from the CMS reference frame to the test beam one  
void TBPositionCalc::computeRotation(int MyCrystal, CLHEP::HepRotation &CMStoTB){
  
  // taking eta/phi of the crystal

  double myEta   = 999.;
  double myPhi   = 999.;
  double myTheta = 999.;
  theTestMap->findCrystalAngles(MyCrystal, myEta, myPhi);
  myTheta = 2.0*atan(exp(-myEta));

  // matrix
  CLHEP::HepRotation * fromCMStoTB = new CLHEP::HepRotation();
  double angle1 = 90.*deg - myPhi;
  CLHEP::HepRotationZ * r1 = new CLHEP::HepRotationZ(angle1);
  double angle2 = myTheta;
  CLHEP::HepRotationX * r2 = new CLHEP::HepRotationX(angle2);
  double angle3 = 90.*deg;
  CLHEP::HepRotationZ * r3 = new CLHEP::HepRotationZ(angle3);
  (*fromCMStoTB) *= (*r3);
  (*fromCMStoTB) *= (*r2);
  (*fromCMStoTB) *= (*r1);
  
  CMStoTB = (*fromCMStoTB);

  delete fromCMStoTB;
  delete r1;
  delete r2;
  delete r3;
}


