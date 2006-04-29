#include "RecoEcal/EgammaClusterAlgos/interface/PreshowerCluster.h"



#include "RecoCaloTools/Navigation/interface/EcalPreshowerNavigator.h"
#include "Geometry/Vector/interface/GlobalPoint.h"


using namespace reco;

PreshowerCluster::PreshowerCluster() {
}

PreshowerCluster::~PreshowerCluster() { }

PreshowerCluster::PreshowerCluster(std::vector<EcalRecHit*> rhits, int layer_,
				   edm::ESHandle<CaloGeometry> geometry_h,
				   edm::ESHandle<CaloTopology> theCaloTopology)
{

  
  init();

  

  const CaloSubdetectorGeometry *geometry_p = 
    (*geometry_h).getSubdetectorGeometry(DetId::Ecal, EcalPreshower);
  geometry = *geometry_p;

  topology_h = theCaloTopology;
  
  energy = 0.;
  theta = 0.;

  double sinphi = 0.;
  double cosphi = 0.;

  double r;
  double p;
  double t;

  GlobalPoint pos;

  if (rhits.size()>0.)
    {
      for (std::vector<EcalRecHit*>::iterator it=rhits.begin();
	   it != rhits.end();
	   it++)
	{
	  energy += (*it)->energy();
	  
	  ESDetId id = (*it)->id();
	  const CaloCellGeometry *this_cell = geometry.getGeometry(id);
	  pos = this_cell->getPosition();

	  r = sqrt(pos.x()*pos.x()+pos.y()*pos.y()+pos.z()*pos.z());
	  p = pos.phi();
	  t = pos.theta();

	  radius += (*it)->energy()*r;
	  sinphi += (*it)->energy()*sin(p);
	  cosphi += (*it)->energy()*cos(p);
	  theta += (*it)->energy()*pos.x();
	  x += (*it)->energy()*pos.x();
	  y += (*it)->energy()*pos.y();
	}

      if (energy < 1.0e-10) {
	energy = 0.;
      }

      if (energy > 0.) {
	radius /= energy;
	sinphi /= energy;
	cosphi /= energy;
	theta /= energy;
	x /= energy;
	y /= energy;

	// 2pi issues
	if (cosphi > M_SQRT1_2) {
	  if (sinphi > 0) phi = asin(sinphi);
	  else phi = 2*M_PI+asin(sinphi);
	}
	else if (cosphi < -M_SQRT1_2) phi = M_PI-asin(sinphi);
	else if (sinphi > 0) phi = acos(cosphi);
	else phi = 2*M_PI - acos(cosphi);
      }

      euncorrected = energy;
      et = energy*sin(theta);
      eta = -log(tan(theta/2.));
      
      nhits = rhits.size();
    }
  else {
    euncorrected = 0.;
    et = 0.;
    eta = -999.;
    nhits = 0;
  }
      



}


PreshowerCluster::PreshowerCluster(const PreshowerCluster &b) {
  init(b);
  rhits = b.rhits;
  energy = b.energy;
  theta = b.theta;
  euncorrected = b.euncorrected;
}

// Initialize variables
void PreshowerCluster::init() {
  et = 0;
  radius = 0;
  eta = 0;
  phi = 0;
  nhits = 0;
  x = 0;
  y = 0;
}



void PreshowerCluster::init(const PreshowerCluster &b) {
  et = b.et;
  radius = b.radius;
  eta = b.eta;
  phi = b.phi;
  nhits = b.nhits;
  x = b.x;
  y = b.y;
}





// Methods that return information about the cluster

double PreshowerCluster::Energy() const {return energy;}

double PreshowerCluster::EnergyUncorrected() const {return euncorrected;}

double PreshowerCluster::Radius() const {return radius;}

double PreshowerCluster::Theta() const {return theta;}

double PreshowerCluster::Eta() const {return eta;}

double PreshowerCluster::Phi() const {return phi;}

int PreshowerCluster::Nhits() const {return rhits.size();}

// Cluster correction





// Comparisons

int PreshowerCluster::operator==(const PreshowerCluster &b) const {
  if (Theta()==b.Theta() && Phi()==b.Phi())
    return 1;
  else
    return 0;
}

int PreshowerCluster::operator<(const PreshowerCluster &b) const {
  return energy*sin(Theta()) < b.energy*sin(Theta()) ? 1 : 0;
}

void PreshowerCluster::Correct() 
{
  if (nhits != 0) {
    
    double c_energy = 0;
    double energy_pos = 0;

    double c_x = 0;
    double c_y = 0;
    double c_z = 0;

    Point pos;

    EcalRecHit* max_hit = *(rhits.begin());

    std::vector<EcalRecHit*>::const_iterator cp;

    for (cp = rhits.begin(); cp != rhits.end(); cp++) {
      if ((*cp)->energy() > max_hit->energy()) max_hit = *cp;
    }

    ESDetId strip = max_hit->id();
    
    EcalPreshowerNavigator theESNav(strip,topology_h->getSubdetectorTopology(DetId::Ecal,EcalPreshower));
    theESNav.setHome(strip);

    ESDetId strip_east = theESNav.east();
    ESDetId strip_west = theESNav.west();

    for (cp = rhits.begin(); cp != rhits.end(); cp++) {
      ESDetId strip_it = (*cp)->id();
      c_energy += (*cp)->energy();

      if ((strip_it == strip_east || strip_it == strip_west) 
	  || strip_it == strip) {
	energy_pos += (*cp)->energy();

	ESDetId id = (*cp)->id();	
	const CaloCellGeometry *this_cell = geometry.getGeometry(id);
	pos = this_cell->getPosition();     

	c_x += (*cp)->energy()*pos.x();
	c_y += (*cp)->energy()*pos.y();
	c_z += (*cp)->energy()*pos.z();
      }
    }

    if (energy_pos > 0) {
      
      x = c_x / energy_pos;
      y = c_y / energy_pos;
      z = c_z / energy_pos;

      energy = c_energy;

      radius = sqrt(x*x+y*y+z*z);
      theta = acos(z/radius);
      eta = -log(tan(theta/2));
      phi = acos(x/(radius*sin(theta)));

      if (y < 0) {
	phi = 2*M_PI-phi;
      }
    }
  }
}
