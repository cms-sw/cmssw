//  ProtJet.cc
//  Revision History:  R. Harris 10/19/05  Modified to work with real CaloTowers from Jeremy Mans
//

#include "PhysicsTools/Candidate/interface/Candidate.h"
#include "PhysicsTools/JetExamples/interface/ProtoJet.h"

#include <vector>
#include <algorithm>               // include STL algorithm implementations
#include <numeric>		   // For the use of numeric

const double PI=3.14159265;

using std::vector;
using namespace aod;

ProtoJet::ProtoJet(): 
  m_constituents(), m_e(0), m_px(0), m_py(0), m_pz(0) {
}

ProtoJet::ProtoJet(const CandidateRefs & theConstituents) : 
  m_constituents(theConstituents), m_e(0), m_px(0), m_py(0), m_pz(0) {
  calculateLorentzVector(); 
}//end of constructor


// Old code
//ProtoJet::ProtoJet(CaloTowerCollection aTowerCollection)
//{
 
// std::vector<const CaloTower*> constituentsPtr;
// for(unsigned int i = 0; i < aTowerCollection.size(); i++)
// {
//    constituentsPtr.push_back(&(aTowerCollection[i]));						  
// }
//  m_constituents = constituentsPtr;
//  calculateLorentzVector(); 
//}//end of constructor


ProtoJet::~ProtoJet() {
}

/*
double ProtoJet::maxEInEmTowers() const 
{
  std::vector<double> energy_i;
  for(vector<const Candidate *>::const_iterator i = m_constituents.begin(); i !=  m_constituents.end(); ++i) {
    assert(*i);
    energy_i.push_back((*i)->e_em()); 
  }//end of for
  
  //Sort the array
  sort(energy_i.begin(), energy_i.end());
  
  //Return the highest value in the array, i.e last element of the array
  return energy_i.back();
}

double ProtoJet::maxEInHadTowers() const 
{
  std::vector<double> energy_i;
  
  for(CandidateRefs::iterator i = m_constituents.begin(); i != m_constituents.end(); ++i) {
    assert(*i);
    energy_i.push_back((*i)->e_had()); 
  }//end of for
  
  //Sort the array
  sort(energy_i.begin(), energy_i.end());
  
  //Return the highest value in the array, i.e last element of the array
  return energy_i.back();
}

double ProtoJet::emFraction() const {
  std::vector<double> emEnergy_i;
  std::vector<double> hadEnergy_i;
  
  for(CandidateRefs::iterator i = m_constituents.begin(); i != m_constituents.end(); ++i) {
    assert(*i);
    emEnergy_i.push_back((*i)->e_em());
    hadEnergy_i.push_back((*i)->e_had());
  }
  //Calculate the ratio between the total EM and HAD energy:
  return accumulate(emEnergy_i.begin(), emEnergy_i.end(), 0.)/(accumulate(hadEnergy_i.begin(),
         hadEnergy_i.end(), 0.) + accumulate(emEnergy_i.begin(), emEnergy_i.end(), 0.));  
}


int ProtoJet::n90() const {
  vector<double> eList;
  double sum = 0.;
  int counter = 0;
  double e90 = et()*0.9;
  double et = 0.;
  
  eList.clear();
  for(CandidatrRefs::iterator i = m_constituents.begin(); i != m_constituents.end(); ++i) {
    assert(*i);
    eList.push_back((*i)->e());
  }
    
  //Make sure that we have a sorted list of constituents:  
  sort(eList.begin(), eList.end());
  
  const vector<double> & ecList = eList;
  for(vector<double>::const_reverse_iterator ePtr = ecList.rbegin(); 
      ePtr != ecList.rend();
      ++ePtr) {
    sum += *ePtr;
    ++counter;  
    et = sum*pt()/p();
    if(et >= e90)
     break;
  }
  return counter++;
}

double ProtoJet::phi() const {
  double px_pos=fabs(m_px);
  double py_pos=fabs(m_py);
  double phi= 0.;

  phi = atan(py_pos / (px_pos + 1.e-20));

  if(signum(m_px)==-1 && signum(m_py)== 1)
    phi = PI - phi;
  else if(signum(m_px)== 1 && signum(m_py) == -1)
    phi = 2.*PI - phi;
  else if(signum(m_px)==-1 && signum(m_py)==-1)
    phi = PI + phi;

  return phi;
}
*/

/*

double ProtoJet::energyFractionInHO() const {
  double totalHoEnergy = 0.;
  
  //Loop over the array of CaloTower*
  for(vector<const CaloTower *>::const_iterator i = m_constituents.begin(); i !=  m_constituents.end(); ++i) {
    assert (*i);
    //For each of the consituents, loop over the array of 
    //rhits the tower is made of:
    for(vector<const CaloRecHit *>::const_iterator RHitItr = (*i)->recHitsBegin();
        RHitItr != (*i)->recHitsEnd(); ++RHitItr){
	
      if((*RHitItr)->getMyCell().baseLayer() == 3) {
	//Sum all rhit energy:
	totalHoEnergy += (*RHitItr)->energy();
      }
    }
  }
  return totalHoEnergy/m_e;
}


double ProtoJet::energyFraction(const string detector, const string 
subdetector, int layer) const {
  double energyInDetector = 0.;
  
  //Loop over the array of CaloTower*
  for(vector<const CaloTower *>::const_iterator i = m_constituents.begin(); i !=  m_constituents.end(); ++i) {
    assert(*i);
    //For each of the consituents, loop over the array of 
    //rhits the tower is made of:
    for(vector<const CaloRecHit *>::const_iterator RHitItr = (*i)->recHitsBegin();
        RHitItr != (*i)->recHitsEnd(); ++RHitItr) {

      //Select only the detector we are interested in:
      if(detector == "HCAL"){
        if(detector == (*RHitItr)->getMyCell().whichDetector()){
          //Select only rhits in the subdetector we are interested in
          if(subdetector == "ALL"){
            //Select only the specific layers	
            if(layer == -1){
              energyInDetector += (*RHitItr)->energy();
            }
            else if((*RHitItr)->getMyCell().baseLayer() == layer){
	      //Sum all rhit energy:
	      energyInDetector += (*RHitItr)->energy();
            }
          }
	  else if(subdetector == (*RHitItr)->getMyCell().whichSubDetector()){
	     if(layer == -1){
                energyInDetector += (*RHitItr)->energy();
              }
              else if((*RHitItr)->getMyCell().baseLayer() == layer){
	        //Sum all rhit energy:
	        energyInDetector += (*RHitItr)->energy();
              }
          }
        }
      }
      else if(detector == "ECAL"){
        if("EFRY" == (*RHitItr)->getMyCell().whichDetector() ||
	   "EBRY" == (*RHitItr)->getMyCell().whichDetector()){
	  energyInDetector += (*RHitItr)->energy();
	}
      }
      else
        cout << "[ProtoJet]->getEnergyFraction(detector, subdetector, layer): Detector unkown: " << detector << endl;
    }
  }
  return energyInDetector/m_e;
}
*/

HepLorentzVector ProtoJet::getLorentzVector() const {
  HepLorentzVector theLorentzVector;

  theLorentzVector.setPx(px());
  theLorentzVector.setPy(py());
  theLorentzVector.setPz(pz());
  theLorentzVector.setE(e());

 return theLorentzVector;
}

void ProtoJet::calculateLorentzVector() {
  m_e = 0; m_px = 0; m_py = 0; m_pz = 0;
  for(CandidateRefs::iterator i = m_constituents.begin(); i !=  m_constituents.end(); ++i) {
    assert(*i);
    const Candidate &t = **i;
    m_e += t.energy();
    m_px += t.energy() * (1.0/cosh(t.eta())) * cos(t.phi());
    m_py += t.energy() * (1.0/cosh(t.eta())) * sin(t.phi());
    m_pz += t.energy() * tanh(t.eta());
  } //end of loop over the jet constituents
}



int signum(double x) {
  if     ( x < 0. ) return -1;
  else if( x > 0. ) return  1;
  else              return  0;
}
