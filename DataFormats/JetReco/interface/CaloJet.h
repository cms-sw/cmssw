#ifndef JetReco_CaloJet_h
#define JetReco_CaloJet_h

/** \class CaloJet
 * $Id$
 *
 * Ported from original version in JetObjects package
 *
 * \short Jets made from CaloTowers
 *
 * CaloJet represents Jets made from CaloTowers
 * More to be added...
 *
 * \author Fernando Varela Rodriguez, Boston University
 *
 * \version   1st Version April 22, 2005.
 * 
 * \version   2nd Version Oct 19, 2005, R. Harris, modified to work 
 *            with real CaloTowers. No energy fractions yet.
 ************************************************************/
#include <vector>
#include "TLorentzVector.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

class CaloTowerDetId;

namespace reco {
  
  class CaloJet {
  public:
    
    /** Default constructor*/
    CaloJet();
    
    /** Constructor from values*/
    CaloJet( double px, double py, double pz, double e, 
	     const CaloTowerCollection & caloTowerColl, 
	     const std::vector<CaloTowerDetId> & indices );
    
    /** Default destructor*/
    ~CaloJet();
    
    /** The Jet four-vector as a true Lorentz vector **/
    const TLorentzVector & p4() const { return p4_; }
    /* Return the jet momentum component along the x axis */
    double px() const { return p4_.Px(); }
    /** Returns the jet momentum component along the y axis */
    double py() const { return p4_.Py(); }
    /** Returns the jet momentum component along the z axis */
    double pz() const { return p4_.Pz(); }
    /** Returns the total energy of the jet*/
    double energy() const { return p4_.Energy(); }
    
    /** Standard quantities derived from the Jet Lorentz vector
	/\return the modulus of the momentum of the jet */
    double p() const { return p4_.P(); }
    /** Returns the transverse momentum of the jet*/
    double pt() const { return p4_.Pt(); }
    /** Returns the transverse energy of the jet*/
    double et() const { return p4_.Et(); }
    /** Returns the jet mass of the jet*/
    double mass() const { return p4_.M(); }
    /** Returns the azimuthal angle of the jet, Phi*/
    double phi() const { return p4_.Phi(); }
    /** Returns the pseudorapidity of the jet*/
    double eta() const { return p4_.Eta(); }
    /** Returns the rapidity of the jet*/
    double y() const { return p4_.Rapidity(); }
    /** Returns the number of constituents of the jet*/
    int numberOfConstituents() const { return towerIndices_.size(); }
    
    //These methods are specific to the CaloJet class
    
    /** Returns the list of CaloTower IDs forming the Jet*/
    const std::vector<CaloTowerDetId> & towerIndices() const {return towerIndices_;};
    
    /**Sets tower indices */
    void setTowerIndices(const std::vector<CaloTowerDetId > & towerIndices);
    
    /** Returns the maximum energy deposited in ECAL towers*/
    double maxEInEmTowers() const { return maxEInEmTowers_; }
    /** Returns the maximum energy deposited in HCAL towers*/
    double maxEInHadTowers() const { return maxEInHadTowers_; }
    /** Returns the jet energy fraction in HCAL*/
    double energyFractionInHCAL() const { return energyFractionInHCAL_; }
    /** Returns the jet energy fraction in ECAL*/
    double energyFractionInECAL() const { return energyFractionInECAL_; }
    /** Returns the number of constituents carrying a 90% of the total Jet
	energy*/
    int n90() const { return n90_; }
    
  private:
    // Data members
    TLorentzVector p4_;
    /** List of CaloTower IDs the Jet consists of*/
    std::vector<CaloTowerDetId> towerIndices_;
    //Variables specific to to the CaloJet class
    /** Maximum energy in EM towers*/
    double maxEInEmTowers_;
    /** Maximum energy in EM towers*/
    double maxEInHadTowers_;
    /** Energy fraction in HCAL*/
    double energyFractionInHCAL_;
    /** Energy fraction in ECAL*/
    double energyFractionInECAL_;
    /**Number of constituents carrying 90% of the Jet energy*/
    int n90_;
  };
  
typedef std::vector<CaloJet> CaloJetCollection;

}

#endif
