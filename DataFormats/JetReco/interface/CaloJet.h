#ifndef JetReco_CaloJet_h
#define JetReco_CaloJet_h

/** \class CaloJet
 * $Id: CaloJet.h,v 1.7 2006/01/10 09:07:53 llista Exp $
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
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include <vector>
class CaloTower;

namespace reco {
  
  class CaloJet {
  public:
    typedef math::PtEtaPhiELorentzVector LorentzVector;
    /** Default constructor*/
    CaloJet();
    
    /** Constructor from values*/
    CaloJet( double px, double py, double pz, double e, 
	     double maxEInEmTowers, double maxEInHadTowers, 
	     double energyFractionInHCAL, double energyFractionInECAL,
	     int n90 );
    
    /** Default destructor*/
    ~CaloJet();
    
    /** The Jet four-vector as a true Lorentz vector **/
    const LorentzVector & p4() const { return p4_; }
    /* Return the jet momentum component along the x axis */
    double px() const { return p4_.Px(); }
    /** Returns the jet momentum component along the y axis */
    double py() const { return p4_.Py(); }
    /** Returns the jet momentum component along the z axis */
    double pz() const { return p4_.Pz(); }
    /** Returns the total energy of the jet*/
    double energy() const { return p4_.E(); }
    
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
    
    //These methods are specific to the CaloJet class
    
    typedef edm::SortedCollection<CaloTower> ConstituentCollection; 
    typedef edm::Ref<ConstituentCollection> ConstituentRef;
    typedef edm::RefVector<ConstituentCollection> ConstituentRefs;
    typedef ConstituentRefs::iterator constituents_iterator;

    /** Returns/add the list of CaloTower IDs forming the Jet*/
    void add( const ConstituentRef & r ) { constituents_.push_back( r ); }
    constituents_iterator constituents_begin() const { return constituents_.begin(); }
    constituents_iterator constituents_end() const { return constituents_.end(); }
    size_t constituentsSize() const { return constituents_.size(); }
    
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
    LorentzVector p4_;
    /** List of CaloTower IDs the Jet consists of*/
    ConstituentRefs constituents_;
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
  
}

#endif
