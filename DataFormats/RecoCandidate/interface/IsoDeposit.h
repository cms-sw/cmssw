#ifndef RecoCandidate_IsoDeposit_H
#define RecoCandidate_IsoDeposit_H

/** \class IsoDeposit
 *  Class representing the dR profile of deposits around a muon, i.e.
 *  the differential deposits around the muon as a function of dR.
 *  
 *  Each instance should describe deposits of homogeneous type (e.g. ECAL,
 *  HCAL...); it carries information about
 *  the cone axis, the muon pT.
 *
 *  \author N. Amapane - M. Konecki
 *  Ported with an alternative interface to CMSSW by J. Alcaraz
 *  AbsVetos added by G. Petrucciani
 *  Moved to RecoCandidate  by S. Krutelyov
 */

#include "DataFormats/RecoCandidate/interface/IsoDepositDirection.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <map>
#include <cmath>
#include <string>
#include <vector>


namespace reco { 
  namespace isodeposit {
    struct AbsVeto {
      virtual ~AbsVeto() { }
      //! Return "true" if a deposit at specific (eta,phi) with that value must be vetoed in the sum
      virtual bool veto(double eta, double phi, float value) const = 0;
      /** Relocates this veto so that the new center is at some (eta,phi).
	  Must be implemented on the specific AbsVeto subclass: in this mother class it just throws exception */
      virtual void centerOn(double eta, double phi) {
	throw cms::Exception("Not Implemented") << "This AbsVeto implementation (" << typeid(this).name() << ") does not support the centerOn(eta,phi) method";
      }
    };
    typedef std::vector<AbsVeto*> AbsVetos;
  } 
}

namespace reco {

  class IsoDeposit {
  public:

    typedef isodeposit::Direction Direction;
    typedef isodeposit::AbsVeto AbsVeto;
    typedef isodeposit::AbsVetos AbsVetos;

    // old style vetos
    struct Veto  { 
      Direction vetoDir; float dR; 
      Veto() {}
      Veto(Direction dir, double d):vetoDir(dir), dR(d) {}
    };
    typedef std::vector<Veto> Vetos;

    //! Constructor
    IsoDeposit(double eta=0, double phi=0); 
    IsoDeposit(const Direction & candDirection);

    //! Destructor
    virtual ~IsoDeposit(){};

    //! Get direction of isolation cone
    const Direction & direction() const { return theDirection; }
    double eta() const {return theDirection.eta();}
    double phi() const {return theDirection.phi();}

    //! Get veto area
    const Veto & veto() const { return  theVeto; }
    //! Set veto
    void setVeto(const Veto & aVeto) { theVeto = aVeto; }

    //! Add deposit (ie. transverse energy or pT)
    void addDeposit(double dr, double deposit); // FIXME - temporary for backward compatibility
    void addDeposit(const Direction & depDir, double deposit);

    //! Get deposit 
    double depositWithin( 
			 double coneSize,                                        //dR in which deposit is computed
			 const Vetos & vetos = Vetos(),                          //additional vetos 
			 bool skipDepositVeto = false                            //skip exclusion of veto 
			 ) const;

    //! Get deposit wrt other direction
    double depositWithin( Direction dir,
			  double coneSize,                                        //dR in which deposit is computed
			  const Vetos & vetos = Vetos(),                          //additional vetos 
			  bool skipDepositVeto = false                            //skip exclusion of veto 
			  ) const;

    //! Get deposit 
    std::pair<double,int> 
      depositAndCountWithin( 
			    double coneSize,                   //dR in which deposit is computed
			    const Vetos & vetos = Vetos(),     //additional vetos 
			    double threshold = -1e+36,         //threshold on counted deposits
			    bool skipDepositVeto = false       //skip exclusion of veto 
			    ) const;
    
    //! Get deposit wrt other direction
    std::pair<double,int> 
      depositAndCountWithin( 
			    Direction dir,                     //wrt another direction
			    double coneSize,                   //dR in which deposit is computed
			    const Vetos & vetos = Vetos(),     //additional vetos 
			    double threshold = -1e+36,         //threshold on deposits
			    bool skipDepositVeto = false       //skip exclusion of veto 
			    ) const;

    //! Get deposit with new style vetos
    double depositWithin( 
			 double coneSize,                            //dR in which deposit is computed
			 const AbsVetos & vetos,                     //additional vetos 
			 bool skipDepositVeto = false                //skip exclusion of veto 
			 ) const;

    //! Get deposit 
    std::pair<double,int> 
      depositAndCountWithin( 
			    double coneSize,                            //dR in which deposit is computed
			    const AbsVetos & vetos,                     //additional vetos 
			    bool skipDepositVeto = false                //skip exclusion of veto 
			    ) const;

 
    //! Get energy or pT attached to cand trajectory
    double candEnergy() const {return theCandTag;}

    //! Set energy or pT attached to cand trajectory
    void addCandEnergy(double et) { theCandTag += et;}

    std::string print() const;

    class const_iterator {
    public:
      const const_iterator & operator++() { ++it_; cacheReady_ = false; return *this; }
      const const_iterator * operator->() const { return this; }
      float dR() const { return it_->first.deltaR; }
      float eta() const { if (!cacheReady_) doDir(); return cache_.eta(); }
      float phi() const { if (!cacheReady_) doDir(); return cache_.phi(); }
      float value() const { return it_->second; }
      bool  operator!=(const const_iterator &it2) { return it2.it_ != it_; }
      friend class IsoDeposit;
    private:
      typedef Direction::Distance Distance;
      void doDir() const { cache_ = parent_->direction() + it_->first; cacheReady_ = true; } 
      const_iterator(const IsoDeposit* parent, std::multimap<Distance, float>::const_iterator it) : 
	parent_(parent), it_(it), cache_(), cacheReady_(false) { } 
	const reco::IsoDeposit* parent_;
	mutable std::multimap<Distance, float>::const_iterator it_;
	mutable Direction cache_;
	mutable bool      cacheReady_;
    };
    const_iterator begin() const { return const_iterator(this, theDeposits.begin()); } 
    const_iterator end() const { return const_iterator(this, theDeposits.end()); } 
  private:

    //! direcion of deposit (center of isolation cone)
    Direction theDirection;

    //! area to be excluded in computaion of depositWithin 
    Veto      theVeto;
    
    //! float tagging cand, ment to be transverse energy or pT attached to cand,
    float theCandTag; 

    //! the deposits identifed by relative position to center of cone and deposit value
    typedef Direction::Distance Distance;
    typedef std::multimap<Distance, float> DepositsMultimap;

    DepositsMultimap theDeposits;
  };

}


#endif
