#ifndef MuIsoDeposit_H
#define MuIsoDeposit_H

/** \class MuIsoDeposit
 *  Class representing the dR profile of deposits around a muon, i.e.
 *  the differential deposits around the muon as a function of dR.
 *  
 *  Each instance should describe deposits of homogeneous type (e.g. ECAL,
 *  HCAL...); it is labelled with a string, and carries information about
 *  the cone axis, the muon pT.
 *
 *  \author N. Amapane - M. Konecki
 *  Ported with an alternative interface to CMSSW by J. Alcaraz
 */

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/MuonReco/interface/Direction.h"
#include <map>
#include <string>
#include <vector>


namespace reco {

  class MuIsoDeposit {
  public:

    typedef muonisolation::Direction Direction;
    struct Veto { Direction vetoDir; float dR; 
      Veto() {}
      Veto(Direction dir, double d):vetoDir(dir), dR(d) {}
    };
    typedef std::vector<Veto> Vetos;

    /// Constructor
    MuIsoDeposit(const std::string type="", double eta=0, double phi=0); 
    MuIsoDeposit(const std::string type, const Direction & muonDirection);

    /// Destructor
    virtual ~MuIsoDeposit(){};

    /// Get type of deposit (ECAL, HCAL, Tracker)
    const std::string getType() const {return theType;}

    /// Get direction of isolation cone
    const Direction & direction() const { return theDirection; }
    double eta() const {return theDirection.eta();}
    double phi() const {return theDirection.phi();}

    /// Get veto area
    const Veto & veto() const { return  theVeto; }
    /// Set veto
    void setVeto(const Veto & aVeto) { theVeto = aVeto; }

    /// Add deposit (ie. transverse energy or pT)
    void addDeposit(double dr, double deposit); // FIXME - temporary for backward compatibility
    void addDeposit(const Direction & depDir, double deposit);

    /// Get deposit 
    double depositWithin( 
        double coneSize,                                        //dR in which deposit is computed
        const Vetos & vetos = Vetos(),                          //additional vetos 
        bool skipDepositVeto = false                            //skip exclusion of veto 
        ) const;

    /// Get deposit wrt other direction
      double depositWithin( Direction dir,
        double coneSize,                                        //dR in which deposit is computed
        const Vetos & vetos = Vetos(),                          //additional vetos 
        bool skipDepositVeto = false                            //skip exclusion of veto 
        ) const;

    /// Get deposit 
	std::pair<double,int> depositAndCountWithin( 
	double coneSize,                                        //dR in which deposit is computed
        const Vetos & vetos = Vetos(),                          //additional vetos 
	double threshold = -1e+36,                              //threshold on counted deposits
        bool skipDepositVeto = false                            //skip exclusion of veto 
        ) const;

    /// Get deposit wrt other direction
	std::pair<double,int> depositAndCountWithin( 
        Direction dir,                                          //wrt another direction
        double coneSize,                                        //dR in which deposit is computed
	const Vetos & vetos = Vetos(),                          //additional vetos 
	double threshold = -1e+36,                              //threshold on deposits
        bool skipDepositVeto = false                            //skip exclusion of veto 
        ) const;

    /// Get energy or pT attached to muon trajectory
    double muonEnergy() const {return theMuonTag;}

    /// Set energy or pT attached to muon trajectory
    void addMuonEnergy(double et) { theMuonTag += et;}

    std::string print() const;

  private:

    /// type of deposit
    std::string theType;  

    /// direcion of deposit (center of isolation cone)
    Direction theDirection;

    /// area to be excluded in computaion of depositWithin 
    Veto      theVeto;
    
    /// float tagging muon, ment to be transverse energy or pT attached to muon,
    float theMuonTag; 

    /// the deposits identifed by relative position to center of cone and deposit value
    typedef muonisolation::Direction::Distance Distance;
    typedef std::multimap<Distance, float> DepositsMultimap;
//    struct Closer { bool operator()(const Distance&, const Distance& ) const; };
//    typedef std::multimap<Distance, double, Closer> DepositsMultimap;
    DepositsMultimap theDeposits;
  };

}


#endif
