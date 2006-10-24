#ifndef MuIsoDeposit_H
#define MuIsoDeposit_H

/** \class MuIsoDeposit
 *  Class representing the dR profile of deposits around a muon, i.e.
 *  the differential and integral deposits around the muon as a function of dR.
 *  
 *  Each instance should describe deposits of omogeneous type (e.g. ECAL,
 *  HCAL...); it is labelled with a string, and carries information about
 *  the cone axis, the muon pT and Z of vertex, and a weight.
 *
 *  \author N. Amapane - M. Konecki
 *  Ported with an alternative interface to CMSSW by J. Alcaraz
 */

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include <string>
#include <map>

namespace reco {

  class MuIsoDeposit {
  public:
    /// Default constructor
    MuIsoDeposit();

    /// Constructor
    MuIsoDeposit(const std::string type, double eta=0, double phi=0);

    /// Destructor
    virtual ~MuIsoDeposit(){};

    /// Get type of deposit (ECAL, HCAL, Tracker)
    const std::string getType() const {return type_;};

    /// Get eta of isolation cone
    double eta() const {return eta_;};

    /// Get phi of isolation cone
    double phi() const {return phi_;};

    /// Add energy or pT
    void addDeposit(double dr, double ene);

    /// Get deposit in a cone of dR=coneSize (muon energy excluded)
    double depositWithin(double coneSize) const;

    /// Get energy or pT attached to muon trajectory
    double muonEnergy() const {return depositFromMuon;}

    /// Set energy or pT attached to muon trajectory
    void addMuonEnergy(double et) {depositFromMuon += et;}

  private:
    std::string type_;  // type
    double eta_;  // eta
    double phi_;  // phi
    double depositFromMuon; // energy or pT attached to muon
    std::multimap<double,double> deposits; // Stores values sorted by increasing DR

  };

}

#include "DataFormats/MuonReco/interface/MuIsoDepositFwd.h"

#endif
