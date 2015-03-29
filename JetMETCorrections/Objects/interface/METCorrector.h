//
// Generic interface for JetCorrection services
//
#ifndef METCorrector_h
#define METCorrector_h

#include <string>
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/METReco/interface/MET.h"

/// classes declaration
namespace edm {
  class Event;
  class EventSetup;
}

class METCorrector
{
 public:

  typedef reco::Particle::LorentzVector LorentzVector;
 
  METCorrector (){};
  virtual ~METCorrector (){};

  /// get correction using Jet information only
  virtual double correction (const LorentzVector& fJet) const = 0;

  /// apply correction using Jet information only
  virtual double correction (const reco::Jet& fJet) const = 0;

  /// apply correction using all event information
  virtual double correction (const reco::Jet& fJet,
			     const edm::Event& fEvent, 
			     const edm::EventSetup& fSetup) const;

  /// apply correction using all event information
  virtual double correction (const reco::Jet& fJet,
			     const edm::RefToBase<reco::Jet>& fJetRef,
			     const edm::Event& fEvent, 
			     const edm::EventSetup& fSetup) const;

  /// Apply vectorial correction using all event information
  virtual double correction ( const reco::Jet& fJet, 
			      const edm::RefToBase<reco::Jet>& fJetRef,
			      const edm::Event& fEvent, 
			      const edm::EventSetup& fSetup, 
			      LorentzVector& corrected ) const;

   /// apply correction using Met information only
   //virtual double correction (const reco::MET& fMet) const = 0;

   /// get corrections using MET information only
   virtual double correction (const reco::MET& fMet,
                             const edm::Event& fEvent, 
			     const edm::EventSetup& fSetup) const;

   /// get corrections by parameters propagation
   virtual std::vector<double> correction () const = 0;
  
  
  /// if correction needs event information
  virtual bool eventRequired () const = 0;

  /// if correction needs the jet reference
  virtual bool refRequired () const = 0;
  
  /// if vectorial correction is provided
  inline virtual bool vectorialCorrection () const;

  /// retrieve corrector from the event setup. troughs exception if something is missing
  //static const METCorrector* getJetCorrector (const std::string& fName, const edm::EventSetup& fSetup); 
};

// inline method
inline bool METCorrector::vectorialCorrection () const { return false; }

#endif
