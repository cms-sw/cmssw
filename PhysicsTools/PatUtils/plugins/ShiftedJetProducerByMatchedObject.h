#ifndef PhysicsTools_PatUtils_ShiftedJetProducerByMatchedObject_h
#define PhysicsTools_PatUtils_ShiftedJetProducerByMatchedObject_h

/** \class ShiftedJetProducerByMatchedObject
 *
 * Vary energy of jets coinciding in eta-phi with selected electrons/muons/tau-jets
 * by electron/muon/tau-jet energy uncertainty.
 *
 * \author Christian Veelken, LLR
 *
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

#include <string>
#include <vector>

template <typename T>
class ShiftedJetProducerByMatchedObjectT : public edm::stream::EDProducer<>  
{
  typedef std::vector<T> JetCollection;

 public:

  explicit ShiftedJetProducerByMatchedObjectT(const edm::ParameterSet&);
  ~ShiftedJetProducerByMatchedObjectT();
    
 private:

  void produce(edm::Event&, const edm::EventSetup&);

  std::string moduleLabel_;

  edm::EDGetTokenT<JetCollection> srcJets_; 
  edm::EDGetTokenT<edm::View<reco::Candidate> > srcUnshiftedObjects_; 
  edm::EDGetTokenT<edm::View<reco::Candidate> > srcShiftedObjects_; 

  double dRmatch_Jet_;
  double dRmatch_Object_;

  double dR2match_Jet_;
  double dR2match_Object_;

  struct objectEntryType
  {
    objectEntryType(const reco::Candidate::LorentzVector& shiftedObjectP4, 
		    const reco::Candidate::LorentzVector& unshiftedObjectP4, double dRmatch)
      : shiftedObjectP4_(shiftedObjectP4),
	unshiftedObjectP4_(unshiftedObjectP4),
	dRmatch_(dRmatch),
	isValidMatch_(false)
    {
      if ( unshiftedObjectP4.energy() > 0. ) {
	shift_ = (shiftedObjectP4.energy()/unshiftedObjectP4.energy()) - 1.;
	isValidMatch_ = true;
      }
    }
    ~objectEntryType() {}
    reco::Candidate::LorentzVector shiftedObjectP4_;
    reco::Candidate::LorentzVector unshiftedObjectP4_;
    double shift_;
    double dRmatch_;
    bool isValidMatch_;
  };

  std::vector<objectEntryType> objects_;
};

#endif


 

