#ifndef JetMETCorrections_Type1MET_MultShiftMETcorrInputProducer_h
#define JetMETCorrections_Type1MET_MultShiftMETcorrInputProducer_h

/** \class MultShiftMETcorrInputProducer
 *
 * Compute MET correction to compensate systematic shift of MET in x/y-direction
 * (cf. https://indico.cern.ch/getFile.py/access?contribId=1&resId=0&materialId=slides&confId=174318 )
 *
 * \authors Christian Veelken, LLR
 *
 *
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

#include <TF1.h>

#include <string>
#include <vector>

class MultShiftMETcorrInputProducer : public edm::EDProducer  
{
 public:

  explicit MultShiftMETcorrInputProducer(const edm::ParameterSet&);
  ~MultShiftMETcorrInputProducer();
  std::vector<edm::ParameterSet> cfgCorrParameters_;
    
 private:

  void produce(edm::Event&, const edm::EventSetup&);

  std::string moduleLabel_;

//  edm::EDGetTokenT<edm::View<reco::MET> > token_;
  edm::EDGetTokenT<std::vector<reco::PFCandidate> > pflowToken_;

  std::vector<double> etaMin_, etaMax_;
  std::vector<int> type_, counts_;
  std::vector<TF1*> formula_x_;
  std::vector<TF1*> formula_y_;
};

#endif


 

