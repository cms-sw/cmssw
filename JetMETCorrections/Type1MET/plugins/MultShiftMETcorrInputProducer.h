#ifndef JetMETCorrections_Type1MET_MultShiftMETcorrInputProducer_h
#define JetMETCorrections_Type1MET_MultShiftMETcorrInputProducer_h

/** \class MultShiftMETcorrInputProducer
 *
 * Compute MET correction to compensate systematic shift of MET in x/y-direction
 * (cf. https://indico.cern.ch/getFile.py/access?contribId=1&resId=0&materialId=slides&confId=174318 )
 *
 * \authors Robert Schoefbeck, Vienna
 *
 *
 *
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

#include <TF1.h>

#include <string>
#include <vector>

class MultShiftMETcorrInputProducer : public edm::stream::EDProducer<>  
{
 public:

  explicit MultShiftMETcorrInputProducer(const edm::ParameterSet&);
  ~MultShiftMETcorrInputProducer();
    
 private:

  void produce(edm::Event&, const edm::EventSetup&) override;
  static int translateTypeToAbsPdgId( reco::PFCandidate::ParticleType type );


  edm::EDGetTokenT<edm::View<reco::Candidate> > pflow_;
  edm::EDGetTokenT<edm::View<reco::Vertex>> vertices_;
  std::string moduleLabel_;

  std::vector<edm::ParameterSet> cfgCorrParameters_;

  std::vector<double> etaMin_, etaMax_;
  std::vector<int> type_, counts_, varType_;
  std::vector<double> sumPt_;
  std::vector<std::unique_ptr<TF1> > formula_x_;
  std::vector<std::unique_ptr< TF1> > formula_y_;
};

#endif


 

