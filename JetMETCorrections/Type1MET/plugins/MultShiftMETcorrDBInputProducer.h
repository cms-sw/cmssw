#ifndef JetMETCorrections_Type1MET_MultShiftMETcorrDBInputProducer_h
#define JetMETCorrections_Type1MET_MultShiftMETcorrDBInputProducer_h

/** \class MultShiftMETcorrDBInputProducer
 *
 * Compute MET correction to compensate systematic shift of MET in x/y-direction
 * (cf. https://indico.cern.ch/getFile.py/access?contribId=1&resId=0&materialId=slides&confId=174318 )
 *
 * \authors SangEun Lee,
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

class MultShiftMETcorrDBInputProducer : public edm::stream::EDProducer<>  
{
 public:

  explicit MultShiftMETcorrDBInputProducer(const edm::ParameterSet&);
  ~MultShiftMETcorrDBInputProducer();
    
 private:

  void produce(edm::Event&, const edm::EventSetup&) override;
  static int translateTypeToAbsPdgId( reco::PFCandidate::ParticleType type );


  edm::EDGetTokenT<edm::View<reco::Candidate> > pflow_;
  edm::EDGetTokenT<edm::View<reco::Vertex>> vertices_;
  std::string moduleLabel_;
  std::string mPayloadName;
  std::string mSampleType;
  bool mIsData;

  std::vector<edm::ParameterSet> cfgCorrParameters_;

  std::vector<double> etaMin_, etaMax_;
  int counts_;
  double sumPt_;
  std::unique_ptr< TF1 > formula_x_;
  std::unique_ptr< TF1 > formula_y_;
};

#endif


 

