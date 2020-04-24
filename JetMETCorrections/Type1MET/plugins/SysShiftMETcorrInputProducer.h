#ifndef JetMETCorrections_Type1MET_SysShiftMETcorrInputProducer_h
#define JetMETCorrections_Type1MET_SysShiftMETcorrInputProducer_h

/** \class SysShiftMETcorrInputProducer
 *
 * Compute MET correction to compensate systematic shift of MET in x/y-direction
 * (cf. https://indico.cern.ch/getFile.py/access?contribId=1&resId=0&materialId=slides&confId=174318 )
 *
 * \authors Christian Veelken, LLR
 *
 *
 *
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include <TFormula.h>

#include <string>

class SysShiftMETcorrInputProducer : public edm::stream::EDProducer<>  
{
 public:

  explicit SysShiftMETcorrInputProducer(const edm::ParameterSet&);
  ~SysShiftMETcorrInputProducer();
    
 private:

  void produce(edm::Event&, const edm::EventSetup&);

  std::string moduleLabel_;

  edm::EDGetTokenT<edm::View<reco::MET> > token_;
  edm::EDGetTokenT<reco::VertexCollection> verticesToken_;

  bool useNvtx;

  TFormula* corrPx_;
  TFormula* corrPy_;
};

#endif


 

