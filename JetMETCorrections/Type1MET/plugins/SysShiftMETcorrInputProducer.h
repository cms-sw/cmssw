#ifndef JetMETCorrections_Type1MET_SysShiftMETcorrInputProducer_h
#define JetMETCorrections_Type1MET_SysShiftMETcorrInputProducer_h

/** \class SysShiftMETcorrInputProducer
 *
 * Compute MET correction to compensate systematic shift of MET in x/y-direction
 * (cf. https://indico.cern.ch/getFile.py/access?contribId=1&resId=0&materialId=slides&confId=174318 )
 *
 * \authors Christian Veelken, LLR
 *
 * \version $Revision: 1.3 $
 *
 * $Id: SysShiftMETcorrInputProducer.h,v 1.3 2012/08/31 08:00:54 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "JetMETCorrections/Type1MET/interface/SysShiftMETcorrExtractor.h"

#include <string>

class SysShiftMETcorrInputProducer : public edm::EDProducer  
{
 public:

  explicit SysShiftMETcorrInputProducer(const edm::ParameterSet&);
  ~SysShiftMETcorrInputProducer();
    
 private:

  void produce(edm::Event&, const edm::EventSetup&);

  std::string moduleLabel_;

  edm::InputTag srcMEt_;
  edm::InputTag srcVertices_;
  edm::InputTag srcJets_; 
  double jetPtThreshold_;

  SysShiftMETcorrExtractor* extractor_;
};

#endif
