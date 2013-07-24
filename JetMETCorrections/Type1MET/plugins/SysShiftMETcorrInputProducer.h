#ifndef JetMETCorrections_Type1MET_SysShiftMETcorrInputProducer_h
#define JetMETCorrections_Type1MET_SysShiftMETcorrInputProducer_h

/** \class SysShiftMETcorrInputProducer
 *
 * Compute MET correction to compensate systematic shift of MET in x/y-direction
 * (cf. https://indico.cern.ch/getFile.py/access?contribId=1&resId=0&materialId=slides&confId=174318 )
 *
 * \authors Christian Veelken, LLR
 *
 * \version $Revision: 1.2 $
 *
 * $Id: SysShiftMETcorrInputProducer.h,v 1.2 2012/04/09 14:19:01 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <TFormula.h>

#include <string>

class SysShiftMETcorrInputProducer : public edm::EDProducer  
{
 public:

  explicit SysShiftMETcorrInputProducer(const edm::ParameterSet&);
  ~SysShiftMETcorrInputProducer();
    
 private:

  void produce(edm::Event&, const edm::EventSetup&);

  std::string moduleLabel_;

  edm::InputTag src_; // PFCandidate input collection
  edm::InputTag srcVertices_; // Vertex input collection

  TFormula* corrPx_;
  TFormula* corrPy_;
};

#endif


 

