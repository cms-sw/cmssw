#ifndef JetMETCorrections_Type1MET_PFchsMETcorrInputProducer_h
#define JetMETCorrections_Type1MET_PFchsMETcorrInputProducer_h

/** \class PFchsMETcorrInputProducer
 *
 * Sum PF Charged Particles Originating from the primary vertices which are
 * not primary vertex of the high-pT events 
 * needed as input for Type 0 MET corrections
 *
 * \authors Anne-Maria Visuri, Mikko Voutilainen
 *          Tai Sakuma
 *
 * \version $Revision: 1.2 $
 *
 * $Id: PFchsMETcorrInputProducer.h,v 1.2 2011/10/14 10:14:35 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/METReco/interface/CorrMETData.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include <string>

class PFchsMETcorrInputProducer : public edm::EDProducer  
{
 public:

  explicit PFchsMETcorrInputProducer(const edm::ParameterSet&);
  ~PFchsMETcorrInputProducer();
    
 private:

  void produce(edm::Event&, const edm::EventSetup&);

  std::string moduleLabel_;

  edm::InputTag src_; // input vertex collection

  unsigned goodVtxNdof_;
  double goodVtxZ_;
 

};

#endif


 

