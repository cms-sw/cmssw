#ifndef RecoTauTag_RecoTau_PFRecoTauDiscriminationAgainstMuon_H_
#define RecoTauTag_RecoTau_PFRecoTauDiscriminationAgainstMuon_H_

/* class PFRecoTauDiscriminationAgainstMuon
 * created : May 07 2008,
 * revised : ,
 * Authorss : Sho Maruyama
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTauTag/TauTagTools/interface/TauTagTools.h"

#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "RecoMuon/MuonIdentification/interface/IdGlobalFunctions.h"

using namespace std; 
using namespace edm;
using namespace edm::eventsetup; 
using namespace reco;

class PFRecoTauDiscriminationAgainstMuon : public EDProducer {
 public:
  explicit PFRecoTauDiscriminationAgainstMuon(const ParameterSet& iConfig){   
    PFTauProducer_        = iConfig.getParameter<string>("PFTauProducer");
    discriminatorOption_  = iConfig.getParameter<string>("discriminatorOption");  
    a  = iConfig.getParameter<double>("a");  
    b  = iConfig.getParameter<double>("b");  

    produces<PFTauDiscriminator>();
  }
  ~PFRecoTauDiscriminationAgainstMuon(){} 
  virtual void produce(Event&, const EventSetup&);
 private:  
  string PFTauProducer_;
  string discriminatorOption_;
  double a;
  double b;
};
#endif
