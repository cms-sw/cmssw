#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

/**
   \class   JetEnergyShift JetEnergyShift.h "PhysicsTools/PatExamples/plugins/JetEnergyShift.h"

   \brief   Plugin to shift the jet energy scale and recalculate the MET accordingly

   Plugin to shift the jet energy scale and recalculate the MET accordingly. The module 
   mimics the assumption that the jet energy scale (JES) has been estimated wrong by a
   factor of _scaleFactor_, corresponding to a L2L3 corrected jet. The p4 of the patJet 
   is beeing rescaled. All other patJet properties stay the same. The MET is recalculated 
   taking the shifted JES into account for the Type1 MET correction. For the patMET the 
   rescaled sumET and the p4 are stored. The different correction levels are lost for 
   the new collection. The module has the following parameters: 

  inputJets            : input collection for  MET (expecting patMET).
  inputMETs            : input collection for jets (expecting patJets).
  scaleFactor          : scale factor to which to shift the JES.
  jetPTThresholdForMET : pt threshold for (uncorrected!) jets considered for Type1 MET 
                         corrections. 
  jetEMLimitForMET     : limit in em fraction for Type1 MET correction. 

  For expected parameters for _jetPTThresholdForMET_ and _jetEMLimitForMET_ have a look 
  at: JetMETCorrections/Type1MET/python/MetType1Corrections_cff.py. Two output collections 
  are written to file with instance label corresponding to the input label of the jet 
  and met input collections. 
*/

class JetEnergyShift : public edm::EDProducer {

 public:
  /// default constructor
  explicit JetEnergyShift(const edm::ParameterSet&);
  /// default destructor
  ~JetEnergyShift(){};
  
 private:
  /// rescale jet energy and recalculated MET
  virtual void produce(edm::Event&, const edm::EventSetup&);

 private:
  /// jet input collection 
  edm::InputTag inputJets_;
  /// met input collection
  edm::InputTag inputMETs_;
  /// jet output collection 
  std::string outputJets_;
  /// MET output collection 
  std::string outputMETs_;
  /// scale factor for the rescaling
  double scaleFactor_;
  /// threshold on (raw!) jet pt for Type1 MET corrections 
  double jetPTThresholdForMET_;
  /// limit on the emf of the jet for Type1 MET corrections 
  double jetEMLimitForMET_;
};


#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"

JetEnergyShift::JetEnergyShift(const edm::ParameterSet& cfg):
  inputJets_           (cfg.getParameter<edm::InputTag>("inputJets"           )),
  inputMETs_           (cfg.getParameter<edm::InputTag>("inputMETs"           )),
  scaleFactor_         (cfg.getParameter<double>       ("scaleFactor"         )),
  jetPTThresholdForMET_(cfg.getParameter<double>       ("jetPTThresholdForMET")),
  jetEMLimitForMET_    (cfg.getParameter<double>       ("jetEMLimitForMET"    ))
{
  // use label of input to create label for output
  outputJets_ = inputJets_.label();
  outputMETs_ = inputMETs_.label();
  // register products
  produces<std::vector<pat::Jet> >(outputJets_);
  produces<std::vector<pat::MET> >(outputMETs_);
}

void
JetEnergyShift::produce(edm::Event& event, const edm::EventSetup& setup)
{
  edm::Handle<std::vector<pat::Jet> > jets;
  event.getByLabel(inputJets_, jets);

  edm::Handle<std::vector<pat::MET> > mets;
  event.getByLabel(inputMETs_, mets);
  
  std::auto_ptr<std::vector<pat::Jet> > pJets(new std::vector<pat::Jet>);
  std::auto_ptr<std::vector<pat::MET> > pMETs(new std::vector<pat::MET>);

  double dPx    = 0.;
  double dPy    = 0.;
  double dSumEt = 0.;

  for(std::vector<pat::Jet>::const_iterator jet = jets->begin(); jet != jets->end(); ++jet) {
    pat::Jet scaledJet = *jet;
    scaledJet.scaleEnergy( scaleFactor_ );
    pJets->push_back( scaledJet );
    // consider jet scale shift only if the raw jet pt and emf 
    // is above the thresholds given in the module definition
    if(jet->correctedJet("raw").pt() > jetPTThresholdForMET_
       && jet->emEnergyFraction() < jetEMLimitForMET_) {
      dPx    += scaledJet.px() - jet->px();
      dPy    += scaledJet.py() - jet->py();
      dSumEt += scaledJet.et() - jet->et();
    }
  }

  // scale MET accordingly
  pat::MET met = *(mets->begin());
  double scaledMETPx = met.px() - dPx;
  double scaledMETPy = met.py() - dPy;
  pat::MET scaledMET(reco::MET(met.sumEt()+dSumEt, reco::MET::LorentzVector(scaledMETPx, scaledMETPy, 0, sqrt(scaledMETPx*scaledMETPx+scaledMETPy*scaledMETPy)), reco::MET::Point(0,0,0)));
  pMETs->push_back( scaledMET );
  event.put(pJets, outputJets_);
  event.put(pMETs, outputMETs_);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE( JetEnergyShift );
