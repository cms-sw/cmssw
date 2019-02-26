#include "RecoJets/JetProducers/interface/NjettinessAdder.h"


#include "FWCore/Framework/interface/MakerMacros.h"

NjettinessAdder::NjettinessAdder(const edm::ParameterSet& iConfig) :
  src_(iConfig.getParameter<edm::InputTag>("src")),
  src_token_(consumes<edm::View<reco::Jet>>(src_)),
  Njets_(iConfig.getParameter<std::vector<unsigned> >("Njets")),
  measureDefinition_(iConfig.getParameter<unsigned >("measureDefinition")),
  beta_(iConfig.getParameter<double>("beta")),
  R0_(iConfig.getParameter<double>("R0")),
  Rcutoff_(iConfig.getParameter<double>("Rcutoff")),
  axesDefinition_(iConfig.getParameter< unsigned >("axesDefinition")),
  nPass_(iConfig.getParameter<int>("nPass")),
  akAxesR0_(iConfig.getParameter<double>("akAxesR0"))
{
  for ( std::vector<unsigned>::const_iterator n = Njets_.begin(); n != Njets_.end(); ++n )
    {
      std::ostringstream tauN_str;
      tauN_str << "tau" << *n;

      produces<edm::ValueMap<float> >(tauN_str.str());
    }


  // Get the measure definition
  fastjet::contrib::NormalizedMeasure          normalizedMeasure        (beta_,R0_);
  fastjet::contrib::UnnormalizedMeasure        unnormalizedMeasure      (beta_);
  fastjet::contrib::OriginalGeometricMeasure   geometricMeasure         (beta_);// changed in 1.020
  fastjet::contrib::NormalizedCutoffMeasure    normalizedCutoffMeasure  (beta_,R0_,Rcutoff_);
  fastjet::contrib::UnnormalizedCutoffMeasure  unnormalizedCutoffMeasure(beta_,Rcutoff_);
  //fastjet::contrib::GeometricCutoffMeasure     geometricCutoffMeasure   (beta_,Rcutoff_); // removed in 1.020

  fastjet::contrib::MeasureDefinition const * measureDef = nullptr;
  switch ( measureDefinition_ ) {
  case UnnormalizedMeasure : measureDef = &unnormalizedMeasure; break;
  case OriginalGeometricMeasure    : measureDef = &geometricMeasure; break;// changed in 1.020
  case NormalizedCutoffMeasure : measureDef = &normalizedCutoffMeasure; break;
  case UnnormalizedCutoffMeasure : measureDef = &unnormalizedCutoffMeasure; break;
  //case GeometricCutoffMeasure : measureDef = &geometricCutoffMeasure; break; // removed in 1.020
  case NormalizedMeasure : default : measureDef = &normalizedMeasure; break;
  } 

  // Get the axes definition
  fastjet::contrib::KT_Axes             kt_axes; 
  fastjet::contrib::CA_Axes             ca_axes; 
  fastjet::contrib::AntiKT_Axes         antikt_axes   (akAxesR0_);
  fastjet::contrib::WTA_KT_Axes         wta_kt_axes; 
  fastjet::contrib::WTA_CA_Axes         wta_ca_axes; 
  fastjet::contrib::OnePass_KT_Axes     onepass_kt_axes;
  fastjet::contrib::OnePass_CA_Axes     onepass_ca_axes;
  fastjet::contrib::OnePass_AntiKT_Axes onepass_antikt_axes   (akAxesR0_);
  fastjet::contrib::OnePass_WTA_KT_Axes onepass_wta_kt_axes;
  fastjet::contrib::OnePass_WTA_CA_Axes onepass_wta_ca_axes;
  fastjet::contrib::MultiPass_Axes      multipass_axes (nPass_);

  fastjet::contrib::AxesDefinition const * axesDef = nullptr;
  switch ( axesDefinition_ ) {
  case  KT_Axes : default : axesDef = &kt_axes; break;
  case  CA_Axes : axesDef = &ca_axes; break; 
  case  AntiKT_Axes : axesDef = &antikt_axes; break;
  case  WTA_KT_Axes : axesDef = &wta_kt_axes; break; 
  case  WTA_CA_Axes : axesDef = &wta_ca_axes; break; 
  case  OnePass_KT_Axes : axesDef = &onepass_kt_axes; break;
  case  OnePass_CA_Axes : axesDef = &onepass_ca_axes; break; 
  case  OnePass_AntiKT_Axes : axesDef = &onepass_antikt_axes; break;
  case  OnePass_WTA_KT_Axes : axesDef = &onepass_wta_kt_axes; break; 
  case  OnePass_WTA_CA_Axes : axesDef = &onepass_wta_ca_axes; break; 
  case  MultiPass_Axes : axesDef = &multipass_axes; break;
  };

  routine_ = std::unique_ptr<fastjet::contrib::Njettiness> ( new fastjet::contrib::Njettiness( *axesDef, *measureDef ) );
      
}

void NjettinessAdder::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
  // read input collection
  edm::Handle<edm::View<reco::Jet> > jets;
  iEvent.getByToken(src_token_, jets);
  
  for ( std::vector<unsigned>::const_iterator n = Njets_.begin(); n != Njets_.end(); ++n )
    {
      std::ostringstream tauN_str;
      tauN_str << "tau" << *n;

      // prepare room for output
      std::vector<float> tauN;
      tauN.reserve(jets->size());

      for ( typename edm::View<reco::Jet>::const_iterator jetIt = jets->begin() ; jetIt != jets->end() ; ++jetIt ) {

	edm::Ptr<reco::Jet> jetPtr = jets->ptrAt(jetIt - jets->begin());

	float t=getTau( *n, jetPtr );

	tauN.push_back(t);
      }

      auto outT = std::make_unique<edm::ValueMap<float>>();
      edm::ValueMap<float>::Filler fillerT(*outT);
      fillerT.insert(jets, tauN.begin(), tauN.end());
      fillerT.fill();

      iEvent.put(std::move(outT),tauN_str.str());
    }
}

float NjettinessAdder::getTau(unsigned num, const edm::Ptr<reco::Jet> & object) const
{
  std::vector<fastjet::PseudoJet> FJparticles;
  for (unsigned k = 0; k < object->numberOfDaughters(); ++k)
    {
      const reco::CandidatePtr & dp = object->daughterPtr(k);
      if ( dp.isNonnull() && dp.isAvailable() )
	FJparticles.push_back( fastjet::PseudoJet( dp->px(), dp->py(), dp->pz(), dp->energy() ) );
      else
	edm::LogWarning("MissingJetConstituent") << "Jet constituent required for N-subjettiness computation is missing!";
    }

  return routine_->getTau(num, FJparticles); 
}



DEFINE_FWK_MODULE(NjettinessAdder);
