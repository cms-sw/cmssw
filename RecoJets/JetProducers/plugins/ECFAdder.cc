#include "RecoJets/JetProducers/interface/ECFAdder.h"
#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequence.hh"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

ECFAdder::ECFAdder(const edm::ParameterSet& iConfig) :
  src_(iConfig.getParameter<edm::InputTag>("src")),
  src_token_(consumes<edm::View<reco::Jet>>(src_)),
  Njets_(iConfig.getParameter<std::vector<unsigned> >("Njets")),
  ecftype_(iConfig.getParameter<std::string>("ecftype")),
  beta_(iConfig.getParameter<double>("beta"))
{
  if ( iConfig.exists("alpha") ) {
    alpha_ = iConfig.getParameter<double>("alpha");
  } else {
    alpha_ = beta_;
  }

    for ( std::vector<unsigned>::const_iterator n = Njets_.begin(); n != Njets_.end(); ++n )
      {
	std::ostringstream ecfN_str;
	std::shared_ptr<fastjet::FunctionOfPseudoJet<double> > pfunc;

	if ( ecftype_ == "ECF" || ecftype_ == "" ) {
	  ecfN_str << "ecf" << *n;
	  pfunc.reset( new fastjet::contrib::EnergyCorrelator( *n, beta_, fastjet::contrib::EnergyCorrelator::pt_R ) );
	}
	else if ( ecftype_ == "C" ) {
	  ecfN_str << "ecfC" << *n;
	  pfunc.reset( new fastjet::contrib::EnergyCorrelatorCseries( *n, beta_, fastjet::contrib::EnergyCorrelator::pt_R ) );
	}
	else if ( ecftype_ == "D" ) {
	  ecfN_str << "ecfD" << *n;
	  pfunc.reset( new fastjet::contrib::EnergyCorrelatorGeneralizedD2( alpha_, beta_, fastjet::contrib::EnergyCorrelator::pt_R ) );
	}
	else if ( ecftype_ == "N" ) {
	  ecfN_str << "ecfN" << *n;
	  pfunc.reset( new fastjet::contrib::EnergyCorrelatorNseries( *n, beta_, fastjet::contrib::EnergyCorrelator::pt_R ) );
	}
	else if ( ecftype_ == "M" ) {
	  ecfN_str << "ecfM" << *n;
	  pfunc.reset( new fastjet::contrib::EnergyCorrelatorMseries( *n, beta_, fastjet::contrib::EnergyCorrelator::pt_R ) );
	}
	else if ( ecftype_ == "U" ) {
	  ecfN_str << "ecfU" << *n;
	  pfunc.reset( new fastjet::contrib::EnergyCorrelatorUseries( *n, beta_, fastjet::contrib::EnergyCorrelator::pt_R ) );
	}
	variables_.push_back(ecfN_str.str());
	produces<edm::ValueMap<float> >(ecfN_str.str().c_str());
	routine_.push_back(pfunc);
      }

}

void ECFAdder::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
  // read input collection
  edm::Handle<edm::View<reco::Jet> > jets;
  iEvent.getByToken(src_token_, jets);
  
  unsigned i=0;
  for ( std::vector<unsigned>::const_iterator n = Njets_.begin(); n != Njets_.end(); ++n )
    {
      // prepare room for output
      std::vector<float> ecfN;
      ecfN.reserve(jets->size());

      for ( typename edm::View<reco::Jet>::const_iterator jetIt = jets->begin() ; jetIt != jets->end() ; ++jetIt ) {

	edm::Ptr<reco::Jet> jetPtr = jets->ptrAt(jetIt - jets->begin());

	float t=getECF( i, jetPtr );

	ecfN.push_back(t);
      }

      auto outT = std::make_unique<edm::ValueMap<float>>();
      edm::ValueMap<float>::Filler fillerT(*outT);
      fillerT.insert(jets, ecfN.begin(), ecfN.end());
      fillerT.fill();

      iEvent.put(std::move(outT),variables_[i].c_str());
      ++i;
    }
}

float ECFAdder::getECF(unsigned index, const edm::Ptr<reco::Jet> & object) const
{
  std::vector<fastjet::PseudoJet> FJparticles;
  for (unsigned k = 0; k < object->numberOfDaughters(); ++k)
    {
      const reco::CandidatePtr & dp = object->daughterPtr(k);
      if ( dp.isNonnull() && dp.isAvailable() ){

	// Here, the daughters are the "end" node, so this is a PFJet
	if ( dp->numberOfDaughters() == 0 ) {
	  FJparticles.push_back( fastjet::PseudoJet( dp->px(), dp->py(), dp->pz(), dp->energy() ) );
	} else { // Otherwise, this is a BasicJet, so you need to descend further.
	  auto subjet = dynamic_cast<reco::Jet const * >( dp.get() );
	  for ( unsigned l = 0; l < subjet->numberOfDaughters(); ++l ) {
	    if ( subjet != 0 ) {
	      const reco::CandidatePtr & ddp = subjet->daughterPtr(l);
	      FJparticles.push_back( fastjet::PseudoJet( ddp->px(), ddp->py(), ddp->pz(), ddp->energy() ) );	      
	    } else {
	      edm::LogWarning("MissingJetConstituent") << "BasicJet constituent required for ECF computation is missing!";
	    }
	  }
	} // end if basic jet
      } // end if daughter pointer is nonnull and available
      else
	edm::LogWarning("MissingJetConstituent") << "Jet constituent required for ECF computation is missing!";
    }
    return routine_[index]->result(join(FJparticles)); 
}


// ParameterSet description for module
void ECFAdder::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription iDesc;
  iDesc.setComment("Energy Correlation Functions adder");

  iDesc.add<edm::InputTag>("src", edm::InputTag("no default"))->setComment("input collection");
  iDesc.add<std::vector<unsigned> >("Njets", {1,2,3} )->setComment("Number of jets to emulate");
  iDesc.addOptional<double>("alpha",1.0)->setComment("alpha factor, only valid for N2");
  iDesc.add<double>("beta",1.0)->setComment("angularity factor");
  iDesc.add<std::string>("ecftype","")->setComment("ECF type: ECF or empty; C; D; N; M; U;");

  descriptions.add("ECFAdder", iDesc);
}

DEFINE_FWK_MODULE(ECFAdder);
