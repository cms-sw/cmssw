#include "RecoJets/JetProducers/interface/NjettinessAdder.h"
#include "fastjet/contrib/Njettiness.hh"

#include "FWCore/Framework/interface/MakerMacros.h"

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

    std::auto_ptr<edm::ValueMap<float> > outT(new edm::ValueMap<float>());
    edm::ValueMap<float>::Filler fillerT(*outT);
    fillerT.insert(jets, tauN.begin(), tauN.end());
    fillerT.fill();

    iEvent.put(outT,tauN_str.str().c_str());
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

  fastjet::contrib::NsubParameters paraNsub = fastjet::contrib::NsubParameters(1.0, cone_);
  fastjet::contrib::Njettiness routine(fastjet::contrib::Njettiness::onepass_kt_axes, paraNsub);
  return routine.getTau(num, FJparticles); 
}



DEFINE_FWK_MODULE(NjettinessAdder);
