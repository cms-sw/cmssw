#include "RecoJets/JetProducers/interface/QjetsAdder.h"
#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequence.hh"

#include "FWCore/Framework/interface/MakerMacros.h"

using namespace std;

void QjetsAdder::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
  // read input collection
  //edm::Handle<edm::View<pat::Jet> > jets;
  edm::Handle<edm::View<reco::Jet> > jets;
  iEvent.getByToken(src_token_, jets);

  // prepare room for output
  std::vector<float> QjetsVolatility;       QjetsVolatility.reserve(jets->size());

  for ( typename edm::View<reco::Jet>::const_iterator jetIt = jets->begin() ; jetIt != jets->end() ; ++jetIt ) {
    reco::Jet newCand(*jetIt);

    if(newCand.pt()<cutoff_)
    {
      QjetsVolatility.push_back(-1);
      continue;
    }

    //refill and recluster
    vector<fastjet::PseudoJet> allconstits;
    //for (unsigned k=0; k < newCand.getPFConstituents().size(); k++){
    for (unsigned k=0; k < newCand.getJetConstituents().size(); k++){
      const edm::Ptr<reco::Candidate> thisParticle = newCand.getJetConstituents().at(k);
      allconstits.push_back( fastjet::PseudoJet( thisParticle->px(), thisParticle->py(), thisParticle->pz(), thisParticle->energy() ) );
    }

    fastjet::JetDefinition jetDef(fastjet::cambridge_algorithm, jetRad_);
    if (mJetAlgo_== "AK") jetDef.set_jet_algorithm( fastjet::antikt_algorithm );
    else if (mJetAlgo_ == "CA") jetDef.set_jet_algorithm( fastjet::cambridge_algorithm );
    else throw cms::Exception("GroomedJetFiller") << " unknown jet algorithm " << std::endl;

    fastjet::ClusterSequence thisClustering_basic(allconstits, jetDef);
    std::vector<fastjet::PseudoJet> out_jets_basic = sorted_by_pt(thisClustering_basic.inclusive_jets(cutoff_));
    //std::cout << newCand.pt() << " " << out_jets_basic.size() <<std::endl;
    if(out_jets_basic.size()!=1){ // jet reclustering did not return exactly 1 jet, most likely due to the higher cutoff or large cone size. Use a recognizeable default value for this jet
      QjetsVolatility.push_back(-1);
      continue;
    }

    // setup objects for qjets computation
    fastjet::JetDefinition qjet_def(&qjetsAlgo_);

    std::vector<double> qjetmass;

    vector<fastjet::PseudoJet> constits;
    unsigned int nqjetconstits = out_jets_basic.at(0).constituents().size(); // there should always be exactly one reclsutered jet => always "at(0)"
    if (nqjetconstits < (unsigned int) QjetsPreclustering_) constits = out_jets_basic.at(0).constituents();
    else constits = out_jets_basic.at(0).associated_cluster_sequence()->exclusive_subjets_up_to(out_jets_basic.at(0),QjetsPreclustering_);

    edm::Service<edm::RandomNumberGenerator> rng;
    CLHEP::HepRandomEngine* engine = &rng->getEngine(iEvent.streamID());
    qjetsAlgo_.SetRNEngine(engine);
    // create probabilistic recusterings
    for(unsigned int ii = 0 ; ii < (unsigned int) ntrial_ ; ii++){
      //qjetsAlgo_.SetRandSeed(iEvent.id().event()*100 + (jetIt - jets->begin())*ntrial_ + ii );// set random seed for reprudcibility. We need a smarted scheme
      fastjet::ClusterSequence qjet_seq(constits, qjet_def);
      vector<fastjet::PseudoJet> inclusive_jets2 = sorted_by_pt(qjet_seq.inclusive_jets(cutoff_));
      if (inclusive_jets2.size()>0){ // fill the massvalue only if the reclustering was successfull
	qjetmass.push_back(inclusive_jets2[0].m());
      }

    }//end loop on trials

    if (qjetmass.size()<1) {//protection against dummy case
      QjetsVolatility.push_back(-1);
      continue;
    }

    double mean = std::accumulate( qjetmass.begin( ) , qjetmass.end( ) , 0 ) /qjetmass.size() ;
    float totalsquared = 0.;
    for (unsigned int i = 0; i < qjetmass.size(); i++){
      totalsquared += (qjetmass[i] - mean)*(qjetmass[i] - mean) ;
    }
    float variance = sqrt( totalsquared/qjetmass.size() );  
    
    QjetsVolatility.push_back(variance/mean);
  }//end loop on jets

  std::auto_ptr<edm::ValueMap<float> > outQJV(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler fillerQJV(*outQJV);
  fillerQJV.insert(jets, QjetsVolatility.begin(), QjetsVolatility.end());
  fillerQJV.fill();

  iEvent.put(outQJV,"QjetsVolatility");
}




DEFINE_FWK_MODULE(QjetsAdder);
