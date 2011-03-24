////////////////////////////////////////////////////////////////////////////////
//
// SubjetFilterJetProducer
// -----------------------
//
// For a description of the Subjet/Filter algorithm, see e.g.
// http://arXiv.org/abs/0802.2470 
//
// This implementation is largely based on fastjet_boosted_higgs.cc provided
// with the fastjet package.
//
// If you did, you know that the algorithm produces a 'fatjet', two
// associated 'subjets', and two or three associated 'filterjets'. This producer
// will therefore write three corresponding jet collections. The association
// between a fatjet and its subjets and filterjets is done by making all (!)
// of them daughters of the fatjet, such that the first two daughter jets always
// correspond to the subjets, while all remaining correspond to the filterjets.
//
// The real work is done in RecoJets/JetAlgorithms/src/SubjetFilterAlgorithm.cc
//       
// see: https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSubjetFilterJetProducer
//
//                       David Lopes-Pegna                 <david.lopes@cern.ch>
//            25/11/2009 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
////////////////////////////////////////////////////////////////////////////////


#include "SubjetFilterJetProducer.h"
#include "RecoJets/JetProducers/interface/JetSpecific.h"
#include "FWCore/Framework/interface/MakerMacros.h"


using namespace std;


////////////////////////////////////////////////////////////////////////////////
// construction / destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
SubjetFilterJetProducer::SubjetFilterJetProducer(const edm::ParameterSet& iConfig)
  : VirtualJetProducer(iConfig)
  , alg_(iConfig.getParameter<string>  ("@module_label"),
	 iConfig.getParameter<string>  ("jetAlgorithm"),
	 iConfig.getParameter<unsigned>("nFatMax"),
	 iConfig.getParameter<double>  ("rParam"),
	 iConfig.getParameter<double>  ("rFilt"),
	 iConfig.getParameter<double>  ("jetPtMin"),
	 iConfig.getParameter<double>  ("massDropCut"),
	 iConfig.getParameter<double>  ("asymmCut"),
	 iConfig.getParameter<bool>    ("asymmCutLater"),
	 iConfig.getParameter<bool>    ("doAreaFastjet"),
	 iConfig.getParameter<double>  ("Ghost_EtaMax"),
	 iConfig.getParameter<int>     ("Active_Area_Repeats"),
	 iConfig.getParameter<double>  ("GhostArea"),
	 iConfig.getUntrackedParameter<bool>("verbose",false))
{
  produces<reco::BasicJetCollection>("fat");
  makeProduces(moduleLabel_,"sub");
  makeProduces(moduleLabel_,"filter");
}


//______________________________________________________________________________
SubjetFilterJetProducer::~SubjetFilterJetProducer()
{
  
}


////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
void SubjetFilterJetProducer::produce(edm::Event& iEvent,
				      const edm::EventSetup& iSetup)
{
  VirtualJetProducer::produce(iEvent,iSetup);
}


//______________________________________________________________________________
void SubjetFilterJetProducer::endJob()
{
  cout<<alg_.summary()<<endl;
}


//______________________________________________________________________________
void SubjetFilterJetProducer::runAlgorithm(edm::Event& iEvent,
					   const edm::EventSetup& iSetup)
{
  alg_.run(fjInputs_, fjCompoundJets_, iSetup);
}


//______________________________________________________________________________
void SubjetFilterJetProducer::inputTowers()
{
  fjCompoundJets_.clear();
  VirtualJetProducer::inputTowers();
}


//______________________________________________________________________________
void SubjetFilterJetProducer::output(edm::Event& iEvent,
				     edm::EventSetup const& iSetup)
{
  // Write jets and constitutents. Will use fjCompoundJets_. 
  switch( jetTypeE ) {
  case JetType::CaloJet :
    writeCompoundJets<reco::CaloJet>( iEvent, iSetup );
    break;
  case JetType::PFJet :
    writeCompoundJets<reco::PFJet>( iEvent, iSetup );
    break;
  case JetType::GenJet :
    writeCompoundJets<reco::GenJet>( iEvent, iSetup );
    break;
  case JetType::BasicJet :
    writeCompoundJets<reco::BasicJet>( iEvent, iSetup );
    break;
  default:
    edm::LogError("InvalidInput")<<" invalid jet type in SubjetFilterJetProducer\n";
    break;
  };
  
}


//______________________________________________________________________________
template< class T>
void SubjetFilterJetProducer::writeCompoundJets(edm::Event& iEvent,
						const edm::EventSetup& iSetup)
{
  auto_ptr<reco::BasicJetCollection> fatJets( new reco::BasicJetCollection() );
  auto_ptr<vector<T> >  subJets( new vector<T>() );
  auto_ptr<vector<T> >  filterJets( new vector<T>() );

  edm::OrphanHandle< vector<T> > subJetsAfterPut;
  edm::OrphanHandle< vector<T> > filterJetsAfterPut;
  
  vector< vector<int> > subIndices(fjCompoundJets_.size());
  vector< vector<int> > filterIndices(fjCompoundJets_.size());
  
  vector<math::XYZTLorentzVector> p4FatJets;
  vector<double>                  areaFatJets;
  
  vector<CompoundPseudoJet>::const_iterator itBegin(fjCompoundJets_.begin());
  vector<CompoundPseudoJet>::const_iterator itEnd(fjCompoundJets_.end());
  vector<CompoundPseudoJet>::const_iterator it(itBegin);
  
  for (;it!=itEnd;++it) {

    int jetIndex = it-itBegin;
    fastjet::PseudoJet fatJet = it->hardJet();
    p4FatJets.push_back(math::XYZTLorentzVector(fatJet.px(),fatJet.py(),
						fatJet.pz(),fatJet.e()));
    areaFatJets.push_back(it->hardJetArea());
    
    vector<CompoundPseudoSubJet>::const_iterator itSubBegin(it->subjets().begin());
    vector<CompoundPseudoSubJet>::const_iterator itSubEnd(it->subjets().end());
    vector<CompoundPseudoSubJet>::const_iterator itSub(itSubBegin);
    
    for (; itSub!=itSubEnd;++itSub) {
      int subJetIndex = itSub-itSubBegin;
      fastjet::PseudoJet fjSubJet = itSub->subjet();
      math::XYZTLorentzVector p4SubJet(fjSubJet.px(),fjSubJet.py(),
				       fjSubJet.pz(),fjSubJet.e());
      reco::Particle::Point point(0,0,0);
      
      vector<reco::CandidatePtr> subJetConstituents;
      const vector<int>& subJetConstituentIndices = itSub->constituents();
      vector<int>::const_iterator itIndexBegin(subJetConstituentIndices.begin());
      vector<int>::const_iterator itIndexEnd(subJetConstituentIndices.end());
      vector<int>::const_iterator itIndex(itIndexBegin);
      for (;itIndex!=itIndexEnd;++itIndex)
	if ((*itIndex) < static_cast<int>(inputs_.size())) 
	  subJetConstituents.push_back(inputs_[*itIndex]);
      
      T subJet;
      reco::writeSpecific(subJet,p4SubJet,point,subJetConstituents,iSetup);
      subJet.setJetArea(itSub->subjetArea());
      
      if (subJetIndex<2) {
	subIndices[jetIndex].push_back(subJets->size());
	subJets->push_back(subJet);
      }
      else {
	filterIndices[jetIndex].push_back(filterJets->size());
	filterJets->push_back(subJet);
      }
    }
  }
  
  subJetsAfterPut    = iEvent.put(subJets,   "sub");
  filterJetsAfterPut = iEvent.put(filterJets,"filter");
  
  vector<math::XYZTLorentzVector>::const_iterator itP4Begin(p4FatJets.begin());
  vector<math::XYZTLorentzVector>::const_iterator itP4End(p4FatJets.end());
  vector<math::XYZTLorentzVector>::const_iterator itP4(itP4Begin);
  for (;itP4!=itP4End;++itP4) {
    int fatIndex = itP4-itP4Begin;
    vector<int>& fatToSub    = subIndices[fatIndex];
    vector<int>& fatToFilter = filterIndices[fatIndex];

    vector<reco::CandidatePtr> i_fatJetConstituents;

    vector<int>::const_iterator itSubBegin(fatToSub.begin());
    vector<int>::const_iterator itSubEnd(fatToSub.end());
    vector<int>::const_iterator itSub(itSubBegin);
    for(;itSub!=itSubEnd;++itSub) {
      reco::CandidatePtr candPtr(subJetsAfterPut,(*itSub),false);
      i_fatJetConstituents.push_back(candPtr);
    }

    vector<int>::const_iterator itFilterBegin(fatToFilter.begin());
    vector<int>::const_iterator itFilterEnd(fatToFilter.end());
    vector<int>::const_iterator itFilter(itFilterBegin);
    for(;itFilter!=itFilterEnd;++itFilter) {
      reco::CandidatePtr candPtr(filterJetsAfterPut,(*itFilter),false);
      i_fatJetConstituents.push_back(candPtr);
    }

    reco::Particle::Point point(0,0,0);
    fatJets->push_back(reco::BasicJet((*itP4),point,i_fatJetConstituents));
    fatJets->back().setJetArea(areaFatJets[fatIndex]);
  }
  
  iEvent.put(fatJets,"fat");
}


////////////////////////////////////////////////////////////////////////////////
// DEFINE THIS AS A CMSSW FWK PLUGIN
////////////////////////////////////////////////////////////////////////////////

DEFINE_FWK_MODULE(SubjetFilterJetProducer);

