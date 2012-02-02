// $Id:$

#include "RecoTauTag/HLTProducers/interface/HLTCaloJetPairDzMatchFilter.h"
//#include "HLTrigger/special/interface/HLTCaloJetPairDzMatchFilter.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/Math/interface/deltaPhi.h"


// all these debug printouts will need to be removed at some point
//#define DBG_PRINT(arg) (arg)
#define DBG_PRINT(arg) 


HLTCaloJetPairDzMatchFilter::HLTCaloJetPairDzMatchFilter(const edm::ParameterSet& conf)
{ 
  m_saveTags	= conf.getParameter<bool>("saveTags");
  m_jetSrc	= conf.getParameter<edm::InputTag>("JetSrc");
  m_jetMinPt	= conf.getParameter<double>("JetMinPt");
  m_jetMaxEta	= conf.getParameter<double>("JetMaxEta");
  m_jetMinDR	= conf.getParameter<double>("JetMinDR");
  m_jetMaxDZ	= conf.getParameter<double>("JetMaxDZ");

  // set the minimum DR between jets, so that one never has a chance 
  // to create a pair out of the same CaloJet replicated with two different vertices
  if (m_jetMinDR < 0.1f) m_jetMinDR = 0.1f;
  
  produces<trigger::TriggerFilterObjectWithRefs>();
}

HLTCaloJetPairDzMatchFilter::~HLTCaloJetPairDzMatchFilter(){}

bool HLTCaloJetPairDzMatchFilter::filter(edm::Event& ev, const edm::EventSetup& es)
{
  using namespace std;
  using namespace reco;
  
  DBG_PRINT(cout<<endl);
  
  // The resuilting filter object to store in the Event
  std::auto_ptr<trigger::TriggerFilterObjectWithRefs> filterproduct (new trigger::TriggerFilterObjectWithRefs(path(),module()));
  if (m_saveTags) filterproduct->addCollectionTag(m_jetSrc);

  // Ref to Candidate object to be recorded in the filter object
  CaloJetRef ref;


  // *** Pick up L2 tau jets which have been equipped with some meaningful vertices before that ***

  edm::Handle<CaloJetCollection> jetsHandle;
  ev.getByLabel( m_jetSrc, jetsHandle );
  const CaloJetCollection & jets = *jetsHandle;
  const size_t n_jets = jets.size();

  // *** Combine jets into pairs and check the dz matching ***

  size_t npairs = 0, nfail_dz = 0;
  if (n_jets > 1) for(size_t j1 = 0; j1 < n_jets; ++j1)
  { 
    if ( jets[j1].pt() < m_jetMinPt || std::abs(jets[j1].eta()) > m_jetMaxEta ) continue;
    
    float mindz = 99.f;
    for(size_t j2 = j1+1; j2 < n_jets; ++j2)
    {
      if ( jets[j2].pt() < m_jetMinPt || std::abs(jets[j2].eta()) > m_jetMaxEta ) continue;

      float deta = jets[j1].eta() - jets[j2].eta();
      float dphi = reco::deltaPhi(jets[j1].phi(), jets[j2].phi());
      float dr2 = dphi*dphi + deta*deta;
      float dz = jets[j1].vz() - jets[j2].vz();

      DBG_PRINT(cout<<"@ "<<j1<<" "<<j2<<" : ( "<<jets[j1].eta()<<" "<<jets[j1].phi()<<" "<<jets[j1].vz()<<" ) ( "<<jets[j2].eta()<<" "<<jets[j2].phi()<<" "<<jets[j2].vz()<<" )  "<<sqrt(dr2)<<" "<<!( dr2 < m_jetMinDR*m_jetMinDR )<<"   "<<std::abs(dz)<<" "<<!( std::abs(dz) > m_jetMaxDZ ));

      // skip pairs of jets that are close
      if ( dr2 < m_jetMinDR*m_jetMinDR ) { DBG_PRINT(cout<<endl); continue;}
      
      if (std::abs(dz) < std::abs(mindz)) mindz = dz;

      // do not form a pair if dz is too large
      if ( std::abs(dz) > m_jetMaxDZ ) { DBG_PRINT(cout<<endl); ++nfail_dz; continue;}
      
      // add references to both jets
      ref = CaloJetRef(jetsHandle, j1);
      filterproduct->addObject(trigger::TriggerTau, ref);
      
      ref = CaloJetRef(jetsHandle, j2);
      filterproduct->addObject(trigger::TriggerTau, ref);

      ++npairs;
      
      DBG_PRINT(cout<<"  --> added"<<endl);
    }
    DBG_PRINT(cout<<"mindz  "<<mindz<<endl);
  }
  
  if (npairs==0 && nfail_dz>0) DBG_PRINT(cout<<"no-pass-dz"<<endl);
  DBG_PRINT(cout<<"filter npairs "<<npairs<<endl);

  // put filter object into the Event
  ev.put(filterproduct);
  
  return npairs>0;
}
