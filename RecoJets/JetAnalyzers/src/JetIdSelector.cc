/** \class CaloJetIdSelector
 *
 * Select a subset of the CaloJet collection based on the quality stored in a ValueMap
 *
 * \author: Kalanand Mishra, Fermilab
 *
 * usage:
 *
 * from RecoJets.JetProducers.ak5JetID_cfi import *
 * module CaloJetsLooseId = cms.EDProducer("CaloJetIdSelector",
 *    src     = cms.InputTag( "ak5CaloJets" ),
 *    idLevel = cms.string("LOOSE"),
 *    jetIDMap = cms.untracked.InputTag("ak5JetID") 
 *               ### must provide jet ID value map for CaloJets
 * )
 *
 *
 * module PFJetsLooseId = cms.EDProducer("PFJetIdSelector",
 *    src     = cms.InputTag( "ak5PFJets" ),
 *    idLevel = cms.string("LOOSE"),
 * )
 *
 *
 *
 */

#include "PhysicsTools/SelectorUtils/interface/JetIDSelectionFunctor.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
 
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
 
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/JetReco/interface/JetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"  
#include "DataFormats/JetReco/interface/CaloJetCollection.h"  
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/JPTJet.h"
#include "DataFormats/JetReco/interface/JPTJetCollection.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"

#include <memory>
#include <vector>
#include <sstream>

////////////////////////////////////////////////////////////////////////////////
// class definition
////////////////////////////////////////////////////////////////////////////////
template<typename T>
class JetIdSelector : public edm::EDProducer
{
public:
  typedef std::vector<T> JetCollection;
  // construction/destruction
  explicit JetIdSelector(const edm::ParameterSet& iConfig);
  virtual ~JetIdSelector();
  
  // member functions
  void produce(edm::Event& iEvent,const edm::EventSetup& iSetup);
  void endJob();

private:  
  // member data
  edm::InputTag src_;
  std::string  qualityStr;
  edm::InputTag jetIDMap_;
  std::string  moduleLabel_;


  unsigned int nJetsTot_;
  unsigned int nJetsPassed_;
  JetIDSelectionFunctor* jetIDFunctor;
  bool use_pfloose;
  bool use_pfmedium;
  bool use_pftight;
};


using namespace std;


////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////
//______________________________________________________________________________
template<typename T>
JetIdSelector<T>::JetIdSelector(const edm::ParameterSet& iConfig)
  : src_    (iConfig.getParameter<edm::InputTag>         ("src"))
  , qualityStr  (iConfig.getParameter<string>            ("idLevel"))
  , jetIDMap_(iConfig.getUntrackedParameter<edm::InputTag> ("jetIDMap", edm::InputTag("ak5JetID")))
  , moduleLabel_(iConfig.getParameter<string>                ("@module_label"))
  , nJetsTot_(0)
  , nJetsPassed_(0)
{
  produces<JetCollection>();

  use_pfloose = false;
  use_pfmedium = false;
  use_pftight = false;

  if ( qualityStr == "MINIMAL" ) {
    jetIDFunctor 
      = new JetIDSelectionFunctor( JetIDSelectionFunctor::PURE09, JetIDSelectionFunctor::MINIMAL);
    use_pfloose = true;
  }
  else if ( qualityStr == "LOOSE_AOD" ) {
    jetIDFunctor 
      = new JetIDSelectionFunctor( JetIDSelectionFunctor::PURE09, JetIDSelectionFunctor::LOOSE_AOD);
    use_pfloose = true;
  }
  else if ( qualityStr == "LOOSE" ) {
    jetIDFunctor 
      = new JetIDSelectionFunctor( JetIDSelectionFunctor::PURE09, JetIDSelectionFunctor::LOOSE);   
    use_pfloose = true;
  } 
  else if ( qualityStr == "MEDIUM" ) {
    jetIDFunctor 
      = new JetIDSelectionFunctor( JetIDSelectionFunctor::PURE09, JetIDSelectionFunctor::LOOSE);  
    // There is no medium quality for CaloJet !!
    use_pfmedium = true;
  }
  else if ( qualityStr == "TIGHT" ) {
    jetIDFunctor 
      = new JetIDSelectionFunctor( JetIDSelectionFunctor::PURE09, JetIDSelectionFunctor::TIGHT);    
    use_pftight = true;
  }
  else   
    throw cms::Exception("InvalidInput") 
      << "Expect quality to be one of MINIMAL, LOOSE_AOD, LOOSE, MEDIUM, TIGHT" << std::endl;
}


//______________________________________________________________________________
template<typename T>
JetIdSelector<T>::~JetIdSelector(){
  if(jetIDFunctor) delete jetIDFunctor;
}



////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
template<typename T>
void JetIdSelector<T>::produce(edm::Event& iEvent,const edm::EventSetup& iSetup)
{
  auto_ptr<JetCollection> selectedJets(new JetCollection);
  edm::Handle<reco::JetView> jets;  // uncorrected jets!
  iEvent.getByLabel(src_,jets);


 // handle to the jet ID variables
   edm::Handle<reco::JetIDValueMap> hJetIDMap;

   if(typeid((*jets)[0]) == typeid(reco::CaloJet)) 
     iEvent.getByLabel( jetIDMap_, hJetIDMap );

   unsigned int idx=0;
   bool passed = false;

   for ( edm::View<reco::Jet>::const_iterator ibegin = jets->begin(),
           iend = jets->end(), iJet = ibegin;
         iJet != iend; ++iJet ) {

     // initialize the boolean flag to false
     passed = false;

     //calculate the Calo jetID
     const std::type_info & type = typeid((*jets)[idx]);
     if( type == typeid(reco::CaloJet) ) {
       const reco::CaloJet calojet = static_cast<const reco::CaloJet &>((*jets)[idx]);
       edm::RefToBase<reco::Jet> jetRef = jets->refAt(idx);
       reco::JetID const & jetId = (*hJetIDMap)[ jetRef ];
       passed = (*jetIDFunctor)( calojet, jetId);
     }

     //calculate the PF jetID
     if ( type == typeid(reco::PFJet) ) {
       const reco::PFJet pfjet = static_cast<const reco::PFJet &>((*jets)[idx]);
       bool passingLoose=false;
       bool passingMedium=false;
       bool passingTight=false;
       bool ThisIsClean=true;
       //apply following only if |eta|<2.4: CHF>0, CEMF<0.99, chargedMultiplicity>0   
       if(( pfjet.chargedHadronEnergy()/ pfjet.energy())<= 0.0  
	  && fabs(pfjet.eta())<2.4) ThisIsClean=false; 
       if( (pfjet.chargedEmEnergy()/pfjet.energy())>= 0.99 
	   && fabs(pfjet.eta())<2.4 ) ThisIsClean=false;
       if( pfjet.chargedMultiplicity()<=0 && fabs(pfjet.eta())<2.4 ) 
	 ThisIsClean=false;
       
       // always require #Constituents > 1
       if( pfjet.nConstituents() <=1 ) ThisIsClean=false;
       
       if(ThisIsClean && 
	  (pfjet.neutralHadronEnergy()/pfjet.energy())< 0.99 
	  && (pfjet.neutralEmEnergy()/pfjet.energy())<0.99) 
	 passingLoose=true;
       
       if(ThisIsClean && 
	  (pfjet.neutralHadronEnergy()/pfjet.energy())< 0.95 
	  && (pfjet.neutralEmEnergy()/pfjet.energy())<0.95) 
	 passingMedium=true;
       
       if(ThisIsClean && 
	  (pfjet.neutralHadronEnergy()/pfjet.energy())< 0.90 
	  && (pfjet.neutralEmEnergy()/pfjet.energy())<0.90) 
	 passingTight=true;


       if ( use_pfloose && passingLoose)  passed = true;
       if ( use_pfmedium && passingMedium) passed = true;
       if ( use_pftight && passingTight) passed = true;
     }

     if ( type == typeid(reco::GenJet) || type == typeid(reco::JPTJet)) {
       edm::LogWarning( "JetId" )<< 
	 "Criteria for jets other than CaloJets and PFJets are not yet implemented";
       passed = true;
     } // close GenJet, JPT jet

 
     const T& goodJet = static_cast<const T&>((*jets)[idx]);
     if(passed) selectedJets->push_back( goodJet );

     idx++;
   } // close jet iterator



  nJetsTot_  +=jets->size();
  nJetsPassed_+=selectedJets->size();  
  iEvent.put(selectedJets);
}



//______________________________________________________________________________
template<typename T>
void JetIdSelector<T>::endJob()
{
  stringstream ss;
  ss<<"nJetsTot="<<nJetsTot_<<" nJetsPassed="<<nJetsPassed_
    <<" fJetsPassed="<<100.*(nJetsPassed_/(double)nJetsTot_)<<"%\n";
  cout<<"++++++++++++++++++++++++++++++++++++++++++++++++++"
      <<"\n"<< moduleLabel_ << "(JetIdSelector) SUMMARY:\n"<<ss.str()
      <<"++++++++++++++++++++++++++++++++++++++++++++++++++"
      <<endl;
}


////////////////////////////////////////////////////////////////////////////////
// plugin definition
////////////////////////////////////////////////////////////////////////////////
typedef JetIdSelector<reco::CaloJet> CaloJetIdSelector;
typedef JetIdSelector<reco::PFJet>   PFJetIdSelector;
typedef JetIdSelector<reco::JPTJet>  JPTJetIdSelector;
typedef JetIdSelector<reco::GenJet>  GenJetIdSelector;

DEFINE_FWK_MODULE(CaloJetIdSelector);
DEFINE_FWK_MODULE(PFJetIdSelector);
DEFINE_FWK_MODULE(JPTJetIdSelector);
DEFINE_FWK_MODULE(GenJetIdSelector);


