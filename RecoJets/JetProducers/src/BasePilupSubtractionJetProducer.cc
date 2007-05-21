// File: BasePilupSubtractionJetProducer.cc
// Author: F.Ratnikov UMd Aug 22, 2006
// $Id: BasePilupSubtractionJetProducer.cc,v 1.7 2007/05/04 23:23:02 fedor Exp $
//--------------------------------------------
#include <memory>
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/View.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/BasicJet.h"
#include "DataFormats/JetReco/interface/GenericJet.h"
#include "RecoJets/JetAlgorithms/interface/JetMaker.h"
#include "RecoJets/JetAlgorithms/interface/JetAlgoHelper.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoCaloTowerCandidate.h"
#include "RecoJets/JetProducers/interface/BasePilupSubtractionJetProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/Provenance/interface/Provenance.h"

using namespace std;
using namespace reco;
using namespace JetReco;

namespace {
  const bool debug = false;
  bool makeCaloJetPU (const string& fTag) {
    return fTag == "CaloJetPileupSubtraction";
  }

  bool makeCaloJet (const string& fTag) {
    return fTag == "CaloJet";
  }
  bool makeGenJet (const string& fTag) {
    return fTag == "GenJet";
  }
  bool makeBasicJet (const string& fTag) {
    return fTag == "BasicJet";
  }
  bool makeGenericJet (const string& fTag) {
    return !makeCaloJet (fTag) && !makeGenJet (fTag) && !makeBasicJet (fTag);
  }

  class FakeCandidate : public RecoCandidate {
  public:
     FakeCandidate( Charge q , const LorentzVector& p4, const Point& vtx) : RecoCandidate( q, p4, vtx ) {}
  private:
    virtual bool overlap( const Candidate & ) const {return false;}
  };

  class FakeHandle {
  public:
    FakeHandle (const CandidateCollection* fCollection, edm::ProductID fId) : mCollection (fCollection), mId (fId) {}
    edm::ProductID id () const {return mId;}
    const CandidateCollection* product () const {return mCollection;}
  private:
    const CandidateCollection* mCollection;
    edm::ProductID mId;
  };

 template <class HandleC>
  void fillInputs (const HandleC& fData, JetReco::InputCollection* fInput) {
    for (unsigned i = 0; i < fData.product ()->size (); i++) {
        fInput->push_back (InputItem (fData, i));
    }
  }

}

namespace cms
{

  

  // Constructor takes input parameters now: to be replaced with parameter set.
  BasePilupSubtractionJetProducer::BasePilupSubtractionJetProducer(const edm::ParameterSet& conf)
    : mSrc (conf.getParameter<edm::InputTag>( "src" )),
      mJetType (conf.getUntrackedParameter<string>( "jetType", "CaloJet")),
      mVerbose (conf.getUntrackedParameter<bool>("verbose", false)),
      mEtInputCut (conf.getParameter<double>("inputEtMin")),
      mEInputCut (conf.getParameter<double>("inputEMin")),
      mEtJetInputCut (conf.getParameter<double>("inputEtJetMin")),
      nSigmaPU (conf.getParameter<double>("nSigmaPU"))
  {
    std::cout<<" Number of sigmas "<<nSigmaPU<<std::endl;
    std::string alias = conf.getUntrackedParameter<string>( "alias", conf.getParameter<std::string>("@module_label"));
    if (makeCaloJetPU (mJetType)) produces<CaloJetCollection>().setBranchAlias (alias);
    if (makeGenJet (mJetType)) produces<GenJetCollection>().setBranchAlias (alias);
    if (makeBasicJet (mJetType)) produces<BasicJetCollection>().setBranchAlias (alias);
    if (makeGenericJet (mJetType)) produces<GenericJetCollection>().setBranchAlias (alias);
  }

  // Virtual destructor needed.
  BasePilupSubtractionJetProducer::~BasePilupSubtractionJetProducer() { }
   
  void BasePilupSubtractionJetProducer::beginJob( const edm::EventSetup& iSetup)
  {
//      edm::ESHandle<CaloGeometry> pG;
//      iSetup.get<IdealGeometryRecord>().get(pG);
//      geo = pG.product();
  }
  
  // Functions that gets called by framework every event
  void BasePilupSubtractionJetProducer::produce(edm::Event& e, const edm::EventSetup& fSetup)
  {
//    std::cout<<"BasePilupSubtractionJetProducer::produce::start"<<std::endl;
    const CaloSubdetectorGeometry* towerGeometry = 0; // cache geometry
    // Provenance
/*
  std::vector<edm::Provenance const*> theProvenance;
  e.getAllProvenance(theProvenance);
  for( std::vector<edm::Provenance const*>::const_iterator ip = theProvenance.begin();
                                                      ip != theProvenance.end(); ip++)
  {
     cout<<" Print all module/label names "<<(**ip).moduleName()<<" "<<(**ip).moduleLabel()<<
     " "<<(**ip).productInstanceName()<<endl;
  }
*/
    // get input
    InputCollection inputs; 
    edm::Handle<CandidateCollection> concreteInputs;
    e.getByLabel( mSrc, concreteInputs );
    fillInputs (concreteInputs, &inputs);

//
// Create the initial vector for Candidates
//    
//    std::cout<<" Before calculate pedestal "<<std::endl;
    calculate_pedestal(inputs); 
    std::vector<ProtoJet> output;
//    std::cout<<" After calculate pedestal "<<std::endl;
    CandidateCollection inputTMPN = subtract_pedestal(inputs);
//    std::cout<<" After pedestal subtraction "<<inputTMPN.size()<<std::endl;

// create input for runAlgorithm
   JetReco::InputCollection fInput;
   FakeHandle handle (&inputTMPN, concreteInputs.id ());
//   std::cout<<" After FakeHandle "<<std::endl;
   fillInputs (handle, &fInput);
//   std::cout<<" After fill inputs "<<std::endl; 
    // run algorithm
    vector <ProtoJet> firstoutput;
    runAlgorithm (fInput, &firstoutput);
    
//
// Now we find jets and need to recalculate their energy
// mark towers participated in jet
// remove occupied towers from the list and recalculate mean and sigma
// put the initial towers collection to the jet   
// 
   
      InputCollection jettowers;
      vector <ProtoJet>::iterator protojetTMP = firstoutput.begin ();

      for (; protojetTMP != firstoutput.end (); protojetTMP++) {

        if( (*protojetTMP).et() < mEtJetInputCut) continue;

        const ProtoJet::Constituents towers = (*protojetTMP).getTowerList();

        ProtoJet::Constituents newtowers; 
           
        for(ProtoJet::Constituents::const_iterator itn = towers.begin(); itn != towers.end(); itn++)
        {
         
         double eta2 = (**itn).eta();
	 double phi2 = (**itn).phi();

           for (InputCollection::const_iterator it = inputs.begin(); it != inputs.end(); it++ ) {
              double eta1 = (**it).eta();
              double phi1 = (**it).phi();

              double dphi = fabs(phi1-phi2);
	      double deta = eta1-eta2;
	       if (dphi > 4.*atan(1.)) dphi = 8.*atan(1.) - dphi;
	      double dr = sqrt(dphi*dphi+deta*deta);
	      if( dr < 0.001) {
	         newtowers.push_back(*it);
                 jettowers.push_back(*it); 
	      }
                
           } // initial input collection
        } // towers in jets
	(*protojetTMP).putTowers(newtowers);  // put the reference of the towers from initial map
	
       } // protojets

//
// Create a new collections from the towersnot included in jets 
//
        InputCollection input;   
        for(InputCollection::const_iterator it = inputs.begin(); it != inputs.end(); it++ ) {
          InputCollection::const_iterator itjet = find(jettowers.begin(),jettowers.end(),*it);
          if( itjet == jettowers.end() ) input.push_back(*it); 
        }
//
// Recalculate pedestal
//
    calculate_pedestal(input);

//    
// Reestimate energy of jet
//
    protojetTMP = firstoutput.begin ();
    
    for (; protojetTMP != firstoutput.end (); protojetTMP++) {
    
      if( (*protojetTMP).et() < mEtInputCut) continue;
      
      const ProtoJet::Constituents towers = (*protojetTMP).getTowerList();
      
      double offset = 0.;
      
      for(ProtoJet::Constituents::const_iterator ito = towers.begin(); ito != towers.end(); ito++)
      {
       int it = ieta(&(**ito));
//       offset = offset + (*emean.find(it)).second + (*esigma.find(it)).second;
// Temporarily for test       
        double etnew = (**ito).et() - (*emean.find(it)).second - (*esigma.find(it)).second; 
        if( etnew <0.) etnew = 0.;
// 
        offset = offset + etnew;
      }
//      double mScale = ((*protojetTMP).et()-offset)/(*protojetTMP).et();
// Temporarily for test only
      double mScale = offset/(*protojetTMP).et();
//////
      Jet::LorentzVector fP4((*protojetTMP).px()*mScale, (*protojetTMP).py()*mScale,
                            (*protojetTMP).pz()*mScale, (*protojetTMP).energy()*mScale);      
      
      
      ProtoJet pj(fP4, towers);
      output.push_back(pj);
   }    
    
    // produce output collection
    auto_ptr<CaloJetCollection> caloJets;
    if (makeCaloJet (mJetType)) caloJets.reset (new CaloJetCollection);
    auto_ptr<GenJetCollection> genJets;
    if (makeGenJet (mJetType)) genJets.reset (new GenJetCollection);
    auto_ptr<BasicJetCollection> basicJets;
    if (makeBasicJet (mJetType)) basicJets.reset (new BasicJetCollection);
    vector <ProtoJet>::const_iterator protojet = output.begin ();
    JetMaker jetMaker;
    for (; protojet != output.end (); protojet++) {
      if (caloJets.get ()) {
	if (!towerGeometry) {
	  edm::ESHandle<CaloGeometry> geometry;
	  fSetup.get<IdealGeometryRecord>().get(geometry);
	  towerGeometry = geometry->getSubdetectorGeometry(DetId::Calo, CaloTowerDetId::SubdetId);
	}
	caloJets->push_back (jetMaker.makeCaloJet (*protojet, *towerGeometry));
      }
      if (genJets.get ()) { 
	genJets->push_back (jetMaker.makeGenJet (*protojet));
      }
      if (basicJets.get ()) { 
	basicJets->push_back (jetMaker.makeBasicJet (*protojet));
      }
    }
    // sort and store output
    if (caloJets.get ()) {
      GreaterByPt<CaloJet> compJets;
      std::sort (caloJets->begin (), caloJets->end (), compJets);
      e.put(caloJets);  //Puts Jet Collection into event
    }
    if (genJets.get ()) {
      GreaterByPt<GenJet> compJets;
      std::sort (genJets->begin (), genJets->end (), compJets);
      e.put(genJets);  //Puts Jet Collection into event
    }
    if (basicJets.get ()) {
      GreaterByPt<BasicJet> compJets;
      std::sort (basicJets->begin (), basicJets->end (), compJets);
      e.put(basicJets);  //Puts Jet Collection into event
    }
  }
  
void BasePilupSubtractionJetProducer::calculate_pedestal(JetReco::InputCollection& inputs)
{
    map<int,double> emean;
    map<int,double> emean2;
    map<int,double> ntowers;
    
    int ietaold = -10000;
    int ieta0 = -100;
    int itower = 0;
    
    for (JetReco::InputCollection::const_iterator input_object = inputs.begin ();  input_object != inputs.end (); input_object++) {
      
//      if (makeCaloJet (mJetType)) ieta = dynamic_cast<const CaloTower*>(&(**input_object))->id().ieta(); 
      ieta0 = ieta(&(**input_object));
       if( ieta0-ietaold != 0 )
      {

        emean[ieta0] = emean[ieta0]+(**input_object).et();
        emean2[ieta0] = emean2[ieta0]+((**input_object).et())*((**input_object).et());
	ntowers[ieta0]++;
      } 
        else
	{
           emean[ieta0] = emean[ieta0]+(**input_object).et();
           emean2[ieta0] = emean2[ieta0]+((**input_object).et())*((**input_object).et());
	   ntowers[ieta0]=1.;
	   itower++;	
	}
    }
    
    for(int it = 0; it< itower; it++)
    {
       emean[it] = emean[it]/ntowers[it];
       esigma[it] = nSigmaPU*sqrt(emean2[it]/ntowers[it] - emean[it]*emean[it]);
    }

}

CandidateCollection BasePilupSubtractionJetProducer::subtract_pedestal(JetReco::InputCollection& inputs)
{
//
// Subtract mean and sigma and prepare collection for jet finder
//    
    CandidateCollection inputCache;
    
    Candidate * mycand;

    JetReco::InputCollection inputTMP;
    int it = -100;
    
    for (JetReco::InputCollection::const_iterator input_object = inputs.begin (); input_object != inputs.end (); input_object++) {
         
       it = ieta(&(**input_object));
       double etnew = (**input_object).et() - (*emean.find(it)).second - (*esigma.find(it)).second;
       float mScale = etnew/(**input_object).et(); 
// Temporarily //////
       if(etnew < 0.) mScale = 0.;
//////
       math::PtEtaPhiELorentzVector p4((**input_object).px()*mScale, (**input_object).py()*mScale,
                                         (**input_object).pz()*mScale, (**input_object).energy()*mScale);
//       std::cout<<" CaloJet "<<makeCaloJetPU (mJetType)<<" "<<mJetType<<std::endl;
       if (makeCaloJetPU (mJetType)) mycand = new RecoCaloTowerCandidate( 0, Candidate::LorentzVector( p4 ) );
       inputCache.push_back (mycand);          
    }
    return inputCache;
}

int BasePilupSubtractionJetProducer::ieta(const reco::Candidate* in)
{
//   std::cout<<" Start BasePilupSubtractionJetProducer::ieta "<<std::endl;
   int it = 0;
   if (makeCaloJetPU (mJetType)) {
//     std::cout<<" PU type "<<std::endl;
     const RecoCaloTowerCandidate* ctc = dynamic_cast<const RecoCaloTowerCandidate*>(in);
     if(ctc)
     {
          it = ctc->caloTower()->id().ieta(); 
     } else
     {
          throw cms::Exception("Invalid Constituent") << "CaloJet constituent is not of RecoCandidate type";
     }
   }  
//   std::cout<<" BasePilupSubtractionJetProducer::ieta "<<it<<std::endl; 
   return it;
}

} // namespace cms
