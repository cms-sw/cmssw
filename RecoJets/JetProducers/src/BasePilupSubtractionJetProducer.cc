// File: BasePilupSubtractionJetProducer.cc
// Author: F.Ratnikov UMd Aug 22, 2006
// $Id: BasePilupSubtractionJetProducer.cc,v 1.11 2007/07/13 15:28:46 llista Exp $
//--------------------------------------------
#include <memory>
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/BasicJetCollection.h"
#include "DataFormats/JetReco/interface/GenericJetCollection.h"

#include "RecoJets/JetAlgorithms/interface/JetMaker.h"
#include "RecoJets/JetAlgorithms/interface/JetAlgoHelper.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoCaloTowerCandidate.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"

#include "BasePilupSubtractionJetProducer.h"

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
      nSigmaPU (conf.getParameter<double>("nSigmaPU")),
      radiusPU (conf.getParameter<double>("radiusPU")),geo(0)
  {
//    std::cout<<" Number of sigmas "<<nSigmaPU<<std::endl;
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
  }
  
  // Functions that gets called by framework every event
  void BasePilupSubtractionJetProducer::produce(edm::Event& e, const edm::EventSetup& fSetup)
  {
//    std::cout<<"========================BasePilupSubtractionJetProducer::produce::start"<<std::endl;

    const CaloSubdetectorGeometry* towerGeometry = 0; // cache geometry

    // Provenance
/*
  std::vector<edm::Provenance const*> theProvenance;
  e.getAllProvenance(theProvenance);
  for( std::vector<edm::Provenance const*>::const_iterator ip = theProvenance.begin();
                                                      ip != theProvenance.end(); ip++)
  {
     
     std::cout<<" Print all module/label names "<<(**ip).moduleName()<<" "<<(**ip).moduleLabel()<<
     " "<<(**ip).productInstanceName()<<std::endl;
     edm::ParameterSetID pset = (**ip).psetID();
     std::cout<<" Try to print "<<pset<<std::endl; 
  }
*/
 
    if(geo == 0)
    {
      edm::ESHandle<CaloGeometry> pG;
      fSetup.get<IdealGeometryRecord>().get(pG);
      geo = pG.product();
      std::vector<DetId> alldid =  geo->getValidDetIds();
      
      int ietaold = -10000;
      ietamax = -10000;
      ietamin = 10000;   
   for(std::vector<DetId>::const_iterator did=alldid.begin(); did != alldid.end(); did++)
   {
      if( (*did).det() == DetId::Hcal )
      {
                 HcalDetId hid = HcalDetId(*did);
                 if( (hid).depth() == 1 )
                 { 
//                  std::cout<<" Hcal detector eta,phi,depth "<<(hid).ieta()<<" "<<(hid).iphi()<<" "<<(hid).depth()<<std::endl; 
                  allgeomid.push_back(*did);

                  if((hid).ieta() != ietaold)
                  {
                    ietaold = (hid).ieta();
                    geomtowers[(hid).ieta()] = 1;
                    if((hid).ieta() > ietamax) ietamax = (hid).ieta();
                    if((hid).ieta() < ietamin) ietamin = (hid).ieta();
                  }
                    else
                    {
                      geomtowers[(hid).ieta()]++;
                    } 
                 }
      }
  }       
    }
// Clear ntowers_with_jets ====================================================
    for (int i = ietamin; i<ietamax+1; i++)
    {
        ntowers_with_jets[i] = 0;
    }
//==============================================================================
    // get input
    InputCollection inputs; 
    edm::Handle<CandidateCollection> concreteInputs;
    e.getByLabel( mSrc, concreteInputs );
    fillInputs (concreteInputs, &inputs);

//
// Create the initial vector for Candidates
//  
//    std::cout<<"============================================= Before calculate pedestal "<<std::endl;  

    calculate_pedestal(inputs); 
    std::vector<ProtoJet> output;

//    std::cout<<"============================================= After calculate pedestal "<<std::endl;

    CandidateCollection inputTMPN = subtract_pedestal(inputs);

//    std::cout<<"============================================= After pedestal subtraction "<<inputTMPN.size()<<std::endl;

// create input for runAlgorithm
   JetReco::InputCollection fInput;
   FakeHandle handle (&inputTMPN, concreteInputs.id ());

//   std::cout<<" After FakeHandle "<<std::endl;

   fillInputs (handle, &fInput);
//   std::cout<<" After fill inputs "<<std::endl; 

    // run algorithm
   vector <ProtoJet> firstoutput;

//   std::cout<<" We are here at Point 0 "<<std::endl;   

//   runAlgorithm (inputs, &firstoutput);
   runAlgorithm (fInput, &firstoutput);

//   std::cout<<" We are here at Point 1 with firstoutput size (Njets) "<<firstoutput.size()<<std::endl; 
//
// Now we find jets and need to recalculate their energy,
// mark towers participated in jet,
// remove occupied towers from the list and recalculate mean and sigma
// put the initial towers collection to the jet,   
// and subtract from initial towers in jet recalculated mean and sigma of towers 
  
      InputCollection jettowers;
      vector <ProtoJet>::iterator protojetTMP = firstoutput.begin ();

      for (; protojetTMP != firstoutput.end (); protojetTMP++) {

//         std::cout<<" Before mEtJetInputCut, firstoutput.size()="<<firstoutput.size()
//                  <<" (*protojetTMP).et()="<<(*protojetTMP).et()<<std::endl;

        if( (*protojetTMP).et() < mEtJetInputCut) continue;

         ProtoJet::Constituents newtowers;

//        std::cout<<" First passed cut, (*protojetTMP).et()= "<<(*protojetTMP).et()
//                 <<" Eta_jet= "<< (*protojetTMP).eta()<<" Phi_jet="<<(*protojetTMP).phi()<<std::endl;

          double eta2 = (*protojetTMP).eta();
          double phi2 = (*protojetTMP).phi();
          for(vector<HcalDetId>::const_iterator im = allgeomid.begin(); im != allgeomid.end(); im++)
          {
              double eta1 = geo->getPosition((DetId)(*im)).eta();
              double phi1 = geo->getPosition((DetId)(*im)).phi();

              double dphi = fabs(phi1-phi2);
              double deta = eta1-eta2;
               if (dphi > 4.*atan(1.)) dphi = 8.*atan(1.) - dphi;
              double dr = sqrt(dphi*dphi+deta*deta);

//               std::cout<<" dr="<<dr<<std::endl;

              if( dr < radiusPU) {
                ntowers_with_jets[(*im).ieta()]++; 

//               std::cout<<"Towers WITH jets, eta1="<<eta1<<" phi1="<<phi1
//                        <<" ntowers_with_jets="<<ntowers_with_jets[(*im).ieta()]
//                        <<"(DetId)(*im)="<<(*im)<<std::endl;

              }

          }

//          std::cout<<" Number of towers in input collection "<<inputs.size()<<std::endl;

         for (InputCollection::const_iterator it = inputs.begin(); it != inputs.end(); it++ ) {

              double eta1 = (**it).eta();
              double phi1 = (**it).phi();

              double dphi = fabs(phi1-phi2);
	      double deta = eta1-eta2;
	       if (dphi > 4.*atan(1.)) dphi = 8.*atan(1.) - dphi;
	      double dr = sqrt(dphi*dphi+deta*deta);

	      if( dr < radiusPU) {
                newtowers.push_back(*it);
                jettowers.push_back(*it); 
                int ieta1 = ieta(&(**it));
                int iphi1 = iphi(&(**it));
          
//       std::cout<<" Take Et of tower inputs, (dr < 0.5), (**it).et()= "<<(**it).et()<<" eta= "<<(**it).eta()
//                <<" phi= "<<(**it).phi()<<" ieta1= "<<ieta1<<" iphi1= "<<iphi1<<std::endl;

	      } //dr < 0.5
                
           } // initial input collection

//         std::cout<<" Jet with new towers before putTowers (after background subtraction) "<<(*protojetTMP).et()<<std::endl;

	(*protojetTMP).putTowers(newtowers);  // put the reference of the towers from initial map

//         std::cout<<" Jet with new towers (Initial tower energy)"<<(*protojetTMP).et()<<std::endl;	


//========> PRINT  Tower after Subtraction
        for (InputCollection::const_iterator itt = fInput.begin(); itt !=  fInput.end(); itt++ ) {
        
              double eta_pu1 = (**itt).eta();
              double phi_pu1 = (**itt).phi();

              double dphi = fabs(phi_pu1-phi2);
              double deta = eta_pu1-eta2;
               if (dphi > 4.*atan(1.)) dphi = 8.*atan(1.) - dphi;
              double dr = sqrt(dphi*dphi+deta*deta);

              if( dr < radiusPU) {        
                int ieta_pu1 = ieta(&(**itt));
                int iphi_pu1 = iphi(&(**itt));

//       std::cout<<" Take Et of tower after Subtraction, (**itt).et()= "<<(**itt).et()
//                <<" eta= "<<(**itt).eta()<<" phi= "<<(**itt).phi()
//                <<" ieta_pu1= "<<ieta_pu1<<" iphi_pu1= "<<iphi_pu1<<std::endl;

              } //dr < 0.5

           } //  input collection after Subtraction


//=====================>


       } // protojets

//       std::cout<<" We are at Point 2 with "<<firstoutput.size()<<std::endl;
//
// Create a new collections from the towers not included in jets 
//
        InputCollection input;   
        for(InputCollection::const_iterator it = inputs.begin(); it != inputs.end(); it++ ) {
          InputCollection::const_iterator itjet = find(jettowers.begin(),jettowers.end(),*it);
          if( itjet == jettowers.end() ) input.push_back(*it); 
        }
//       std::cout<<" We are at Point 3, Number of tower not included in jets= "<<input.size()<<std::endl;

/*
//======================> PRINT NEW InputCollection without jets

       for (InputCollection::const_iterator it = input.begin(); it != input.end(); it++ ) {

              int ieta1 = ieta(&(**it));
              int iphi1 = iphi(&(**it));

       std::cout<<" Take Et of tower WITHOUT jet, (**it).et()= "<<(**it).et()
                <<" eta= "<<(**it).eta()<<" phi= "<<(**it).phi()
                <<" ieta1= "<<ieta1<<" iphi1="<<iphi1<<std::endl;
       }
//======================>
*/

//
// Recalculate pedestal
//
    calculate_pedestal(input);

//    std::cout<<" We are at Point 4, After Recalculation of pedestal"<<std::endl;
//    
// Reestimate energy of jet (energy of jet with initial map)
//
    protojetTMP = firstoutput.begin ();
    int kk = 0; 
    for (; protojetTMP != firstoutput.end (); protojetTMP++) {

//      std::cout<<" ++++++++++++++Jet with initial map energy "<<kk<<" "<<(*protojetTMP).et()
//               <<" mEtJetInputCut="<<mEtJetInputCut<<std::endl;
 
      if( (*protojetTMP).et() < mEtJetInputCut) continue;

//      std::cout<<" Jet with energyi passed condition "<<kk<<" "<<(*protojetTMP).et()<<std::endl;
      
      const ProtoJet::Constituents towers = (*protojetTMP).getTowerList();

//      std::cout<<" List of candidates "<<towers.size()<<std::endl;
      
      double offset = 0.;
      
      for(ProtoJet::Constituents::const_iterator ito = towers.begin(); ito != towers.end(); ito++)
      {
//       std::cout<<" start towers list "<<std::endl;

       int it = ieta(&(**ito));

//       std::cout<<" Reference to tower : "<<it<<std::endl;
//       offset = offset + (*emean.find(it)).second + (*esigma.find(it)).second;
// Temporarily for test       

        double etnew = (**ito).et() - (*emean.find(it)).second - (*esigma.find(it)).second; 
        if( etnew <0.) etnew = 0.;
 
        offset = offset + etnew;
      }
//      double mScale = ((*protojetTMP).et()-offset)/(*protojetTMP).et();
// Temporarily for test only

      double mScale = offset/(*protojetTMP).et();
      
//////
      Jet::LorentzVector fP4((*protojetTMP).px()*mScale, (*protojetTMP).py()*mScale,
                            (*protojetTMP).pz()*mScale, (*protojetTMP).energy()*mScale);      
      
//      std::cout<<" Final energy of jet, fP4.pt()= "<<fP4.pt()<<" Eta "<<fP4.eta()<<" Phi "<<fP4.phi()<<std::endl;      
///
///!!! Change towers to rescaled towers///
///
      ProtoJet pj(fP4, towers);
      kk++;
      output.push_back(pj);
   }    

//   std::cout<<" Size of final collection "<<output.size()<<std::endl;
    
    // produce output collection
    auto_ptr<CaloJetCollection> caloJets;
    if (makeCaloJetPU (mJetType)) caloJets.reset (new CaloJetCollection);
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
//   std::cout<<"========== Start BasePilupSubtractionJetProducer::calculate_pedestal"<<std::endl;
//   std::cout<<" ietamax="<<ietamax<<" ietamin="<<ietamin<<std::endl;

    map<int,double> emean2;
    map<int,int> ntowers;
    
    int ietaold = -10000;
    int ieta0 = -100;
   
// Initial values for emean, emean2, esigma, ntowers

    for(int i = ietamin; i < ietamax+1; i++)
    {
       emean[i] = 0.;
       emean2[i] = 0.;
       esigma[i] = 0.;
       ntowers[i] = 0;
    }

//=>
    int iphi0 = -100;
    
    for (JetReco::InputCollection::const_iterator input_object = inputs.begin ();  input_object != inputs.end (); input_object++) {
       
       ieta0 = ieta(&(**input_object));

//       std::cout<<"+++calculate pedestal, Et_tower="<<(**input_object).et()
//                <<" ieta0 ="<<ieta(&(**input_object))
//                <<" iphi0 ="<<iphi(&(**input_object))<<"ietaold="<<ietaold<<std::endl;

//=> 
       if( ieta0-ietaold != 0 )
      {

        emean[ieta0] = emean[ieta0]+(**input_object).et();
        emean2[ieta0] = emean2[ieta0]+((**input_object).et())*((**input_object).et());
        ntowers[ieta0] = 1;

        ietaold = ieta0;

///        std::cout<<"--NEW ETA, emean[ieta0]="<<emean[ieta0]
///                 <<" ntowers[ieta0]="<<ntowers[ieta0]<<" ieta0="<<ieta0<<std::endl;
      }
        else
        {
           emean[ieta0] = emean[ieta0]+(**input_object).et();
           emean2[ieta0] = emean2[ieta0]+((**input_object).et())*((**input_object).et());
           ntowers[ieta0]++;

///           std::cout<<"--OLD ETA, emean[ieta0]="<<emean[ieta0]
///                 <<" ntowers[ieta0]="<<ntowers[ieta0]<<" ieta0="<<ieta0<<std::endl;
        }

//=>


    }

//======================>
//    std::cout<<" Geom towers begin "<<geomtowers.size()<<std::endl;

    for(map<int,int>::const_iterator gt = geomtowers.begin(); gt != geomtowers.end(); gt++)    
    {

       int it = (*gt).first;
       
       double e1 = (*emean.find(it)).second;
       double e2 = (*emean2.find(it)).second;
       int nt = (*gt).second - (*ntowers_with_jets.find(it)).second;
        
       if(nt == 0) {
          emean[it] = 0.;
          esigma[it] = 0.;
       }
          else
         {
            emean[it] = e1/nt;
            esigma[it] = nSigmaPU*sqrt(e2/nt - e1*e1/(nt*nt));
         }

//          std::cout<<"---calculate_pedestal, emean[it]= "
//                   <<(*emean.find(it)).second<<"esigma[it]="<<(*esigma.find(it)).second
//                   <<" ntowers_without_jets{it]="<<nt<<" it="<<it
//                   <<" ntowers_map(it)="<<(*ntowers.find(it)).second
//                   <<" geomtowers(it)="<<(*geomtowers.find(it)).second<<" gt.second "<<(*gt).second<<
//                   " ntow_with_jets "<<(*ntowers_with_jets.find(it)).second<<std::endl;

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

//       std::cout<<" Subtraction from tower with eta "<<it<<" phi "<<iphi(&(**input_object))<<" OLD energy "<<
//                                                 (**input_object).et()<<" NEW energy "<<etnew<<" mScale "<<mScale<<
//                                                 " Mean energy "<<(*emean.find(it)).second<<" Sigma "<<(*esigma.find(it)).second<<std::endl;


//////
       math::XYZTLorentzVectorD p4((**input_object).px()*mScale, (**input_object).py()*mScale,
                                         (**input_object).pz()*mScale, (**input_object).energy()*mScale);

//       std::cout<<"NEW energy from p4 "<<p4.pt()<<std::endl;
//       std::cout<<" CaloJet "<<makeCaloJetPU (mJetType)<<" "<<mJetType<<std::endl;

       if (makeCaloJetPU (mJetType)) {
       mycand = new RecoCaloTowerCandidate( 0, Candidate::LorentzVector( p4 ) );
       const RecoCaloTowerCandidate* ct = dynamic_cast<const RecoCaloTowerCandidate*>(&(**input_object));
       if(ct)
       {
          dynamic_cast<RecoCaloTowerCandidate*>(mycand)->setCaloTower(ct->caloTower());
       }
        else
       {
            throw cms::Exception("Invalid Constituent") << "CaloJet constituent is not of RecoCandidate type";
       }      
       }
       inputCache.push_back (mycand);          
    }

//    std::cout<<" OLD size "<<inputs.size()<<" NEW size "<<inputCache.size()<<std::endl;

/*
//===> Print NEW tower energy after background Subtraction
        for (CandidateCollection::const_iterator itt = inputCache.begin(); itt != inputCache.end(); itt++ ) {
              double et_pu = (*itt).et();
              double eta_pu = (*itt).eta();
              double phi_pu = (*itt).phi();

              int ieta_pu = ieta(&(*itt));
              int iphi_pu = iphi(&(*itt));    

//         std::cout<<"---inputCache, Subtraction from tower with eta= "<<ieta_pu
//                 <<" phi="<<iphi_pu<<" NEW et="<<et_pu<<std::endl;

          }
//===>
*/
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

int BasePilupSubtractionJetProducer::iphi(const reco::Candidate* in)
{
//   std::cout<<" Start BasePilupSubtractionJetProducer::ieta "<<std::endl;
   int it = 0;
   if (makeCaloJetPU (mJetType)) {
//     std::cout<<" PU type "<<std::endl;
     const RecoCaloTowerCandidate* ctc = dynamic_cast<const RecoCaloTowerCandidate*>(in);
     if(ctc)
     {
          it = ctc->caloTower()->id().iphi();
     } else
     {
          throw cms::Exception("Invalid Constituent") << "CaloJet constituent is not of RecoCandidate type";
     }
   }
//   std::cout<<" BasePilupSubtractionJetProducer::ieta "<<it<<std::endl;
   return it;
}


} // namespace cms
