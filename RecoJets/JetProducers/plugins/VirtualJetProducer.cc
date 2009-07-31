////////////////////////////////////////////////////////////////////////////////
//
// VirtualJetProducer
// ------------------
//
//            04/21/2009 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
////////////////////////////////////////////////////////////////////////////////

#include "RecoJets/JetProducers/plugins/VirtualJetProducer.h"

#include "RecoJets/JetProducers/interface/JetSpecific.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/CodedException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/BasicJetCollection.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "RecoJets/JetProducers/interface/JetSpecific.h"

#include "fastjet/SISConePlugin.hh"
#include "fastjet/CMSIterativeConePlugin.hh"
#include "fastjet/ATLASConePlugin.hh"
#include "fastjet/CDFMidPointPlugin.hh"


#include <iostream>
#include <memory>
#include <algorithm>
#include <limits>
#include <cmath>

using namespace std;


namespace reco {
  namespace helper {
    struct GreaterByPtPseudoJet {
      bool operator()( const fastjet::PseudoJet & t1, const fastjet::PseudoJet & t2 ) const {
	return t1.perp() > t2.perp();
      }
    };

  }
}

//______________________________________________________________________________
const char *VirtualJetProducer::JetType::names[] = {
  "BasicJet","GenJet","CaloJet","PFJet"
};


//______________________________________________________________________________
VirtualJetProducer::JetType::Type
VirtualJetProducer::JetType::byName(const string &name)
{
  const char **pos = std::find(names, names + LastJetType, name);
  if (pos == names + LastJetType) {
    std::string errorMessage="Requested jetType not supported: "+name+"\n"; 
    throw cms::Exception("Configuration",errorMessage);
  }
  return (Type)(pos-names);
}


void VirtualJetProducer::makeProduces( std::string alias, std::string tag ) 
{
  if (makeCaloJet(jetTypeE)) {
    produces<reco::CaloJetCollection>(tag).setBranchAlias(alias);
  }
  else if (makePFJet(jetTypeE)) {
    produces<reco::PFJetCollection>(tag).setBranchAlias(alias);
  }
  else if (makeGenJet(jetTypeE)) {
    produces<reco::GenJetCollection>(tag).setBranchAlias(alias);
  }
  else if (makeBasicJet(jetTypeE)) {
    produces<reco::BasicJetCollection>(tag).setBranchAlias(alias);
  }
}

////////////////////////////////////////////////////////////////////////////////
// construction / destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
VirtualJetProducer::VirtualJetProducer(const edm::ParameterSet& iConfig)
  : moduleLabel_   (iConfig.getParameter<string>       ("@module_label"))
  , src_           (iConfig.getParameter<edm::InputTag>("src"))
  , srcPVs_        (iConfig.getParameter<edm::InputTag>("srcPVs"))
  , jetType_       (iConfig.getParameter<string>       ("jetType"))
  , jetAlgorithm_  (iConfig.getParameter<string>       ("jetAlgorithm"))
  , rParam_        (iConfig.getParameter<double>       ("rParam"))
  , inputEtMin_    (iConfig.getParameter<double>       ("inputEtMin"))
  , inputEMin_     (iConfig.getParameter<double>       ("inputEMin"))
  , jetPtMin_      (iConfig.getParameter<double>       ("jetPtMin"))
  , doPVCorrection_(iConfig.getParameter<bool>         ("doPVCorrection"))
  , restrictInputs_(false)
  , maxInputs_(99999999)
  , doPUFastjet_   (iConfig.getParameter<bool>         ("doPUFastjet"))
  , doPUOffsetCorr_(iConfig.getParameter<bool>         ("doPUOffsetCorr"))
  , maxBadEcalCells_        (iConfig.getParameter<unsigned>("maxBadEcalCells"))
  , maxRecoveredEcalCells_  (iConfig.getParameter<unsigned>("maxRecoveredEcalCells"))
  , maxProblematicEcalCells_(iConfig.getParameter<unsigned>("maxProblematicEcalCells"))
  , maxBadHcalCells_        (iConfig.getParameter<unsigned>("maxBadHcalCells"))
  , maxRecoveredHcalCells_  (iConfig.getParameter<unsigned>("maxRecoveredHcalCells"))
  , maxProblematicHcalCells_(iConfig.getParameter<unsigned>("maxProblematicHcalCells"))
  , jetCollInstanceName_ ("")
{

  //
  // additional parameters to think about:
  // - overlap threshold (set to 0.75 for the time being)
  // - p parameter for generalized kT (set to -2 for the time being)
  // - fastjet PU subtraction parameters (not yet considered)
  //
  if (jetAlgorithm_=="SISCone") {
    fjPlugin_ = PluginPtr( new fastjet::SISConePlugin(rParam_,0.75,0,0.0,false,
						      fastjet::SISConePlugin::SM_pttilde) );
    fjJetDefinition_= JetDefPtr( new fastjet::JetDefinition(&*fjPlugin_) );
  }
  else if (jetAlgorithm_=="IterativeCone") {
    fjPlugin_ = PluginPtr(new fastjet::CMSIterativeConePlugin(rParam_,1.0));
    fjJetDefinition_= JetDefPtr(new fastjet::JetDefinition(&*fjPlugin_));
  }
  else if (jetAlgorithm_=="CDFMidPoint") {
    fjPlugin_ = PluginPtr(new fastjet::CDFMidPointPlugin(rParam_,0.75));
    fjJetDefinition_= JetDefPtr(new fastjet::JetDefinition(&*fjPlugin_));
  }
  else if (jetAlgorithm_=="ATLASCone") {
    fjPlugin_ = PluginPtr(new fastjet::ATLASConePlugin(rParam_));
    fjJetDefinition_= JetDefPtr(new fastjet::JetDefinition(&*fjPlugin_));
  }
  else if (jetAlgorithm_=="Kt")
    fjJetDefinition_= JetDefPtr(new fastjet::JetDefinition(fastjet::kt_algorithm,rParam_));
  else if (jetAlgorithm_=="CambridgeAachen")
    fjJetDefinition_= JetDefPtr(new fastjet::JetDefinition(fastjet::cambridge_algorithm,
							   rParam_) );
  else if (jetAlgorithm_=="AntiKt")
    fjJetDefinition_= JetDefPtr( new fastjet::JetDefinition(fastjet::antikt_algorithm,rParam_) );
  else if (jetAlgorithm_=="GeneralizedKt")
    fjJetDefinition_= JetDefPtr( new fastjet::JetDefinition(fastjet::genkt_algorithm,
							    rParam_,-2) );
  else
    throw cms::Exception("Invalid jetAlgorithm")
      <<"Jet algorithm for VirtualJetProducer is invalid, Abort!\n";
  
  jetTypeE=JetType::byName(jetType_);

  if ( iConfig.exists("jetCollInstanceName") ) {
    jetCollInstanceName_ = iConfig.getParameter<string>("jetCollInstanceName");
  }

  // do UE subtraction? 
  if ( doPUFastjet_ ) {           // accept pilup subtraction parameters
    double ghostEtaMax = iConfig.getParameter<double> ("Ghost_EtaMax");          //default Ghost_EtaMax should be 6
    int activeAreaRepeats = iConfig.getParameter<int> ("Active_Area_Repeats");   //default Active_Area_Repeats 5
    double ghostArea = iConfig.getParameter<double> ("GhostArea");               //default GhostArea 0.01
    fjActiveArea_ =  ActiveAreaSpecPtr( new fastjet::ActiveAreaSpec (ghostEtaMax, activeAreaRepeats, ghostArea) );
    fjRangeDef_ = RangeDefPtr( new fastjet::RangeDefinition(ghostEtaMax) );
  } 

  if ( doPUOffsetCorr_ ) {
    nSigmaPU_ = iConfig.getParameter<double>("nSigmaPU");
    radiusPU_ = iConfig.getParameter<double>("radiusPU");
    if ( jetTypeE != JetType::CaloJet ) {
      throw cms::Exception("InvalidInput") << "Can only offset correct jets of type CaloJet";
    }
  }

  // restrict inputs to first "maxInputs" towers?
  if ( iConfig.exists("restrictInputs") ) {
    restrictInputs_ = iConfig.getParameter<bool>("restrictInputs");
    maxInputs_      = iConfig.getParameter<unsigned int>("maxInputs");
  }
 

  string alias=iConfig.getUntrackedParameter<string>("alias",moduleLabel_);

  // make the "produces" statements
  makeProduces( alias, jetCollInstanceName_ );
}


//______________________________________________________________________________
VirtualJetProducer::~VirtualJetProducer()
{
} 


////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
void VirtualJetProducer::produce(edm::Event& iEvent,const edm::EventSetup& iSetup)
{
  LogDebug("VirtualJetProducer") << "Entered produce\n";
  //determine signal vertex
  vertex_=reco::Jet::Point(0,0,0);
  if (makeCaloJet(jetTypeE)&&doPVCorrection_) {
    LogDebug("VirtualJetProducer") << "Adding PV info\n";
    edm::Handle<reco::VertexCollection> pvCollection;
    iEvent.getByLabel(srcPVs_,pvCollection);
    if (pvCollection->size()>0) vertex_=pvCollection->begin()->position();
  }

  // For Pileup subtraction using offset correction:
  // set up geometry map
  if ( doPUOffsetCorr_ ) {
    setupGeometryMap(iEvent, iSetup);
  }

  // clear data
  LogDebug("VirtualJetProducer") << "Clear data\n";
  fjInputs_.clear();
  fjJets_.clear();
  
  
  // get inputs and convert them to the fastjet format (fastjet::PeudoJet)
  edm::Handle<reco::CandidateView> inputsHandle;
  iEvent.getByLabel(src_,inputsHandle);
  inputs_ = *inputsHandle;
  LogDebug("VirtualJetProducer") << "Got inputs\n";
  
  // Convert candidates to fastjet::PseudoJets.
  // Also correct to Primary Vertex. Will modify fjInputs_
  // and use inputs_
  fjInputs_.reserve(inputs_.size());
  inputTowers();
  LogDebug("VirtualJetProducer") << "Inputted towers\n";

  // For Pileup subtraction using offset correction:
  // Subtract pedestal. 
  if ( doPUOffsetCorr_ ) {
    calculatePedestal(fjInputs_); 
    subtractPedestal(fjInputs_);    
    LogDebug("VirtualJetProducer") << "Subtracted pedestal\n";
  }

  // Run algorithm. Will modify fjJets_ and allocate fjClusterSeq_. 
  // This will use fjInputs_
  runAlgorithm( iEvent, iSetup );
  LogDebug("VirtualJetProducer") << "Ran algorithm\n";

  // For Pileup subtraction using offset correction:
  // Now we find jets and need to recalculate their energy,
  // mark towers participated in jet,
  // remove occupied towers from the list and recalculate mean and sigma
  // put the initial towers collection to the jet,   
  // and subtract from initial towers in jet recalculated mean and sigma of towers 
  if ( doPUOffsetCorr_ ) {
    vector<fastjet::PseudoJet> orphanInput;
    calculateOrphanInput(orphanInput);
    calculatePedestal(orphanInput);
    offsetCorrectJets(orphanInput);
  }

  
  // Write the output jets.
  // This will (by default) call the member function template
  // "writeJets", but can be overridden. 
  // this will use inputs_
  output( iEvent, iSetup );
  LogDebug("VirtualJetProducer") << "Wrote jets\n";
  
  return;
}




//______________________________________________________________________________
  
void VirtualJetProducer::setupGeometryMap(edm::Event& iEvent,const edm::EventSetup& iSetup)    
{
  if(geo_ == 0) {
    edm::ESHandle<CaloGeometry> pG;
    iSetup.get<CaloGeometryRecord>().get(pG);
    geo_ = pG.product();
    std::vector<DetId> alldid =  geo_->getValidDetIds();
    
    int ietaold = -10000;
    ietamax_ = -10000;
    ietamin_ = 10000;   
    for(std::vector<DetId>::const_iterator did=alldid.begin(); did != alldid.end(); did++){
      if( (*did).det() == DetId::Hcal ){
	HcalDetId hid = HcalDetId(*did);
	if( (hid).depth() == 1 ) { 
	  allgeomid_.push_back(*did);
	  
	  if((hid).ieta() != ietaold){
	    ietaold = (hid).ieta();
	    geomtowers_[(hid).ieta()] = 1;
	    if((hid).ieta() > ietamax_) ietamax_ = (hid).ieta();
	    if((hid).ieta() < ietamin_) ietamin_ = (hid).ieta();
	  }
	  else{
	    geomtowers_[(hid).ieta()]++;
	  } 
	}
      }
    }       
  }

  for (int i = ietamin_; i<ietamax_+1; i++) {
    ntowersWithJets_[i] = 0;
  }
}

//______________________________________________________________________________
  
void VirtualJetProducer::inputTowers( )
{
  reco::CandidateView::const_iterator inBegin = inputs_.begin(),
    inEnd = inputs_.end(), i = inBegin;
  for (; i != inEnd; ++i ) {
    reco::CandidatePtr input = inputs_.ptrAt( i - inBegin );
    if (isnan(input->pt()))           continue;
    if (input->et()    <inputEtMin_)  continue;
    if (input->energy()<inputEMin_)   continue;
    if (isAnomalousTower(input))      continue;
    if (makeCaloJet(jetTypeE)&&doPVCorrection_) {
      const CaloTower* tower=dynamic_cast<const CaloTower*>(input.get());
      math::PtEtaPhiMLorentzVector ct(tower->p4(vertex_));
      fjInputs_.push_back(fastjet::PseudoJet(ct.px(),ct.py(),ct.pz(),ct.energy()));
    }
    else {
      fjInputs_.push_back(fastjet::PseudoJet(input->px(),input->py(),input->pz(),
					    input->energy()));
    }
    fjInputs_.back().set_user_index(i - inBegin);
  }

  if ( restrictInputs_ && inputs_.size() > maxInputs_ ) {
    reco::helper::GreaterByPtPseudoJet   pTComparator;
    std::sort(fjInputs_.begin(), fjInputs_.end(), pTComparator);
    fjInputs_.resize(maxInputs_);
    edm::LogWarning("JetRecoTooManyEntries") << "Too many inputs in the event, limiting to first " << maxInputs_ << ". Output is suspect.";
  }
}

//______________________________________________________________________________
bool VirtualJetProducer::isAnomalousTower(reco::CandidatePtr input)
{
  if (!makeCaloJet(jetTypeE)) return false;
  const CaloTower* tower=dynamic_cast<const CaloTower*>(input.get());
  if (0==tower) return false;
  if (tower->numBadEcalCells()        >maxBadEcalCells_        ||
      tower->numRecoveredEcalCells()  >maxRecoveredEcalCells_  ||
      tower->numProblematicEcalCells()>maxProblematicEcalCells_||
      tower->numBadHcalCells()        >maxBadHcalCells_        ||
      tower->numRecoveredHcalCells()  >maxRecoveredHcalCells_  ||
      tower->numProblematicHcalCells()>maxProblematicHcalCells_) return true;
  return false;
}


//------------------------------------------------------------------------------
// This is pure virtual. 
//______________________________________________________________________________
// void VirtualJetProducer::runAlgorithm( edm::Event & iEvent, edm::EventSetup const& iSetup,
// 				       reco::CandidateView const & inputs_);

//______________________________________________________________________________
void VirtualJetProducer::copyConstituents(const vector<fastjet::PseudoJet>& fjConstituents,
					  reco::Jet* jet)
{
  for (unsigned int i=0;i<fjConstituents.size();++i)
    jet->addDaughter(inputs_.ptrAt(fjConstituents[i].user_index()));
}


//______________________________________________________________________________
vector<reco::CandidatePtr>
VirtualJetProducer::getConstituents(const vector<fastjet::PseudoJet>&fjConstituents)
{
  vector<reco::CandidatePtr> result;
  for (unsigned int i=0;i<fjConstituents.size();i++) {
    int index = fjConstituents[i].user_index();
    reco::CandidatePtr candidate = inputs_.ptrAt(index);
    result.push_back(candidate);
  }
  return result;
}


void VirtualJetProducer::output(edm::Event & iEvent, edm::EventSetup const& iSetup)
{
  // Write jets and constitutents. Will use fjJets_, inputs_
  // and fjClusterSeq_
  switch( jetTypeE ) {
  case JetType::CaloJet :
    writeJets<reco::CaloJet>( iEvent, iSetup);
    break;
  case JetType::PFJet :
    writeJets<reco::PFJet>( iEvent, iSetup);
    break;
  case JetType::GenJet :
    writeJets<reco::GenJet>( iEvent, iSetup);
    break;
  case JetType::BasicJet :
    writeJets<reco::BasicJet>( iEvent, iSetup);
    break;
  default:
    edm::LogError("InvalidInput") << " invalid jet type in VirtualJetProducer\n";
    break;
  };
  
}

template< typename T >
void VirtualJetProducer::writeJets( edm::Event & iEvent, edm::EventSetup const& iSetup )
{
  // produce output jet collection

  using namespace reco;

  std::auto_ptr<std::vector<T> > jets(new std::vector<T>() );
  jets->reserve(fjJets_.size());
      
  for (unsigned int ijet=0;ijet<fjJets_.size();++ijet) {
    // allocate this jet
    T jet;
    // get the fastjet jet
    const fastjet::PseudoJet& fjJet = fjJets_[ijet];
    // get the constituents from fastjet
    std::vector<fastjet::PseudoJet> fjConstituents =
      sorted_by_pt(fjClusterSeq_->constituents(fjJet));
    // convert them to CandidatePtr vector
    std::vector<CandidatePtr> constituents =
      getConstituents(fjConstituents);

    // Get the PU subtraction
    double px=fjJet.px();
    double py=fjJet.py();
    double pz=fjJet.pz();
    double E=fjJet.E();
    double jetArea=0.0;
    double pu=0.;

    // write the jet areas
    if ( doPUFastjet_ ) {
      // get PU pt
      fastjet::ClusterSequenceArea const * clusterSequenceWithArea = dynamic_cast<fastjet::ClusterSequenceArea const *> ( &*fjClusterSeq_ );
      double median_Pt_Per_Area = clusterSequenceWithArea->median_pt_per_unit_area_4vector( *fjRangeDef_ );
      fastjet::PseudoJet pu_p4 = median_Pt_Per_Area * clusterSequenceWithArea->area_4vector(fjJet);
      pu = pu_p4.E();
      if (pu_p4.perp2() >= fjJet.perp2() || pu_p4.E() >= fjJet.E()) { // if the correction is too large, set the jet to zero
	px = py = pz = E = 0.;
      } 
      else {   // otherwise do an E-scheme subtraction
	px -= pu_p4.px();
	py -= pu_p4.py();
	pz -= pu_p4.pz();
	E -= pu_p4.E();
      }
      jetArea = clusterSequenceWithArea->area(fjJet);
    }
    
    // write the specifics to the jet (simultaneously sets 4-vector, vertex).
    // These are overridden functions that will call the appropriate
    // specific allocator. 
    writeSpecific( jet,
		   Particle::LorentzVector(px, py, pz, E),
		   vertex_, 
		   constituents, iSetup);

    jet.setJetArea (jetArea);
    jet.setPileup (pu);

    // add to the list
    jets->push_back(jet);	
  }
  
  // put the jets in the collection
  iEvent.put(jets);
}



//
// Calculate mean E and sigma from jet collection "coll".  
//
void VirtualJetProducer::calculatePedestal( vector<fastjet::PseudoJet> const & coll )
{

  map<int,double> emean2;
  map<int,int> ntowers;
    
  int ietaold = -10000;
  int ieta0 = -100;
   
  // Initial values for emean_, emean2, esigma_, ntowers

  for(int i = ietamin_; i < ietamax_+1; i++)
    {
      emean_[i] = 0.;
      emean2[i] = 0.;
      esigma_[i] = 0.;
      ntowers[i] = 0;
    }
    
  for (vector<fastjet::PseudoJet>::const_iterator input_object = coll.begin (),
	 fjInputsEnd = coll.end();  
       input_object != fjInputsEnd; ++input_object) {
    ieta0 = ieta( inputs_.ptrAt( input_object->user_index() ) );

    if( ieta0-ietaold != 0 )
      {
        emean_[ieta0] = emean_[ieta0]+input_object->Et();
        emean2[ieta0] = emean2[ieta0]+(input_object->Et())*(input_object->Et());
        ntowers[ieta0] = 1;
        ietaold = ieta0;
      }
    else
      {
	emean_[ieta0] = emean_[ieta0]+input_object->Et();
	emean2[ieta0] = emean2[ieta0]+(input_object->Et())*(input_object->Et());
	ntowers[ieta0]++;
      }
  }

  for(map<int,int>::const_iterator gt = geomtowers_.begin(); gt != geomtowers_.end(); gt++)    
    {

      int it = (*gt).first;
       
      double e1 = (*emean_.find(it)).second;
      double e2 = (*emean2.find(it)).second;
      int nt = (*gt).second - (*ntowersWithJets_.find(it)).second;
        
      if(nt > 0) {
	emean_[it] = e1/nt;
	double eee = e2/nt - e1*e1/(nt*nt);	    
	if(eee<0.) eee = 0.;
	esigma_[it] = nSigmaPU_*sqrt(eee);
      }
      else
	{
          emean_[it] = 0.;
          esigma_[it] = 0.;
	}
    }

}


//
// Subtract mean and sigma from fjInputs_
//    
void VirtualJetProducer::subtractPedestal(vector<fastjet::PseudoJet> & coll)
{
  int it = -100;
  int ip = -100;
        
  for (vector<fastjet::PseudoJet>::iterator input_object = coll.begin (),
	 fjInputsEnd = coll.end(); 
       input_object != fjInputsEnd; ++input_object) {
    
    reco::CandidatePtr const & itow =  inputs_.ptrAt( input_object->user_index() );
    
    it = ieta( itow );
    ip = iphi( itow );
    
    double etnew = input_object->Et() - (*emean_.find(it)).second - (*esigma_.find(it)).second;
    float mScale = etnew/input_object->Et(); 
    
    if(etnew < 0.) mScale = 0.;
    
    math::XYZTLorentzVectorD towP4(input_object->px()*mScale, input_object->py()*mScale,
				   input_object->pz()*mScale, input_object->e()*mScale);
    
    input_object->reset ( towP4.px(),
			  towP4.py(),
			  towP4.pz(),
			  towP4.energy() );
  }
}



void VirtualJetProducer::calculateOrphanInput(vector<fastjet::PseudoJet> & orphanInput) 
{
  vector<int> jettowers; // vector of towers indexed by "user_index"
  vector <fastjet::PseudoJet>::iterator pseudojetTMP = fjJets_.begin (),
    fjJetsEnd = fjJets_.end();
    
  for (; pseudojetTMP != fjJetsEnd ; ++pseudojetTMP) {
      
    vector<fastjet::PseudoJet> newtowers;
      
    // get eta, phi of this jet
    double eta2 = pseudojetTMP->eta();
    double phi2 = pseudojetTMP->phi();
    // find towers within radiusPU_ of this jet
    for(vector<HcalDetId>::const_iterator im = allgeomid_.begin(); im != allgeomid_.end(); im++)
      {
	double eta1 = geo_->getPosition((DetId)(*im)).eta();
	double phi1 = geo_->getPosition((DetId)(*im)).phi();
	double dphi = fabs(phi1-phi2);
	double deta = eta1-eta2;
	if (dphi > 4.*atan(1.)) dphi = 8.*atan(1.) - dphi;
	double dr = sqrt(dphi*dphi+deta*deta);	  
	if( dr < radiusPU_) {
	  ntowersWithJets_[(*im).ieta()]++; 	    
	}
      }

    vector<fastjet::PseudoJet>::const_iterator it = fjInputs_.begin(),
      fjInputsBegin = fjInputs_.begin(),
      fjInputsEnd = fjInputs_.end();
      
    // 
    for (; it != fjInputsEnd; ++it ) {
	
      double eta1 = it->eta();
      double phi1 = it->phi();
	
      double dphi = fabs(phi1-phi2);
      double deta = eta1-eta2;
      if (dphi > 4.*atan(1.)) dphi = 8.*atan(1.) - dphi;
      double dr = sqrt(dphi*dphi+deta*deta);
	
      if( dr < radiusPU_) {
	newtowers.push_back(*it);
	jettowers.push_back(it->user_index());
      } //dr < 0.5

    } // initial input collection  

  } // pseudojets

  //
  // Create a new collections from the towers not included in jets 
  //
  for(vector<fastjet::PseudoJet>::const_iterator it = fjInputs_.begin(),
	fjInputsEnd = fjInputs_.end(); it != fjInputsEnd; ++it ) {
    vector<int>::const_iterator itjet = find(jettowers.begin(),jettowers.end(),it->user_index());
    if( itjet == jettowers.end() ) orphanInput.push_back(*it); 
  }

}


void VirtualJetProducer::offsetCorrectJets(vector<fastjet::PseudoJet> & orphanInput) 
{

  using namespace reco;

  //    
  // Reestimate energy of jet (energy of jet with initial map)
  //
  vector<fastjet::PseudoJet>::iterator pseudojetTMP = fjJets_.begin (),
    jetsEnd = fjJets_.end();
  for (; pseudojetTMP != jetsEnd; ++pseudojetTMP) {

    // get the constituents from fastjet
    std::vector<fastjet::PseudoJet> towers =
      sorted_by_pt(fjClusterSeq_->constituents(*pseudojetTMP));
    
    double offset = 0.;
      
    for(vector<fastjet::PseudoJet>::const_iterator ito = towers.begin(),
	  towEnd = towers.end(); 
	ito != towEnd; 
	++ito)
      {
	  
	int it = ieta( inputs_.ptrAt( ito->user_index() ) );
	  
	//       offset = offset + (*emean_.find(it)).second + (*esigma_.find(it)).second;
	// Temporarily for test       
	  
	double etnew = (*ito).Et() - (*emean_.find(it)).second - (*esigma_.find(it)).second; 
	  
	if( etnew <0.) etnew = 0.;
	  
	offset = offset + etnew;

      }
    //      double mScale = (pseudojetTMP->Et()-offset)/pseudojetTMP->Et();
    // Temporarily for test only
      
    double mScale = offset/pseudojetTMP->Et();

    ///
    ///!!! Change towers to rescaled towers///
    ///      
    pseudojetTMP->reset(pseudojetTMP->px()*mScale, pseudojetTMP->py()*mScale,
			pseudojetTMP->pz()*mScale, pseudojetTMP->e()*mScale);
     
  }    
}



int VirtualJetProducer::ieta(const reco::CandidatePtr & in)
{
  //   std::cout<<" Start BasePilupSubtractionJetProducer::ieta "<<std::endl;
  int it = 0;
  const CaloTower* ctc = dynamic_cast<const CaloTower*>(in.get());
     
  if(ctc)
    {
      it = ctc->id().ieta(); 
    } else
    {
      throw cms::Exception("Invalid Constituent") << "CaloJet constituent is not of CaloTower type";
    }
   
  return it;
}

int VirtualJetProducer::iphi(const reco::CandidatePtr & in)
{
  int it = 0;
  const CaloTower* ctc = dynamic_cast<const CaloTower*>(in.get());
  if(ctc)
    {
      it = ctc->id().iphi();
    } else
    {
      throw cms::Exception("Invalid Constituent") << "CaloJet constituent is not of CaloTower type";
    }
   
  return it;
}

