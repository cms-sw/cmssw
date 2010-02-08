#include "RecoJets/JetAlgorithms/interface/JetIDHelper.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitFwd.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include "TMath.h"
#include <vector>
#include <iostream>

using namespace std;

// defining stuff useful only for this class (not needed in header)
namespace reco {

  namespace helper {

    // select2nd exists only in some std and boost implementations, so let's control our own fate
    // and it can't be a non-static member function.
    static double select2nd (std::map<int,double>::value_type const &pair) {return pair.second;}
    
    bool hasNonPositiveE( reco::helper::JetIDHelper::subtower x ) {
      return x.E <= 0;
    }
    
    bool subtower_has_greater_E( reco::helper::JetIDHelper::subtower i, 
				 reco::helper::JetIDHelper::subtower j ) { return i.E > j.E; }
  }
}


reco::helper::JetIDHelper::JetIDHelper( edm::ParameterSet const & pset )
{
  useRecHits_ = pset.getParameter<bool>("useRecHits");
  if( useRecHits_ ) {
    hbheRecHitsColl_ = pset.getParameter<edm::InputTag>("hbheRecHitsColl");
    hoRecHitsColl_   = pset.getParameter<edm::InputTag>("hoRecHitsColl");
    hfRecHitsColl_   = pset.getParameter<edm::InputTag>("hfRecHitsColl");
    ebRecHitsColl_   = pset.getParameter<edm::InputTag>("ebRecHitsColl");
    eeRecHitsColl_   = pset.getParameter<edm::InputTag>("eeRecHitsColl");   
  }
  initValues();
}
  
void reco::helper::JetIDHelper::initValues()
{
  fHPD_= -1.0;
  fRBX_= -1.0;
  n90Hits_ = -1;
  fSubDetector1_= 0.0;
  fSubDetector2_= 0.0;
  fSubDetector3_= 0.0;
  fSubDetector4_= 0.0;
  restrictedEMF_= 0.0;
  nHCALTowers_ = 0;
  nECALTowers_ = 0;
  approximatefHPD_ = 0.0;
  approximatefRBX_ = 0.0;
  hitsInN90_ = 0;
}

void reco::helper::JetIDHelper::fillDescription(edm::ParameterSetDescription& iDesc)
{
  iDesc.ifValue( edm::ParameterDescription<bool>("useRecHits", true, true),
		 true >> (edm::ParameterDescription<edm::InputTag>("hbheRecHitsColl", edm::InputTag(), true) and
			  edm::ParameterDescription<edm::InputTag>("hoRecHitsColl", edm::InputTag(), true) and
			  edm::ParameterDescription<edm::InputTag>("hfRecHitsColl", edm::InputTag(), true) and
			  edm::ParameterDescription<edm::InputTag>("ebRecHitsColl", edm::InputTag(), true) and
			  edm::ParameterDescription<edm::InputTag>("eeRecHitsColl", edm::InputTag(), true)
			  ) 
                 )->setComment("If using RecHits to calculate the precise jet ID variables that need them, "
			       "their sources need to be specified");
}


void reco::helper::JetIDHelper::calculate( const edm::Event& event, const reco::CaloJet &jet, const int iDbg )
{
  initValues();

  // ---------------------------------------------------
  // 1) jet ID variables derived from existing fractions
  // ---------------------------------------------------

  double E_EM = TMath::Max( float(0.), jet.emEnergyInHF() )
              + TMath::Max( float(0.), jet.emEnergyInEB() )
              + TMath::Max( float(0.), jet.emEnergyInEE() );
  double E_Had = TMath::Max( float(0.), jet.hadEnergyInHB() )
               + TMath::Max( float(0.), jet.hadEnergyInHE() )
               + TMath::Max( float(0.), jet.hadEnergyInHO() )
               + TMath::Max( float(0.), jet.hadEnergyInHF() );
  if( E_Had + E_EM > 0 ) this->restrictedEMF_ = E_EM / ( E_EM + E_Had );
  if( iDbg > 1 ) cout<<"jet pT: "<<jet.pt()<<", eT: "<<jet.et()<<", E: "
		     <<jet.energy()<<" rEMF: "<<this->restrictedEMF_<<endl;

  if( jet.energy() > 0 ) {
    vector<double> subdet_energies;
    calculateSubDetectorEnergies( event, jet, subdet_energies );
    if( subdet_energies.size() > 0 ) this->fSubDetector1_ = subdet_energies.at( 0 ) / jet.energy();
    if( subdet_energies.size() > 1 ) this->fSubDetector2_ = subdet_energies.at( 1 ) / jet.energy();
    if( subdet_energies.size() > 2 ) this->fSubDetector3_ = subdet_energies.at( 2 ) / jet.energy();
    if( subdet_energies.size() > 3 ) this->fSubDetector4_ = subdet_energies.at( 3 ) / jet.energy();
    if( iDbg > 1 ) {
      cout<<"subdet_E:";
      for (unsigned int i=0; i<subdet_energies.size(); ++i) cout<<" "<<subdet_energies[i];
      cout<<endl;
    }
  }

  // ------------------------
  // 2) tower based variables
  // ------------------------
  vector<subtower> subtowers, Ecal_subtowers, Hcal_subtowers, HO_subtowers;
  vector<double> HPD_energies, RBX_energies;

  classifyJetTowers( event, jet, 
		     subtowers, Ecal_subtowers, Hcal_subtowers, HO_subtowers, 
		     HPD_energies, RBX_energies, iDbg );
  if( iDbg > 1 ) {
    cout<<"E:";
    for (unsigned int i=0; i<subtowers.size(); ++i) cout<<" "<<subtowers[i].E<<","<<subtowers[i].Nhit;
    cout<<"\nECal_E:";
    for (unsigned int i=0; i<Ecal_subtowers.size(); ++i) cout<<" "<<Ecal_subtowers[i].E<<","<<Ecal_subtowers[i].Nhit;
    cout<<"\nHCal_E:";
    for (unsigned int i=0; i<Hcal_subtowers.size(); ++i) cout<<" "<<Hcal_subtowers[i].E<<","<<Hcal_subtowers[i].Nhit;
    cout<<"\nHO_E:";
    for (unsigned int i=0; i<HO_subtowers.size(); ++i) cout<<" "<<HO_subtowers[i].E<<","<<HO_subtowers[i].Nhit;
    cout<<"\nHPD_E:";
    for (unsigned int i=0; i<HPD_energies.size(); ++i) cout<<" "<<HPD_energies[i];
    cout<<"\nRBX_E:";
    for (unsigned int i=0; i<RBX_energies.size(); ++i) cout<<" "<<RBX_energies[i];
    cout<<endl;
  }

  // counts
  this->hitsInN90_ = hitsInNCarrying( 0.9, subtowers );
  this->nHCALTowers_ = Hcal_subtowers.size();
  vector<subtower>::const_iterator it;
  it = find_if( Ecal_subtowers.begin(), Ecal_subtowers.end(), hasNonPositiveE );
  this->nECALTowers_ = it - Ecal_subtowers.begin(); // ignores negative energies from HF!

  // energy fractions
  this->approximatefHPD_ = this->approximatefRBX_ = 0;
  if( jet.energy() > 0 ) {
    if( HPD_energies.size() > 0 ) this->approximatefHPD_ = HPD_energies.at( 0 ) / jet.energy();
    if( RBX_energies.size() > 0 ) this->approximatefRBX_ = RBX_energies.at( 0 ) / jet.energy();
  }

  // -----------------------
  // 3) cell based variables
  // -----------------------
  if( useRecHits_ ) {
    vector<double> energies, Ecal_energies, Hcal_energies, HO_energies;
    classifyJetComponents( event, jet, 
			   energies, Ecal_energies, Hcal_energies, HO_energies, 
			   HPD_energies, RBX_energies,
			   true, iDbg );

    // counts
    this->n90Hits_ = nCarrying( 0.9, energies );

    // energy fractions
    this->fHPD_ = this->fRBX_ = 0;
    if( jet.energy() > 0 ) {
      if( HPD_energies.size() > 0 ) this->fHPD_ = HPD_energies.at( 0 ) / jet.energy();
      if( RBX_energies.size() > 0 ) this->fRBX_ = RBX_energies.at( 0 ) / jet.energy();
    }

    if( iDbg > 1 ) {
      cout<<"DBG - fHPD: "<<this->fHPD_<<", fRBX: "<<this->fRBX_<<", nh90: "<<this->n90Hits_<<endl;
      cout<<"    -~fHPD: "<<this->approximatefHPD_<<", ~fRBX: "<<this->approximatefRBX_
	  <<", hits in n90: "<<this->hitsInN90_<<endl;
      cout<<"    - nHCALTowers: "<<this->nHCALTowers_<<", nECALTowers: "<<this->nECALTowers_<<endl;
    }
  }
}


unsigned int reco::helper::JetIDHelper::nCarrying( double fraction, vector< double > descending_energies )
{
  double totalE = 0;
  for( unsigned int i = 0; i < descending_energies.size(); ++i ) totalE += descending_energies[ i ];

  double runningE = 0;
  unsigned int NC = descending_energies.size();
  
  // slightly odd loop structure avoids round-off problems when runningE never catches up with totalE
  for( unsigned int i = descending_energies.size(); i > 0; --i ) {
    runningE += descending_energies[ i-1 ];
    if (runningE < ( 1-fraction ) * totalE) NC = i-1;
  }
  return NC;
}


unsigned int reco::helper::JetIDHelper::hitsInNCarrying( double fraction, vector< subtower > descending_towers )
{
  double totalE = 0;
  for( unsigned int i = 0; i < descending_towers.size(); ++i ) totalE += descending_towers[ i ].E;

  double runningE = 0;
  unsigned int NH = 0;
  
  // slightly odd loop structure avoids round-off problems when runningE never catches up with totalE
  for( unsigned int i = descending_towers.size(); i > 0; --i ) {
    runningE += descending_towers[ i-1 ].E;
    if (runningE >= ( 1-fraction ) * totalE) NH += descending_towers[ i-1 ].Nhit;
  }
  return NH;
}


  
void reco::helper::JetIDHelper::calculateSubDetectorEnergies( const edm::Event& event, const reco::CaloJet &jet, 
							      vector< double > &subdet_energies )
{
  subdet_energies.clear(); 
  // sub detector energies were already sorted out by JetMaker.
  // JetMaker respects the "physics" interpretation of HF readount ( (short, long) --> (em, had) )
  // while we need HF long and HF short separately. Luckily, this is all additive...
  subdet_energies.push_back( jet.hadEnergyInHB() );
  subdet_energies.push_back( jet.hadEnergyInHE() );
  subdet_energies.push_back( jet.hadEnergyInHO() );
  subdet_energies.push_back( jet.emEnergyInEB() );
  subdet_energies.push_back( jet.emEnergyInEE() );
  // from CaloTowers_CreationAlgo: E_short = 0.5 * newE_had; E_long  = newE_em + 0.5 * newE_had;
  subdet_energies.push_back( 0.5 * jet.hadEnergyInHF() );
  subdet_energies.push_back( jet.emEnergyInHF() + 0.5 * jet.hadEnergyInHF() );
  std::sort( subdet_energies.begin(), subdet_energies.end(), greater<double>() );
}

void reco::helper::JetIDHelper::classifyJetComponents( const edm::Event& event, const reco::CaloJet &jet, 
						       vector< double > &energies,      
						       vector< double > &Ecal_energies, 
						       vector< double > &Hcal_energies, 
						       vector< double > &HO_energies,
						       vector< double > &HPD_energies,  
						       vector< double > &RBX_energies,
						       const bool recHitBased, const int iDbg )
{
  energies.clear(); Ecal_energies.clear(); Hcal_energies.clear(); HO_energies.clear();
  HPD_energies.clear(); RBX_energies.clear();

  std::map< int, double > HPD_energy_map, RBX_energy_map;
  edm::Handle<HBHERecHitCollection> HBHERecHits;
  edm::Handle<HORecHitCollection> HORecHits;
  edm::Handle<HFRecHitCollection> HFRecHits;
  edm::Handle<EBRecHitCollection> EBRecHits;
  edm::Handle<EERecHitCollection> EERecHits;
  if( recHitBased ) {
    // the jet only contains DetIds, so first read recHit collection
    event.getByLabel( hbheRecHitsColl_, HBHERecHits );
    event.getByLabel( hoRecHitsColl_, HORecHits );
    event.getByLabel( hfRecHitsColl_, HFRecHits );
    event.getByLabel( ebRecHitsColl_, EBRecHits );
    event.getByLabel( eeRecHitsColl_, EERecHits );
    if( iDbg > 2 ) cout<<"# of rechits found - HBHE: "<<HBHERecHits->size()
		       <<", HO: "<<HORecHits->size()<<", HF: "<<HFRecHits->size()
		       <<", EB: "<<EBRecHits->size()<<", EE: "<<EERecHits->size()<<endl;
  }
  //  double totHcalE = 0;

  vector< CaloTowerPtr > towers = jet.getCaloConstituents ();
  int nTowers = towers.size();
  if( iDbg > 9 ) cout<<"In classifyJetComponents. # of towers found: "<<nTowers<<endl;

  for( int iTower = 0; iTower <nTowers ; iTower++ ) {

    CaloTowerPtr& tower = towers[iTower];

    int nCells = tower->constituentsSize();
    if( iDbg ) cout<<"tower #"<<iTower<<" has "<<nCells<<" cells. "
		   <<"It's at iEta: "<<tower->ieta()<<", iPhi: "<<tower->iphi()<<endl;

    if( ! recHitBased ) {

      double E_em = tower->emEnergy();
      if( E_em != 0 ) Ecal_energies.push_back( E_em );
      
      double E_HO = tower->outerEnergy();
      if( E_HO != 0 ) HO_energies.push_back( E_HO );
      
      double E_had = tower->hadEnergy();
      if( E_had != 0 ) {
	Hcal_energies.push_back( E_had );
	// totHcalE += E_had;

	int iEta = tower->ieta();
	Region reg = region( iEta );
	int iPhi = tower->iphi();
	if( iDbg>3 ) cout<<"tower has E_had: "<<E_had<<" iEta: "<<iEta
			 <<", iPhi: "<<iPhi<<" -> "<<reg;

	if( reg == HEneg || reg == HBneg || reg == HBpos || reg == HEpos ) {
	  int oddnessIEta = HBHE_oddness( iEta );
	  if( oddnessIEta < 0 ) break; // can't assign this tower to a single readout component

	  int iHPD = 100 * reg;
	  int iRBX = 100 * reg + ((iPhi + 1) % 72) / 4; // 71,72,1,2 are in the same RBX module
	    
	  if(( reg == HEneg || reg == HEpos ) && std::abs( iEta ) >= 21 ) { // at low-granularity edge of HE
	    if( (0x1 & iPhi) == 0 ) {
	      throw cms::Exception("CodeAssumptionsViolated")<<
		"Jet ID code assumes no even iPhi recHits at HE edges";
	      return;
	    }
	    bool boolOddnessIEta = oddnessIEta;
	    bool upperIPhi = (( iPhi%4 ) == 1 || ( iPhi%4 ) == 2); // the upper iPhi indices in the HE wedge
	    // remap the iPhi so it fits the one in the inner HE regions, change in needed in two cases:
	    // 1) in the upper iPhis of the module, the even IEtas belong to the higher iPhi
	    // 2) in the loewr iPhis of the module, the odd  IEtas belong to the higher iPhi
	    if( upperIPhi != boolOddnessIEta ) ++iPhi; 
	    // note that iPhi could not be 72 before, so it's still in the legal range [1,72]
	  } // if at low-granularity edge of HE
	  iHPD += iPhi;
	    
	  // book the energies
	  HPD_energy_map[ iHPD ] += E_had;
	  RBX_energy_map[ iRBX ] += E_had;
	  if( iDbg > 5 ) cout<<" --> H["<<iHPD<<"]="<<HPD_energy_map[iHPD]
			     <<", R["<<iRBX<<"]="<<RBX_energy_map[iRBX];
	} // HBHE
      } // E_had > 0

    } else { // RecHit based:

      const vector< DetId >& cellIDs = tower->constituents();  // cell == recHit
  
      for( int iCell = 0; iCell < nCells; ++iCell ) {
	DetId::Detector detNum = cellIDs[iCell].det();
	if( detNum == DetId::Hcal ) {
	  HcalDetId HcalID = cellIDs[ iCell ];
	  HcalSubdetector HcalNum = HcalID.subdet();
	  double hitE = 0;
	  if( HcalNum == HcalOuter ) {
	    HORecHitCollection::const_iterator theRecHit=HORecHits->find(HcalID);
	    if (theRecHit == HORecHits->end()) {
	      edm::LogWarning("UnexpectedEventContents")<<"Can't find the HO recHit with ID: "<<HcalID;
	      continue;
	    }
	    hitE = theRecHit->energy();
	    HO_energies.push_back( hitE );
	    
	  } else if( HcalNum == HcalForward ) {
	    
	    HFRecHitCollection::const_iterator theRecHit=HFRecHits->find( HcalID );	    
	    if( theRecHit == HFRecHits->end() ) {
	      edm::LogWarning("UnexpectedEventContents")<<"Can't find the HF recHit with ID: "<<HcalID;
	      continue;
	    }
	    hitE = theRecHit->energy();
	    if( iDbg>4 ) cout 
	      << "hit #"<<iCell<<" is  HF , E: "<<hitE<<" iEta: "<<theRecHit->id().ieta()
	      <<", depth: "<<theRecHit->id().depth()<<", iPhi: "
	      <<theRecHit->id().iphi();
	    
	    Hcal_energies.push_back( hitE ); // not clear what is the most useful here. This is a preliminary guess.
	    
	  } else { // HBHE
	    
	    HBHERecHitCollection::const_iterator theRecHit = HBHERecHits->find( HcalID );	    
	    if( theRecHit == HBHERecHits->end() ) {
	      edm::LogWarning("UnexpectedEventContents")<<"Can't find the HBHE recHit with ID: "<<HcalID; 
	      continue;
	    }
	    hitE = theRecHit->energy();
	    int iEta = theRecHit->id().ieta();
	    int depth = theRecHit->id().depth();
	    Region region = HBHE_region( iEta, depth );
	    int hitIPhi = theRecHit->id().iphi();
	    if( iDbg>3 ) cout<<"hit #"<<iCell<<" is HBHE, E: "<<hitE<<" iEta: "<<iEta
			     <<", depth: "<<depth<<", iPhi: "<<theRecHit->id().iphi()
			     <<" -> "<<region;
	    int absIEta = TMath::Abs( theRecHit->id().ieta() );
	    if( depth == 3 && (absIEta == 28 || absIEta == 29) ) {
	      hitE /= 2; // Depth 3 at the HE forward edge is split over tower 28 & 29, and jet reco. assigns half each
	    }
	    int iHPD = 100 * region;
	    int iRBX = 100 * region + ((hitIPhi + 1) % 72) / 4; // 71,72,1,2 are in the same RBX module
	    
	    if( std::abs( iEta ) >= 21 ) {
	      if( (0x1 & hitIPhi) == 0 ) {
		edm::LogError("CodeAssumptionsViolated")<<"Bug?! Jet ID code assumes no even iPhi recHits at HE edges";
		return;
	      }
	      bool oddnessIEta = HBHE_oddness( iEta, depth );
	      bool upperIPhi = (( hitIPhi%4 ) == 1 || ( hitIPhi%4 ) == 2); // the upper iPhi indices in the HE wedge
	      // remap the iPhi so it fits the one in the inner HE regions, change in needed in two cases:
	      // 1) in the upper iPhis of the module, the even iEtas belong to the higher iPhi
	      // 2) in the loewr iPhis of the module, the odd  iEtas belong to the higher iPhi
	      if( upperIPhi != oddnessIEta ) ++hitIPhi; 
	      // note that hitIPhi could not be 72 before, so it's still in the legal range [1,72]
	    }
	    iHPD += hitIPhi;
	    
	    // book the energies
	    HPD_energy_map[ iHPD ] += hitE;
	    RBX_energy_map[ iRBX ] += hitE;
	    if( iDbg > 5 ) cout<<" --> H["<<iHPD<<"]="<<HPD_energy_map[iHPD]
			       <<", R["<<iRBX<<"]="<<RBX_energy_map[iRBX];
	    if( iDbg > 1 ) cout<<endl;
	    // totHcalE += hitE;
	    
	    Hcal_energies.push_back( hitE );
	    
	  } // if HBHE
	  if( hitE == 0 ) edm::LogWarning("UnexpectedEventContents")<<"HCal hitE==0? (or unknown subdetector?)";
	  
	} // if HCAL 
	
	else if (detNum == DetId::Ecal) {
	  
	  int EcalNum =  cellIDs[iCell].subdetId();
	  double hitE = 0;
	  if( EcalNum == 1 ){
	    EBDetId EcalID = cellIDs[iCell];
	    EBRecHitCollection::const_iterator theRecHit=EBRecHits->find(EcalID);
	    if( theRecHit == EBRecHits->end() ) {edm::LogWarning("UnexpectedEventContents")
	      <<"Can't find the EB recHit with ID: "<<EcalID; continue;}
	    hitE = theRecHit->energy();
	  } else if(  EcalNum == 2 ){
	    EEDetId EcalID = cellIDs[iCell];
	    EERecHitCollection::const_iterator theRecHit=EERecHits->find(EcalID);
	    if( theRecHit == EERecHits->end() ) {edm::LogWarning("UnexpectedEventContents")
	      <<"Can't find the EE recHit with ID: "<<EcalID; continue;}
	    hitE = theRecHit->energy();
	  }
	  if( hitE == 0 ) edm::LogWarning("UnexpectedEventContents")<<"ECal hitE==0? (or unknown subdetector?)";
	  if( iDbg > 6 ) cout<<"EcalNum: "<<EcalNum<<" hitE: "<<hitE<<endl;
	  Ecal_energies.push_back (hitE);
	} // 
      } // loop on cells
    } // if RecHit based

  } // loop on towers

  /* Disabling check until HO is accounted for in EMF. Check was used in CMSSW_2, where HE was excluded.
  double expHcalE = jet.energy() * (1-jet.emEnergyFraction());
  if( totHcalE + expHcalE > 0 && 
      TMath::Abs( totHcalE - expHcalE ) > 0.01 && 
      ( totHcalE - expHcalE ) / ( totHcalE + expHcalE ) > 0.0001 ) {
    edm::LogWarning("CodeAssumptionsViolated")<<"failed to account for all Hcal energies"
					      <<totHcalE<<"!="<<expHcalE;
					      } */   

  // sort the energies
  std::sort( Hcal_energies.begin(), Hcal_energies.end(), greater<double>() );
  std::sort( Ecal_energies.begin(), Ecal_energies.end(), greater<double>() );

  // put the energy sums (the 2nd entry in each pair of the maps) into the output vectors and sort them
  std::transform( HPD_energy_map.begin(), HPD_energy_map.end(), 
		  std::inserter (HPD_energies, HPD_energies.end()), select2nd ); 
  //		  std::select2nd<std::map<int,double>::value_type>());
  std::sort( HPD_energies.begin(), HPD_energies.end(), greater<double>() );
  std::transform( RBX_energy_map.begin(), RBX_energy_map.end(), 
		  std::inserter (RBX_energies, RBX_energies.end()), select2nd );
  //		  std::select2nd<std::map<int,double>::value_type>());
  std::sort( RBX_energies.begin(), RBX_energies.end(), greater<double>() );

  energies.insert( energies.end(), Hcal_energies.begin(), Hcal_energies.end() );
  energies.insert( energies.end(), Ecal_energies.begin(), Ecal_energies.end() );
  energies.insert( energies.end(), HO_energies.begin(), HO_energies.end() );
  std::sort( energies.begin(), energies.end(), greater<double>() );
}

void reco::helper::JetIDHelper::classifyJetTowers( const edm::Event& event, const reco::CaloJet &jet, 
						   vector< subtower > &subtowers,      
						   vector< subtower > &Ecal_subtowers, 
						   vector< subtower > &Hcal_subtowers, 
						   vector< subtower > &HO_subtowers,
						   vector< double > &HPD_energies,  
						   vector< double > &RBX_energies,
						   const int iDbg )
{
  subtowers.clear(); Ecal_subtowers.clear(); Hcal_subtowers.clear(); HO_subtowers.clear();
  HPD_energies.clear(); RBX_energies.clear();

  std::map< int, double > HPD_energy_map, RBX_energy_map;

  vector< CaloTowerPtr > towers = jet.getCaloConstituents ();
  int nTowers = towers.size();
  if( iDbg > 9 ) cout<<"classifyJetTowers started. # of towers found: "<<nTowers<<endl;

  for( int iTower = 0; iTower <nTowers ; iTower++ ) {

    CaloTowerPtr& tower = towers[iTower];

    int nEM = 0, nHad = 0, nHO = 0;
    const vector< DetId >& cellIDs = tower->constituents();  // cell == recHit
    int nCells = cellIDs.size();
    if( iDbg ) cout<<"tower #"<<iTower<<" has "<<nCells<<" cells. "
		   <<"It's at iEta: "<<tower->ieta()<<", iPhi: "<<tower->iphi()<<endl;
  
    for( int iCell = 0; iCell < nCells; ++iCell ) {
      DetId::Detector detNum = cellIDs[iCell].det();
      if( detNum == DetId::Hcal ) {
	HcalDetId HcalID = cellIDs[ iCell ];
	HcalSubdetector HcalNum = HcalID.subdet();
	if( HcalNum == HcalOuter ) {
	  ++nHO;
	} else {
	  ++nHad;
	}
      }	else if (detNum == DetId::Ecal) {
	++nEM;
      }
    }

    double E_em = tower->emEnergy();
    if( E_em != 0 ) Ecal_subtowers.push_back( subtower( E_em, nEM ) );
      
    double E_HO = tower->outerEnergy();
    if( E_HO != 0 ) HO_subtowers.push_back( subtower( E_HO, nHO ) );
      
    double E_had = tower->hadEnergy();
    if( E_had != 0 ) {
      Hcal_subtowers.push_back( subtower( E_had, nHad ) );
      // totHcalE += E_had;
      
      int iEta = tower->ieta();
      Region reg = region( iEta );
      int iPhi = tower->iphi();
      if( iDbg>3 ) cout<<"tower has E_had: "<<E_had<<" iEta: "<<iEta
		       <<", iPhi: "<<iPhi<<" -> "<<reg;
      
      if( reg == HEneg || reg == HBneg || reg == HBpos || reg == HEpos ) {
	int oddnessIEta = HBHE_oddness( iEta );
	if( oddnessIEta < 0 ) break; // can't assign this tower to a single readout component
	
	int iHPD = 100 * reg;
	int iRBX = 100 * reg + ((iPhi + 1) % 72) / 4; // 71,72,1,2 are in the same RBX module
	
	if(( reg == HEneg || reg == HEpos ) && std::abs( iEta ) >= 21 ) { // at low-granularity edge of HE
	  if( (0x1 & iPhi) == 0 ) {
	    edm::LogError("CodeAssumptionsViolated")<<
	      "Bug?! Jet ID code assumes no even iPhi recHits at HE edges";
	    return;
	  }
	  bool boolOddnessIEta = oddnessIEta;
	  bool upperIPhi = (( iPhi%4 ) == 1 || ( iPhi%4 ) == 2); // the upper iPhi indices in the HE wedge
	  // remap the iPhi so it fits the one in the inner HE regions, change in needed in two cases:
	  // 1) in the upper iPhis of the module, the even IEtas belong to the higher iPhi
	  // 2) in the loewr iPhis of the module, the odd  IEtas belong to the higher iPhi
	  if( upperIPhi != boolOddnessIEta ) ++iPhi; 
	  // note that iPhi could not be 72 before, so it's still in the legal range [1,72]
	} // if at low-granularity edge of HE
	iHPD += iPhi;
	
	// book the energies
	HPD_energy_map[ iHPD ] += E_had;
	RBX_energy_map[ iRBX ] += E_had;
	if( iDbg > 5 ) cout<<" --> H["<<iHPD<<"]="<<HPD_energy_map[iHPD]
			   <<", R["<<iRBX<<"]="<<RBX_energy_map[iRBX];
      } // HBHE
    } // E_had > 0
  } // loop on towers

  // sort the subtowers
  std::sort( Hcal_subtowers.begin(), Hcal_subtowers.end(), subtower_has_greater_E );
  std::sort( Ecal_subtowers.begin(), Ecal_subtowers.end(), subtower_has_greater_E );

  // put the energy sums (the 2nd entry in each pair of the maps) into the output vectors and sort them
  std::transform( HPD_energy_map.begin(), HPD_energy_map.end(), 
		  std::inserter (HPD_energies, HPD_energies.end()), select2nd ); 
  //		  std::select2nd<std::map<int,double>::value_type>());
  std::sort( HPD_energies.begin(), HPD_energies.end(), greater<double>() );
  std::transform( RBX_energy_map.begin(), RBX_energy_map.end(), 
		  std::inserter (RBX_energies, RBX_energies.end()), select2nd );
  //		  std::select2nd<std::map<int,double>::value_type>());
  std::sort( RBX_energies.begin(), RBX_energies.end(), greater<double>() );

  subtowers.insert( subtowers.end(), Hcal_subtowers.begin(), Hcal_subtowers.end() );
  subtowers.insert( subtowers.end(), Ecal_subtowers.begin(), Ecal_subtowers.end() );
  subtowers.insert( subtowers.end(), HO_subtowers.begin(), HO_subtowers.end() );
  std::sort( subtowers.begin(), subtowers.end(), subtower_has_greater_E );
}

// ---------------------------------------------------------------------------------------------------------
// private helper functions to figure out detector & readout geometry as needed

int reco::helper::JetIDHelper::HBHE_oddness( int iEta, int depth )
{
 int ae = TMath::Abs( iEta );
 if( ae == 29 && depth == 1 ) ae += 1; // observed that: hits are at depths 1 & 2; 1 goes with the even pattern
 return ae & 0x1;
}

reco::helper::JetIDHelper::Region reco::helper::JetIDHelper::HBHE_region( int iEta, int depth )
{
  // no error checking for HO indices (depth 2 & |ieta|<=14 or depth 3 & |ieta|=15)
  if( iEta <= -17 || ( depth == 3 && iEta == -16 ) ) return HEneg;
  if( iEta >=  17 || ( depth == 3 && iEta ==  16 ) ) return HEpos;
  if( iEta < 0 ) return HBneg;
  return HBpos;
}

int reco::helper::JetIDHelper::HBHE_oddness( int iEta )
{
 int ae = TMath::Abs( iEta );
 if( ae == 29 ) return -1; // can't figure it out without RecHits
 return ae & 0x1;
}

reco::helper::JetIDHelper::Region reco::helper::JetIDHelper::region( int iEta )
{
  if( iEta == 16 || iEta == -16 ) return unknown_region; // both HB and HE cells belong to these towers
  if( iEta == 29 || iEta == -29 ) return unknown_region; // both HE and HF cells belong to these towers
  if( iEta <= -30 ) return HFneg;
  if( iEta >=  30 ) return HFpos;
  if( iEta <= -17 ) return HEneg;
  if( iEta >=  17 ) return HEpos;
  if( iEta < 0 ) return HBneg;
  return HBpos;
}
