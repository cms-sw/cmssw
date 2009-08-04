
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



// ------------------------------ Implementation ---------------------------------------------------------

reco::helper::JetIDHelper::JetIDHelper( edm::ParameterSet const & pset )
{

  hbheRecHitsColl_ = pset.getParameter<edm::InputTag>("hbheRecHitsColl");
  hoRecHitsColl_   = pset.getParameter<edm::InputTag>("hoRecHitsColl");
  hfRecHitsColl_   = pset.getParameter<edm::InputTag>("hfRecHitsColl");
  ebRecHitsColl_   = pset.getParameter<edm::InputTag>("ebRecHitsColl");
  eeRecHitsColl_   = pset.getParameter<edm::InputTag>("eeRecHitsColl");   
  
  fHPD_= 0.0;
  fRBX_= 0.0;
  n90Hits_ = 0;
  fSubDetector1_= 0.0;
  fSubDetector2_= 0.0;
  fSubDetector3_= 0.0;
  fSubDetector4_= 0.0;
  restrictedEMF_= 0.0;
  nHCALTowers_ = 0;
  nECALTowers_ = 0;
}


int reco::helper::JetIDHelper::HBHE_oddness (int iEta, int depth)
{
 int ae = TMath::Abs (iEta);
 if (ae == 29 && depth == 1) ae += 1; // observed that: hits are at depths 1 & 2; 1 goes with the even pattern
 return ae & 0x1;
}

int reco::helper::JetIDHelper::HBHE_region (int iEta, int depth)
{
  // no error checking for HO indices (depth 2 & |ieta|<=14 or depth 3 & |ieta|=15)
  if( iEta <= -17 || ( depth == 3 && iEta == -16 ) ) return 0; // HE-
  if( iEta >=  17 || ( depth == 3 && iEta ==  16 ) ) return 3; // HE+
  if( iEta < 0 ) return 1; // HB-
  return 2; // HB+
}

void reco::helper::JetIDHelper::calculate( const edm::Event& event, const reco::CaloJet &jet, const int iDbg )
{
  vector<double> energies, subdet_energies, Ecal_energies, Hcal_energies, HO_energies, HPD_energies, RBX_energies;
  unsigned int nHadTowers, nEMTowers;
  classifyJetComponents( event, jet, 
			 energies, subdet_energies, Ecal_energies, Hcal_energies, HO_energies, 
			 HPD_energies, RBX_energies,
			 nHadTowers, nEMTowers,
			 iDbg );
  if( iDbg ) cout<<"Got "<<energies.size()<<" hits"<<endl;
  if( iDbg > 1 ) {
    cout<<"E:";
    for (unsigned int i=0; i<energies.size(); ++i) cout<<" "<<energies[i];
    cout<<"\nsubdet_E:";
    for (unsigned int i=0; i<subdet_energies.size(); ++i) cout<<" "<<subdet_energies[i];
    cout<<"\nECal_E:";
    for (unsigned int i=0; i<Ecal_energies.size(); ++i) cout<<" "<<Ecal_energies[i];
    cout<<"\nHCal_E:";
    for (unsigned int i=0; i<Hcal_energies.size(); ++i) cout<<" "<<Hcal_energies[i];
    cout<<"\nHO_E:";
    for (unsigned int i=0; i<HO_energies.size(); ++i) cout<<" "<<HO_energies[i];
    cout<<"\nHPD_E:";
    for (unsigned int i=0; i<HPD_energies.size(); ++i) cout<<" "<<HPD_energies[i];
    cout<<"\nRBX_E:";
    for (unsigned int i=0; i<RBX_energies.size(); ++i) cout<<" "<<RBX_energies[i];
    cout<<endl;
  }

  // counts
  this->n90Hits_ = nCarrying( 0.9, energies );
  this->nHCALTowers_ = nHadTowers;
  this->nECALTowers_ = nEMTowers;

  // energy fractions
  this->fHPD_ = this->fRBX_ = 0;
  this->fSubDetector1_ = this->fSubDetector2_ = this->fSubDetector3_ = this->fSubDetector4_ = 0;
  if( jet.energy() > 0 ) {
    if( HPD_energies.size() > 0 ) this->fHPD_ = HPD_energies.at( 0 ) / jet.energy();
    if( RBX_energies.size() > 0 ) this->fRBX_ = RBX_energies.at( 0 ) / jet.energy();
    if( subdet_energies.size() > 0 ) this->fSubDetector1_ = subdet_energies.at( 0 ) / jet.energy();
    if( subdet_energies.size() > 1 ) this->fSubDetector2_ = subdet_energies.at( 1 ) / jet.energy();
    if( subdet_energies.size() > 2 ) this->fSubDetector3_ = subdet_energies.at( 2 ) / jet.energy();
    if( subdet_energies.size() > 3 ) this->fSubDetector4_ = subdet_energies.at( 3 ) / jet.energy();
  }

  // restricted energy fraction
  this->restrictedEMF_ = 0;
  double E_EM = TMath::Max( float(0.), jet.emEnergyInHF() )
              + TMath::Max( float(0.), jet.emEnergyInEB() )
              + TMath::Max( float(0.), jet.emEnergyInEE() );
  double E_Had = TMath::Max( float(0.), jet.hadEnergyInHB() )
               + TMath::Max( float(0.), jet.hadEnergyInHE() )
               + TMath::Max( float(0.), jet.hadEnergyInHO() )
               + TMath::Max( float(0.), jet.hadEnergyInHF() );
  if( E_Had + E_EM > 0 ) this->restrictedEMF_ = E_EM / ( E_EM + E_Had );

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

  
void reco::helper::JetIDHelper::classifyJetComponents( const edm::Event& event, const reco::CaloJet &jet, 
						       vector< double > &energies,      vector< double > &subdet_energies,
						       vector< double > &Ecal_energies, vector< double > &Hcal_energies, 
						       vector< double > &HO_energies,
						       vector< double > &HPD_energies,  vector< double > &RBX_energies,
						       unsigned int& nHadTowers, unsigned int& nEMTowers, int iDbg )
{
  energies.clear(); subdet_energies.clear(); Ecal_energies.clear(); Hcal_energies.clear(); HO_energies.clear();
  HPD_energies.clear(); RBX_energies.clear();
  nHadTowers = nEMTowers = 0;

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

  std::map< int, double > HPD_energy_map, RBX_energy_map, subdet_energy_map;
  // the jet only contains DetIds, so first read recHit collection
  edm::Handle<HBHERecHitCollection> HBHERecHits;
  event.getByLabel( hbheRecHitsColl_, HBHERecHits );
  edm::Handle<HORecHitCollection> HORecHits;
  event.getByLabel( hoRecHitsColl_, HORecHits );
  edm::Handle<HFRecHitCollection> HFRecHits;
  event.getByLabel( hfRecHitsColl_, HFRecHits );
  edm::Handle<EBRecHitCollection> EBRecHits;
  event.getByLabel( ebRecHitsColl_, EBRecHits );
  edm::Handle<EERecHitCollection> EERecHits;
  event.getByLabel( eeRecHitsColl_, EERecHits );
  double totHcalE = 0;
  if( iDbg > 2 ) cout<<"# of rechits found - HBHE: "<<HBHERecHits->size()
		  /*<<", HO: "<<HORecHits->size()*/<<", HF: "<<HFRecHits->size()
		    <<", EB: "<<EBRecHits->size()<<", EE: "<<EERecHits->size()<<endl;

  vector< CaloTowerPtr > towers = jet.getCaloConstituents ();
  int nTowers_ = towers.size();
  if( iDbg > 9 ) cout<<"# of towers found: "<<nTowers_<<endl;

  for( int iTower = 0; iTower <nTowers_ ; iTower++ ) {
    const vector< DetId >& cellIDs = towers[iTower]->constituents();  // cell == recHit
    int nCells = cellIDs.size();
    if( iDbg ) cout<<"tower #"<<iTower<<" has "<<nCells<<" cells. "
		  <<"It's at iEta: "<<towers[iTower]->ieta()<<", iPhi: "<<towers[iTower]->iphi()<<endl;
  
    for( int iCell = 0; iCell < nCells; ++iCell ) {
      DetId::Detector detNum = cellIDs[iCell].det();
      if( detNum == DetId::Hcal ) {
	HcalDetId HcalID = cellIDs[ iCell ];
	HcalSubdetector HcalNum = HcalID.subdet();
	double hitE = 0;
	if( HcalNum == HcalOuter ) {
	  HORecHitCollection::const_iterator theRecHit=HORecHits->find(HcalID);
	  if (theRecHit == HORecHits->end()) {edm::LogWarning("UnexpectedEventContents")<<"Can't find the HO recHit"
											<<" with ID: "<<HcalID; continue;}
	  hitE = theRecHit->energy();
	  HO_energies.push_back( hitE );

	} else if( HcalNum == HcalForward ) {

	  HFRecHitCollection::const_iterator theRecHit=HFRecHits->find( HcalID );	    
	  if( theRecHit == HFRecHits->end() ) {edm::LogWarning("UnexpectedEventContents")<<"Can't find the HF recHit"
											 <<" with ID: "<<HcalID; continue;}
	  hitE = theRecHit->energy();
	  if( iDbg>4 ) cout 
	    << "hit #"<<iCell<<" is  HF , E: "<<hitE<<" iEta: "<<theRecHit->id().ieta()
	    <<", depth: "<<theRecHit->id().depth()<<", iPhi: "
	    <<theRecHit->id().iphi();

	  Hcal_energies.push_back( hitE ); // not clear what is the most useful here. This is a preliminary guess.

	} else { // HBHE

	  HBHERecHitCollection::const_iterator theRecHit = HBHERecHits->find( HcalID );	    
	  if( theRecHit == HBHERecHits->end() ) {edm::LogWarning("UnexpectedEventContents")<<"Can't find the HBHE recHit"
											 <<" with ID: "<<HcalID; continue;}
	  hitE = theRecHit->energy();
	  int iEta = theRecHit->id().ieta();
	  int depth = theRecHit->id().depth();
	  int region = HBHE_region( iEta, depth );
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
	    // 1) in the upper iPhis of the module, the even IEtas belong to the higher iPhi
	    // 2) in the loewr iPhis of the module, the odd  IEtas belong to the higher iPhi
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
	  totHcalE += hitE;

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
	  if( theRecHit == EBRecHits->end() ) {edm::LogWarning("UnexpectedEventContents")<<"Can't find the EB recHit"
											 <<" with ID: "<<EcalID; continue;}
	  hitE = theRecHit->energy();
	} else if(  EcalNum == 2 ){
	  EEDetId EcalID = cellIDs[iCell];
	  EERecHitCollection::const_iterator theRecHit=EERecHits->find(EcalID);
	  if( theRecHit == EERecHits->end() ) {edm::LogWarning("UnexpectedEventContents")<<"Can't find the EE recHit"
											 <<" with ID: "<<EcalID; continue;}
	  hitE = theRecHit->energy();
	}
	if( hitE == 0 ) edm::LogWarning("UnexpectedEventContents")<<"ECal hitE==0? (or unknown subdetector?)";
	if (iDbg > 6) cout<<"EcalNum: "<<EcalNum<<" hitE: "<<hitE<<endl;
	Ecal_energies.push_back (hitE);
      } // 
    } // loop on cells

    if( towers[iTower]->emEnergy() > 0 ) ++nEMTowers; // Note this reject negative HF EM energies
    if( towers[iTower]->hadEnergy() > 0 ) ++nHadTowers;

  } // loop on towers

  /* Disabling check until HO is accounted for. Check was used in CMSSW_2, where HE was excluded.
  double expHcalE = jet.energy() * (1-jet.emEnergyFraction());
  if( totHcalE + expHcalE > 0 && 
      TMath::Abs( totHcalE - expHcalE ) > 0.01 && 
      ( totHcalE - expHcalE ) / ( totHcalE + expHcalE ) > 0.0001 ) {
    edm::LogWarning("CodeAssumptionsViolated")<<"failed to account for all Hcal energies"
					      <<totHcalE<<"!="<<expHcalE;
					      } */   

  // sort the energies
  std::sort( subdet_energies.begin(), subdet_energies.end(), greater<double>() );
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

  //vector<double> recHit_energies (Hcal_energies.begin(), Hcal_energies.end());
  energies.insert( energies.end(), Hcal_energies.begin(), Hcal_energies.end() );
  if( iDbg>7 ) cout<<"DBG # energies "<<energies.size();
  energies.insert( energies.end(), Ecal_energies.begin(), Ecal_energies.end() );
  energies.insert( energies.end(), HO_energies.begin(), HO_energies.end() );
  std::sort( energies.begin(), energies.end(), greater<double>() );
}

