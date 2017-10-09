//Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/InputTag.h"

//DataFormats
#include <DataFormats/TrackReco/interface/Track.h>
#include <DataFormats/METReco/interface/CaloMET.h>
#include <DataFormats/Math/interface/deltaPhi.h>

//STL
#include <math.h>
//ROOT
#include "TLorentzVector.h"

#include "Alignment/CommonAlignmentProducer/interface/AlignmentTwoBodyDecayTrackSelector.h"
//TODO put those namespaces into functions?
using namespace std;
using namespace edm; 
// constructor ----------------------------------------------------------------

AlignmentTwoBodyDecayTrackSelector::AlignmentTwoBodyDecayTrackSelector(const edm::ParameterSet & cfg, edm::ConsumesCollector& iC)
{
 LogDebug("Alignment")   << "> applying two body decay Trackfilter ...";
  theMassrangeSwitch = cfg.getParameter<bool>( "applyMassrangeFilter" );
  if (theMassrangeSwitch){
    theMinMass = cfg.getParameter<double>( "minXMass" );
    theMaxMass = cfg.getParameter<double>( "maxXMass" );
    theDaughterMass = cfg.getParameter<double>( "daughterMass" );
    theCandNumber = cfg.getParameter<unsigned int>( "numberOfCandidates" );//Number of candidates to keep
    secThrBool = cfg.getParameter<bool> ( "applySecThreshold" );
    thesecThr = cfg.getParameter<double>( "secondThreshold" );
    LogDebug("Alignment") << ">  Massrange min,max         :   " << theMinMass   << "," << theMaxMass 
			 << "\n>  Mass of daughter Particle :   " << theDaughterMass;

  }else{
    theMinMass = 0;
    theMaxMass = 0;
    theDaughterMass = 0;
  }
  theChargeSwitch = cfg.getParameter<bool>( "applyChargeFilter" );
  if(theChargeSwitch){
    theCharge = cfg.getParameter<int>( "charge" );
    theUnsignedSwitch = cfg.getParameter<bool>( "useUnsignedCharge" );
    if(theUnsignedSwitch) 
      theCharge=std::abs(theCharge);
    LogDebug("Alignment") << ">  Desired Charge, unsigned: "<<theCharge<<" , "<<theUnsignedSwitch;
  }else{
    theCharge =0;
    theUnsignedSwitch = true;
  }
  theMissingETSwitch = cfg.getParameter<bool>( "applyMissingETFilter" );
  if(theMissingETSwitch){
    edm::InputTag theMissingETSource = cfg.getParameter<InputTag>( "missingETSource" );
    theMissingETToken = iC.consumes<reco::CaloMETCollection>(theMissingETSource);
    LogDebug("Alignment") << ">  missing Et Source: "<< theMissingETSource;
  }
  theAcoplanarityFilterSwitch = cfg.getParameter<bool>( "applyAcoplanarityFilter" );
  if(theAcoplanarityFilterSwitch){
    theAcoplanarDistance = cfg.getParameter<double>( "acoplanarDistance" );
    LogDebug("Alignment") << ">  Acoplanar Distance: "<<theAcoplanarDistance;
  }
  
}

// destructor -----------------------------------------------------------------

AlignmentTwoBodyDecayTrackSelector::~AlignmentTwoBodyDecayTrackSelector()
{}


///returns if any of the Filters is used.
bool AlignmentTwoBodyDecayTrackSelector::useThisFilter()
{
  return theMassrangeSwitch || theChargeSwitch || theAcoplanarityFilterSwitch;
}

// do selection ---------------------------------------------------------------

AlignmentTwoBodyDecayTrackSelector::Tracks 
AlignmentTwoBodyDecayTrackSelector::select(const Tracks& tracks, const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  Tracks result = tracks;

  if (theMassrangeSwitch) {  
    if (theMissingETSwitch)
      result = checkMETMass(result,iEvent); 
    else
      result = checkMass(result); 
  }

  LogDebug("Alignment") << ">  TwoBodyDecay tracks all,kept: " << tracks.size() << "," << result.size();
  return result;
}

template<class T>
struct higherTwoBodyDecayPt : public std::binary_function<T,T,bool>
{
  bool operator()( const T& a, const T& b ) 
  { 
    return a.first > b.first ; 
  }
};

///checks if the mass of the X is in the mass region
AlignmentTwoBodyDecayTrackSelector::Tracks 
AlignmentTwoBodyDecayTrackSelector::checkMass(const Tracks& cands) const
{
  Tracks result;
  
  LogDebug("Alignment") <<">  cands size : "<< cands.size();
  
  if (cands.size()<2) return result;

  TLorentzVector track0;
  TLorentzVector track1;
  TLorentzVector mother;
  typedef pair<const reco::Track*,const reco::Track*> constTrackPair;
  typedef pair<double,constTrackPair> candCollectionItem;
  vector<candCollectionItem> candCollection;
  
  for (unsigned int iCand = 0; iCand < cands.size(); iCand++) {
    
    track0.SetXYZT(cands.at(iCand)->px(),
		   cands.at(iCand)->py(),
		   cands.at(iCand)->pz(),
		   sqrt( cands.at(iCand)->p()*cands.at(iCand)->p() + theDaughterMass*theDaughterMass ));
    
    for (unsigned int jCand = iCand+1; jCand < cands.size(); jCand++) {

      track1.SetXYZT(cands.at(jCand)->px(),
		     cands.at(jCand)->py(),
		     cands.at(jCand)->pz(),
		     sqrt( cands.at(jCand)->p()*cands.at(jCand)->p() + theDaughterMass*theDaughterMass ));
      if (secThrBool==true && track1.Pt() < thesecThr && track0.Pt()< thesecThr) continue;          
      mother = track0 + track1;
      
      const reco::Track *trk1 = cands.at(iCand);
      const reco::Track *trk2 = cands.at(jCand);

      bool correctCharge = true;
      if (theChargeSwitch) correctCharge = this->checkCharge(trk1, trk2);

      bool acoplanarTracks = true;
      if (theAcoplanarityFilterSwitch) acoplanarTracks = this->checkAcoplanarity(trk1, trk2);

      if (mother.M() > theMinMass &&
	  mother.M() < theMaxMass &&
	  correctCharge &&
	  acoplanarTracks) {
	candCollection.push_back(candCollectionItem(mother.Pt(),
						    constTrackPair(trk1, trk2)));
      }
    }
  }

  if (candCollection.size()==0) return result;

  sort(candCollection.begin(), candCollection.end(), 
       higherTwoBodyDecayPt<candCollectionItem>());

  std::map<const reco::Track*,unsigned int> uniqueTrackIndex;
  std::map<const reco::Track*,unsigned int>::iterator it;
  for (unsigned int i=0;
       i<candCollection.size() && i<theCandNumber;
       i++) {
    constTrackPair & trackPair = candCollection[i].second;
    
    it = uniqueTrackIndex.find(trackPair.first);
    if (it==uniqueTrackIndex.end()) {
      result.push_back(trackPair.first);
      uniqueTrackIndex[trackPair.first] = i;
    }
    
    it = uniqueTrackIndex.find(trackPair.second);
    if (it==uniqueTrackIndex.end()) {
      result.push_back(trackPair.second);
      uniqueTrackIndex[trackPair.second] = i;
    }
  }

  return result;
}

///checks if the mass of the X is in the mass region adding missing E_T
AlignmentTwoBodyDecayTrackSelector::Tracks 
AlignmentTwoBodyDecayTrackSelector::checkMETMass(const Tracks& cands,const edm::Event& iEvent) const
{
  Tracks result;
  
  LogDebug("Alignment") <<">  cands size : "<< cands.size();
  
  if (cands.size()==0) return result;

  TLorentzVector track;
  TLorentzVector met4;
  TLorentzVector mother;

  Handle<reco::CaloMETCollection> missingET;
  iEvent.getByToken(theMissingETToken ,missingET);
  if (!missingET.isValid()) {
    LogError("Alignment")<< "@SUB=AlignmentTwoBodyDecayTrackSelector::checkMETMass"
			 << ">  could not optain missingET Collection!";
    return result;
  }

  typedef pair<double,const reco::Track*> candCollectionItem;
  vector<candCollectionItem> candCollection;

  for (reco::CaloMETCollection::const_iterator itMET = missingET->begin();
       itMET != missingET->end();
       ++itMET) {
    
    met4.SetXYZT((*itMET).px(),
		 (*itMET).py(),
		 (*itMET).pz(),
		 (*itMET).p());
  
    for (unsigned int iCand = 0; iCand < cands.size(); iCand++) {
    
      track.SetXYZT(cands.at(iCand)->px(),
		    cands.at(iCand)->py(),
		    cands.at(iCand)->pz(),
		    sqrt( cands.at(iCand)->p()*cands.at(iCand)->p() + theDaughterMass*theDaughterMass ));
      
      mother = track + met4;
      
      const reco::Track *trk = cands.at(iCand);
      const reco::CaloMET *met = &(*itMET);

      bool correctCharge = true;
      if (theChargeSwitch) correctCharge = this->checkCharge(trk);

      bool acoplanarTracks = true;
      if (theAcoplanarityFilterSwitch) acoplanarTracks = this->checkMETAcoplanarity(trk, met);

      if (mother.M() > theMinMass &&
	  mother.M() < theMaxMass &&
	  correctCharge &&
	  acoplanarTracks) {
	candCollection.push_back(candCollectionItem(mother.Pt(), trk));
      }
    }
  }

  if (candCollection.size()==0) return result;

  sort(candCollection.begin(), candCollection.end(), 
       higherTwoBodyDecayPt<candCollectionItem>());
  
  std::map<const reco::Track*,unsigned int> uniqueTrackIndex;
  std::map<const reco::Track*,unsigned int>::iterator it;
  for (unsigned int i=0;
       i<candCollection.size() && i<theCandNumber;
       i++) {
    it = uniqueTrackIndex.find(candCollection[i].second);
    if (it==uniqueTrackIndex.end()) {
      result.push_back(candCollection[i].second);
      uniqueTrackIndex[candCollection[i].second] = i;
    }
  }

  return result;
}

///checks if the mother has charge = [theCharge]
bool
AlignmentTwoBodyDecayTrackSelector::checkCharge(const reco::Track* trk1, const reco::Track* trk2)const
{
  int sumCharge = trk1->charge();
  if (trk2) sumCharge += trk2->charge();
  if (theUnsignedSwitch) sumCharge = std::abs(sumCharge);
  if (sumCharge == theCharge) return true;
  return false;
}

///checks if the [cands] are acoplanar (returns empty set if not)
bool
AlignmentTwoBodyDecayTrackSelector::checkAcoplanarity(const reco::Track* trk1, const reco::Track* trk2)const
{
  if (fabs(deltaPhi(trk1->phi(),trk2->phi()-M_PI)) < theAcoplanarDistance) return true;
  return false;
}

///checks if the [cands] are acoplanar (returns empty set if not)
bool
AlignmentTwoBodyDecayTrackSelector::checkMETAcoplanarity(const reco::Track* trk1, const reco::CaloMET* met)const
{
  if (fabs(deltaPhi(trk1->phi(),met->phi()-M_PI)) < theAcoplanarDistance) return true;
  return false;
}

//===================HELPERS===================

///print Information on Track-Collection
void AlignmentTwoBodyDecayTrackSelector::printTracks(const Tracks& col) const
{
  int count = 0;
  LogDebug("Alignment") << ">......................................";
  for(Tracks::const_iterator it = col.begin();it < col.end();++it,++count){
    LogDebug("Alignment") 
      <<">  Track No. "<< count <<": p = ("<<(*it)->px()<<","<<(*it)->py()<<","<<(*it)->pz()<<")\n"
      <<">                        pT = "<<(*it)->pt()<<" eta = "<<(*it)->eta()<<" charge = "<<(*it)->charge();    
  }
  LogDebug("Alignment") << ">......................................";
}


