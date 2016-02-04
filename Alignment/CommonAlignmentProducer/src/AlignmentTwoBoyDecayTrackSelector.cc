//Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/EDMException.h"

//DataFormats
#include <DataFormats/TrackReco/interface/Track.h>
#include <DataFormats/METReco/interface/CaloMET.h>
#include <DataFormats/METReco/interface/CaloMETFwd.h>
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

AlignmentTwoBodyDecayTrackSelector::AlignmentTwoBodyDecayTrackSelector(const edm::ParameterSet & cfg) :
  theMissingETSource("met")
{
 LogDebug("Alignment")   << "> applying two body decay Trackfilter ...";
  theMassrangeSwitch = cfg.getParameter<bool>( "applyMassrangeFilter" );
  if (theMassrangeSwitch){
    theMinMass = cfg.getParameter<double>( "minXMass" );
    theMaxMass = cfg.getParameter<double>( "maxXMass" );
    theDaughterMass = cfg.getParameter<double>( "daughterMass" );
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
    theMissingETSource = cfg.getParameter<InputTag>( "missingETSource" );
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
AlignmentTwoBodyDecayTrackSelector::select(const Tracks& tracks, const edm::Event& iEvent) 
{
  Tracks result=tracks;

  if(theMassrangeSwitch){  
    if(theMissingETSwitch)
      result = checkMETMass(result,iEvent); 
    else
      result = checkMass(result); 
  }
  if(theChargeSwitch)
    result = checkCharge(result);
  if(theAcoplanarityFilterSwitch){
    if(theMissingETSwitch)
      result = checkMETAcoplanarity(result,iEvent);
    else
      result = checkAcoplanarity(result);
  }
  LogDebug("Alignment") << ">  TwoBodyDecay tracks all,kept: " << tracks.size() << "," << result.size();
  //  LogDebug("AlignmentTwoBodyDecayTrackSelector")<<">  o kept:";
  //printTracks(result);
  return result;

}

///checks if the mass of the X is in the mass region
AlignmentTwoBodyDecayTrackSelector::Tracks 
AlignmentTwoBodyDecayTrackSelector::checkMass(const Tracks& cands) const
{
  Tracks result;  result.clear();
  //TODO perhaps try combinations if there are more than 2 tracks ....
  if(cands.size() == 2){
    //TODO use other vectors here
    TLorentzVector track0(cands.at(0)->px(),cands.at(0)->py(),cands.at(0)->pz(),
			  sqrt((cands.at(0)->p()*cands.at(0)->p())+theDaughterMass*theDaughterMass));
    TLorentzVector track1(cands.at(1)->px(),cands.at(1)->py(),cands.at(1)->pz(),
			  sqrt((cands.at(1)->p()*cands.at(1)->p())+theDaughterMass*theDaughterMass));
    TLorentzVector mother = track0+track1;
    if(mother.M() > theMinMass && mother.M() < theMaxMass)
      result = cands;
    LogDebug("Alignment") <<">  mass of mother: "<<mother.M()<<"GeV";
  }
  return result;
}

///checks if the mass of the X is in the mass region adding missing E_T
AlignmentTwoBodyDecayTrackSelector::Tracks 
AlignmentTwoBodyDecayTrackSelector::checkMETMass(const Tracks& cands,const edm::Event& iEvent) const
{
  Tracks result;  result.clear();
  if(cands.size() == 1){
    Handle<reco::CaloMETCollection> missingET;
    iEvent.getByLabel(theMissingETSource ,missingET);
    if(missingET.isValid()){
      //TODO use the one with highest pt instead of the first one?
      //      for(reco::CaloMETCollection::const_iterator itMET = missingET->begin(); itMET != missingET->end() ; ++itMET){
      //      cout <<"missingET p = ("<<(*itMET).px()<<","<<(*itMET).py()<<","<<(*itMET).pz()<<")"<<endl;
      //}
      TLorentzVector track(cands.at(0)->px(),cands.at(0)->py(),cands.at(0)->pz(),
			    sqrt((cands.at(0)->p()*cands.at(0)->p())+theDaughterMass*theDaughterMass));
      TLorentzVector met((*missingET).at(0).px(),(*missingET).at(0).py(),(*missingET).at(0).pz(),
			 (*missingET).at(0).p());//ignoring nuetralino masses for now ;)
      TLorentzVector motherSystem = track + met;
      if(motherSystem.M() > theMinMass && motherSystem.M() < theMaxMass)
	result = cands;
      LogDebug("Alignment") <<">  mass of motherSystem: "<<motherSystem.M()<<"GeV";
     }else  
      LogError("Alignment")<<"@SUB=AlignmentTwoBodyDecayTrackSelector::checkMETMass"
			   <<">  could not optain missingET Collection!";
  }
  return cands;
}

///checks if the mother has charge = [theCharge]
AlignmentTwoBodyDecayTrackSelector::Tracks 
AlignmentTwoBodyDecayTrackSelector::checkCharge(const Tracks& cands) const
{
  Tracks result;  result.clear();
  int sumCharge = 0;
  for(Tracks::const_iterator it = cands.begin();it < cands.end();++it)
	sumCharge += (*it)->charge();
  if(theUnsignedSwitch)
    sumCharge = std::abs(sumCharge);
  if(sumCharge == theCharge)
    result = cands;

  return result;
}

///checks if the [cands] are acoplanar (returns empty set if not)
AlignmentTwoBodyDecayTrackSelector::Tracks 
AlignmentTwoBodyDecayTrackSelector::checkAcoplanarity(const Tracks& cands) const
{
  Tracks result;  result.clear();  
  //TODO return the biggest set of acoplanar tracks or two tracks with smallest distance?
  if(cands.size() == 2){
    LogDebug("Alignment") <<">  Acoplanarity: "<<fabs(fabs(deltaPhi(cands.at(0)->phi(),cands.at(1)->phi()))-M_PI)<<endl;
    if(fabs(fabs(deltaPhi(cands.at(0)->phi(),cands.at(1)->phi()))-M_PI)<theAcoplanarDistance) 
      result = cands;
  }  
  return result;
}
///checks if [cands] contains a acoplanar track w.r.t missing ET (returns empty set if not)
AlignmentTwoBodyDecayTrackSelector::Tracks 
AlignmentTwoBodyDecayTrackSelector::checkMETAcoplanarity(const Tracks& cands,const edm::Event& iEvent)const
{
  Tracks result;  result.clear();  
  if(cands.size() == 1){
    Handle<reco::CaloMETCollection> missingET;
    iEvent.getByLabel(theMissingETSource ,missingET);
    if(missingET.isValid()){     
      //TODO return the biggest set of acoplanar tracks or the one with smallest distance?
      LogDebug("Alignment") <<">  METAcoplanarity: "<<fabs(fabs(deltaPhi(cands.at(0)->phi(),(*missingET).at(0).phi()))-M_PI)<<endl;
      if(fabs(fabs(deltaPhi(cands.at(0)->phi(),(*missingET).at(0).phi()))-M_PI)<theAcoplanarDistance)
	result = cands;

     }else  
      LogError("Alignment")<<"@SUB=AlignmentTwoBodyDecayTrackSelector::checkMETAcoplanarity"
			   <<">  could not optain missingET Collection!";
  }
  return result;
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


