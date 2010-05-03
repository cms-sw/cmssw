/** \class HLTMuonOverlap
 *
 *  \author M. Vander Donckt  (copied from J. Alcaraz)
 */
//
#include "DQMOffline/Trigger/interface/HLTMuonOverlap.h"

// Collaborating Class Header
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefToBase.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


HLTMuonOverlap::HLTMuonOverlap(const edm::ParameterSet& pset)
{
  TrigResultsIn=true;
  Ntp = 0;
  Nall_trig = Nevents = 0;
  theCrossSection = pset.getParameter<double>("CrossSection");
  // Convert it already into /nb/s)
  theLuminosity = pset.getUntrackedParameter<double>("Luminosity",1.e+32)*1.e-33;
  TrigResLabel_=pset.getParameter<edm::InputTag>("TriggerResultLabel");
  init_=false;
  size=0;
}



void HLTMuonOverlap::begin() {
}



void HLTMuonOverlap::analyze(const edm::Event & event ) {
  using namespace edm;  

  if (!TrigResultsIn)return;
  ++Nevents;
  
  Handle<TriggerResults> trigRes;
  event.getByLabel(TrigResLabel_, trigRes);
  if (!trigRes.isValid()){
  edm::InputTag triggerResultsLabelFU(TrigResLabel_.label(),TrigResLabel_.instance(), "FU");
  event.getByLabel(triggerResultsLabelFU,trigRes);
  if(!trigRes.isValid()) {
    LogWarning("HLTMuonVal")<< "No trigger Results";
    TrigResultsIn=false;
    // Do nothing
    return;
   }
  }
  size=trigRes->size();
  LogTrace("HLTMuonVal")<< "Ntp="<<Ntp<<" Size of trigger results="<<size;
  const edm::TriggerNames & triggerNames = event.triggerNames(*trigRes);

  if(Ntp)
    assert(Ntp == size);
  else
    Ntp = size;
  
  // has any trigger fired this event?
  if(trigRes->accept())++Nall_trig;
  else return;
  LogTrace("HLTMuonVal")<<" This event has fired ";
  // loop over all paths, get trigger decision
  for(unsigned i = 0; i != size; ++i)
    {
      std::string name = triggerNames.triggerName(i);
      LogTrace("HLTMuonVal") << name << " has decision "<<trigRes->accept(i);
      fired[name] = trigRes->accept(i);
      if(fired[name]) ++(Ntrig[name]);
    }
  
  // NOTE: WE SHOULD MAKE THIS A SYMMETRIC MATRIX...
  // double-loop over all paths, get trigger overlaps
  for(unsigned i = 0; i != size; ++i)
    {
      std::string name = triggerNames.triggerName(i);
      if(!fired[name])continue;
      
      bool correlation = false;
      
      for(unsigned j = 0; j != size; ++j)
	{
	  // skip the same name; 
	  // this entry correponds to events triggered by single trigger
	  if(i == j) continue;
	  std::string name2 = triggerNames.triggerName(j);
	  if(fired[name2])
	    {
	      correlation = true;
	      ++(Ncross[name][name2]);
	    }
	} // loop over j-trigger paths
      
      if(!correlation) // events triggered by single trigger
	++(Ncross[name][name]);
      
    } //  // loop over i-trigger paths
  
}


void HLTMuonOverlap::finish()
{
  using namespace edm;
  if (!TrigResultsIn || Nevents == 0 )return;
  LogVerbatim("HLTMuonVal") << " Total trigger fraction: " << Nall_trig << "/" << Nevents
			    << " events (" << 100*Nall_trig/Nevents<<"%), the Rate="<< Nall_trig*theLuminosity*theCrossSection/Nevents << "Hz ";
  
  LogVerbatim("HLTMuonVal") << " Individual path rates: " ;
  typedef trigPath::iterator It;
  int ix = 1;
  for(It it = Ntrig.begin(); 
      it != Ntrig.end(); ++it, ++ix)
    {
      LogVerbatim("HLTMuonVal") << " Trigger path \"" << it->first << "\": " 
				<< it->second << "/"
				<< Nevents << " events, Rate=" << (it->second)*theLuminosity*theCrossSection/Nevents 
				<< "Hz " ;
    }
    
  LogVerbatim("HLTMuonVal") << " Trigger path correlations: " ;
  typedef std::map<std::string, trigPath>::iterator IIt;
    
  ix = 1;
  for(IIt it = Ncross.begin(); 
      it != Ncross.end(); ++it, ++ix)
    { // loop over first trigger of pair
	
      trigPath & cross = it->second;
      int iy = 1;
      for(It it2 = cross.begin(); 
	  it2 != cross.end(); ++it2, ++iy)
	{ // loop over second trigger of pair
	  // skip symmetric pairs: 1st pass does "path1", "path2";
	  // 2nd pass should skip "path2", "path1"
	  if(it->first > it2->first)continue;

	  // if second trigger = first trigger, 
	  // this corresponds to unique rate (ie. no correlation)
	  if(it->first == it2->first)
	    LogVerbatim("HLTMuonVal") << " \"" << it->first << "\"" 
				      << " (unique rate): "<< it2->second << "/"
				      << Nevents << " events, Rate=" 
				      << (it2->second)*theLuminosity*theCrossSection/Nevents 
				      << "Hz ";
	  else
	    LogVerbatim("HLTMuonVal") << " \"" << it->first << "\""<< " x \"" 
				      << it2->first << "\": "<< it2->second 
				      << "/"<< Nevents << " events, Rate=" 
				      << (it2->second)*theLuminosity*theCrossSection/Nevents 
				      << "Hz ";

	}
    }

}
