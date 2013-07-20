#include "RecoMuon/MuonSeedGenerator/plugins/MuonSeedMerger.h"

/** \class MuonSeedMerger
 *  Module to merge two or more muon seed collections.
 *  Currently it does not contain any seed cleaner, so the number of ghosts seed
 *  can be high.
 *  This is still a preliminary implementation.
 *
 *  $Date: 2008/10/17 22:36:09 $
 *  $Revision: 1.1 $
 *  \author R. Bellan - CERN <riccardo.bellan@cern.ch>
 */

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Data Formats 
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/Common/interface/View.h"

using namespace std;
using namespace edm;

MuonSeedMerger::MuonSeedMerger(const ParameterSet& parameterSet){
  const string metname = "Muon|RecoMuon|MuonSeedMerger";  

  theSeedCollectionLabels = parameterSet.getParameter<vector<InputTag> >("SeedCollections");

  LogTrace(metname)<< "MuonSeedMerger will Merge the following seed collections:";
  for(vector<InputTag>::const_iterator label = theSeedCollectionLabels.begin();
      label != theSeedCollectionLabels.end(); ++label)
    LogTrace(metname) << *label;
  

  produces<TrajectorySeedCollection>();
}


MuonSeedMerger:: ~MuonSeedMerger(){}

void MuonSeedMerger::produce(Event& event, const EventSetup& eventSetup){
  
  const string metname = "Muon|RecoMuon|MuonSeedMerger";  

  std::auto_ptr<TrajectorySeedCollection> output(new TrajectorySeedCollection());
  
  Handle<View<TrajectorySeed> > seeds; 
  
  for(vector<InputTag>::const_iterator label = theSeedCollectionLabels.begin();
      label != theSeedCollectionLabels.end(); ++label){ 
    event.getByLabel(*label,seeds);
    LogTrace(metname) << *label << " has "<< seeds->size() <<" seeds";
    for(View<TrajectorySeed>::const_iterator seed = seeds->begin(); seed != seeds->end(); ++seed)
      output->push_back(*seed);
  }
  
  event.put(output);
}
