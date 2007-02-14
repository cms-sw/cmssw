/*----------------------------------------------------------------------
$Id: NUSource.cc,v 1.2 2006/04/26 13:02:07 pjanot Exp $
----------------------------------------------------------------------*/

#include "FastSimulation/MaterialEffects/interface/NUSource.h"
#include "IOPool/Input/src/RootFile.h"
#include "IOPool/Common/interface/ClassFiller.h"

#include "DataFormats/Common/interface/BranchDescription.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "DataFormats/Common/interface/ProductRegistry.h"
#include "DataFormats/Common/interface/ProductID.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"


namespace edm {

NUSource::NUSource(ParameterSet const& pset, InputSourceDescription const& desc) :
  VectorInputSource(pset, desc),
  rootFile_(),
  rootFiles_(),
  eventsInRootFiles(),
  totalNbEvents(0),
  localEntry(0)
{
  ClassFiller();
  init();
}
  
void 
NUSource::init() {
  
  // For the moment, we keep all old files open.
  // FIX: We will need to limit the number of open files.
  // rootFiles[file]_.reset();
  
  // To save the product registry from the current file
  boost::shared_ptr<ProductRegistry const> pReg; 
  RootFileMap::const_iterator it;
  
  // Loop over the files 
  for ( std::vector<std::string>::const_iterator 
 	  fileIter = fileNames().begin();
	  fileIter != fileNames().end();
	++fileIter ) {
    
    // Check if the file has already been mentioned and stored
    it = rootFiles_.find(*fileIter);
    
    std::cout << "Nuclear interaction event file " << *fileIter << " open." << std::endl;
    // If not, store it. 
    // FIX : Shouldn't we check that the file indeed exists?
    if (it == rootFiles_.end()) {
      
      // Set the shared_ptr of the RooFile
      rootFile_ = RootFileSharedPtr(new RootFile(*fileIter, catalog().url()));

      // Update the map of stored files
      rootFiles_[*fileIter] = rootFile_;

      // Update the total number of event
      totalNbEvents = rootFile_->entries();
      eventsInRootFiles[rootFile_] = totalNbEvents;
      
    }

  }

}

NUSource::~NUSource() {

  // close the files
  RootFileMap::iterator it;
  for (   it = rootFiles_.begin(); it != rootFiles_.end(); ++it ) {
    it->second.reset();
  }

}

std::auto_ptr<EventPrincipal> 
NUSource::read() {
    
  return rootFile_->read(rootFile_->productRegistry()); 
  
}
  
  
std::auto_ptr<EventPrincipal> 
NUSource::readIt(int entry) {
  
  // The first file in the list
  std::map<boost::shared_ptr<RootFile>,int>::const_iterator 
    it = eventsInRootFiles.begin();
  rootFile_ = it->first;
  
  // The file to be read
  int file = (int) (entry/1E8);
  int nbEvents = 0;
  for ( int ifile=0; ifile<=file && it != eventsInRootFiles.end(); ++ifile ) {
    rootFile_ = it->first;
    nbEvents =  it->second;
    ++it;
  }

  //  localEntry = (int) (nbEvents * (entry-file*1E8) / 1E8);

  // Set the entry number
  rootFile_->setEntryNumber(localEntry-localEntry/nbEvents*nbEvents);
  ++localEntry;

  // Return the event
  return read();

}


// Warning - this readMany_ reads only one entry !
// "number" just sets the location of this entry.
void
NUSource::readMany_(int number, EventPrincipalVector& result) {

  // Read that entry
  std::auto_ptr<EventPrincipal> ev = readIt(number);
  if (ev.get() == 0) {
    return;
  }

  // Return the event
  EventPrincipalVectorElement e(ev.release());
  result.push_back(e);
}

// end namespace edm
}
