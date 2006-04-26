/*----------------------------------------------------------------------
$Id: PUSource.cc,v 1.1 2006/04/24 17:02:16 pjanot Exp $
----------------------------------------------------------------------*/

#include "FastSimulation/PileUpProducer/interface/PUSource.h"
#include "IOPool/Input/src/RootFile.h"
#include "IOPool/Common/interface/ClassFiller.h"

#include "DataFormats/Common/interface/BranchDescription.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "DataFormats/Common/interface/ProductRegistry.h"
#include "DataFormats/Common/interface/ProductID.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"


namespace edm {

PUSource::PUSource(ParameterSet const& pset, InputSourceDescription const& desc) :
  VectorInputSource(pset, desc),
  rootFile_(),
  rootFiles_(),
  eventsInRootFiles(),
  totalNbEvents(0)
{
  ClassFiller();
  init();
}
  
void 
PUSource::init() {
  
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
    
    std::cout << "MinBias event file " << *fileIter << " open." << std::endl;
    // If not, store it. 
    // FIX : Shouldn't we check that the file indeed exists?
    if (it == rootFiles_.end()) {
      
      // Set the shared_ptr of the RooFile
      rootFile_ = RootFileSharedPtr(new RootFile(*fileIter, catalog().url()));

      // make sure the new product registry is identical to the old one (if any)
      //      if ( pReg && *pReg != rootFile_->productRegistry()) {
      //	throw cms::Exception("MismatchedInput","PoolSource::init()")
      //	  << "File " << *fileIter 
      //	  << "\nhas different product registry than previous files\n";
      //      }
      
      // Update the map of stored files
      rootFiles_[*fileIter] = rootFile_;
      // Update the total number of event
      totalNbEvents += rootFile_->entries();
      std::cout << " Number of events : " << totalNbEvents << std::endl;
      eventsInRootFiles[rootFile_] = totalNbEvents;
      
      // save the product registry from the current file, temporarily
      //      pReg = rootFile_->productRegistrySharedPtr();
      //      rootFile_ = it->second;
      //      rootFile_->setEntryNumber(-1);

      
    }
  }

}

PUSource::~PUSource() {

  // close the files
  RootFileMap::iterator it;
  for (   it = rootFiles_.begin(); it != rootFiles_.end(); ++it ) {
    it->second.reset();
  }

}

std::auto_ptr<EventPrincipal> 
PUSource::read() {
    
  return rootFile_->read(rootFile_->productRegistry()); 
  
}
  
  
std::auto_ptr<EventPrincipal> 
PUSource::readIt(int entry) {
  
  // The first file in the list
  std::map<boost::shared_ptr<RootFile>,int>::const_iterator 
    it = eventsInRootFiles.begin();
  rootFile_ = it->first;

  // Check that the entry is not beyond the total number of events 
  entry -= entry/totalNbEvents*totalNbEvents;
  int localEntry = entry;
 
  // Loop over the files to find the corresponding entry
  while ( entry > it->second ) {
    localEntry -= rootFile_->entries();
    ++it;
    rootFile_ = it->first;
  }

  // Set the entry number
  rootFile_->setEntryNumber(localEntry);

  // Return the event
  return read();

}


// Warning - this readMany_ reads only one entry !
// "number" just sets the location of this entry.
void
PUSource::readMany_(int number, EventPrincipalVector& result) {

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
