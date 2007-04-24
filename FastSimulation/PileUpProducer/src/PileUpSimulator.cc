//Framework Headers
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "FastSimulation/PileUpProducer/interface/PileUpSimulator.h"
#include "FastSimulation/PileUpProducer/interface/PUEvent.h"
#include "FastSimulation/Event/interface/FSimEvent.h"
#include "FastSimulation/Particle/interface/RawParticle.h"
#include "FastSimulation/Utilities/interface/RandomEngine.h"

#include "CLHEP/Vector/LorentzVector.h"
#include "CLHEP/Vector/Rotation.h"
#include "CLHEP/Units/PhysicalConstants.h"

#include <iostream>
#include <sys/stat.h>
#include <cmath>
#include <string>
#include "TFile.h"
#include "TTree.h"

PileUpSimulator::PileUpSimulator(FSimEvent* aSimEvent, 
				 edm::ParameterSet const & p,  
				 const RandomEngine* engine) :
  averageNumber_(p.getParameter<double>("averageNumber")),
  mySimEvent(aSimEvent),
  random(engine),
  theFileNames(p.getParameter<std::vector<std::string> >("fileNames")),
  inputFile(p.getParameter<std::string>("inputFile")),
  theFiles(theFileNames.size(),static_cast<TFile*>(0)),
  theTrees(theFileNames.size(),static_cast<TTree*>(0)),
  theBranches(theFileNames.size(),static_cast<TBranch*>(0)),
  thePUEvents(theFileNames.size(),static_cast<PUEvent*>(0)),
  theCurrentEntry(theFileNames.size(),static_cast<unsigned>(0)),
  theCurrentMinBiasEvt(theFileNames.size(),static_cast<unsigned>(0)),
  theNumberOfEntries(theFileNames.size(),static_cast<unsigned>(0)),
  theNumberOfMinBiasEvts(theFileNames.size(),static_cast<unsigned>(0))
  
{
  
  gROOT->cd();
  
  std::string fullPath;
  
  // Read the information from a previous run (to keep reproducibility)
  this->read(inputFile);

  // Open the file for saving the information of the current run
  myOutputFile.open ("PileUpOutputFile.txt");
  myOutputBuffer = 0;

  // Open the root files
  for ( unsigned file=0; file<theFileNames.size(); ++file ) {

    edm::FileInPath myDataFile("FastSimulation/PileUpProducer/data/"+theFileNames[file]);
    fullPath = myDataFile.fullPath();
    //    theFiles[file] = TFile::Open(theFileNames[file].c_str());
    theFiles[file] = TFile::Open(fullPath.c_str());
    if ( !theFiles[file] ) throw cms::Exception("FastSimulation/PileUpProducer") 
      << "File " << theFileNames[file] << " " << fullPath <<  " not found ";
    //
    theTrees[file] = (TTree*) theFiles[file]->Get("MinBiasEvents"); 
    if ( !theTrees[file] ) throw cms::Exception("FastSimulation/PileUpProducer") 
      << "Tree with name MinBiasEvents not found in " << theFileNames[file];
    //
    theBranches[file] = theTrees[file]->GetBranch("puEvent");
    if ( !theBranches[file] ) throw cms::Exception("FastSimulation/PileUpProducer") 
      << "Branch with name puEvent not found in " << theFileNames[file];
    //
    thePUEvents[file] = new PUEvent();
    theBranches[file]->SetAddress(&thePUEvents[file]);
    //
    theNumberOfEntries[file] = theTrees[file]->GetEntries();
    std::cout << "File " << theFileNames[file]
	      << " is opened with " << theNumberOfEntries[file] 
	      << " entries and will be read from Entry/Event "
	      << theCurrentEntry[file] << "/" << theCurrentMinBiasEvt[file]
	      << std::endl;

    theTrees[file]->GetEntry(theCurrentEntry[file]);
    unsigned NMinBias = thePUEvents[file]->nMinBias();
    theNumberOfMinBiasEvts[file] = NMinBias;
    
  }
  
  // Return Loot in the same state as it was when entering. 
  gROOT->cd();
  
}
	       
PileUpSimulator::~PileUpSimulator() {
  
  // Close all local files
  // Among other things, this allows the TROOT destructor to end up 
  // without crashing, while trying to close these files from outside
  for ( unsigned file=0; file<theFiles.size(); ++file ) {
    
    std::cout << "Closing " << theFileNames[file] << std::endl;
    theFiles[file]->Close();
    
  }
  
  // Close the output file
  myOutputFile.close();
  
  // And return Loot in the same state as it was when entering. 
  gROOT->cd();
  
}

void PileUpSimulator::produce()
{

  // Draw a file in a ramdom manner 
  unsigned file = 1;

  // Some rotation around the z axis, for more randomness
  Hep3Vector theAxis(0.,0.,1.);
  double theAngle = random->flatShoot() * 2. * 3.14159265358979323;
  HepRotation theRotation(theAxis,theAngle);

  //      std::cerr << "File chosen : " << file 
  //		<< " Current interaction = " << theCurrentMinBiasEvt[file] 
  //		<< " Total interactions = " << theNumberOfMinBiasEvts[file] << std::endl;
  //      theFiles[file]->cd();
  //      gDirectory->ls();
  // Check we are not either at the end of an interaction bunch 
  // or at the end of a file
  if ( theCurrentMinBiasEvt[file] == theNumberOfMinBiasEvts[file] ) {
    //	std::cerr << "End of interaction bunch ! ";
    ++theCurrentEntry[file];
    //	std::cerr << "Read the next entry " << theCurrentEntry[file] << std::endl;
    theCurrentMinBiasEvt[file] = 0;
    if ( theCurrentEntry[file] == theNumberOfEntries[file] ) { 
      theCurrentEntry[file] = 0;
      //	  std::cerr << "End of file - Rewind! " << std::endl;
    }
    //	std::cerr << "The PUEvent is reset ... "; 
    //	thePUEvents[file]->reset();
    unsigned myEntry = theCurrentEntry[file];
    //	std::cerr << "The new entry " << myEntry << " is read ... in TTree " << theTrees[file] << " "; 
    theTrees[file]->GetEntry(myEntry);
    //	std::cerr << "The number of interactions in the new entry is ... "; 	
    theNumberOfMinBiasEvts[file] = thePUEvents[file]->nMinBias();
    //	std::cerr << theNumberOfMinBiasEvts[file] << std::endl;
  }
  
  // Read the interaction
  PUEvent::PUMinBiasEvt aMinBiasEvt 
    = thePUEvents[file]->thePUMinBiasEvts()[theCurrentMinBiasEvt[file]];
  
  unsigned firstTrack = aMinBiasEvt.first; 
  unsigned lastTrack = firstTrack + aMinBiasEvt.size;
  //      std::cerr << "First and last tracks are " << firstTrack << " " << lastTrack << std::endl;
  
  for ( unsigned iTrack=firstTrack; iTrack<lastTrack; ++iTrack ) {
    
    PUEvent::PUParticle aParticle = thePUEvents[file]->thePUParticles()[iTrack];
    //	std::cerr << "Track " << iTrack 
    //		  << " id/px/py/pz/mass "
    //		  << aParticle.id << " " 
    //		  << aParticle.px << " " 
    //		  << aParticle.py << " " 
    //		  << aParticle.pz << " " 
    //		  << aParticle.mass << " " << endl; 
    
    // Create a RawParticle with the proper energy in the c.m frame of 
    // the nuclear interaction
    double energy = std::sqrt( aParticle.px*aParticle.px
			       + aParticle.py*aParticle.py
			       + aParticle.pz*aParticle.pz
			       + aParticle.mass*aParticle.mass );
    RawParticle * myPart 
      = new  RawParticle (aParticle.id,
			  HepLorentzVector(aParticle.px,aParticle.py,
					   aParticle.pz,energy));
    
    // Rotate around the boost axis
    (*myPart) *= theRotation;
    
  }
  
  // Increment for next time
  ++theCurrentMinBiasEvt[file];
  
}

void
PileUpSimulator::save() {

  // Size of buffer
  ++myOutputBuffer;

  // Periodically close the current file and open a new one
  if ( myOutputBuffer/1000*1000 == myOutputBuffer ) { 
    myOutputFile.close();
    myOutputFile.open ("PileUpOutputFile.txt");
    //    myOutputFile.seekp(0); // No need to rewind in that case
  }

  // Save the current position to file
  myOutputFile.write((const char*)(&theCurrentEntry.front()),
		     theCurrentEntry.size()*sizeof(unsigned));
  myOutputFile.write((const char*)&theCurrentMinBiasEvt.front(),
		     theCurrentMinBiasEvt.size()*sizeof(unsigned));
  myOutputFile.flush();

}

void
PileUpSimulator::read(std::string inputFile) {

  ifstream myInputFile;
  struct stat results;
  unsigned size1 = theCurrentEntry.size()*sizeof(unsigned);
  unsigned size2 = theCurrentMinBiasEvt.size()*sizeof(unsigned);
  unsigned size = 0;


  // Open the file (if any)
  myInputFile.open (inputFile.c_str());
  if ( myInputFile.is_open() ) { 

    // Get the size of the file
    if ( stat(inputFile.c_str(), &results) == 0 ) size = results.st_size;
    else return; // Something is wrong with that file !
  
    // Position the pointer just before the last record
    myInputFile.seekg(size-size1-size2);

    // Read the information
    myInputFile.read((char*)(&theCurrentEntry.front()),size1);
    myInputFile.read((char*)&theCurrentMinBiasEvt.front(),size2);
    myInputFile.close();

  }
  
}
