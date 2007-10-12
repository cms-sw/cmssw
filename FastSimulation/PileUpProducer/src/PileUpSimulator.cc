//Framework Headers
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FastSimulation/PileUpProducer/interface/PileUpSimulator.h"
#include "FastSimDataFormats/PileUpEvents/interface/PUEvent.h"
#include "FastSimulation/Event/interface/FSimEvent.h"
#include "FastSimulation/Event/interface/PrimaryVertexGenerator.h"
#include "FastSimulation/Particle/interface/RawParticle.h"
#include "FastSimulation/Utilities/interface/RandomEngine.h"

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
  theFileNames(p.getUntrackedParameter<std::vector<std::string> >("fileNames")),
  inputFile(p.getUntrackedParameter<std::string>("inputFile")),
  theNumberOfFiles(theFileNames.size()),
  theFiles(theNumberOfFiles,static_cast<TFile*>(0)),
  theTrees(theNumberOfFiles,static_cast<TTree*>(0)),
  theBranches(theNumberOfFiles,static_cast<TBranch*>(0)),
  thePUEvents(theNumberOfFiles,static_cast<PUEvent*>(0)),
  theCurrentEntry(theNumberOfFiles,static_cast<unsigned>(0)),
  theCurrentMinBiasEvt(theNumberOfFiles,static_cast<unsigned>(0)),
  theNumberOfEntries(theNumberOfFiles,static_cast<unsigned>(0)),
  theNumberOfMinBiasEvts(theNumberOfFiles,static_cast<unsigned>(0))
  
{
  
  gROOT->cd();
  
  std::string fullPath;
  
  // Read the information from a previous run (to keep reproducibility)
  bool input = this->read(inputFile);

  // Open the file for saving the information of the current run
  myOutputFile.open ("PileUpOutputFile.txt");
  myOutputBuffer = 0;

  // Open the root files
  std::cout << "Opening minimum-bias event files ... " << std::endl;
  for ( unsigned file=0; file<theNumberOfFiles; ++file ) {

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
    // Add some randomness (if there was no input file)
    if ( !input ) 
      theCurrentEntry[file] 
	= (unsigned) (theNumberOfEntries[file] * random->flatShoot());

    /*
    std::cout << "File " << theFileNames[file]
	      << " is opened with " << theNumberOfEntries[file] 
	      << " entries and will be read from Entry/Event "
	      << theCurrentEntry[file] << "/" << theCurrentMinBiasEvt[file]
	      << std::endl;
    */
    theTrees[file]->GetEntry(theCurrentEntry[file]);
    unsigned NMinBias = thePUEvents[file]->nMinBias();
    theNumberOfMinBiasEvts[file] = NMinBias;
    // Add some randomness (if there was no input file)
    if ( !input )
	theCurrentMinBiasEvt[file] = 
	  (unsigned) (theNumberOfMinBiasEvts[file] * random->flatShoot());
    
  }
  
  // Return Loot in the same state as it was when entering. 
  gROOT->cd();
  
}
	       
PileUpSimulator::~PileUpSimulator() {
  
  // Close all local files
  // Among other things, this allows the TROOT destructor to end up 
  // without crashing, while trying to close these files from outside
  std::cout << "Closing minimum-bias event files... " << std::endl;
  for ( unsigned file=0; file<theFiles.size(); ++file ) {
    
    // std::cout << "Closing " << theFileNames[file] << std::endl;
    theFiles[file]->Close();
    
  }
  
  // Close the output file
  myOutputFile.close();
  
  // And return Loot in the same state as it was when entering. 
  gROOT->cd();
  
}

void PileUpSimulator::produce()
{

  //  bool debug = mySimEvent->id().event() >= 621;
  //  if ( debug ) mySimEvent->print();

  // How many pile-up events?
  int PUevts = (int) random->poissonShoot(averageNumber_);

  // Get N events from random files
  for ( int ievt=0; ievt<PUevts; ++ievt ) { 

    
    // Draw a file in a ramdom manner 
    unsigned file = (unsigned) (theNumberOfFiles * random->flatShoot());
    /*
    if ( debug )  
      std::cout << "The file chosen for event " << ievt 
		<< " is the file number " << file << std::endl; 
    */

    // Smear the primary vertex
    mySimEvent->thePrimaryVertexGenerator()->generate();
    XYZTLorentzVector smearedVertex =  
      XYZTLorentzVector(mySimEvent->thePrimaryVertexGenerator()->X(),
			mySimEvent->thePrimaryVertexGenerator()->Y(),
			mySimEvent->thePrimaryVertexGenerator()->Z(),
			0.);
    int mainVertex = mySimEvent->addSimVertex(smearedVertex);

    // Some rotation around the z axis, for more randomness
    XYZVector theAxis(0.,0.,1.);
    double theAngle = random->flatShoot() * 2. * 3.14159265358979323;
    RawParticle::Rotation theRotation(theAxis,theAngle);
    
    /*
    if ( debug ) 
      std::cout << "File chosen : " << file 
		<< " Current entry in this file " << theCurrentEntry[file] 
		<< " Current minbias in this chunk= " << theCurrentMinBiasEvt[file] 
		<< " Total number of minbias in this chunk = " << theNumberOfMinBiasEvts[file] << std::endl;
    */

    //      theFiles[file]->cd();
    //      gDirectory->ls();
    // Check we are not either at the end of a minbias bunch 
    // or at the end of a file
    if ( theCurrentMinBiasEvt[file] == theNumberOfMinBiasEvts[file] ) {
      // if ( debug ) std::cout << "End of MinBias bunch ! ";
      ++theCurrentEntry[file];
      // if ( debug) std::cout << "Read the next entry " << theCurrentEntry[file] << std::endl;
      theCurrentMinBiasEvt[file] = 0;
      if ( theCurrentEntry[file] == theNumberOfEntries[file] ) { 
	theCurrentEntry[file] = 0;
	// if ( debug ) std::cout << "End of file - Rewind! " << std::endl;
      }
      //if ( debug ) std::cout << "The PUEvent is reset ... "; 
      thePUEvents[file]->reset();
      unsigned myEntry = theCurrentEntry[file];
      /* 
      if ( debug ) std::cout << "The new entry " << myEntry 
			     << " is read ... in TTree " << theTrees[file] << " "; 
      */
      theTrees[file]->GetEntry(myEntry);
      /*
      if ( debug ) 
	std::cout << "The number of interactions in the new entry is ... "; 	
      */
      theNumberOfMinBiasEvts[file] = thePUEvents[file]->nMinBias();
      // if ( debug ) std::cout << theNumberOfMinBiasEvts[file] << std::endl;
  }
  
    // Read a minbias event chunk
    const PUEvent::PUMinBiasEvt& aMinBiasEvt 
      = thePUEvents[file]->thePUMinBiasEvts()[theCurrentMinBiasEvt[file]];
  
    // Find corresponding particles
    unsigned firstTrack = aMinBiasEvt.first; 
    unsigned trackSize = firstTrack + aMinBiasEvt.size;
    /*
    if ( debug ) std::cout << "First and last+1 tracks are " 
			   << firstTrack << " " << trackSize << std::endl;
    */

    // Loop on particles
    for ( unsigned iTrack=firstTrack; iTrack<trackSize; ++iTrack ) {
      
      const PUEvent::PUParticle& aParticle 
	= thePUEvents[file]->thePUParticles()[iTrack];
      /* 
      if ( debug) 
	std::cout << "Track " << iTrack 
		  << " id/px/py/pz/mass "
		  << aParticle.id << " " 
		  << aParticle.px << " " 
		  << aParticle.py << " " 
		  << aParticle.pz << " " 
		  << aParticle.mass << " " << std::endl; 
      */
      
      // Create a RawParticle 
      double energy = std::sqrt( aParticle.px*aParticle.px
				 + aParticle.py*aParticle.py
				 + aParticle.pz*aParticle.pz
				 + aParticle.mass*aParticle.mass );
      RawParticle myPart(XYZTLorentzVector(aParticle.px,
					   aParticle.py,
					   aParticle.pz,
					   energy), 
			 smearedVertex);
      myPart.setID(aParticle.id);
      
      // Rotate around the z axis
      myPart.rotate(theRotation);
      
      // Add the particle to the event (with a genpartIndex 
      // indicating the pileup event index)
      mySimEvent->addSimTrack(&myPart,mainVertex,-ievt-2);

    }
    // End of particle loop
    
    // Increment for next time
    ++theCurrentMinBiasEvt[file];
    
  }
  // End of pile-up event loop
  //  if ( debug ) mySimEvent->print();

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

bool
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
    else return false; // Something is wrong with that file !
  
    // Position the pointer just before the last record
    myInputFile.seekg(size-size1-size2);

    // Read the information
    myInputFile.read((char*)(&theCurrentEntry.front()),size1);
    myInputFile.read((char*)&theCurrentMinBiasEvt.front(),size2);
    myInputFile.close();

    return true;

  } 

  return false;

}
