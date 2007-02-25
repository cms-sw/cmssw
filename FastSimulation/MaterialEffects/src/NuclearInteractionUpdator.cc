//Framework Headers
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "FastSimulation/MaterialEffects/interface/NuclearInteractionUpdator.h"
#include "FastSimulation/MaterialEffects/interface/NUEvent.h"

#include "CLHEP/Vector/LorentzVector.h"
#include "CLHEP/Vector/Rotation.h"
#include "CLHEP/Units/PhysicalConstants.h"

#include <iostream>
#include <sys/stat.h>
#include <cmath>
#include "TFile.h"
#include "TTree.h"

using namespace std;

NuclearInteractionUpdator::NuclearInteractionUpdator(
  std::vector<std::string>& listOfFiles,
  std::vector<double>& pionEnergies,
  double pionEnergy,
  double lengthRatio,
  std::vector<double> ratioRatio,
  std::string inputFile,
  const RandomEngine* engine) 
  :
  MaterialEffectsUpdator(engine),
  theFileNames(listOfFiles),
  thePionCM(pionEnergies),
  thePionEnergy(pionEnergy),
  theLengthRatio(lengthRatio),
  theRatios(ratioRatio),
  theFiles(theFileNames.size(),static_cast<TFile*>(0)),
  theTrees(theFileNames.size(),static_cast<TTree*>(0)),
  theBranches(theFileNames.size(),static_cast<TBranch*>(0)),
  theNUEvents(theFileNames.size(),static_cast<NUEvent*>(0)),
  theCurrentEntry(theFileNames.size(),static_cast<unsigned>(0)),
  theCurrentInteraction(theFileNames.size(),static_cast<unsigned>(0)),
  theNumberOfEntries(theFileNames.size(),static_cast<unsigned>(0)),
  theNumberOfInteractions(theFileNames.size(),static_cast<unsigned>(0))

{

  gROOT->cd();

  string fullPath;

  // Read the information from a previous run (to keep reproducibility)
  this->read(inputFile);

  // Open the file for saving the information of the current run
  myOutputFile.open ("NuclearInteractionOutputFile.txt");
  myOutputBuffer = 0;

  // Open the root files
  for ( unsigned file=0; file<theFileNames.size(); ++file ) {

    edm::FileInPath myDataFile("FastSimulation/MaterialEffects/data/"+theFileNames[file]);
    fullPath = myDataFile.fullPath();
    //    theFiles[file] = TFile::Open(theFileNames[file].c_str());
    theFiles[file] = TFile::Open(fullPath.c_str());
    if ( !theFiles[file] ) throw cms::Exception("FastSimulation/MaterialEffects") 
      << "File " << theFileNames[file] << " " << fullPath <<  " not found ";
    //
    theTrees[file] = (TTree*) theFiles[file]->Get("NuclearInteractions"); 
    if ( !theTrees[file] ) throw cms::Exception("FastSimulation/MaterialEffects") 
      << "Tree with name NuclearInteractions not found in " << theFileNames[file];
    //
    theBranches[file] = theTrees[file]->GetBranch("nuEvent");
    if ( !theBranches[file] ) throw cms::Exception("FastSimulation/MaterialEffects") 
      << "Branch with name nuEvent not found in " << theFileNames[file];
    //
    theNUEvents[file] = new NUEvent();
    theBranches[file]->SetAddress(&theNUEvents[file]);
    //
    theNumberOfEntries[file] = theTrees[file]->GetEntries();
    std::cout << "File " << theFileNames[file]
	      << " is opened with " << theNumberOfEntries[file] 
	      << " entries and will be read from Entry/Interaction "
	      << theCurrentEntry[file] << "/" << theCurrentInteraction[file]
	      << std::endl;

    theTrees[file]->GetEntry(theCurrentEntry[file]);
    unsigned NInteractions = theNUEvents[file]->nInteractions();
    theNumberOfInteractions[file] = NInteractions;

  }

  if ( theFileNames.size() != thePionCM.size() ||
       theFileNames.size() != theRatios.size() )
    throw cms::Exception("FastSimulation/MaterialEffects") 
      << "There must be as many NU files as CM energies in the cfg file!";


  // Compute the corresponding cm energies of the nuclear interactions
  HepLorentzVector Proton(0.,0.,0.,0.986);
  for ( unsigned file=0; file<thePionCM.size(); ++file ) {

    double thePionMass2 = 0.13957*0.13957;
    double thePionMomentum2 = thePionCM[file]*thePionCM[file] - thePionMass2;
    HepLorentzVector Reference(0.,0.,sqrt(thePionMomentum2),thePionCM[file]);
    thePionCM[file] = (Reference+Proton).mag();

  }

  // Return Loot in the same state as it was when entering. 
  gROOT->cd();


}

NuclearInteractionUpdator::~NuclearInteractionUpdator() {

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

void NuclearInteractionUpdator::compute(ParticlePropagator& Particle)
{

  // Read a Nuclear Interaction in a random manner
  using namespace edm; 

  double pHadron = Particle.momentum().vect().mag(); 

  // The hadron has enough momentum to create some relevant final state
  if ( pHadron > thePionEnergy ) { 

    // Probability to interact is dl/L0 (maximum for 4 GeV pion)
    double aNuclInteraction 
      = -log(random->flatShoot()) / (radLengths * theLengthRatio);
    if ( aNuclInteraction <= 1.0 ) {

      // Find the file with the closest c.m energy
      HepLorentzVector Proton(0.,0.,0.,0.986);
      HepLorentzVector Hadron(Particle.momentum());
      double ecm = (Proton+Hadron).mag();
      // Get the files of interest (closest c.m. energies)
      unsigned file1=0;
      unsigned file2=0;
      double ecm1=1.63; 
      double ecm2=thePionCM[0]; 
      double ratio1=0.;
      double ratio2=theRatios[0];
      if ( ecm>thePionCM[0] && ecm<thePionCM[thePionCM.size()-1] ) {
	for ( unsigned file=1; 
	      file<thePionCM.size()&&ecm>thePionCM[file-1]; 
	      ++file ) {
	  if ( ecm<thePionCM[file] ) { 
	    file2 = file;
	    file1 = file2-1;
	    ecm1 = thePionCM[file1];
	    ecm2 = thePionCM[file2];
	    ratio1 = theRatios[file1];
	    ratio2 = theRatios[file2];
	  } 
	}
      } else if ( ecm>thePionCM[thePionCM.size()-1] ) { 
	file1 = thePionCM.size()-1;
	file2 = thePionCM.size()-2;
	ecm1 = thePionCM[file1];
	ecm2 = thePionCM[file2];
	ratio1 = theRatios[file2];
	ratio2 = theRatios[file2];
      } 
      
      // The inelastic part of the cross section
      double slope = (log10(ecm)-log10(ecm1)) / (log10(ecm2)-log10(ecm1));
      double inelastic = ratio1 + (ratio2-ratio1) * slope;
      //      std::cout << "Energy/Inelastic : " 
      //		<< Hadron.e() << " " << inelastic << std::endl;
      
      // Simulate an inelastic interaction
      if ( aNuclInteraction < inelastic ) { 
	
	// Choice of the file to read according the the log10(ecm) distance
	unsigned file;
	if ( random->flatShoot() < slope )  
	  file = file2;
	else
	  file = file1;
       	//	std::cout << "Pion energy = " << Hadron.e() 
	//		  << "File chosen " << theFileNames[file]
	//		  << std::endl;
	
	// The boost characteristics
	HepLorentzVector theBoost = Proton + Hadron;
	theBoost /= theBoost.e();
	
	// Some rotation arount the boost axis, for more randomness
	Hep3Vector theAxis = theBoost.vect().unit();
	double theAngle = random->flatShoot() * 2. * 3.14159265358979323;
	HepRotation theRotation(theAxis,theAngle);
	//      std::cerr << "File chosen : " << file 
	//		<< " Current interaction = " << theCurrentInteraction[file] 
	//		<< " Total interactions = " << theNumberOfInteractions[file] << std::endl;
	//      theFiles[file]->cd();
	//      gDirectory->ls();
	// Check we are not either at the end of an interaction bunch 
	// or at the end of a file
	if ( theCurrentInteraction[file] == theNumberOfInteractions[file] ) {
	  //	std::cerr << "End of interaction bunch ! ";
	  ++theCurrentEntry[file];
	  //	std::cerr << "Read the next entry " << theCurrentEntry[file] << std::endl;
	  theCurrentInteraction[file] = 0;
	  if ( theCurrentEntry[file] == theNumberOfEntries[file] ) { 
	    theCurrentEntry[file] = 0;
	    //	  std::cerr << "End of file - Rewind! " << std::endl;
	  }
	  //	std::cerr << "The NUEvent is reset ... "; 
	  //	theNUEvents[file]->reset();
	  unsigned myEntry = theCurrentEntry[file];
	  //	std::cerr << "The new entry " << myEntry << " is read ... in TTree " << theTrees[file] << " "; 
	  theTrees[file]->GetEntry(myEntry);
	  //	std::cerr << "The number of interactions in the new entry is ... "; 	
	  theNumberOfInteractions[file] = theNUEvents[file]->nInteractions();
	  //	std::cerr << theNumberOfInteractions[file] << std::endl;
	}
	
	// Read the interaction
	NUEvent::NUInteraction anInteraction 
	  = theNUEvents[file]->theNUInteractions()[theCurrentInteraction[file]];
		
	unsigned firstTrack = anInteraction.first; 
	unsigned lastTrack = anInteraction.last;
	//      std::cerr << "First and last tracks are " << firstTrack << " " << lastTrack << std::endl;
	
	for ( unsigned iTrack=firstTrack; iTrack<=lastTrack; ++iTrack ) {
	  
	  NUEvent::NUParticle aParticle = theNUEvents[file]->theNUParticles()[iTrack];
	  //	std::cerr << "Track " << iTrack 
	  //		  << " id/px/py/pz/mass "
	  //		  << aParticle.id << " " 
	  //		  << aParticle.px << " " 
	  //		  << aParticle.py << " " 
	  //		  << aParticle.pz << " " 
	  //		  << aParticle.mass << " " << endl; 
	  
	  // Create a RawParticle with the proper energy in the c.m frame of 
	  // the nuclear interaction
	  double energy = sqrt( aParticle.px*aParticle.px
			      + aParticle.py*aParticle.py
			      + aParticle.pz*aParticle.pz
			      + aParticle.mass*aParticle.mass/(ecm*ecm) );
	  RawParticle * myPart 
	    = new  RawParticle (aParticle.id,
				HepLorentzVector(aParticle.px,aParticle.py,
						 aParticle.pz,energy)*ecm);
	  
	  // Rotate around the boost axis
	  (*myPart) *= theRotation;
	  
	  // Boost it in the lab frame
	  myPart->boost(theBoost.x(),theBoost.y(),theBoost.z());
	  
	  // Update the daughter list
	  _theUpdatedState.push_back(myPart);
	  
	}
	
	// Increment for next time
	++theCurrentInteraction[file];
	
      } else if ( ecm < 3.3 ) {
	
	// Some code is needed here !
	//	std::cout << "Elastic scattering !" << std::endl;
	
      }
      
    }

  }

}

void
NuclearInteractionUpdator::save() {

  // Size of buffer
  ++myOutputBuffer;

  // Periodically close the current file and open a new one
  if ( myOutputBuffer/1000*1000 == myOutputBuffer ) { 
    myOutputFile.close();
    myOutputFile.open ("NuclearInteractionOutputFile.txt");
    //    myOutputFile.seekp(0); // No need to rewind in that case
  }

  // Save the current position to file
  myOutputFile.write((const char*)(&theCurrentEntry.front()),
		     theCurrentEntry.size()*sizeof(unsigned));
  myOutputFile.write((const char*)&theCurrentInteraction.front(),
		     theCurrentInteraction.size()*sizeof(unsigned));
  myOutputFile.flush();

}

void
NuclearInteractionUpdator::read(std::string inputFile) {

  ifstream myInputFile;
  struct stat results;
  unsigned size1 = theCurrentEntry.size()*sizeof(unsigned);
  unsigned size2 = theCurrentInteraction.size()*sizeof(unsigned);
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
    myInputFile.read((char*)&theCurrentInteraction.front(),size2);
    myInputFile.close();

  }
  
}
