//Framework Headers
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "FastSimulation/MaterialEffects/interface/NuclearInteractionSimulator.h"
#include "FastSimulation/Utilities/interface/RandomEngineAndDistribution.h"

#include "FastSimDataFormats/NuclearInteractions/interface/NUEvent.h"

//#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
//#include "FWCore/ServiceRegistry/interface/Service.h"

#include <iostream>
#include <sys/stat.h>
#include <cmath>
#include "TFile.h"
#include "TTree.h"
#include "TROOT.h"

// Internal variable and vectors with name started frm "thePion" means
// vectors/variable not only for pions but for all type of hadrons
// treated inside this code

NuclearInteractionSimulator::NuclearInteractionSimulator(
  std::vector<double>& hadronEnergies,
  std::vector<int>& hadronTypes,
  std::vector<std::string>& hadronNames,
  std::vector<double>& hadronMasses,
  std::vector<double>& hadronPMin,
  double pionEnergy,
  std::vector<double>& lengthRatio,
  std::vector< std::vector<double> >& ratios,
  std::map<int,int >& idMap,
  std::string inputFile,
  unsigned int distAlgo,
  double distCut)
  :
  MaterialEffectsSimulator(),
  thePionEN(hadronEnergies),
  thePionID(hadronTypes),
  thePionNA(hadronNames),
  thePionMA(hadronMasses),
  thePionPMin(hadronPMin),
  thePionEnergy(pionEnergy),
  theLengthRatio(lengthRatio),
  theRatios(ratios),
  theIDMap(idMap),
  theDistAlgo(distAlgo),
  theDistCut(distCut),
  currentValuesWereSet(false)
{
  std::string fullPath;

  // Prepare the map of files
  // Loop over the particle names
  std::vector<TFile*> aVFile(thePionEN.size(),static_cast<TFile*>(0));
  std::vector<TTree*> aVTree(thePionEN.size(),static_cast<TTree*>(0));
  std::vector<TBranch*> aVBranch(thePionEN.size(),static_cast<TBranch*>(0));
  std::vector<NUEvent*> aVNUEvents(thePionEN.size(),static_cast<NUEvent*>(0));
  std::vector<unsigned> aVCurrentEntry(thePionEN.size(),static_cast<unsigned>(0));
  std::vector<unsigned> aVCurrentInteraction(thePionEN.size(),static_cast<unsigned>(0));
  std::vector<unsigned> aVNumberOfEntries(thePionEN.size(),static_cast<unsigned>(0));
  std::vector<unsigned> aVNumberOfInteractions(thePionEN.size(),static_cast<unsigned>(0));
  std::vector<std::string> aVFileName(thePionEN.size(),static_cast<std::string>(""));
  std::vector<double> aVPionCM(thePionEN.size(),static_cast<double>(0));
  theFiles.resize(thePionNA.size());
  theTrees.resize(thePionNA.size());
  theBranches.resize(thePionNA.size());
  theNUEvents.resize(thePionNA.size());
  theCurrentEntry.resize(thePionNA.size());
  theCurrentInteraction.resize(thePionNA.size());
  theNumberOfEntries.resize(thePionNA.size());
  theNumberOfInteractions.resize(thePionNA.size());
  theFileNames.resize(thePionNA.size());
  thePionCM.resize(thePionNA.size());
  for ( unsigned iname=0; iname<thePionNA.size(); ++iname ) { 
    theFiles[iname] = aVFile;
    theTrees[iname] = aVTree;
    theBranches[iname] = aVBranch;
    theNUEvents[iname] = aVNUEvents;
    theCurrentEntry[iname] = aVCurrentEntry;
    theCurrentInteraction[iname] = aVCurrentInteraction;
    theNumberOfEntries[iname] = aVNumberOfEntries;
    theNumberOfInteractions[iname] = aVNumberOfInteractions;
    theFileNames[iname] = aVFileName;
    thePionCM[iname] = aVPionCM;
  } 

  // Read the information from a previous run (to keep reproducibility)
  currentValuesWereSet = this->read(inputFile);
  if ( currentValuesWereSet )
    std::cout << "***WARNING*** You are reading nuclear-interaction information from the file "
	      << inputFile << " created in an earlier run."
	      << std::endl;

  // Open the file for saving the information of the current run
  myOutputFile.open ("NuclearInteractionOutputFile.txt");
  myOutputBuffer = 0;


  // Open the root files
  //  for ( unsigned file=0; file<theFileNames.size(); ++file ) {
  unsigned fileNb = 0;
  for ( unsigned iname=0; iname<thePionNA.size(); ++iname ) {
    for ( unsigned iene=0; iene<thePionEN.size(); ++iene ) {
      //std::cout << "iname/iene " << iname << " " << iene << std::endl; 
      std::ostringstream filename;
      double theEne = thePionEN[iene];
      filename << "NuclearInteractionsVal_" << thePionNA[iname] << "_E"<< theEne << ".root";
      theFileNames[iname][iene] = filename.str();
      //std::cout << "thePid/theEne " << thePionID[iname] << " " << theEne << std::endl; 

      edm::FileInPath myDataFile("FastSimulation/MaterialEffects/data/"+theFileNames[iname][iene]);
      fullPath = myDataFile.fullPath();
      //    theFiles[file] = TFile::Open(theFileNames[file].c_str());
      theFiles[iname][iene] = TFile::Open(fullPath.c_str());
      if ( !theFiles[iname][iene] ) throw cms::Exception("FastSimulation/MaterialEffects") 
	<< "File " << theFileNames[iname][iene] << " " << fullPath <<  " not found ";
      ++fileNb;
      //
      theTrees[iname][iene] = (TTree*) theFiles[iname][iene]->Get("NuclearInteractions"); 
      if ( !theTrees[iname][iene] ) throw cms::Exception("FastSimulation/MaterialEffects") 
	<< "Tree with name NuclearInteractions not found in " << theFileNames[iname][iene];
      //
      theBranches[iname][iene] = theTrees[iname][iene]->GetBranch("nuEvent");
      //std::cout << "The branch = " << theBranches[iname][iene] << std::endl;
      if ( !theBranches[iname][iene] ) throw cms::Exception("FastSimulation/MaterialEffects") 
	<< "Branch with name nuEvent not found in " << theFileNames[iname][iene];
      //
      theNUEvents[iname][iene] = new NUEvent();
      //std::cout << "The branch = " << theBranches[iname][iene] << std::endl;
      theBranches[iname][iene]->SetAddress(&theNUEvents[iname][iene]);
      //
      theNumberOfEntries[iname][iene] = theTrees[iname][iene]->GetEntries();

      if(currentValuesWereSet) {
        theTrees[iname][iene]->GetEntry(theCurrentEntry[iname][iene]);
        unsigned NInteractions = theNUEvents[iname][iene]->nInteractions();
        theNumberOfInteractions[iname][iene] = NInteractions;
      }

      //
      // Compute the corresponding cm energies of the nuclear interactions
      XYZTLorentzVector Proton(0.,0.,0.,0.986);
      XYZTLorentzVector 
	Reference(0.,
		  0.,
		  std::sqrt(thePionEN[iene]*thePionEN[iene]
			    -thePionMA[iname]*thePionMA[iname]),
		  thePionEN[iene]);
      thePionCM[iname][iene] = (Reference+Proton).M();

    }

  }

  // Find the index for which EN = 4. (or thereabout)
  ien4 = 0;
  while ( thePionEN[ien4] < 4.0 ) ++ien4;

  gROOT->cd();

  // Information (Should be on LogInfo)
//  std::cout << " ---> A total of " << fileNb 
//	    << " nuclear-interaction files was sucessfully open" << std::endl;

  //  dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  //  htot = dbe->book1D("Total", "All particles",150,0.,150.);
  //  helas = dbe->book1D("Elastic", "Elastic interactions",150,0.,150.);
  //  hinel = dbe->book1D("Inelastic", "Inelastic interactions",150,0.,150.);
  //  hscatter = dbe->book1D("Scattering","Elastic Scattering angle",200,0.,2.); 
  //  hscatter2 = dbe->book2D("Scattering2","Elastic Scattering angle vs p",100,0.,10.,200,0.,2.); 
  //  hAfter = dbe->book1D("eAfter","Energy after collision",200,0.,4.); 
  //  hAfter2 = dbe->book2D("eAfter2","Energy after collision",100,-2.5,2.5,100,0.,4); 
  //  hAfter3 = dbe->book2D("eAfter3","Energy after collision",100,0.,1000.,100,0.,4); 

}

NuclearInteractionSimulator::~NuclearInteractionSimulator() {

  // Close all local files
  // Among other things, this allows the TROOT destructor to end up 
  // without crashing, while trying to close these files from outside
  for ( unsigned ifile=0; ifile<theFiles.size(); ++ifile ) { 
    for ( unsigned iene=0; iene<theFiles[ifile].size(); ++iene ) {
      // std::cout << "Closing file " << iene << " with name " << theFileNames[ifile][iene] << std::endl;
      theFiles[ifile][iene]->Close(); 
    }
  }

  // Close the output file
  myOutputFile.close();

  //  dbe->save("test.root");

}

void NuclearInteractionSimulator::compute(ParticlePropagator& Particle, RandomEngineAndDistribution const* random)
{
  if(!currentValuesWereSet) {
    currentValuesWereSet = true;
    for ( unsigned iname=0; iname<thePionNA.size(); ++iname ) {
      for ( unsigned iene=0; iene<thePionEN.size(); ++iene ) {
        theCurrentEntry[iname][iene] = (unsigned) (theNumberOfEntries[iname][iene] * random->flatShoot());

        theTrees[iname][iene]->GetEntry(theCurrentEntry[iname][iene]);
        unsigned NInteractions = theNUEvents[iname][iene]->nInteractions();
        theNumberOfInteractions[iname][iene] = NInteractions;

        theCurrentInteraction[iname][iene] = (unsigned) (theNumberOfInteractions[iname][iene] * random->flatShoot());
      }
    }
  }

  // Read a Nuclear Interaction in a random manner

  double pHadron = std::sqrt(Particle.Vect().Mag2()); 
  //  htot->Fill(pHadron);

  // The hadron has enough momentum to create some relevant final state
  if ( pHadron > thePionEnergy ) { 

    // The particle type
    std::map<int,int>::const_iterator thePit = theIDMap.find(Particle.pid());
    
    int thePid = thePit != theIDMap.end() ? thePit->second : Particle.pid(); 

    // Is this particle type foreseen?
    unsigned fPid = abs(thePid);
    if ( fPid != 211 && fPid != 130 && fPid != 321 && fPid != 2112 && fPid != 2212 ) { 
      return;
      //std::cout << "Unknown particle type = " << thePid << std::endl;
      //thePid = 211;
    }

    // The inelastic interaction length at p(pion) = 5 GeV/c
    unsigned thePidIndex = index(thePid);
    double theInelasticLength = radLengths * theLengthRatio[thePidIndex];

    // The elastic interaction length
    // The baroque parameterization is a fit to Fig. 40.13 of the PDG
    double ee = pHadron > 0.6 ? 
      exp(-std::sqrt((pHadron-0.6)/1.122)) : exp(std::sqrt((0.6-pHadron)/1.122));
    double theElasticLength = ( 0.8753 * ee + 0.15 )
    //    double theElasticLength = ( 0.15 + 0.195 / log(pHadron/0.4) )
    //    double theElasticLength = ( 0.15 + 0.305 / log(pHadron/0.35) )
                            * theInelasticLength;

    // The total interaction length
    double theTotalInteractionLength = theInelasticLength + theElasticLength;

    // Probability to interact is dl/L0 (maximum for 4 GeV pion)
    double aNuclInteraction = -std::log(random->flatShoot());
    if ( aNuclInteraction < theTotalInteractionLength ) { 

      // The elastic part
      double elastic = random->flatShoot();
      if ( elastic < theElasticLength/theTotalInteractionLength ) {

	//       	helas->Fill(pHadron);
	
	// Characteristic scattering angle for the elastic part
	double theta0 = std::sqrt(3.)/ std::pow(theA(),1./3.) * Particle.mass()/pHadron; 
	
	// Draw an angle with theta/theta0*exp[(-theta/2theta0)**2] shape 
	double theta = theta0 * std::sqrt(-2.*std::log(random->flatShoot()));
	double phi = 2. * 3.14159265358979323 * random->flatShoot();
	
	// Rotate the particle accordingly
	RawParticle::Rotation rotation1(orthogonal(Particle.Vect()),theta);
	RawParticle::Rotation rotation2(Particle.Vect(),phi);
	Particle.rotate(rotation1);
	Particle.rotate(rotation2);
	
	// Distance 
	double distance = std::sin(theta);

	// Create a daughter if the kink is large engough 
	if ( distance > theDistCut ) { 
	  _theUpdatedState.resize(1);
	  _theUpdatedState[0].SetXYZT(Particle.Px(), Particle.Py(),
				      Particle.Pz(), Particle.E());
	  _theUpdatedState[0].setID(Particle.pid());
	}

	//	hscatter->Fill(myTheta);
	//	hscatter2->Fill(pHadron,myTheta);
	
      } 

      // The inelastic part
      else {
    
	// Avoid multiple map access
	const std::vector<double>& aPionCM = thePionCM[thePidIndex];
	const std::vector<double>& aRatios = theRatios[thePidIndex];
	// Find the file with the closest c.m energy
	// The target nucleon
	XYZTLorentzVector Proton(0.,0.,0.,0.939);
	// The current particle
	const XYZTLorentzVector& Hadron = (const XYZTLorentzVector&)Particle;
	// The smallest momentum for inelastic interactions
	double pMin = thePionPMin[thePidIndex]; 
	// The correspong smallest four vector
	XYZTLorentzVector Hadron0(0.,0.,pMin,std::sqrt(pMin*pMin+Particle.M2()));

	// The current centre-of-mass energy
	double ecm = (Proton+Hadron).M();
	//std::cout << "Proton = " << Proton << std::endl;
	//std::cout << "Hadron = " << Hadron << std::endl;
	//std::cout << "ecm = " << ecm << std::endl;
	// Get the files of interest (closest c.m. energies)
	unsigned ene1=0;
	unsigned ene2=0;
	// The smallest centre-of-mass energy
	//	double ecm1=1.63; 
	double ecm1= (Proton+Hadron0).M();
	//std::cout << "ecm1 = " << ecm1 << std::endl;
       	//std::cout << "ecm[0] = " << aPionCM[0] << std::endl;
	//std::cout << "ecm[11] = " << aPionCM [ aPionCM.size()-1 ] << std::endl;
	double ecm2=aPionCM[0]; 
	double ratio1=0.;
	double ratio2=aRatios[0];
	if ( ecm > aPionCM[0] && ecm < aPionCM [ aPionCM.size()-1 ] ) {
	  for ( unsigned ene=1; 
		ene < aPionCM.size() && ecm > aPionCM[ene-1]; 
		++ene ) {
	    if ( ecm<aPionCM[ene] ) { 
	      ene2 = ene;
	      ene1 = ene2-1;
	      ecm1 = aPionCM[ene1];
	      ecm2 = aPionCM[ene2];
	      ratio1 = aRatios[ene1];
	      ratio2 = aRatios[ene2];
	    } 
	  }
	} else if ( ecm > aPionCM[ aPionCM.size()-1 ] ) { 
	  ene1 = aPionCM.size()-1;
	  ene2 = aPionCM.size()-2;
	  ecm1 = aPionCM[ene1];
	  ecm2 = aPionCM[ene2];
	  ratio1 = aRatios[ene2];
	  ratio2 = aRatios[ene2];
	} 

	
	// The inelastic part of the cross section depends cm energy
	double slope = (std::log10(ecm )-std::log10(ecm1)) 
	             / (std::log10(ecm2)-std::log10(ecm1));
	double inelastic = ratio1 + (ratio2-ratio1) * slope;
	double inelastic4 =  pHadron < 4. ? aRatios[ien4] : 1.;  

	//std::cout << "Inelastic = " << ratio1 << " " << ratio2 << " " << inelastic << std::endl;
	//      std::cout << "Energy/Inelastic : " 
	//		<< Hadron.e() << " " << inelastic << std::endl;
	
	// Simulate an inelastic interaction
	if ( elastic > 1.- (inelastic*theInelasticLength)
	                   /theTotalInteractionLength ) { 

	  // Avoid mutliple map access
	  std::vector<unsigned>& aCurrentInteraction = theCurrentInteraction[thePidIndex];
	  std::vector<unsigned>& aNumberOfInteractions = theNumberOfInteractions[thePidIndex];
	  std::vector<NUEvent*>& aNUEvents = theNUEvents[thePidIndex];
	  //	  hinel->Fill(pHadron);
	  //	  std::cout << "INELASTIC INTERACTION ! " 
	  //	  	    << pHadron << " " << theInelasticLength << " "
	  //		    << inelastic * theInelasticLength << std::endl;
	  // Choice of the file to read according the the log10(ecm) distance
	  // and protection against low momentum proton and neutron that never interacts 
	  // (i.e., empty files)
	  unsigned ene;
	  if ( random->flatShoot() < slope || aNumberOfInteractions[ene1] == 0 )  
	    ene = ene2;
	  else
	    ene = ene1;

	  //std::cout << "Ecm1/2 = " << ecm1 << " " << ecm2 << std::endl;
	  //std::cout << "Ratio1/2 = " << ratio1 << " " << ratio2 << std::endl;
	  //std::cout << "Ene = " << ene << " slope = " << slope << std::endl;

	  //std::cout << "Pion energy = " << Hadron.E() 
	  //	    << "File chosen " << theFileNames[thePidIndex][ene]
	  //	    << std::endl;
	  
	  // The boost characteristics
	  XYZTLorentzVector theBoost = Proton + Hadron;
	  theBoost /= theBoost.e();
	  
	  // std::cout << "File chosen : " << thePid << "/" << ene 
	  //	    << " Current interaction = " << aCurrentInteraction[ene] 
	  //	    << " Total interactions = " << aNumberOfInteractions[ene]
	  //	    << std::endl;
	  //      theFiles[thePidIndex][ene]->cd();
	  //      gDirectory->ls();

	  // Check we are not either at the end of an interaction bunch 
	  // or at the end of a file
	  if ( aCurrentInteraction[ene] == aNumberOfInteractions[ene] ) {
	    // std::cout << "End of interaction bunch ! ";
	    std::vector<unsigned>& aCurrentEntry = theCurrentEntry[thePidIndex];
	    std::vector<unsigned>& aNumberOfEntries = theNumberOfEntries[thePidIndex];
	    std::vector<TTree*>& aTrees = theTrees[thePidIndex];
	    ++aCurrentEntry[ene];
	    // std::cerr << "Read the next entry " 
	    //           << aCurrentEntry[ene] << std::endl;
	    aCurrentInteraction[ene] = 0;
	    if ( aCurrentEntry[ene] == aNumberOfEntries[ene] ) { 
	      aCurrentEntry[ene] = 0;
	      //  std::cout << "End of file - Rewind! " << std::endl;
	    }
	    unsigned myEntry = aCurrentEntry[ene];
	    // std::cout << "The new entry " << myEntry 
	    //           << " is read ... in TTree " << aTrees[ene] << " "; 
	    aTrees[ene]->GetEntry(myEntry);
	    // std::cout << "The number of interactions in the new entry is ... "; 
	    aNumberOfInteractions[ene] = aNUEvents[ene]->nInteractions();
	    // std::cout << aNumberOfInteractions[ene] << std::endl;
	  }
	  
	  // Read the interaction
	  NUEvent::NUInteraction anInteraction 
	    = aNUEvents[ene]->theNUInteractions()[aCurrentInteraction[ene]];
	  
	  unsigned firstTrack = anInteraction.first; 
	  unsigned lastTrack = anInteraction.last;
	  //      std::cout << "First and last tracks are " << firstTrack << " " << lastTrack << std::endl;
	  
	  _theUpdatedState.resize(lastTrack-firstTrack+1);

	  double distMin = 1E99;

	  // Some rotation around the boost axis, for more randomness
	  XYZVector theAxis = theBoost.Vect().Unit();
	  double theAngle = random->flatShoot() * 2. * 3.14159265358979323;
	  RawParticle::Rotation axisRotation(theAxis,theAngle);
	  RawParticle::Boost axisBoost(theBoost.x(),theBoost.y(),theBoost.z());

	  // A rotation to bring the particles back to the pion direction
	  XYZVector zAxis(0.,0.,1.); 
	  XYZVector orthAxis = (zAxis.Cross(theBoost.Vect())).Unit(); 
	  double orthAngle = acos(theBoost.Vect().Unit().Z());
	  RawParticle::Rotation orthRotation(orthAxis,orthAngle);

	  // A few checks
	  // double eAfter = 0.;
	  
	  // Loop on the nuclear interaction products
	  for ( unsigned iTrack=firstTrack; iTrack<=lastTrack; ++iTrack ) {
	    
	    unsigned idaugh = iTrack - firstTrack;
	    NUEvent::NUParticle aParticle = aNUEvents[ene]->theNUParticles()[iTrack];
	    //	std::cout << "Track " << iTrack 
	    //		  << " id/px/py/pz/mass "
	    //		  << aParticle.id << " " 
	    //		  << aParticle.px << " " 
	    //		  << aParticle.py << " " 
	    //		  << aParticle.pz << " " 
	    //		  << aParticle.mass << " " << endl; 
	    
	    // Add a RawParticle with the proper energy in the c.m frame of 
	    // the nuclear interaction
	    double energy = std::sqrt( aParticle.px*aParticle.px
				     + aParticle.py*aParticle.py
				     + aParticle.pz*aParticle.pz
				     + aParticle.mass*aParticle.mass/(ecm*ecm) );

	    RawParticle& aDaughter = _theUpdatedState[idaugh]; 
	    aDaughter.SetXYZT(aParticle.px*ecm,aParticle.py*ecm,
			      aParticle.pz*ecm,energy*ecm);	    
	    aDaughter.setID(aParticle.id);

	    // Rotate to the collision axis
	    aDaughter.rotate(orthRotation);

	    // Rotate around the boost axis for more randomness
	    aDaughter.rotate(axisRotation);
	    
	    // Boost it in the lab frame
	    aDaughter.boost(axisBoost);

	    // Store the closest daughter index (for later tracking purposes, so charged particles only) 
	    double distance = distanceToPrimary(Particle,aDaughter);
	    // Find the closest daughter, if closer than a given upper limit.
	    if ( distance < distMin && distance < theDistCut ) {
	      distMin = distance;
	      theClosestChargedDaughterId = idaugh;
	    }

	    // eAfter += aDaughter.E();

	  }

	  /*
	  double eBefore = Particle.E();
	  double rapid = Particle.momentum().Eta();
	  if ( eBefore > 0. ) { 
	    hAfter->Fill(eAfter/eBefore);
	    hAfter2->Fill(rapid,eAfter/eBefore);
	    hAfter3->Fill(eBefore,eAfter/eBefore);
	  }
	  */

         // ERROR The way this loops through the events breaks
         // replay. Which events are retrieved depends on
         // which previous events were processed.

	  // Increment for next time
	  ++aCurrentInteraction[ene];
	  
	// Simulate a stopping hadron (low momentum)
	} else if ( pHadron < 4. &&  
		    elastic > 1.- (inelastic4*theInelasticLength)
		                  /theTotalInteractionLength ) { 
	  // A fake particle with 0 momentum as a daughter!
	  _theUpdatedState.resize(1);
	  _theUpdatedState[0].SetXYZT(0.,0.,0.,0.);
	  _theUpdatedState[0].setID(22);
	}

      }

    }

  }

}

double 
NuclearInteractionSimulator::distanceToPrimary(const RawParticle& Particle,
					       const RawParticle& aDaughter) const {

  double distance = 2E99;

  // Compute the distance only for charged primaries
  if ( fabs(Particle.charge()) > 1E-12 ) { 

    // The secondary must have the same charge
    double chargeDiff = fabs(aDaughter.charge()-Particle.charge());
    if ( fabs(chargeDiff) < 1E-12 ) {

      // Here are two distance definitions * to be tuned *
      switch ( theDistAlgo ) { 
	
      case 1:
	// sin(theta12)
	distance = (aDaughter.Vect().Unit().Cross(Particle.Vect().Unit())).R();
	break;
	
      case 2: 
	// sin(theta12) * p1/p2
	distance = (aDaughter.Vect().Cross(Particle.Vect())).R()
	  /aDaughter.Vect().Mag2();
	break;
	
      default:
	// Should not happen
	distance = 2E99;
	break;
	
      }

    }

  }
      
  return distance;

}

void
NuclearInteractionSimulator::save() {

  // Size of buffer
  ++myOutputBuffer;

  // Periodically close the current file and open a new one
  if ( myOutputBuffer/1000*1000 == myOutputBuffer ) { 
    myOutputFile.close();
    myOutputFile.open ("NuclearInteractionOutputFile.txt");
    //    myOutputFile.seekp(0); // No need to rewind in that case
  }
  //
  unsigned size1 = 
    theCurrentEntry.size()*
    theCurrentEntry.begin()->size();
  std::vector<unsigned> theCurrentEntries;
  theCurrentEntries.resize(size1);
  size1*=sizeof(unsigned);
  //
  unsigned size2 = 
    theCurrentInteraction.size()*
    theCurrentInteraction.begin()->size();
  std::vector<unsigned> theCurrentInteractions;
  theCurrentInteractions.resize(size2);
  size2 *= sizeof(unsigned);

  // Save the current entries 
  std::vector< std::vector<unsigned> >::const_iterator aCurrentEntry = theCurrentEntry.begin();
  std::vector< std::vector<unsigned> >::const_iterator lastCurrentEntry = theCurrentEntry.end();
  unsigned allEntries=0;
  for ( ; aCurrentEntry!=lastCurrentEntry; ++aCurrentEntry ) { 
    unsigned size = aCurrentEntry->size();
    for ( unsigned iene=0; iene<size; ++iene )
      theCurrentEntries[allEntries++] = (*aCurrentEntry)[iene]; 
  }
  
  // Save the current interactions
  std::vector< std::vector<unsigned> >::const_iterator aCurrentInteraction = theCurrentInteraction.begin();
  std::vector< std::vector<unsigned> >::const_iterator lastCurrentInteraction = theCurrentInteraction.end();
  unsigned allInteractions=0;
  for ( ; aCurrentInteraction!=lastCurrentInteraction; ++aCurrentInteraction ) { 
    unsigned size = aCurrentInteraction->size();
    for ( unsigned iene=0; iene<size; ++iene )
      theCurrentInteractions[allInteractions++] = (*aCurrentInteraction)[iene]; 
  }
  // 
  myOutputFile.write((const char*)(&theCurrentEntries.front()), size1);
  myOutputFile.write((const char*)(&theCurrentInteractions.front()), size2);  
  myOutputFile.flush();

}

bool
NuclearInteractionSimulator::read(std::string inputFile) {

  std::ifstream myInputFile;
  struct stat results;
  //
  unsigned size1 = 
    theCurrentEntry.size()*
    theCurrentEntry.begin()->size();
  std::vector<unsigned> theCurrentEntries;
  theCurrentEntries.resize(size1);
  size1*=sizeof(unsigned);
  //
  unsigned size2 = 
    theCurrentInteraction.size()*
    theCurrentInteraction.begin()->size();
  std::vector<unsigned> theCurrentInteractions;
  theCurrentInteractions.resize(size2);
  size2 *= sizeof(unsigned);
  //
  unsigned size = 0;


  // Open the file (if any)
  myInputFile.open (inputFile.c_str());
  if ( myInputFile.is_open() ) { 

    // Get the size of the file
    if ( stat(inputFile.c_str(), &results) == 0 ) size = results.st_size;
    else return false; // Something is wrong with that file !

    // Position the pointer just before the last record
    myInputFile.seekg(size-size1-size2);
    myInputFile.read((char*)(&theCurrentEntries.front()),size1);
    myInputFile.read((char*)(&theCurrentInteractions.front()),size2);
    myInputFile.close();

    // Read the current entries
    std::vector< std::vector<unsigned> >::iterator aCurrentEntry = theCurrentEntry.begin();
    std::vector< std::vector<unsigned> >::iterator lastCurrentEntry = theCurrentEntry.end();
    unsigned allEntries=0;
    for ( ; aCurrentEntry!=lastCurrentEntry; ++aCurrentEntry ) { 
      unsigned size = aCurrentEntry->size();
      for ( unsigned iene=0; iene<size; ++iene )
	(*aCurrentEntry)[iene] = theCurrentEntries[allEntries++]; 
    }

    // Read the current interactions
    std::vector< std::vector<unsigned> >::iterator aCurrentInteraction = theCurrentInteraction.begin();
    std::vector< std::vector<unsigned> >::iterator lastCurrentInteraction = theCurrentInteraction.end();
    unsigned allInteractions=0;
    for ( ; aCurrentInteraction!=lastCurrentInteraction; ++aCurrentInteraction ) { 
      unsigned size = aCurrentInteraction->size();
      for ( unsigned iene=0; iene<size; ++iene )
	(*aCurrentInteraction)[iene] =  theCurrentInteractions[allInteractions++];
    }
    
    return true;
  }
 
  return false;

}

unsigned 
NuclearInteractionSimulator::index(int thePid) { 
  
  unsigned myIndex=0;
  while ( thePid != thePionID[myIndex] ) ++myIndex; 
  //  std::cout << "pid/index = " << thePid << " " << myIndex << std::endl;
  return myIndex;

}
