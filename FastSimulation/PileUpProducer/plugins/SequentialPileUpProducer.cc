#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "FastSimDataFormats/PileUpEvents/interface/PUEvent.h"

#include "FastSimulation/PileUpProducer/plugins/SequentialPileUpProducer.h"
#include "FastSimulation/Event/interface/BetaFuncPrimaryVertexGenerator.h"
#include "FastSimulation/Event/interface/GaussianPrimaryVertexGenerator.h"
#include "FastSimulation/Event/interface/FlatPrimaryVertexGenerator.h"
#include "FastSimulation/Event/interface/NoPrimaryVertexGenerator.h"
#include "FastSimulation/Utilities/interface/RandomEngine.h"

#include "HepMC/GenEvent.h"
#include "TFile.h"
#include "TTree.h"
#include "TROOT.h"

#include <iostream>
#include <memory>
#include <sys/stat.h>
#include <cmath>

SequentialPileUpProducer::SequentialPileUpProducer(edm::ParameterSet const & p)  
{    

  // This producer produces a HepMCProduct, with all pileup vertices/particles
  produces<edm::HepMCProduct>("PileUpEvents");
  
  // Initialize the random number generator service
  edm::Service<edm::RandomNumberGenerator> rng;
  if ( ! rng.isAvailable() ) {
    throw cms::Exception("Configuration")
      << "SequentialPileUpProducer requires the RandomGeneratorService\n"
         "which is not present in the configuration file.\n"
         "You must add the service in the configuration file\n"
         "or remove the module that requires it";
  }  
  random = new RandomEngine(&(*rng));

  // The pile-up event generation condition
  const edm::ParameterSet& pu = p.getParameter<edm::ParameterSet>("PileUpSimulator");
  averageNumber_ = pu.getParameter<double>("averageNumber");
  theFileNames = pu.getUntrackedParameter<std::vector<std::string> >("fileNames");
  theNumberOfFiles = theFileNames.size();
  // If the event generation is divided in many jobs, we want to make sure
  // that the consecutive jobs pickup a distinct set of sequential events
  // so we allow the user to set the starting event
  theStartingEvent = pu.getUntrackedParameter<unsigned>("startingEvent");
  // If the user rather specify the sequential job number and the number of
  // hard interaction events per job, we can computing the approximate 
  // starting event for him 
  unsigned jobNumber = pu.getUntrackedParameter<unsigned>("jobNumber");
  unsigned nEventsPerJob = pu.getUntrackedParameter<unsigned>("nEventsPerJob");
  if(theStartingEvent == 0 && jobNumber != 0 && nEventsPerJob != 0) {
    theStartingEvent = jobNumber * averageNumber_ * nEventsPerJob;
    theStartingEvent += sqrt(theStartingEvent); // To account for fluctuations
  }
  // Initialize the primary vertex generator
  const edm::ParameterSet& vtx = p.getParameter<edm::ParameterSet>("VertexGenerator");
  std::string vtxType = vtx.getParameter<std::string>("type");
  if ( vtxType == "Gaussian" ) 
    theVertexGenerator = new GaussianPrimaryVertexGenerator(vtx,random);
  else if ( vtxType == "Flat" ) 
    theVertexGenerator = new FlatPrimaryVertexGenerator(vtx,random);
  else if ( vtxType == "BetaFunc" )
    theVertexGenerator = new BetaFuncPrimaryVertexGenerator(vtx,random);
  else
    theVertexGenerator = new NoPrimaryVertexGenerator();

}

SequentialPileUpProducer::~SequentialPileUpProducer() { 

  delete theVertexGenerator;

}

void SequentialPileUpProducer::beginRun(edm::Run & run, edm::EventSetup const& es)
{
  
  gROOT->cd();
  
  std::string fullPath;
  
  // Open the root file
  unsigned nEventsSkipped = 0;
  openFile(0);
  while(nEventsSkipped < theStartingEvent)
    {
      if(theNumberOfMinBiasEvts > (theStartingEvent - nEventsSkipped))
	{
	  theCurrentMinBiasEvt = theStartingEvent - nEventsSkipped;
	  nEventsSkipped += theCurrentMinBiasEvt;
	  break;
	}
      else
	{
	  nEventsSkipped += theNumberOfMinBiasEvts;
	  ++theCurrentEntry;
	  if(theCurrentEntry == theNumberOfEntries)
	    {
	      theFile->Close();
	      ++theCurrentFile;
	      if(theCurrentFile == theNumberOfFiles)
		throw cms::Exception("Configuration")
		  << "SequentialPileUpProducer has fewer events than asked to skip";
	      openFile(theCurrentFile);
	    }
	  else {
	    theTree->GetEntry(theCurrentEntry);
	    theCurrentMinBiasEvt = 0;
	    theNumberOfMinBiasEvts = thePUEvent->nMinBias();
	  }
	}
    }
  // Return Loot in the same state as it was when entering. 
  gROOT->cd();
  
}

void SequentialPileUpProducer::openFile(unsigned file)
{
  gROOT->cd();
  theCurrentFile = file;
  edm::FileInPath myDataFile("FastSimulation/PileUpProducer/data/"+theFileNames[file]);
  std::string fullPath = myDataFile.fullPath();
  theFile = TFile::Open(fullPath.c_str());
  if ( !theFile ) throw cms::Exception("FastSimulation/PileUpProducer") 
    << "File " << theFileNames[file] << " " << fullPath <<  " not found ";
  theTree = (TTree*) theFile->Get("MinBiasEvents"); 
  if ( !theTree ) throw cms::Exception("FastSimulation/PileUpProducer") 
    << "Tree with name MinBiasEvents not found in " << theFileNames[file];
  theBranch = theTree->GetBranch("puEvent");
  if ( !theBranch ) throw cms::Exception("FastSimulation/PileUpProducer") 
    << "Branch with name puEvent not found in " << theFileNames[file];
  thePUEvent = new PUEvent();
  theBranch->SetAddress(&thePUEvent);
  theNumberOfEntries = theTree->GetEntries();
  theCurrentEntry = 0;
  theTree->GetEntry(theCurrentEntry);
  theNumberOfMinBiasEvts = thePUEvent->nMinBias();
  theCurrentMinBiasEvt = 0;
}

void SequentialPileUpProducer::endRun()
{ 
  // Close all local files
  // Among other things, this allows the TROOT destructor to end up 
  // without crashing, while trying to close these files from outside
  theFile->Close();
  // And return Loot in the same state as it was when entering. 
  gROOT->cd();
  
}
 
void SequentialPileUpProducer::produce(edm::Event & iEvent, const edm::EventSetup & es)
{

  // Create the GenEvent and the HepMCProduct
  std::auto_ptr<edm::HepMCProduct> pu_product(new edm::HepMCProduct());  
  HepMC::GenEvent* evt = new HepMC::GenEvent();
  
  // How many pile-up events?
  int PUevts = (int) random->poissonShoot(averageNumber_);

  // Get N sequential events from minbias file(s)
  for ( int ievt=0; ievt<PUevts; ++ievt ) { 

    // Smear the primary vertex and express it in mm (stupid GenEvent convention...)
    theVertexGenerator->generate();
    HepMC::FourVector smearedVertex =  
      HepMC::FourVector(theVertexGenerator->X()*10.,
			theVertexGenerator->Y()*10.,
			theVertexGenerator->Z()*10.,
			0.);
    HepMC::GenVertex* aVertex = new HepMC::GenVertex(smearedVertex);
    evt->add_vertex(aVertex);

    // Some rotation around the z axis, for more randomness
    double theAngle = random->flatShoot() * 2. * 3.14159265358979323;
    double cAngle = std::cos(theAngle);
    double sAngle = std::sin(theAngle);

    if ( theCurrentMinBiasEvt == theNumberOfMinBiasEvts ) {
      ++theCurrentEntry;
      if ( theCurrentEntry == theNumberOfEntries ) { 
	theCurrentEntry = 0;
	theFile->Close();
	++theCurrentFile;
	if(theCurrentFile == theNumberOfFiles) theCurrentFile = 0;
	openFile(theCurrentFile);
      }
      else {
	theTree->GetEntry(theCurrentEntry);
	theCurrentMinBiasEvt = 0;
      }
    }

    // Read a minbias event chunk
    const PUEvent::PUMinBiasEvt& aMinBiasEvt 
      = thePUEvent->thePUMinBiasEvts()[theCurrentMinBiasEvt];
  
    // Find corresponding particles
    unsigned firstTrack = aMinBiasEvt.first; 
    unsigned trackSize = firstTrack + aMinBiasEvt.size;

    // Loop on particles
    for ( unsigned iTrack=firstTrack; iTrack<trackSize; ++iTrack ) {
      
      const PUEvent::PUParticle& aParticle 
	= thePUEvent->thePUParticles()[iTrack];
      // Create a FourVector, with rotation 
      double energy = std::sqrt( aParticle.px*aParticle.px
			       + aParticle.py*aParticle.py
			       + aParticle.pz*aParticle.pz
			       + aParticle.mass*aParticle.mass );

      HepMC::FourVector myPart(cAngle * aParticle.px + sAngle * aParticle.py,
			      -sAngle * aParticle.px + cAngle * aParticle.py,
			       aParticle.pz, energy);

      // Add a GenParticle
      HepMC::GenParticle* aGenParticle = new HepMC::GenParticle(myPart,aParticle.id);
      aVertex->add_particle_out(aGenParticle);

    }
    // End of particle loop
    
    // Increment for next time
    ++theCurrentMinBiasEvt;
    
  }
  // End of pile-up event loop

  // evt->print();

  // Fill the HepMCProduct from the GenEvent
  if ( evt )  { 
    pu_product->addHepMCData( evt );
    // Boost in case of beam crossing angle
    TMatrixD* boost = theVertexGenerator->boost();
    if ( boost ) pu_product->boostToLab(boost,"momentum");
  }

  // Put the HepMCProduct onto the event
  iEvent.put(pu_product,"PileUpEvents");
  // delete evt;

}

DEFINE_FWK_MODULE(SequentialPileUpProducer);
