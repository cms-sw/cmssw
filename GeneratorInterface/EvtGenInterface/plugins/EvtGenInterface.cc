#include "GeneratorInterface/EvtGenInterface/interface/EvtGenInterface.h"

#include "FWCore/PluginManager/interface/PluginManager.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "Utilities/General/interface/FileInPath.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "CLHEP/Random/Random.h"
#include "CLHEP/Random/RandFlat.h"

#include "EvtGen/EvtGen.hh"
#include "EvtGenBase/EvtId.hh"
#include "EvtGenBase/EvtPDL.hh"
#include "EvtGenBase/EvtDecayTable.hh"
#include "EvtGenBase/EvtSpinType.hh"
#include "EvtGenBase/EvtVector4R.hh"
#include "EvtGenBase/EvtParticle.hh"
#include "EvtGenBase/EvtScalarParticle.hh"
#include "EvtGenBase/EvtStringParticle.hh"
#include "EvtGenBase/EvtDiracParticle.hh"
#include "EvtGenBase/EvtVectorParticle.hh"
#include "EvtGenBase/EvtRaritaSchwingerParticle.hh"
#include "EvtGenBase/EvtTensorParticle.hh"
#include "EvtGenBase/EvtHighSpinParticle.hh"
#include "EvtGenBase/EvtStdHep.hh"
#include "EvtGenBase/EvtSecondary.hh"
#include "EvtGenModels/EvtPythia.hh"

#include "GeneratorInterface/EvtGenInterface/interface/myEvtRandomEngine.h"
#include "GeneratorInterface/Pythia6Interface/interface/Pythia6Service.h"

#include "HepMC/GenEvent.h"

#include "DataFormats/GeometryVector/interface/GlobalVector.h"

namespace PhotosRandomVar {
  CLHEP::HepRandomEngine* decayRandomEngine;
}


extern "C"{

  void phoini_( void );
  void photos_( int& );

  double phoran_(int *idummy)
  {
    return PhotosRandomVar::decayRandomEngine->flat();
  }
  extern struct {
    // bool qedrad[NMXHEP];
    bool qedrad[4000]; // hardcoded for now...
  } phoqed_;

}




using namespace gen;
using namespace edm;

EvtGenInterface::EvtGenInterface( const ParameterSet& pset ){

  ntotal = 0;
  nevent = 0;
  std::cout << " EvtGenProducer starting ... " << std::endl;

  // create random engine and initialize seed using Random Number 
  // Generator Service 
  // as configured in the configuration file

  edm::Service<edm::RandomNumberGenerator> rngen;
  
  if ( ! rngen.isAvailable()) {             
    throw cms::Exception("Configuration")
      << "The EvtGenProducer module requires the RandomNumberGeneratorService\n"
      "which is not present in the configuration file.  You must add the service\n"
      "in the configuration file if you  want to run EvtGenProducer";
    }

  CLHEP::HepRandomEngine& m_engine = rngen->getEngine();
  m_flat = new CLHEP::RandFlat(m_engine, 0., 1.);
  myEvtRandomEngine* the_engine = new myEvtRandomEngine(&m_engine); 

  // Get data from parameter set
  edm::FileInPath decay_table = pset.getParameter<edm::FileInPath>("decay_table");
  edm::FileInPath pdt = pset.getParameter<edm::FileInPath>("particle_property_file");
  bool useDefault = pset.getUntrackedParameter<bool>("use_default_decay",true);
  usePythia = pset.getUntrackedParameter<bool>("use_internal_pythia",true);
  polarize_ids = pset.getUntrackedParameter<std::vector<int> >("particles_to_polarize",
							       std::vector<int>());
  polarize_pol = pset.getUntrackedParameter<std::vector<double> >("particle_polarizations",
								  std::vector<double>());
  if (polarize_ids.size() != polarize_pol.size()) {
      throw cms::Exception("Configuration")
	  << "EvtGenProducer requires that the particles_to_polarize and particle_polarization\n"
	  "vectors be the same size.  Please fix this in your configuration.";
  }
  for (unsigned int ndx = 0; ndx < polarize_ids.size(); ndx++) {
      if (polarize_pol[ndx] < -1. || polarize_pol[ndx] > 1.) {
	  throw cms::Exception("Configuration")
	      << "EvtGenProducer error: particle polarizations must be in the range -1 < P < 1";
      }
      polarizations.insert(std::pair<int, float>(polarize_ids[ndx], polarize_pol[ndx]));
  }

  edm::FileInPath user_decay = pset.getParameter<edm::FileInPath>("user_decay_file");
  std::string decay_table_s = decay_table.fullPath();
  std::string pdt_s = pdt.fullPath();
  std::string user_decay_s = user_decay.fullPath();

  //-->pythia_params = pset.getParameter< std::vector<std::string> >("processParameters");
  
  
  // any number of alias names for forced decays can be specified using dynamic std vector of strings 
  std::vector<std::string> forced_names = pset.getParameter< std::vector<std::string> >("list_forced_decays");
    
  m_EvtGen = new EvtGen (decay_table_s.c_str(),pdt_s.c_str(),the_engine);  
  // 4th parameter should be rad cor - set to PHOTOS (default)
 
  if (!useDefault) m_EvtGen->readUDecay( user_decay_s.c_str() );

  std::vector<std::string>::const_iterator i;
  nforced=0;

  for (i=forced_names.begin(); i!=forced_names.end(); ++i)   
    // i will point to strings containing
    // names of particles with forced decay
    {
      nforced++;
      EvtId found = EvtPDL::getId(*i);      
      if (found.getId()==-1)
	{
	  throw cms::Exception("Configuration")
	    << "name in part list for forced decays not found: " << *i; 
	}
      if (found.getId()==found.getAlias())
	{
	  throw cms::Exception("Configuration")
	    << "name in part list for forced decays is not an alias: " << *i; 
	}
      forced_Evt.push_back(found);                      // forced_Evt is the list of EvtId's
      forced_Hep.push_back(EvtPDL::getStdHep(found));   // forced_Hep is the list of stdhep codes
    }

   // fill up default list of particles to be declared stable in the "master generator"
   // these are assumed to be PDG ID's
   // in case of combo with Pythia6, translation is done in Pythia6Hadronizer
   //
   // Note: Pythia6's kc=43, 44, and 84 commented out because they're obsolete (per S.Mrenna)
   //
   m_PDGs.push_back( 300553 ) ;
   m_PDGs.push_back( 511 ) ;
   m_PDGs.push_back( 521 ) ;
   m_PDGs.push_back( 523 ) ;
   m_PDGs.push_back( 513 ) ;
   m_PDGs.push_back( 533 ) ;
   m_PDGs.push_back( 531 ) ;
   
   m_PDGs.push_back( 15 ) ;
   
   m_PDGs.push_back( 413 ) ;
   m_PDGs.push_back( 423 ) ;
   m_PDGs.push_back( 433 ) ;
   m_PDGs.push_back( 411 ) ;
   m_PDGs.push_back( 421 ) ;
   m_PDGs.push_back( 431 ) ;   
   m_PDGs.push_back( 10411 );
   m_PDGs.push_back( 10421 );
   m_PDGs.push_back( 10413 );
   m_PDGs.push_back( 10423 );   
   m_PDGs.push_back( 20413 );
   m_PDGs.push_back( 20423 );
    
   m_PDGs.push_back( 415 );
   m_PDGs.push_back( 425 );
   m_PDGs.push_back( 10431 );
   m_PDGs.push_back( 20433 );
   m_PDGs.push_back( 10433 );
   m_PDGs.push_back( 435 );
   
   m_PDGs.push_back( 310 );
   m_PDGs.push_back( 311 );
   m_PDGs.push_back( 313 );
   m_PDGs.push_back( 323 );
   m_PDGs.push_back( 10321 );
   m_PDGs.push_back( 10311 );
   m_PDGs.push_back( 10313 );
   m_PDGs.push_back( 10323 );
   m_PDGs.push_back( 20323 );
   m_PDGs.push_back( 20313 );
   m_PDGs.push_back( 325 );
   m_PDGs.push_back( 315 );
   
   m_PDGs.push_back( 100313 );
   m_PDGs.push_back( 100323 );
   m_PDGs.push_back( 30313 );
   m_PDGs.push_back( 30323 );
   m_PDGs.push_back( 30343 );
   m_PDGs.push_back( 30353 );
   m_PDGs.push_back( 30363 );

   m_PDGs.push_back( 111 );
   m_PDGs.push_back( 221 );
   m_PDGs.push_back( 113 );
   m_PDGs.push_back( 213 );
   m_PDGs.push_back( 223 );
   m_PDGs.push_back( 331 );
   m_PDGs.push_back( 333 );
   m_PDGs.push_back( 20213 );
   m_PDGs.push_back( 20113 );
   m_PDGs.push_back( 215 );
   m_PDGs.push_back( 115 );
   m_PDGs.push_back( 10213 );
   m_PDGs.push_back( 10113 );
   m_PDGs.push_back( 9000111 ); // PDG ID = 9000111, Pythia6 ID = 10111
   m_PDGs.push_back( 9000211 ); // PDG ID = 9000211, Pythia6 ID = 10211
   m_PDGs.push_back( 9010221 ); // PDG ID = 9010211, Pythia6 ID = ???
   m_PDGs.push_back( 10221 );
   m_PDGs.push_back( 20223 );
   m_PDGs.push_back( 20333 );
   m_PDGs.push_back( 225 );
   m_PDGs.push_back( 9020221 ); // PDG ID = 9020211, Pythia6 ID = ???
   m_PDGs.push_back( 335 );
   m_PDGs.push_back( 10223 );
   m_PDGs.push_back( 10333 );
   m_PDGs.push_back( 100213 );
   m_PDGs.push_back( 100113 );
   
   m_PDGs.push_back( 441 );
   m_PDGs.push_back( 100441 );
   m_PDGs.push_back( 443 );
   m_PDGs.push_back( 100443 );
   m_PDGs.push_back( 9000443 );
   m_PDGs.push_back( 9010443 );
   m_PDGs.push_back( 9020443 );
   m_PDGs.push_back( 10441 );
   m_PDGs.push_back( 20443 );
   m_PDGs.push_back( 445 );

   m_PDGs.push_back( 30443 );
   m_PDGs.push_back( 551 );
   m_PDGs.push_back( 553 );
   m_PDGs.push_back( 100553 );
   m_PDGs.push_back( 200553 );
   m_PDGs.push_back( 10551 );
   m_PDGs.push_back( 20553 );
   m_PDGs.push_back( 555 );
   m_PDGs.push_back( 10553 );

   m_PDGs.push_back( 110551 );
   m_PDGs.push_back( 120553 );
   m_PDGs.push_back( 100555 );
   m_PDGs.push_back( 210551 );
   m_PDGs.push_back( 220553 );
   m_PDGs.push_back( 200555 );
   m_PDGs.push_back( 30553 );
   m_PDGs.push_back( 20555 );

   m_PDGs.push_back( 557 );
   m_PDGs.push_back( 130553 ); 
   m_PDGs.push_back( 120555 );
   m_PDGs.push_back( 100557 );
   m_PDGs.push_back( 110553 );
   m_PDGs.push_back( 210553 );
   m_PDGs.push_back( 10555 );
   m_PDGs.push_back( 110555 );

   m_PDGs.push_back( 4122 );
   m_PDGs.push_back( 4132 );
   // m_PDGs.push_back( 84 ); // obsolete
   m_PDGs.push_back( 4112 );
   m_PDGs.push_back( 4212 );
   m_PDGs.push_back( 4232 );
   m_PDGs.push_back( 4222 );
   m_PDGs.push_back( 4322 );
   m_PDGs.push_back( 4312 );

   m_PDGs.push_back( 13122 );
   m_PDGs.push_back( 13124 );
   m_PDGs.push_back( 23122 );
   m_PDGs.push_back( 33122 );
   m_PDGs.push_back( 43122 );
   m_PDGs.push_back( 53122 );
   m_PDGs.push_back( 13126 );
   m_PDGs.push_back( 13212 );
   m_PDGs.push_back( 13241 );
  
   m_PDGs.push_back( 3126 );
   m_PDGs.push_back( 3124 );
   m_PDGs.push_back( 3122 );
   m_PDGs.push_back( 3222 );
   m_PDGs.push_back( 2214 );
   m_PDGs.push_back( 2224 );
   m_PDGs.push_back( 3324 );
   m_PDGs.push_back( 2114 );
   m_PDGs.push_back( 1114 );
   m_PDGs.push_back( 3112 );
   m_PDGs.push_back( 3212 );
   m_PDGs.push_back( 3114 );
   m_PDGs.push_back( 3224 );
   m_PDGs.push_back( 3214 );
   m_PDGs.push_back( 3216 );
   m_PDGs.push_back( 3322 );
   m_PDGs.push_back( 3312 );
   m_PDGs.push_back( 3314 );
   m_PDGs.push_back( 3334 );
   
   m_PDGs.push_back( 4114 );
   m_PDGs.push_back( 4214 );
   m_PDGs.push_back( 4224 );
   m_PDGs.push_back( 4314 );
   m_PDGs.push_back( 4324 );
   m_PDGs.push_back( 4332 );
   m_PDGs.push_back( 4334 );
   //m_PDGs.push_back( 43 ); // obsolete (?)
   //m_PDGs.push_back( 44 ); // obsolete (?)
   m_PDGs.push_back( 10443 );

   m_PDGs.push_back( 5122 );
   m_PDGs.push_back( 5132 );
   m_PDGs.push_back( 5232 );
   m_PDGs.push_back( 5332 );
   m_PDGs.push_back( 5222 );
   m_PDGs.push_back( 5112 );
   m_PDGs.push_back( 5212 );
   m_PDGs.push_back( 541 );
   m_PDGs.push_back( 14122 );
   m_PDGs.push_back( 14124 );
   m_PDGs.push_back( 5312 );
   m_PDGs.push_back( 5322 );
   m_PDGs.push_back( 10521 );
   m_PDGs.push_back( 20523 );
   m_PDGs.push_back( 10523 );

   m_PDGs.push_back( 525 );
   m_PDGs.push_back( 10511 );
   m_PDGs.push_back( 20513 );
   m_PDGs.push_back( 10513 );
   m_PDGs.push_back( 515 );
   m_PDGs.push_back( 10531 );
   m_PDGs.push_back( 20533 );
   m_PDGs.push_back( 10533 );
   m_PDGs.push_back( 535 );
   m_PDGs.push_back( 543 );
   m_PDGs.push_back( 545 );
   m_PDGs.push_back( 5114 );
   m_PDGs.push_back( 5224 );
   m_PDGs.push_back( 5214 );
   m_PDGs.push_back( 5314 );
   m_PDGs.push_back( 5324 );
   m_PDGs.push_back( 5334 );
   m_PDGs.push_back( 10541 );
   m_PDGs.push_back( 10543 );
   m_PDGs.push_back( 20543 );

   m_PDGs.push_back( 4424 );
   m_PDGs.push_back( 4422 );
   m_PDGs.push_back( 4414 );
   m_PDGs.push_back( 4412 );
   m_PDGs.push_back( 4432 );
   m_PDGs.push_back( 4434 );

   m_PDGs.push_back( 130 );
   
   // now check if we need to override default list of particles/IDs
   if ( pset.exists("operates_on_particles") )
   {
      std::vector<int> tmpPIDs = pset.getParameter< std::vector<int> >("operates_on_particles");
      if ( tmpPIDs.size() > 0 )
      {
         if ( tmpPIDs[0] > 0 ) // 0 means default !!!
	 {
	    m_PDGs.clear();
	    m_PDGs = tmpPIDs;
         }
      }
   } 
   
  m_Py6Service = new Pythia6Service;
} 

EvtGenInterface::~EvtGenInterface()
{
  std::cout << " EvtGenProducer terminating ... " << std::endl; 
  delete m_Py6Service;
}

void EvtGenInterface::SetPhotosDecayRandomEngine(CLHEP::HepRandomEngine* decayRandomEngine){
  PhotosRandomVar::decayRandomEngine=decayRandomEngine;
}


void EvtGenInterface::init()
{

   Pythia6Service::InstanceWrapper guard(m_Py6Service);	// grab Py6 instance
   
   // Do here initialization of EvtPythia then restore original settings
   if (usePythia) EvtPythia::pythiaInit(0);
   
// no need - will be done via Pythia6Hadronizer::declareStableParticles
//
//    for( std::vector<std::string>::const_iterator itPar = pythia_params.begin(); itPar != pythia_params.end(); ++itPar ) {
//      call_pygive(*itPar);
//    }

   return ;

}


HepMC::GenEvent* EvtGenInterface::decay( HepMC::GenEvent* evt )
{
  Pythia6Service::InstanceWrapper guard(m_Py6Service);	// grab Py6 instance

  nevent++;
  npartial = 0;
  // std::cout << "nevent = " << nevent << std::endl ;
  
  int idHep,ipart,status;
  EvtId idEvt;

  nPythia = evt->particles_size();
  // FIX A MEMORY LEAK (RC)
  // HepMC::GenEvent* newEvt = new HepMC::GenEvent( *evt );

  // First pass through undecayed Pythia particles to decay particles known to EvtGen left stable by Pythia
  // except candidates to be forced which will be searched later to include EvtGen decay products 
  nlist = 0;

  // Notice dynamical use of evt
  for (HepMC::GenEvent::particle_const_iterator p= evt->particles_begin(); p != evt->particles_end(); ++p)
    {
      status = (*p)->status();
 
      if(status==1) {           // only not decayed (status = 1) particles
          

	  idHep = (*p)->pdg_id();
	  int do_force=0;
	  for(int i=0;i<nforced;i++)           // First check if part with forced decay
	    {                                  // In that case do not decay immediately 
	      if(idHep == forced_Hep[i])       // (only 1 per event will be forced)	 
		{                              // Fill list
		  update_candlist(i,*p);
		  do_force=1;
		}
	    }
	  if(do_force==0)         // particles with decays not forced are decayed immediately 
	    {
	      idEvt = EvtPDL::evtIdFromStdHep(idHep);
	      ipart = idEvt.getId();
	      if (ipart==-1) continue;                          // particle not known to EvtGen       
	      if (EvtDecayTable::getNMode(ipart)==0) continue;  // particles stable for EvtGen
	      addToHepMC(*p,idEvt,evt,true);                      // generate decay
	    }
	}
    }

  if(nlist!=0)   
     {
      // decide randomly which one to decay as alias
      int which = (int)(nlist*m_flat->fire()); 
      if (which == nlist) which = nlist-1;
  
	  for(int k=0;k < nlist; k++)
	    {
	      if(k == which || !usePythia) {		
		addToHepMC(listp[k],forced_Evt[index[k]],evt,false);  // decay as alias
	      }	
	      else
		{
		  int id_non_alias = forced_Evt[index[k]].getId();
		  EvtId non_alias(id_non_alias,id_non_alias); // create new EvtId with id = alias
		  addToHepMC(listp[k],non_alias,evt,false);     // decay as standard (non alias)
		}
	    }
     }

  return evt;
  
}

void EvtGenInterface::addToHepMC(HepMC::GenParticle* partHep, EvtId idEvt, HepMC::GenEvent* theEvent, bool del_daug )
{
  // Set spin type
  EvtSpinType::spintype stype = EvtPDL::getSpinType(idEvt);
  EvtParticle* partEvt;
    switch (stype){
    case EvtSpinType::SCALAR: 
      partEvt = new EvtScalarParticle();
      break;
    case EvtSpinType::STRING:
      partEvt = new EvtStringParticle();    
      break;
    case EvtSpinType::DIRAC: 
      partEvt = new EvtDiracParticle();
      break;
    case EvtSpinType::VECTOR:
      partEvt = new EvtVectorParticle();
      break;
    case EvtSpinType::RARITASCHWINGER:
      partEvt = new EvtRaritaSchwingerParticle();
      break;
    case EvtSpinType::TENSOR:
      partEvt = new EvtTensorParticle();
      break;
    case EvtSpinType::SPIN5HALF: case EvtSpinType::SPIN3: case EvtSpinType::SPIN7HALF: case EvtSpinType::SPIN4:
      partEvt = new EvtHighSpinParticle();
      break;
    default:
      std::cout << "Unknown spintype in EvtSpinType!" << std::endl;   
      return;
    }

    // Generate decay
    EvtVector4R momEvt;
    HepMC::FourVector momHep = partHep->momentum();
    momEvt.set(momHep.t(),momHep.x(),momHep.y(),momHep.z());
    EvtVector4R posEvt;
    HepMC::GenVertex* initVert = partHep->production_vertex();
    HepMC::FourVector posHep = initVert->position();
    posEvt.set(posHep.t(),posHep.x(),posHep.y(),posHep.z());
    partEvt->init(idEvt,momEvt);
    if (stype == EvtSpinType::DIRAC 
	&& polarizations.find(partHep->pdg_id()) != polarizations.end()) {
         // std::cout << "Polarize particle" << std::endl;
	//Particle is spin 1/2, so we can polarize it.
	//Check polarizations map for particle, grab its polarization if it exists
	// and make the spin density matrix
	float pol = polarizations.find(partHep->pdg_id())->second;
	GlobalVector pPart(momHep.x(), momHep.y(), momHep.z());
	//std::cout << "Polarizing particle with PDG ID "
	//  << partHep->pdg_id()
	//  << " at " << pol*100 << "%" << std::endl;
	GlobalVector zHat(0., 0., 1.);
	GlobalVector zCrossP = zHat.cross(pPart);
	GlobalVector polVec = pol * zCrossP.unit();

	EvtSpinDensity theSpinDensity;
	theSpinDensity.SetDim(2);
	theSpinDensity.Set(0, 0, EvtComplex(1./2. + polVec.z()/2., 0.));
	theSpinDensity.Set(0, 1, EvtComplex(polVec.x()/2., -polVec.y()/2.));
	theSpinDensity.Set(1, 0, EvtComplex(polVec.x()/2., polVec.y()/2.));
	theSpinDensity.Set(1, 1, EvtComplex(1./2. - polVec.z()/2., 0.));

	partEvt->setSpinDensityForwardHelicityBasis(theSpinDensity);

    } else {
	partEvt->setDiagonalSpinDensity();     
    }
    partEvt->decay();
                       
    // extend the search of candidates to be forced to EvtGen decay products and delete their daughters  ** 
    // otherwise they wouldn't get their chance to take part in the forced decay lottery                 **
    if (del_daug) go_through_daughters(partEvt);    // recursive function go_through_daughters will do   **

    // Change particle in stdHEP format
    static EvtStdHep evtstdhep;
    
    evtstdhep.init();
    partEvt->makeStdHep(evtstdhep);

    ntotal++;
    partEvt->deleteTree();

    // ********* Now add to the HepMC Event **********

    // Then loop on evtstdhep to add vertexes... 
    HepMC::GenVertex* theVerts[200];
    for (int ivert = 0; ivert < 200; ivert++) { 
      theVerts[ivert] = 0;
    }

    for (int ipart = 0; ipart < evtstdhep.getNPart(); ipart++) {
      int theMum = evtstdhep.getFirstMother(ipart);
      if (theMum != -1 && !theVerts[theMum]) {
        EvtVector4R theVpos = evtstdhep.getX4(ipart) + posEvt;
	theVerts[theMum] = 
	  new HepMC::GenVertex(HepMC::FourVector(theVpos.get(1),
						 theVpos.get(2),
						 theVpos.get(3),
						 theVpos.get(0)),0);
      }
    }

    // ...then particles
    partHep->set_status(2);
    if (theVerts[0]) theVerts[0]->add_particle_in( partHep );

    for (int ipart2 = 1; ipart2 < evtstdhep.getNPart(); ipart2++) {
      int idHep = evtstdhep.getStdHepID(ipart2);
      HepMC::GenParticle* thePart = 
	new HepMC::GenParticle( HepMC::FourVector(evtstdhep.getP4(ipart2).get(1),
						  evtstdhep.getP4(ipart2).get(2),
						  evtstdhep.getP4(ipart2).get(3),
						  evtstdhep.getP4(ipart2).get(0)),
				idHep,
				evtstdhep.getIStat(ipart2));
      npartial++;
      thePart->suggest_barcode(npartial + nPythia);
      int theMum2 = evtstdhep.getFirstMother(ipart2);
      if (theMum2 != -1 && theVerts[theMum2]) theVerts[theMum2]->add_particle_out( thePart );
      if (theVerts[ipart2]) theVerts[ipart2]->add_particle_in( thePart );
       
    }
    
    for (int ipart3 = 0; ipart3 < evtstdhep.getNPart(); ipart3++) {
      if (theVerts[ipart3]) theEvent->add_vertex( theVerts[ipart3] );
    }
    
}        

/*
void
EvtGenInterface::call_pygive(const std::string& iParm ) {
  
  //call the fortran routine pygive with a fortran string
  PYGIVE( iParm.c_str(), iParm.length() );  
 
}
*/

void 
EvtGenInterface::go_through_daughters(EvtParticle* part) {

  int NDaug=part->getNDaug();
  if(NDaug)
    {
      EvtParticle* Daughter;
      for (int i=0;i<NDaug;i++)
	{
	  Daughter=part->getDaug(i);
          int idHep = EvtPDL::getStdHep(Daughter->getId());
	  int found=0;
	  for(int k=0;k<nforced;k++)         
	    {
              if(idHep == forced_Hep[k])
		{ 
		  found = 1;
		  Daughter->deleteDaughters();
		}
	    }
	  if (!found) go_through_daughters(Daughter);
	}
    }
}

void 
EvtGenInterface::update_candlist( int theIndex, HepMC::GenParticle *thePart )
{
  if(nlist<10)                 // not nice ... but is 10 a reasonable maximum?
     {
       bool isThere = false;
       if (nlist) {
	 for (int check=0; check < nlist; check++) {
           if (listp[check]->barcode() == thePart->barcode()) isThere = true;
	 }
       }
       if (!isThere) { 
	 listp[nlist] = thePart;
	 index[nlist++] = theIndex;
       }
     }
  else
    {
      throw cms::Exception("runtime")
	<< "more than 10 candidates to be forced ";
      return; 
    }  
  return;
}

