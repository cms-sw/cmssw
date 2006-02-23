/*
 *  $Date: 2006/02/13 21:57:56 $
 *  $Revision: 1.5 $
 *  \author Julia Yarba
 */

#include <ostream>

#include "IOMC/ParticleGuns/interface/BaseFlatGunSource.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "IOMC/EventVertexGenerators/interface/EventVertexGeneratorFactory.h"

#include <iostream>

#include "CLHEP/Random/RandFlat.h"

using namespace edm;
using namespace std;

BaseFlatGunSource::BaseFlatGunSource( const ParameterSet& pset,
                                      const InputSourceDescription& desc ) : 
  GeneratedInputSource (pset, desc),
  fEvt(0),
  fPDGTable( new DefaultConfig::ParticleDataTable("PDG Table") )
{

  ParameterSet pgun_params = pset.getParameter<ParameterSet>("PGunParameters") ;
  
  fPartIDs    = pgun_params.getParameter< vector<int> >("PartID");
  
  fMinEta     = pgun_params.getParameter<double>("MinEta");
  fMaxEta     = pgun_params.getParameter<double>("MaxEta");
  fMinPhi     = pgun_params.getParameter<double>("MinPhi");
  fMaxPhi     = pgun_params.getParameter<double>("MaxPhi");

  // hardcoded for now
  //
  fPDGTablePath = "/afs/cern.ch/sw/lcg/external/clhep/1.9.2.1/slc3_ia32_gcc323/data/HepPDT/" ;
  fPDGTableName = "PDG_mass_width_2002.mc"; // should it be 2004 table ?

  string TableFullName = fPDGTablePath + fPDGTableName ;
  ifstream PDFile( TableFullName.c_str() ) ;
  if( !PDFile ) 
  {
      throw cms::Exception("FileNotFound", "BaseFlatGunSource::BaseFlatGunSource()")
	<< "File " << TableFullName << " cannot be opened.\n";
  }

  HepPDT::TableBuilder tb(*fPDGTable) ;
  if ( !addPDGParticles( PDFile, tb ) ) { cout << " Error reading PDG !" << endl; }
  // the tb dtor fills fPDGTable

  // Vtx.Gen. (ideally, should be optional)
  //
  std::vector<std::string> names = pset.getParameterNames();       
  //see if 'VertexGenerator' is a parameter
  if(names.end() != std::find(names.begin(),names.end(), "VertexGenerator") ) 
  {
     ParameterSet pgun_vtxgen = pset.getParameter<ParameterSet>("VertexGenerator") ;
     std::auto_ptr<EventVertexGeneratorMakerBase> vertexGeneratorMaker(
        EventVertexGeneratorFactory::get()->create
          (pgun_vtxgen.getParameter<std::string> ("type")) );
     if(vertexGeneratorMaker.get()==0) 
     {
        // throw SimG4Exception("Unable to find the event vertex generator requested");
        throw edm::Exception(errors::Configuration,"Unable to find the event vertex generator requested") ;
     }
     fEventVertexGenerator = vertexGeneratorMaker->make(pgun_vtxgen) ;

     if (fEventVertexGenerator.get()==0) 
        throw edm::Exception(errors::Configuration,"EventVertexGenerator construction failed!");
    
  }
  
  fVerbosity = pset.getUntrackedParameter<int>( "Verbosity",0 ) ;
  
  cout << "Internal BaseFlatGunSource is initialzed" << endl ;
   
}

BaseFlatGunSource::~BaseFlatGunSource()
{
  if (fEvt != NULL) delete fEvt ; // double check
  delete fPDGTable;
}

HepMC::GenVertex* BaseFlatGunSource::generateEvtVertex() const
{

   double xx=0.;
   double yy=0.;
   double zz=0.;
   if ( fEventVertexGenerator.get()!=0)
   {
      CLHEP::Hep3Vector* VtxPos = fEventVertexGenerator.get()->newVertex() ;
      xx = VtxPos->x() ;
      yy = VtxPos->y() ;
      zz = VtxPos->z() ;
   }

   return new HepMC::GenVertex(CLHEP::HepLorentzVector(xx,yy,zz));

}
