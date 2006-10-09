/// \file
///
///  \author F. Ronga - CERN
///


#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmPluginFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


AlignmentAlgorithmPluginFactory AlignmentAlgorithmPluginFactory::theInstance;

//__________________________________________________________________________________________________
AlignmentAlgorithmPluginFactory::AlignmentAlgorithmPluginFactory() :
  seal::PluginFactory<AlignmentAlgorithmBase*
					  (const edm::ParameterSet&)>("AlignmentAlgorithmPluginFactory")
{
  
}


//__________________________________________________________________________________________________
AlignmentAlgorithmPluginFactory* AlignmentAlgorithmPluginFactory::get ()
{
  return &theInstance; 
}


//__________________________________________________________________________________________________
AlignmentAlgorithmBase* 
AlignmentAlgorithmPluginFactory::getAlgorithm( std::string name, const edm::ParameterSet& config )
{

  return theInstance.create( name, config );
  

}
