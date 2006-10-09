#ifndef Alignment_CommonAlignmentAlgorithm_AlignmentAlgorithmPluginFactory_h
#define Alignment_CommonAlignmentAlgorithm_AlignmentAlgorithmPluginFactory_h

/// \class AlignmentAlgorithmPluginFactory
///  Plugin factory for alignment algorithm
///
///  \author F. Ronga - CERN
///

#include <PluginManager/PluginFactory.h>
#include <Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmBase.h>

// Forward declaration
namespace edm { class ParameterSet; }

class AlignmentAlgorithmPluginFactory : 
  public seal::PluginFactory<AlignmentAlgorithmBase* (const edm::ParameterSet&) >  
{
  
public:
  /// Constructor
  AlignmentAlgorithmPluginFactory();
  
  /// Return the plugin factory (unique instance)
  static AlignmentAlgorithmPluginFactory* get (void);
  
  /// Directly return the algorithm with given name and configuration
  static AlignmentAlgorithmBase* getAlgorithm( std::string name, 
											   const edm::ParameterSet& config );
  
private:
  static AlignmentAlgorithmPluginFactory theInstance;
  
};
#endif

