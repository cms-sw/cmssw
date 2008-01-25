#include "PhysicsTools/PatUtils/interface/ElectronSelector.h"

using pat::ElectronSelector;

//______________________________________________________________________________
ElectronSelector::ElectronSelector( const edm::ParameterSet& config ) :
  selectionCfg_(config),
  selectionType_( config.getParameter<std::string>("type"))
{
}


//______________________________________________________________________________
const bool ElectronSelector::filter( const unsigned int& index, 
                               const edm::View<Electron>& electrons,
                               const ElectronIDmap& electronIDs ) const
{

  // List of possible selections
  if      ( selectionType_ == "none"       ) 
    {
      return true;
    }
  else if ( selectionType_ == "cut"        ) 
    {
      return electronID( index, electrons, electronIDs )->cutBasedDecision();
    }
  else if ( selectionType_ == "likelihood" )
    {
      double value = selectionCfg_.getParameter<double>("value");
      return electronID( index, electrons, electronIDs )->likelihood() > value;
    }
  else if ( selectionType_ == "neuralnet" ) // FIXME: Check sign of comparison!
    {
      double value = selectionCfg_.getParameter<double>("value");
      return electronID( index, electrons, electronIDs )->neuralNetOutput() > value;
    }
  else if ( selectionType_ == "custom"     ) 
    {
      return customSelection_( index, electrons );
    }


   // Throw! unknown configuration
   throw edm::Exception(edm::errors::Configuration) 
         << "Unknown electron ID selection " << selectionType_;

}


//______________________________________________________________________________
const reco::ElectronIDRef& 
ElectronSelector::electronID( const unsigned int& index,
                              const edm::View<Electron>& electrons,
                              const ElectronIDmap& electronIDs
                              ) const
{
  // Find electron ID for electron with index index
  edm::Ref<std::vector<Electron> > elecsRef = electrons.refAt(index).castTo<edm::Ref<std::vector<Electron> > >();
  ElectronIDmap::const_iterator electronID = electronIDs.find( elecsRef );

  // Return corresponding elecID
  return electronID->val;
}

//______________________________________________________________________________
const bool 
ElectronSelector::customSelection_( const unsigned int& index,
                                       const edm::View<Electron>& electrons ) const
{

  throw edm::Exception(edm::errors::UnimplementedFeature) 
        << "Custom selection not implemented yet";

}
