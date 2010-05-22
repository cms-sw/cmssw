/*
 * PFTauMVAInputDiscriminantTranslator
 *
 * Translate a list of given MVA (i.e. TaNC)
 * variables into standard PFTauDiscriminators
 * to facilitate embeddeing them into pat::Taus
 *
 * Author: Evan K. Friis (UC Davis)
 *
 */

// system include files
#include <memory>
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTauTag/TauTagTools/interface/DiscriminantList.h"
#include "RecoTauTag/TauTagTools/interface/PFTauDiscriminantManager.h"
#include "DataFormats/TauReco/interface/PFTauDecayModeAssociation.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"

using namespace PFTauDiscriminants;

class PFTauMVAInputDiscriminantTranslator : public edm::EDProducer {
   public:
      struct DiscriminantInfo {
         PhysicsTools::AtomicId name;
         string collName;
         size_t index;
         float defaultValue;
      };

      PFTauMVAInputDiscriminantTranslator(const edm::ParameterSet&);
      void produce(Event&, const EventSetup&);

   private:
      InputTag pfTauDMSource_;
      InputTag pfTauSource_;
      std::vector<DiscriminantInfo> discriminators_;
      PFTauDiscriminantManager  discriminantManager_;
      DiscriminantList          myDiscriminants_;  
};



PFTauMVAInputDiscriminantTranslator::PFTauMVAInputDiscriminantTranslator(const edm::ParameterSet& pset)
{
   typedef vector<edm::ParameterSet> VPSet;
   pfTauDMSource_ = pset.getParameter<edm::InputTag>("decayModeSource");
   pfTauSource_ = pset.getParameter<edm::InputTag>("pfTauSource");

   VPSet discriminants = pset.getParameter<VPSet>("discriminants");

   for(VPSet::const_iterator iDisc = discriminants.begin(); 
         iDisc != discriminants.end(); ++iDisc)
   {
      edm::ParameterSet discPSet = *iDisc;
      // get discriminant name
      string name = discPSet.getParameter<string>("name");
      double defaultValue = (discPSet.exists("default")) ? discPSet.getParameter<double>("default") : 0.;
      // check if we are getting multiple indices
      bool requestMultiple = discPSet.exists("indices");
      if(requestMultiple)
      {
         // make a discrimiantor for each desired index
         vector<uint32_t> indices = discPSet.getParameter<vector<uint32_t> >("indices");
         for(vector<uint32_t>::const_iterator index = indices.begin();
               index != indices.end(); ++index)
         {
            DiscriminantInfo newDisc;
            newDisc.name = name;
            newDisc.index = *index;
            newDisc.defaultValue = defaultValue;
            // make a nice colleciton name
            stringstream collectionName;
            collectionName << name << *index;
            newDisc.collName = collectionName.str();
            discriminators_.push_back(newDisc);
         }
      } else 
      { 
         //single discriminant
         DiscriminantInfo newDisc;
         newDisc.name = name;
         newDisc.collName = name;
         newDisc.index = 0;
         newDisc.defaultValue = defaultValue;
         discriminators_.push_back(newDisc);
      }
   }
   // register products
   for(vector<DiscriminantInfo>::const_iterator iDisc = discriminators_.begin();
         iDisc != discriminators_.end(); ++iDisc)
   {
      produces<PFTauDiscriminator>(iDisc->collName);
   }

   // Load discriminants into the manager
   for(DiscriminantList::const_iterator aDiscriminant  = myDiscriminants_.begin();
                                        aDiscriminant != myDiscriminants_.end();
                                      ++aDiscriminant)
   {
      //load the discriminants into the discriminant manager
      discriminantManager_.addDiscriminant(*aDiscriminant);
   }
}

void PFTauMVAInputDiscriminantTranslator::produce(Event& evt, const EventSetup& es)
{
   typedef std::vector<PhysicsTools::Variable::Value> VarList;
   // Handle to get PFTaus to associated to
   edm::Handle<PFTauCollection> pfTaus; 
   evt.getByLabel(pfTauSource_, pfTaus);

   // Handle to PFTauDecayModes for given taus
   edm::Handle<PFTauDecayModeAssociation> pfTauDecayModes; 
   evt.getByLabel(pfTauDMSource_, pfTauDecayModes);

   // Make an MVA output object for each of the taus.  (Due to auto_ptr container issues
   // we want to loop over the output discriminators separately)

   size_t nTaus = pfTaus->size();
   // holder for the the MVA inputs for each tau
   vector<VarList> mvaVariablesForTau(nTaus);

   // Setup Disc. mananger for this event
   discriminantManager_.setEvent(evt, 1.0);

   // Compute the variables for each tau
   for(size_t iTau = 0; iTau < nTaus; ++iTau)
   {
      // get tau ref
      PFTauRef tauRef(pfTaus, iTau);
      // get associated PFTauDecayMode
      const PFTauDecayMode& decayMode = (*pfTauDecayModes)[tauRef];
      discriminantManager_.setTau(decayMode);
      discriminantManager_.buildMVAComputerLink(mvaVariablesForTau[iTau]);
   }

   // Now produce the output collections for each discriminant
   for(vector<DiscriminantInfo>::const_iterator iDisc = discriminators_.begin();
         iDisc != discriminators_.end(); ++iDisc)
   {
      auto_ptr<PFTauDiscriminator> output(new PFTauDiscriminator(RefProd<PFTauCollection>(pfTaus)));
      for(size_t iTau = 0; iTau < nTaus; ++iTau)
      {
         // get computed variables for this tau
         const VarList& computedMVADiscriminants = mvaVariablesForTau[iTau];
         // We're looking to pull out the discriminant with the matching name and index
         // get our desired index
         double varForThisTau = iDisc->defaultValue;
         size_t desired_index = iDisc->index + 1;
         // keep count of what instance of this variable we are on
         size_t instancesFound = 0;
         for(VarList::const_iterator mvaVariable = computedMVADiscriminants.begin(); 
               mvaVariable != computedMVADiscriminants.end(); ++mvaVariable)
         {
            // see if the name of this variable matches the desired one
            if(mvaVariable->getName() == iDisc->name)
               instancesFound++;
            // check if this was the desired instance
            if(desired_index == instancesFound)
            {
               varForThisTau = mvaVariable->getValue();
               break;
            }
         }
         output->setValue(iTau, varForThisTau);
      }
      evt.put(output, iDisc->collName);
   }
}

DEFINE_FWK_MODULE(PFTauMVAInputDiscriminantTranslator);
