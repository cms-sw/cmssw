// -*- C++ -*-
//
// Package:    PFTauDecayModeCutMultiplexer
// Class:      PFTauDecayModeCutMultiplexer
// 
/*

 Description: Applies a different cut to a PFTauDiscriminator, depending on the 
              the reconstructed DecayMode (stored by RecoTauTag/RecoTau/PFTauDecayModeIndexProducer) 
              in PFTauDiscriminator form.

              Produces a PFTauDiscriminator output with a binary (0 or 1) output.

              Cuts are specified in the decay mode PSets, which map the cuts 
              to collections of decay mode indices.  These decay mode PSets are defined
              in the same manner as TaNC MVA computers (which also map to specific decay modes)


*/
//
// Original Author:  Evan K. Friis, UC Davis (friis@physics.ucdavis.edu)
//         Created:  Thurs, April 16, 2009
// $Id: PFTauDecayModeCutMultiplexer.cc,v 1.3 2010/10/19 21:29:13 wmtan Exp $
//
//

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"

using namespace reco;

class PFTauDecayModeCutMultiplexer : public PFTauDiscriminationProducerBase {
   public:
      explicit PFTauDecayModeCutMultiplexer(const edm::ParameterSet&);
      ~PFTauDecayModeCutMultiplexer(){}

      struct  ComputerAndCut {
         std::string computerName;
         double userCut;
      };

      typedef std::vector<ComputerAndCut>    CutList;
      typedef std::map<int, CutList::iterator> DecayModeToCutMap;

      double discriminate(const PFTauRef& thePFTau);
      void beginEvent(const edm::Event& event, const edm::EventSetup& eventSetup);

   private:
      // PFTau discriminator continaing the decaymode index of the tau collection
      edm::InputTag                  pfTauDecayModeIndexSrc_;

      // Discriminant to multiplex cut on
      edm::InputTag                  discriminantToMultiplex_;

      DecayModeToCutMap         computerMap_;      //Maps decay mode to MVA implementation
      CutList                   computers_;

      edm::Handle<PFTauDiscriminator> pfTauDecayModeIndices; // holds the decay mode indices for the taus in the current event
      edm::Handle<PFTauDiscriminator> targetDiscriminant;    // holds the discirminant values of the discriminant we will multiplex for the current event
};

PFTauDecayModeCutMultiplexer::PFTauDecayModeCutMultiplexer(const edm::ParameterSet& iConfig):PFTauDiscriminationProducerBase(iConfig)
{
   pfTauDecayModeIndexSrc_  = iConfig.getParameter<edm::InputTag>("PFTauDecayModeSrc");
   discriminantToMultiplex_ = iConfig.getParameter<edm::InputTag>("PFTauDiscriminantToMultiplex");

   //get the computer/decay mode map
   std::vector<edm::ParameterSet> decayModeMap = iConfig.getParameter<std::vector<edm::ParameterSet> >("computers");
   computers_.reserve(decayModeMap.size());

   // for each decay mode MVA implementation (which may correspond to multiple decay modes, map the decay modes to the correct MVA computer
   for(std::vector<edm::ParameterSet>::const_iterator iComputer  = decayModeMap.begin();
                                            iComputer != decayModeMap.end();
                                          ++iComputer)
   {
      ComputerAndCut toInsert;
      toInsert.computerName = iComputer->getParameter<std::string>("computerName");
      toInsert.userCut      = iComputer->getParameter<double>("cut");
      CutList::iterator computerJustAdded = computers_.insert(computers_.end(), toInsert); //add this computer to the end of the list

      //populate the map
      std::vector<int> associatedDecayModes = iComputer->getParameter<std::vector<int> >("decayModeIndices");
      for(std::vector<int>::const_iterator iDecayMode  = associatedDecayModes.begin();
                                      iDecayMode != associatedDecayModes.end();
                                    ++iDecayMode)
      {
         //map this integer specifying the decay mode to the MVA comptuer we just added to the list
         std::pair<DecayModeToCutMap::iterator, bool> insertResult = computerMap_.insert(std::make_pair(*iDecayMode, computerJustAdded));

         //make sure we aren't double mapping a decay mode
         if(insertResult.second == false) { //indicates that the current key (decaymode) has already been entered!
            throw cms::Exception("PFTauDecayModeCutMultiplexer::ctor") << "A tau decay mode: " << *iDecayMode << " has been mapped to two different MVA implementations, "
                                                              << insertResult.first->second->computerName << " and " << toInsert.computerName 
                                                              << ". Please check the appropriate cfi file." << std::endl;
         }
      }
   }
}

// ------------ get the relevant decay mode index handles at the beginning of each event ------------
void
PFTauDecayModeCutMultiplexer::beginEvent(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

   iEvent.getByLabel(pfTauDecayModeIndexSrc_, pfTauDecayModeIndices);
   iEvent.getByLabel(discriminantToMultiplex_, targetDiscriminant);
}

double PFTauDecayModeCutMultiplexer::discriminate(const PFTauRef& pfTau)
{
   // get decay mode for current tau
   int decayMode = lrint( (*pfTauDecayModeIndices)[pfTau] ); //convert to int

   // get value we are trying to multiplex
   float valueToMultiplex = (*targetDiscriminant)[pfTau];

   // Get correct cut
   DecayModeToCutMap::iterator iterToComputer = computerMap_.find(decayMode);
   if(iterToComputer != computerMap_.end()) //if we don't have a MVA mapped to this decay mode, skip it, it fails.
   {
      // use the supplied cut to make a decision
      if (valueToMultiplex > iterToComputer->second->userCut) 
         return 1.0;
      else 
         return 0.0;
   }

   // no computer associated to this decay mode; it fails
   return 0.;
}

DEFINE_FWK_MODULE(PFTauDecayModeCutMultiplexer);
