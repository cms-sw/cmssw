#ifndef RecoTauTag_TauTagTools_PFTauDiscriminantManager_h
#define RecoTauTag_TauTagTools_PFTauDiscriminantManager_h

#include "PhysicsTools/MVAComputer/interface/Variable.h"
#include "DataFormats/TauReco/interface/PFTauDecayMode.h"
#include "FWCore/Framework/interface/Event.h"
#include "RecoTauTag/TauTagTools/interface/PFTauDiscriminantBase.h"
#include "RecoTauTag/TauTagTools/interface/TauTagTools.h"
#include "TTree.h"

namespace PFTauDiscriminants
{

class Discriminant;

class PFTauDiscriminantManager {
   public:
      PFTauDiscriminantManager();
      ~PFTauDiscriminantManager();

      typedef std::vector<const reco::Candidate*> candPtrVector;
      //maps string (discriminant name, ( (discriminantComputer), links)
      typedef std::map<std::string, Discriminant* const> discriminantHolder;
      /// add a discriminant 
      void addDiscriminant(Discriminant* const aDiscriminant);
      /// add a set of branches ot the TTree 
      bool branchTree(TTree* const treeToBranch, bool addTargetBranch = false, bool addWeightBranch = false);
      /// connect to an MVA computer
      void buildMVAComputerLink(std::vector<PhysicsTools::Variable::Value>&);
      /// set objects for this discriminant
      bool setTau(const reco::PFTauDecayMode& theTau, bool prePass = false, bool preFail = false);
      /// in case there is no tau but you wish to fill anyway (for example, to see situations 
      /// where one cone algorithm fails to find a tau but another does not
      bool setNullResult();
      /// set the current event.  Must be called (once per event) before setTau or setNullResult
      void setEvent(const edm::Event&, double eventWeight);

      void setSignalFlag(bool isSignal) { iAmSignal_ = isSignal; };

      //TODO: Discriminant should be a friend and these should be private...

      /// returns associated PFTauDecayMode 
      const reco::PFTauDecayMode* getDecayMode() const { return currentTauDecayMode_; };
      /// returns associated edm::Event
      const edm::Event*           getEvent()     const { return eventData_; };

      /// get the 'main' track (track computed for relevancy to tau decay resonances) (ie pi- in pi+pi+pi-)
      const reco::Candidate*      mainTrack();
      /// accessed by Discriminant classes (caches to prevent multiple sorts)
      const std::vector<const reco::Candidate*>& signalObjectsSortedByPt();
      const std::vector<const reco::Candidate*>& signalObjectsSortedByDR();
      const std::vector<const reco::Candidate*>& outlierObjectsSortedByPt();
      const std::vector<const reco::Candidate*>& outlierObjectsSortedByDR();

      /// return the lowest level constituent candidates of a composite candidate
      static std::vector<const reco::Candidate*> getLeafDaughters(const reco::Candidate* input);

      /// 
      candPtrVector                         filterByCharge(const candPtrVector& input, bool isCharged) const;

   protected:

   private:
      // magic variables
      Bool_t                                                    iAmSignal_;
      Bool_t                                                    iAmNull_;
      Bool_t                                                    prePass_;
      Bool_t                                                    preFail_;

      Double_t                                                  eventWeight_;
      discriminantHolder                                        myDiscriminants_;
      const reco::PFTauDecayMode*                               currentTauDecayMode_;
      const edm::Event*                                         eventData_;

      void                                                      clearCache();
      //cached objects 
      const reco::Candidate*                                    mainTrack_;
      candPtrVector                 				signalObjectsSortedByPt_;
      candPtrVector                 				signalObjectsSortedByDR_;
      candPtrVector                 				outlierObjectsSortedByPt_;
      candPtrVector                 				outlierObjectsSortedByDR_;

      //utility functions for filling caches
      void                                                      fillSignalObjects(candPtrVector& input);                                     
      void                                                      fillOutlierObjects(candPtrVector& input);                                     
      void                                                      computeMainTrack();

};

}
#endif

