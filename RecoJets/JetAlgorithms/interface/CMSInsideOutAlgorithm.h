#ifndef RecoJets_JetAlgorithms_CMSInsideOutAlgorithm_h
#define RecoJets_JetAlgorithms_CMSInsideOutAlgorithm_h

/** \class Inside-Out Algorithm
 *
 * description
 *
 * \author Evan K. Friis, UC Davis (friis@physics.ucdavis.edu)
 * \some code adapted from RecoJets/JetAlgorithms/CMSIterativeConeAlgorithm
 ************************************************************/


#include <list>
#include <algorithm>

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Math/interface/deltaR.h"
#include <limits>
#include <vector>
#include <list>

#include "fastjet/PseudoJet.hh"




class CMSInsideOutAlgorithm {
   public:
      typedef reco::Particle::LorentzVector LorentzVector;
      typedef std::list<fastjet::PseudoJet>::iterator inputListIter;
      // binary predicate to sort a std::list of std::list<InputItem> iterators by increasing deltaR
      // from a eta-phi point specified in the ctor
      class ListIteratorLesserByDeltaR {
         public:            ListIteratorLesserByDeltaR(const double& eta, const double& phi):seedEta_(eta),seedPhi_(phi){}
            bool operator()(const inputListIter& A, const inputListIter& B) const  {
               double deltaR2A = reco::deltaR2( (*A).eta(), seedEta_, (*A).phi(), seedPhi_ );
               double deltaR2B = reco::deltaR2( (*B).eta(), seedEta_, (*B).phi(), seedPhi_ );
               return 
                  fabs(deltaR2A - deltaR2B) > std::numeric_limits<double>::epsilon() ? deltaR2A < deltaR2B :
                  reco::deltaPhi((*A).phi(), seedPhi_) < reco::deltaPhi((*B).phi(), seedPhi_);
            };
         private:
            double seedEta_, seedPhi_;
      };

      /** Constructor
        \param seed defines the minimum ET in GeV of an object that can seed the jet
        \param growthparameter sets the growth parameter X, i.e. [dR < X/Et_jet]
        \param min/max size define the min/max size of jet in deltaR.  Min is included for resolution effects
        \param seedCharge can be the following values; -1 [don't care], 0 neutral only, 1 [tracks only]
        */
      CMSInsideOutAlgorithm(double seedObjectPt, double growthParameter, double maxSize, double minSize): 
         seedThresholdPt_(seedObjectPt),
         growthParameterSquared_(growthParameter*growthParameter),
         maxSizeSquared_(maxSize*maxSize),
         minSizeSquared_(minSize*minSize){};



	 /// Build from input candidate collection
 void run(const std::vector<fastjet::PseudoJet>& fInput, std::vector<fastjet::PseudoJet> & fOutput);

   private:
      double seedThresholdPt_;
      double growthParameterSquared_;
      double maxSizeSquared_;
      double minSizeSquared_;
};

#endif
