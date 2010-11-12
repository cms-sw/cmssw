#ifndef DataFormats_TauReco_RecoTauPiZero_h
#define DataFormats_TauReco_RecoTauPiZero_h

#include "DataFormats/Candidate/interface/CompositePtrCandidate.h"

namespace reco {
   class RecoTauPiZero : public CompositePtrCandidate {
      public:

         RecoTauPiZero():CompositePtrCandidate(),algoName_(""){ this->setPdgId(111); }
         RecoTauPiZero(const std::string& algoName):
           CompositePtrCandidate(), algoName_(algoName) { this->setPdgId(111); }

         /// constructor from values
         RecoTauPiZero(Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ),
               int pdgId = 111, int status = 0, bool integerCharge = true, const std::string& algoName=""):
            CompositePtrCandidate( q, p4, vtx, pdgId, status, integerCharge ),algoName_(algoName) {}

         /// constructor from values
         RecoTauPiZero(Charge q, const PolarLorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ),
               int pdgId = 111, int status = 0, bool integerCharge = true, const std::string& algoName="" ):
            CompositePtrCandidate( q, p4, vtx, pdgId, status, integerCharge ),algoName_(algoName) {}

         /// constructor from a Candidate
         explicit RecoTauPiZero( const Candidate & p, const std::string& algoName=""):
            CompositePtrCandidate(p),algoName_(algoName) { this->setPdgId(111); }

         /// destructor
         ~RecoTauPiZero(){};

         /// Number of PFGamma constituents
         size_t numberOfGammas() const;

         /// Number of electron constituents
         size_t numberOfElectrons() const;

         /// Maximum DeltaPhi between a constituent and the four vector
         double maxDeltaPhi() const;

         /// Maxmum DeltaEta between a constituent and the four vector
         double maxDeltaEta() const;

         /// Algorithm that built this piZero
         const std::string& algo() const;

         /// Check whether a given algo produced this pi zero
         bool algoIs(const std::string& algo) const;

         void print(std::ostream& out=std::cout) const;

      private:
         std::string algoName_;

   };

   std::ostream & operator<<(std::ostream& out, const RecoTauPiZero& c);

}

#endif
