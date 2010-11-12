#include "DataFormats/TauReco/interface/RecoTauPiZero.h"
#include "DataFormats/Math/interface/deltaPhi.h"

namespace reco {

   size_t RecoTauPiZero::numberOfGammas() const
   {
      size_t nGammas = 0;
      size_t nDaughters = numberOfDaughters();
      for(size_t i = 0; i < nDaughters; ++i) {
         if(daughter(i)->pdgId() == 22) ++nGammas;
      }
      return nGammas;
   }

   size_t RecoTauPiZero::numberOfElectrons() const
   {
      size_t nElectrons = 0;
      size_t nDaughters = numberOfDaughters();
      for(size_t i = 0; i < nDaughters; ++i) {
         if(std::abs(daughter(i)->pdgId()) == 11) ++nElectrons;
      }
      return nElectrons;
   }

   double RecoTauPiZero::maxDeltaPhi() const
   {
      double maxDPhi = 0;
      size_t nDaughters = numberOfDaughters();
      for(size_t i = 0; i < nDaughters; ++i) {
         double dPhi = std::fabs(deltaPhi(*this, *daughter(i)));
         if(dPhi > maxDPhi)
            maxDPhi = dPhi;
      }
      return maxDPhi;
   }

   double RecoTauPiZero::maxDeltaEta() const
   {
      double maxDEta = 0;
      size_t nDaughters = numberOfDaughters();
      for(size_t i = 0; i < nDaughters; ++i) {
         double dEta = std::fabs(eta() - daughter(i)->eta());
         if(dEta > maxDEta)
            maxDEta = dEta;
      }
      return maxDEta;
   }

   const std::string& RecoTauPiZero::algo() const {
      return algoName_;
   }

   bool RecoTauPiZero::algoIs(const std::string& algo) const {
      return algoName_ == algo;
   }

   namespace {
      std::ostream& operator<<(std::ostream& out, const reco::Candidate::LorentzVector& p4)
      {
         out << "(mass/pt/eta/phi) ("  << std::setiosflags(std::ios::fixed) << std::setprecision(2)
            << p4.mass() << "/" << std::setprecision(1) << p4.pt() << "/" << std::setprecision(2) << p4.eta()
         << "/" << std::setprecision(2) << p4.phi() << ")";
         return out;
      }
   }

   void RecoTauPiZero::print(std::ostream& out) const {
      if (!out) return;

      out << "RecoTauPiZero: " << this->p4() <<
         " nDaughters: " << this->numberOfDaughters() <<
         " (gamma/e) (" << this->numberOfGammas() << "/" << this->numberOfElectrons() << ")" <<
         " maxDeltaPhi: " << std::setprecision(3) << maxDeltaPhi() <<
         " maxDeltaEta: "  << std::setprecision(3) << maxDeltaEta() <<
         " algo: " << algo() <<
         std::endl;

      for(size_t i = 0; i < this->numberOfDaughters(); ++i)
      {
         out << "--- daughter " << i << ": " << daughterPtr(i)->p4() <<
            " key: " << daughterPtr(i).key() << std::endl;
      }
   }

   std::ostream& operator<<(std::ostream& out, const reco::RecoTauPiZero& piZero)
   {
      if(!out) return out;
      piZero.print(out);
      return out;
   }
}
