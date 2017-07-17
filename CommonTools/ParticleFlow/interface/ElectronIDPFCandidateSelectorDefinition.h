#ifndef CommonTools_ParticleFlow_ElectronIDPFCandidateSelectorDefinition
#define CommonTools_ParticleFlow_ElectronIDPFCandidateSelectorDefinition

/**
   \class    pf2pat::ElectronIDPFCandidateSelectorDefinition ElectronIDPFCandidateSelectorDefinition.h "CommonTools/ParticleFlow/interface/ElectronIDPFCandidateSelectorDefinition.h"
   \brief    Selects PFCandidates basing on cuts provided with string cut parser

   \author   Giovanni Petrucciani
   \version  $Id: ElectronIDPFCandidateSelectorDefinition.h,v 1.1 2011/01/28 20:56:44 srappocc Exp $
*/

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "CommonTools/ParticleFlow/interface/PFCandidateSelectorDefinition.h"
#include <algorithm>

namespace pf2pat {

  struct ElectronIDPFCandidateSelectorDefinition : public PFCandidateSelectorDefinition {

    ElectronIDPFCandidateSelectorDefinition ( const edm::ParameterSet & cfg, edm::ConsumesCollector && iC ) :
      electronsToken_( iC.consumes<reco::GsfElectronCollection>( cfg.getParameter< edm::InputTag >( "recoGsfElectrons" ) ) ),
      electronIdToken_( iC.consumes<edm::ValueMap<float> >( cfg.getParameter< edm::InputTag >( "electronIdMap" ) ) )
    {
        if (cfg.exists("bitsToCheck")) {
            isBitMap_ = true;
            mask_ = 0;
            if (cfg.existsAs<std::vector<std::string> >("bitsToCheck")) {
                std::vector<std::string> strbits = cfg.getParameter<std::vector<std::string> >("bitsToCheck");
                for (std::vector<std::string>::const_iterator istrbit = strbits.begin(), estrbit = strbits.end();
                        istrbit != estrbit; ++istrbit) {
                    if      (*istrbit == "id" )  { mask_ |= 1; }
                    else if (*istrbit == "iso")  { mask_ |= 2; }
                    else if (*istrbit == "conv") { mask_ |= 4; }
                    else if (*istrbit == "ip")   { mask_ |= 8; }
                    else throw cms::Exception("Configuration") << "ElectronIDPFCandidateSelector: " <<
                        "bitsToCheck allowed string values are only id(0), iso(1), conv(2), ip(3).\n" <<
                            "Otherwise, use uint32_t bitmask).\n";
                }
            } else if (cfg.existsAs<uint32_t>("bitsToCheck")) {
                mask_ = cfg.getParameter<uint32_t>("bitsToCheck");
            } else {
                throw cms::Exception("Configuration") << "ElectronIDPFCandidateSelector: " <<
                        "bitsToCheck must be either a vector of strings, or a uint32 bitmask.\n";
            }
        } else {
            isBitMap_ = false;
            value_ = cfg.getParameter<double>("electronIdCut");
        }
    }

    void select( const HandleToCollection & hc,
		 const edm::Event & e,
		 const edm::EventSetup& s) {
      selected_.clear();

      edm::Handle<reco::GsfElectronCollection> electrons;
      e.getByToken(electronsToken_, electrons);

      edm::Handle<edm::ValueMap<float> > electronId;
      e.getByToken(electronIdToken_, electronId);

      unsigned key=0;
      for( collection::const_iterator pfc = hc->begin();
	   pfc != hc->end(); ++pfc, ++key) {

        // Get GsfTrack for matching with reco::GsfElectron objects
        reco::GsfTrackRef PfTk = pfc->gsfTrackRef();

        // skip ones without GsfTrack: they won't be matched anyway
        if (PfTk.isNull()) continue;

        int match = -1;
        // try first the non-ambiguous tracks
        for (reco::GsfElectronCollection::const_iterator it = electrons->begin(), ed = electrons->end(); it != ed; ++it) {
            if (it->gsfTrack() == PfTk) { match = it - electrons->begin(); break; }
        }
        // then the ambiguous ones
        if (match == -1) {
            for (reco::GsfElectronCollection::const_iterator it = electrons->begin(), ed = electrons->end(); it != ed; ++it) {
                if (std::count(it->ambiguousGsfTracksBegin(), it->ambiguousGsfTracksEnd(), PfTk) > 0) {
                    match = it - electrons->begin(); break;
                }
            }
        }
        // if found, make a GsfElectronRef and read electron id
        if (match != -1) {
            reco::GsfElectronRef ref(electrons,match);
            float eleId = (*electronId)[ref];
            bool pass = false;
            if (isBitMap_) {
                uint32_t thisval = eleId;
                pass = ((thisval & mask_) == mask_);
            } else {
                pass = (eleId > value_);
            }
            if (pass) {
                selected_.push_back( reco::PFCandidate(*pfc) );
                reco::PFCandidatePtr ptrToMother( hc, key );
                selected_.back().setSourceCandidatePtr( ptrToMother );
            }
        }
      }
    }

    private:
        edm::EDGetTokenT<reco::GsfElectronCollection>  electronsToken_;
        edm::EDGetTokenT<edm::ValueMap<float> >  electronIdToken_;
        bool isBitMap_;
        uint32_t mask_;
        double   value_;
  };
}

#endif
