#ifndef PhysicsTools_PatUtils_DuplicatedElectronRemover_h
#define PhysicsTools_PatUtils_DuplicatedElectronRemover_h

#include "PhysicsTools/PatUtils/interface/GenericDuplicateRemover.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/Common/interface/View.h"

#include <memory>
#include <vector>


namespace pat { 
    
    /* --- Original comment from TQAF follows ----
     * it is possible that there are multiple electron objects in the collection that correspond to the same
     * real physics object - a supercluster with two tracks reconstructed to it, or a track that points to two different SC
     *  (i would guess the latter doesn't actually happen).
     * NB triplicates also appear in the electron collection provided by egamma group, it is necessary to handle those correctly   
     */
    class DuplicatedElectronRemover {
        public:
            struct SameSuperclusterOrTrack {
                template<typename T1, typename T2>
                bool operator()(const T1 &t1, const T2 &t2) const { 
                    return ((t1.superCluster() == t2.superCluster()) ||
                            (t1.gsfTrack()     == t2.gsfTrack())); 
                }
            }; // struct

            struct BestEoverP {
                template<typename T1, typename T2>
                bool operator()(const T1 &t1, const T2 &t2) const { 
                    float diff1 = fabs(t1.eSuperClusterOverP()-1);
                    float diff2 = fabs(t2.eSuperClusterOverP()-1);
                    return diff1 <= diff2;
                }
            }; //struct

            // List of duplicate electrons to remove 
            // Among those that share the same cluster or track, the one with E/P nearer to 1 is kept
            std::auto_ptr< std::vector<size_t> > duplicatesToRemove(const std::vector<reco::GsfElectron> &electrons) const ;

            // List of duplicate electrons to remove 
            // Among those that share the same cluster or track, the one with E/P nearer to 1 is kept
            std::auto_ptr< std::vector<size_t> > duplicatesToRemove(const edm::View<reco::GsfElectron>   &electrons) const ;

            // Generic method. Collection can be vector, view or whatever you like
            template<typename Collection>
            std::auto_ptr< std::vector<size_t> > duplicatesToRemove(const Collection &electrons) const ;
            
        private:
    }; // class
} // namespace

// implemented here because is templated
template<typename Collection>
std::auto_ptr< std::vector<size_t> >
pat::DuplicatedElectronRemover::duplicatesToRemove(const Collection &electrons) const {
    pat::GenericDuplicateRemover<SameSuperclusterOrTrack,BestEoverP> dups;
    return dups.duplicates(electrons);
}
#endif
