// -*- C++ -*-
//
// Package:    GenHFHadronMatcher
// Class:      GenHFHadronMatcher
//

/**\class GenHFHadronMatcher GenHFHadronMatcher.cc
* @brief Finds the origin of each heavy flavour hadron and associated jets to it
*
* Starting from each consituent of each jet, tracks back in chain to find heavy flavour hadrons.
* From each hadron traces back until finds the b quark and its mother.
* For each hadron identifies the jet to which it was injected as a ghost hadron.
*
* The description of the run-time parameters can be found at fillDescriptions()
*
* The description of the products can be found at GenHFHadronMatcher()
*/

// Original Author:  Nazar Bartosik,DESY


// system include files
#include <memory>
#include <utility>
#include <algorithm>


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"



//
// class declaration
//

class GenHFHadronMatcher : public edm::EDProducer
{
public:
    explicit GenHFHadronMatcher ( const edm::ParameterSet& );
    ~GenHFHadronMatcher();

    static void fillDescriptions ( edm::ConfigurationDescriptions& descriptions );

private:
    virtual void beginJob() ;
    virtual void produce ( edm::Event&, const edm::EventSetup& );
    virtual void endJob() ;

    virtual void beginRun ( edm::Run&, edm::EventSetup const& );
    virtual void endRun ( edm::Run&, edm::EventSetup const& );
    virtual void beginLuminosityBlock ( edm::LuminosityBlock&, edm::EventSetup const& );
    virtual void endLuminosityBlock ( edm::LuminosityBlock&, edm::EventSetup const& );

    typedef const reco::Candidate* pCRC;
    std::vector<int> findHadronJets ( const reco::GenJetCollection& genJets,  std::vector<int> &hadIndex, std::vector<reco::GenParticle> &hadMothersGenPart, 
                                      std::vector<std::vector<int> > &hadMothersIndices, std::vector<int> &hadLeptonIndex, 
                                      std::vector<int> &hadLeptonHadIndex, std::vector<int> &hadFlavour, 
                                      std::vector<int> &hadFromTopWeakDecay, std::vector<int> &hadBHadronId );
    int analyzeMothers ( const reco::Candidate* thisParticle, pCRC *hadron, pCRC *lepton, int& topDaughterQId, int& topBarDaughterQId, 
                         std::vector<const reco::Candidate*> &hadMothers, std::vector<std::vector<int> > &hadMothersIndices, 
                         std::set<const reco::Candidate*> *analyzedParticles, const int prevPartIndex );
    bool putMotherIndex ( std::vector<std::vector<int> > &hadMothersIndices, int partIndex, int mothIndex );
    bool isHadron ( const int flavour, const reco::Candidate* thisParticle );
    bool isHadronPdgId ( const int flavour, const int pdgId );
    bool hasHadronDaughter ( const int flavour, const reco::Candidate* thisParticle );
    int isInList ( std::vector<const reco::Candidate*> particleList, const reco::Candidate* particle );
    int isInList ( std::vector<int> list, const int value );
    int findInMothers ( int idx, std::vector<int> &mothChains, std::vector<std::vector<int> > &hadMothersIndices, 
                        std::vector<reco::GenParticle> &hadMothers, int status, int pdgId, bool pdgAbs, int stopId, int firstLast, bool verbose );
    bool isNeutralPdg ( int pdgId );

    bool checkForLoop ( std::vector<const reco::Candidate*> &particleChain, const reco::Candidate* particle );
    std::string getParticleName ( int id ) const;

    bool fixExtraSameFlavours(const unsigned int hadId, const std::vector<int> &hadIndices, const std::vector<reco::GenParticle> &hadMothers, 
                              const std::vector<std::vector<int> > &hadMothersIndices, const std::vector<int> &isFromTopWeakDecay, 
                              const std::vector<std::vector<int> > &LastQuarkIds, const std::vector<std::vector<int> > &LastQuarkMotherIds, 
                              std::vector<int> &lastQuarkIndices, std::vector<int> &hadronFlavour, std::set<int> &checkedHadronIds, const int lastQuarkIndex);

// ----------member data ---------------------------
    edm::InputTag genJets_;
    int flavour_;
    bool noBBbarResonances_;
    bool onlyJetClusteredHadrons_;

    std::string flavourStr_;  // Name of the flavour specified in config file

    edm::ESHandle<ParticleDataTable> pdt_;


};

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
/**
* @brief constructor initialising producer products and config parameters
*
* Output generated by this producer:
* <TABLE>
* <TR><TH> name                </TH><TH> type                           </TH><TH> description </TH> </TR>
* <TR><TD> BHadJetIndex        </TD><TD> std::vector<int>               </TD><TD> position of the jet identified as b-hadron jet in the input GenJetCollection </TD></TR>
* <TR><TD> BHadrons            </TD><TD> std::vector<reco::GenParticle> </TD><TD> vector of the identified b-hadrons (definition of b-hadron and anti-b-hadron below) </TD></TR>
* <TR><TD> BHadronFromTopB     </TD><TD> std::vector<bool>              </TD><TD> true if the corresponding b-hadron originates from the ttGenEvent b-quark </TD></TR>
* <TR><TD> BHadronVsJet        </TD><TD> std::vector<int>               </TD><TD> matrix of which b-hadron appears in which GenJet, access by [iJet * BHadrons.size() + iBhadron] </TD></TR>
*
* <TR><TD> AntiBHadJetIndex    </TD><TD> std::vector<int>               </TD><TD> position of the jet identified as anti-b-hadron jet in the input GenJetCollection </TD></TR>
* <TR><TD> AntiBHadrons        </TD><TD> std::vector<reco::GenParticle> </TD><TD> vector of the identified anti-b-hadrons (definition of b-hadron and anti-b-hadron below) </TD></TR>
* <TR><TD> AntiBHadronFromTopB </TD><TD> std::vector<bool>              </TD><TD> true if the corresponding anti-b-hadron originates from the ttGenEvent anti-b-quark </TD></TR>
* <TR><TD> AntiBHadronVsJet    </TD><TD> std::vector<int>               </TD><TD> matrix of which anti-b-hadron appears in which GenJet, access by [iJet * AntiBHadrons.size() + iBhadron] </TD></TR>
*
* </TABLE>
*
* @warning Definition of b-hadron and anti-b-hadron: The term b-hadron and anti-b-hadron is in reference to the quark content and <b>not</b> to distinguish particles from anti-particles.
* Here a b-hadron contains a b-quark and an anti-b-hadron contains an anti-b-quark.
* For mesons this means an inversion with respect to the PDG definition, as mesons actually contain anti-b-quarks and anti-mesons contain b-quarks.
*
*/
GenHFHadronMatcher::GenHFHadronMatcher ( const edm::ParameterSet& cfg )
{

    genJets_           = cfg.getParameter<edm::InputTag> ( "genJets" );
    flavour_           = cfg.getParameter<int> ( "flavour" );
    noBBbarResonances_ = cfg.getParameter<bool> ( "noBBbarResonances" );
    onlyJetClusteredHadrons_ = cfg.getParameter<bool> ( "onlyJetClusteredHadrons" );
    
    flavour_ = abs ( flavour_ ); // Make flavour independent of sign given in configuration
    if ( flavour_==5 ) {
        flavourStr_="B";
    } else if ( flavour_==4 ) {
        flavourStr_="C";
    } else {
        edm::LogError ( "GenHFHadronMatcher" ) << "Flavour option must be 4 (c-jet) or 5 (b-jet), but is: " << flavour_ << ". Correct this!";
    }

    // Hadron matching products
    produces< std::vector<reco::GenParticle> > ( "gen"+flavourStr_+"HadPlusMothers" ); // All mothers in all decay chains above any hadron of specified flavour
    produces< std::vector< std::vector<int> > > ( "gen"+flavourStr_+"HadPlusMothersIndices" ); // Indices of mothers of each hadMother
    produces< std::vector<int> > ( "gen"+flavourStr_+"HadIndex" ); // Index of hadron in the vector of hadMothers
    produces< std::vector<int> > ( "gen"+flavourStr_+"HadFlavour" ); // PdgId of the first non-b(c) quark mother with sign corresponding to hadron charge
    produces< std::vector<int> > ( "gen"+flavourStr_+"HadJetIndex" ); // Index of genJet matched to each hadron by jet clustering algorithm
    produces< std::vector<int> > ( "gen"+flavourStr_+"HadLeptonIndex" ); // Index of lepton found among the hadron decay products in the list of mothers
    produces< std::vector<int> > ( "gen"+flavourStr_+"HadLeptonHadronIndex" ); // Index of hadron the lepton is associated to
    produces< std::vector<int> > ( "gen"+flavourStr_+"HadFromTopWeakDecay" ); // Tells whether the hadron appears in the chain after top decay
    produces< std::vector<int> > ( "gen"+flavourStr_+"HadBHadronId" ); // Index of a b-hadron which the current hadron comes from (for c-hadrons)

}

GenHFHadronMatcher::~GenHFHadronMatcher()
{
}


// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
/**
* @brief description of the run-time parameters
*
* <TABLE>
* <TR><TH> name                </TH><TH> description </TH> </TR>
* <TR><TD> genJets             </TD><TD> input GenJet collection </TD></TR>
* <TR><TD> noBBbarResonances   </TD><TD> exclude resonances to be identified as hadrons </TD></TR>
* <TR><TD> onlyJetClusteredHadrons   </TD><TD> Whether only hadrons, injected to jets, shold be analyzed. Runs x1000 faster in Sherpa. Hadrons not clustered to jets will not be identified.
* <TR><TD> resolveParticleName </TD><TD> print particle name during warning and debug output instead of PDG ID </TD></TR>
* <TR><TD> flavour		</TD><TD> flavour of weakly decaying hadron that the jets should be matched to (5-b, 4-c) </TD></TR>
* </TABLE>
*
*/
void GenHFHadronMatcher::fillDescriptions ( edm::ConfigurationDescriptions& descriptions )
{

    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag> ( "genJets",edm::InputTag ( "ak5GenJets","","HLT" ) )->setComment ( "Input GenJet collection" );
    desc.add<bool> ( "noBBbarResonances",true )->setComment ( "Exclude resonances to be identified as hadrons" );
    desc.add<bool> ( "onlyJetClusteredHadrons",false )->setComment ( "Whether only jets, matched to at least one hadron, should be analysed. Runs x1000 faster in Sherpa. Jets that don't have clustered hadrons will be skipped." );
    desc.add<int> ( "flavour",5 )->setComment ( "Flavour of weakly decaying hadron that should be added to the jet for futher tagging. (4-c, 5-b)" );
    descriptions.add ( "matchGenHFHadron",desc );
}



//
// member functions
//

// ------------ method called to produce the data  ------------
void GenHFHadronMatcher::produce ( edm::Event& evt, const edm::EventSetup& setup )
{

    setup.getData ( pdt_ );
//     printf("Run: %d\tLumi: %d\tEvent: %d\n", evt.id().run(), evt.luminosityBlock(), evt.id().event());

    using namespace edm;

    edm::Handle<reco::GenJetCollection> genJets;
    evt.getByLabel ( genJets_, genJets );


    // Defining adron matching variables
    std::auto_ptr<std::vector<reco::GenParticle> > hadMothers ( new std::vector<reco::GenParticle> );
    std::auto_ptr<std::vector<std::vector<int> > > hadMothersIndices ( new std::vector<std::vector<int> > );
    std::auto_ptr<std::vector<int> > hadIndex ( new std::vector<int> );
    std::auto_ptr<std::vector<int> > hadFlavour ( new std::vector<int> );
    std::auto_ptr<std::vector<int> > hadJetIndex ( new std::vector<int> );
    std::auto_ptr<std::vector<int> > hadLeptonIndex ( new std::vector<int> );
    std::auto_ptr<std::vector<int> > hadLeptonHadIndex ( new std::vector<int> );
    std::auto_ptr<std::vector<int> > hadFromTopWeakDecay ( new std::vector<int> );
    std::auto_ptr<std::vector<int> > hadBHadronId ( new std::vector<int> );
    
    LogDebug ( flavourStr_+"Jet (new)" ) << "searching for "<< flavourStr_ <<"-jets in " << genJets_;
    *hadJetIndex = findHadronJets ( *genJets, *hadIndex, *hadMothers, *hadMothersIndices, *hadLeptonIndex, *hadLeptonHadIndex, *hadFlavour, *hadFromTopWeakDecay, *hadBHadronId );

    // Putting products to the event
    evt.put ( hadMothers,         "gen"+flavourStr_+"HadPlusMothers" );
    evt.put ( hadMothersIndices,  "gen"+flavourStr_+"HadPlusMothersIndices" );
    evt.put ( hadIndex,           "gen"+flavourStr_+"HadIndex" );
    evt.put ( hadFlavour,         "gen"+flavourStr_+"HadFlavour" );
    evt.put ( hadJetIndex,        "gen"+flavourStr_+"HadJetIndex" );
    evt.put ( hadLeptonIndex,     "gen"+flavourStr_+"HadLeptonIndex" );
    evt.put ( hadLeptonHadIndex,  "gen"+flavourStr_+"HadLeptonHadronIndex" );
    evt.put ( hadFromTopWeakDecay,"gen"+flavourStr_+"HadFromTopWeakDecay" );
    evt.put ( hadBHadronId,     "gen"+flavourStr_+"HadBHadronId" );
}

// ------------ method called once each job just before starting event loop  ------------
void GenHFHadronMatcher::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void GenHFHadronMatcher::endJob()
{
}

// ------------ method called when starting to processes a run  ------------
void GenHFHadronMatcher::beginRun ( edm::Run&, edm::EventSetup const& )
{
}

// ------------ method called when ending the processing of a run  ------------
void
GenHFHadronMatcher::endRun ( edm::Run&, edm::EventSetup const& )
{
}

// ------------ method called when starting to processes a luminosity block  ------------
void GenHFHadronMatcher::beginLuminosityBlock ( edm::LuminosityBlock&, edm::EventSetup const& )
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void GenHFHadronMatcher::endLuminosityBlock ( edm::LuminosityBlock&, edm::EventSetup const& )
{
}



/**
* @brief identify the jets that contain b-hadrons
*
* All jets originating from a b-hadron with the right b (c) content (b or anti-b) are identified in the GenJetCollection.
* The b (c) jet is identified by searching for a hadron of corresponding flavour in the jet. Hadrons are put into jets
* by "TopAnalysis.TopUtils.GenJetParticles" plugin.
* For each hadron all mothers from all levels and chains are analysed to find the quark or gluon from which the hadron has originated.
* This is done by searching through the generator particle decay tree starting from the hadron, performing checks for flavour and kinematic consistency.
* The hadrons that are not last in the decay chain (don't decay weakly) are skipped.
* Additionally for each hadron it is checked whether it comes from the top weak decay branch and whether it comes from some other b-hadron decay
*
* b-bbar (c-cbar) resonances can either be considered as hadrons or not, depending on the configuration.
*
* @param[in] genJets the GenJetCollection to be searched
* @param[out] hadIndex vector of indices of found hadrons in hadMothers
* @param[out] hadMothers vector of all mothers at all levels of each found hadron
* @param[out] hadMothersIndices connection between each particle from hadMothers and its mothers
* @param[out] hadLeptonIndex index of lepton among the hadMothers
* @param[out] hadLeptonHadIndex index of hadron associated to each lepton
* @param[out] hadFlavour flavour of each found hadron
* @param[out] hadFromTopWeakDecay whether each hadron contains the top quark in its decay chain [works only for B-Hadrons]
* @param[out] hadBHadronId for each hadron - index of the ancestor b-hadron [-1 if hadron doesn't come from another b-hdaron]
* 
* @returns vector of jet indices that were matched to each hadron [by the jet clustering algorithm]
*/
std::vector<int> GenHFHadronMatcher::findHadronJets ( const reco::GenJetCollection& genJets, std::vector<int> &hadIndex, 
                                                      std::vector<reco::GenParticle> &hadMothers, std::vector<std::vector<int> > &hadMothersIndices, 
                                                      std::vector<int> &hadLeptonIndex, std::vector<int> &hadLeptonHadIndex, std::vector<int> &hadFlavour, 
                                                      std::vector<int> &hadFromTopWeakDecay, std::vector<int> &hadBHadronId )
{
    std::vector<int> result;
    std::vector<const reco::Candidate*> hadMothersCand;

    const unsigned int nJets = genJets.size();
    int topDaughterQId = -1;
    int topBarDaughterQId= -1;

    for ( size_t iJet = 0; iJet < nJets; ++iJet )  {
        const reco::GenJet* thisJet = & ( genJets.at(iJet) );
        std::vector<const reco::GenParticle*> particles = thisJet->getGenConstituents();
        
        //############################ WILL WORK X500 FASTER IN SHERPA
        //########## HADRONS NOT CLUSTERED TO ANY JET ARE LOST (~1-2%)
        if(onlyJetClusteredHadrons_) {
            bool hasClusteredHadron = false;
            // Skipping jets that don't have clustered hadrons
            for(const reco::GenParticle* particle : particles) {
                if(!isHadronPdgId(flavour_, particle->pdgId())) continue;
                hasClusteredHadron = true; 
                break;
            }
            if(!hasClusteredHadron) continue;
        }   // If jets without clustered hadrons should be skipped

        for ( unsigned int iParticle = 0; iParticle < particles.size(); ++iParticle ) {
            const reco::GenParticle* thisParticle = particles[iParticle];
            const reco::Candidate* hadron = 0;
            const reco::Candidate* lepton = 0;
            
            if ( thisParticle->status() > 1 ) continue;    // Skipping non-final state particles (e.g. bHadrons)

            int hadronIndex = analyzeMothers ( thisParticle, &hadron, &lepton, topDaughterQId, topBarDaughterQId, hadMothersCand, hadMothersIndices, 0, -1 );
            if ( hadron ) { // Putting hadron index to the list if it is not yet
//                 std::cout << "hadron: " << hadron << "  lepton: " << lepton << std::endl;
                // Storing the index of the hadron and lepton
                int hadListIndex = isInList(hadIndex, hadronIndex);
                if ( hadListIndex<0 ) {
                    hadIndex.push_back ( hadronIndex );
                    hadListIndex = hadIndex.size()-1;
                }
                // Adding the lepton index to the list of leptons
                if ( lepton ) {
                    int leptonId = isInList(hadMothersCand, lepton);
                    if(isInList(hadLeptonIndex, leptonId)<0) {
                        hadLeptonIndex.push_back( leptonId );
                        hadLeptonHadIndex.push_back( hadListIndex );
                    }
                }
            }   // If hadron has been found in the chain
        }   // End of loop over jet consituents
    }   // End of loop over jets

    for ( int i=0; i< ( int ) hadMothersCand.size(); i++ ) {
        hadMothers.push_back ( ( *dynamic_cast<const reco::GenParticle*> ( hadMothersCand.at(i) ) ) );
    }

    // Checking mothers of hadrons in order to assign flags (where the hadron comes from)
    unsigned int nHad = hadIndex.size();

    std::vector<std::vector<int> > LastQuarkMotherIds;
    std::vector<std::vector<int> > LastQuarkIds;
    std::vector<int> lastQuarkIndices(nHad, -1);

    // Looping over all hadrons
    for ( unsigned int hadNum=0; hadNum<nHad; hadNum++ ) {
        
        unsigned int hadIdx = hadIndex.at(hadNum);   // Index of hadron in the hadMothers
        const reco::GenParticle* hadron = &hadMothers.at(hadIdx);
        int jetIndex = -1;

        // Looping over all jets to match them to current hadron
        for ( unsigned int jetNum = 0; jetNum < nJets; jetNum++ ) {
            // Checking whether jet contains this hadron in it (that was put in jet by clustering algorithm)
            std::vector<const reco::GenParticle*> particles = genJets.at(jetNum).getGenConstituents();
            for ( unsigned int partNum=0; partNum<particles.size(); partNum++ ) {
                const reco::GenParticle* particle = particles.at(partNum);
                if ( particle->status() <2 ) {
                    continue;    // Skipping final state particles
                }
                if ( !isHadron ( flavour_, particle ) ) {
                    continue;
                }
                // Checking whether hadron and particle in jet are identical
                if ( hadron->pdgId() !=particle->pdgId() || fabs ( hadron->eta()-particle->eta() ) >0.00001 || fabs ( hadron->phi()-particle->phi() ) >0.00001 ) {
                    continue;
                }
                jetIndex=jetNum;
                break;
            }   // End of loop over jet constituents
            if ( jetIndex>=0 ) {
                break;
            }
        }   // End of loop over jets

        result.push_back ( jetIndex );  // Putting jet index to the result list

        std::vector <int> FirstQuarkId;
        std::vector <int> LastQuarkId;
        std::vector <int> LastQuarkMotherId;

        int hadFlav = hadMothers.at(hadIdx).pdgId() <0?-1:1; // Charge of the hadron (-1,1)
        if ( abs ( hadMothers.at(hadIdx).pdgId() ) /1000 < 1 ) {
            hadFlav*=-1;    // Inverting flavour of hadron if it is a meson
        }

        // Searching only first quark in the chain with the same flavour as hadron
        findInMothers ( hadIdx, FirstQuarkId, hadMothersIndices, hadMothers, 0, hadFlav*flavour_, false, -1, 1, false );

        // Finding last quark for each first quark
        for ( unsigned int qId=0; qId<FirstQuarkId.size(); qId++ ) {
            // Identifying the flavour of the first quark to find the last quark of the same flavour
            int bQFlav = hadMothers.at(FirstQuarkId.at(qId)).pdgId() < 0?-1:1;
            // Finding last quark of the hadron starting from the first quark
            findInMothers ( FirstQuarkId.at(qId), LastQuarkId, hadMothersIndices, hadMothers, 0, bQFlav*flavour_, false, -1, 2, false );
        }		// End of loop over all first quarks of the hadron
        
//         printf("First quarks: %d\n", (int)FirstQuarkId.size());
//         for(int qId : FirstQuarkId) printf("   %d.\tPdg: %d\tPt: %.3f\n", qId, hadMothers.at(qId).pdgId(), hadMothers.at(qId).pt());
//         printf("Last quarks: %d\n", (int)LastQuarkId.size());
//         for(int qId : LastQuarkId) {
//             printf("   %d.\tPdg: %d\tPt: %.3f\n", qId, hadMothers.at(qId).pdgId(), hadMothers.at(qId).pt());
//             int mother1Id = hadMothersIndices.at(qId).at(0);
//             if(mother1Id>=0) {
//                 const reco::GenParticle mother1 = hadMothers.at(mother1Id);
//                 printf("Mother 1: Pdg: %d\tPt: %.3f\n", mother1.pdgId(), mother1.pt());
//                 int mother2Id = hadMothersIndices.at(mother1Id).at(0);
//                 if(mother2Id>=0) {
//                     const reco::GenParticle mother2 = hadMothers.at(mother2Id);
//                     printf("Mother 2: Pdg: %d\tPt: %.3f\n", mother2.pdgId(), mother2.pt());
//                 } else printf("NO MOTHER 2");
//             } else printf("NO MOTHER 1");
//         }
//         if(hadFromTopWeakDecay.at(hadNum)) printf("From TOP\n"); else printf("NOT From TOP\n");
//         printf("\n\n");

        // Setting initial flavour of the hadron
        int hadronFlavour = 0;

        // Initialising pairs of last quark index and its distance from the hadron (to sort by distance)
        std::vector<std::pair<double, int> > lastQuark_dR_id_pairs;

        // Finding the closest quark in dR
        for ( unsigned int qId=0; qId<LastQuarkId.size(); qId++ ) {
            int qIdx = LastQuarkId.at(qId);
            // Calculating the dR between hadron and quark
            float dR = deltaR ( hadMothers.at(hadIdx).eta(),hadMothers.at(hadIdx).phi(),hadMothers.at(qIdx).eta(),hadMothers.at(qIdx).phi() );

            std::pair<double, int> dR_hadId_pair(dR,qIdx);
            lastQuark_dR_id_pairs.push_back(dR_hadId_pair);
        }		// End of loop over all last quarks of the hadron

        std::sort(lastQuark_dR_id_pairs.begin(), lastQuark_dR_id_pairs.end());
        
        if(lastQuark_dR_id_pairs.size()>1) {
            double dRratio = (lastQuark_dR_id_pairs.at(1).first - lastQuark_dR_id_pairs.at(0).first)/lastQuark_dR_id_pairs.at(1).first;
            int qIdx_closest = lastQuark_dR_id_pairs.at(0).second;
            LastQuarkId.clear();
            if(dRratio>0.5) LastQuarkId.push_back(qIdx_closest); 
            else for(std::pair<double, int> qIdDrPair : lastQuark_dR_id_pairs) LastQuarkId.push_back(qIdDrPair.second);
        }
        for(int qIdx : LastQuarkId) {
            int qmIdx = hadMothersIndices.at ( qIdx ).at(0);
            LastQuarkMotherId.push_back( qmIdx );
        }

        if((int)LastQuarkId.size()>0) lastQuarkIndices.at(hadNum) = 0;     // Setting the first quark in array as a candidate if it exists

        LastQuarkIds.push_back( LastQuarkId );

        LastQuarkMotherIds.push_back ( LastQuarkMotherId );

        if(LastQuarkMotherId.size()<1) {
            hadronFlavour = 0;
        } else {
            int qIdx = LastQuarkId.at( lastQuarkIndices.at(hadNum) );
            int qFlav = ( hadMothers.at(qIdx).pdgId() < 0 ) ? -1 : 1;
            hadronFlavour = qFlav*std::abs( hadMothers.at( LastQuarkMotherId.at( lastQuarkIndices.at(hadNum) ) ).pdgId() );
        }
        hadFlavour.push_back(hadronFlavour);    // Adding hadron flavour to the list of flavours
        
        // Checking whether hadron comes from the Top weak decay
        int isFromTopWeakDecay = 1;
        std::vector <int> checkedParticles;
        if(hadFlavour.at(hadNum)!=0) {
            int lastQIndex = LastQuarkId.at(lastQuarkIndices.at(hadNum));
            bool fromTB = topDaughterQId>=0?findInMothers( lastQIndex, checkedParticles, hadMothersIndices, hadMothers, -1, 0, false, topDaughterQId, 2, false ) >= 0 : false;
            checkedParticles.clear();
            bool fromTbarB = topBarDaughterQId>=0?findInMothers( lastQIndex, checkedParticles, hadMothersIndices, hadMothers, -1, 0, false, topBarDaughterQId, 2, false) >= 0:false;
            checkedParticles.clear();
            if(!fromTB && !fromTbarB) {
                isFromTopWeakDecay = 0;
            }
        } else isFromTopWeakDecay = 2;
        hadFromTopWeakDecay.push_back(isFromTopWeakDecay);
        int bHadronMotherId = findInMothers( hadIdx, checkedParticles, hadMothersIndices, hadMothers, 0, 555555, true, -1, 1, false );
        hadBHadronId.push_back(bHadronMotherId);
        

        if(LastQuarkMotherId.size()>0) {
            std::set<int> checkedHadronIds;
            fixExtraSameFlavours(hadNum, hadIndex, hadMothers, hadMothersIndices, hadFromTopWeakDecay, LastQuarkIds, LastQuarkMotherIds, lastQuarkIndices, hadFlavour, checkedHadronIds, 0);
        }
        
    }	// End of loop over all hadrons


    return result;
}


/**
* @brief Check if the cpecified particle is already in the list of particles
*
* @param[in] particleList list of particles to be checked
* @param[in] particle particle that should be checked
*
* @returns the index of the particle in the list [-1 if particle not found]
*/
int GenHFHadronMatcher::isInList ( std::vector<const reco::Candidate*> particleList, const reco::Candidate* particle )
{
    for ( unsigned int i = 0; i<particleList.size(); i++ )
        if ( particleList.at(i)==particle ) {
            return i;
        }

    return -1;
}

int GenHFHadronMatcher::isInList ( std::vector<int> list, const int value )
{
    for ( unsigned int i = 0; i<list.size(); i++ )
        if ( list.at(i)==value ) {
            return i;
        }

    return -1;
}


/**
* @brief Check the pdgId of a given particle if it is a hadron
*
* @param[in] flavour flavour of a hadron that is being searched (5-B, 4-C)
* @param[in] thisParticle a particle that is to be analysed
*
* @returns whether the particle is a hadron of specified flavour
*/
bool GenHFHadronMatcher::isHadron ( const int flavour, const reco::Candidate* thisParticle )
{
    return isHadronPdgId(flavour, thisParticle->pdgId());
}


/**
* @brief Check the pdgId if it represents a hadron of particular flavour
*
* @param[in] flavour flavour of a hadron that is being searched (5-B, 4-C)
* @param[in] pdgId pdgId to be checked
*
* @returns if the pdgId represents a hadron of specified flavour
*/
bool GenHFHadronMatcher::isHadronPdgId ( const int flavour, const int pdgId )
{
    int flavour_abs = std::abs(flavour);
    if(flavour_abs > 5 || flavour_abs < 1) return false;
    int pdgId_abs = std::abs(pdgId);

    if ( pdgId_abs / 1000 == flavour_abs // baryons
            || ( pdgId_abs / 100 % 10 == flavour_abs // mesons
                 && ! ( noBBbarResonances_ && pdgId_abs / 10 % 100 == 11*flavour_abs ) // but not a resonance
               )
       ) {
        return true;
    } else {
        return false;
    }
}


/**
* @brief Check if the particle has bHadron among daughters
*
* @param[in] flavour flavour of a hadron that is being searched (5-B, 4-C)
* @param[in] thisParticle a particle that is to be analysed
*
* @returns whether the particle has a hadron among its daughters
*/
bool GenHFHadronMatcher::hasHadronDaughter ( const int flavour, const reco::Candidate* thisParticle )
{
// Looping through daughters of the particle
    bool hasDaughter = false;
    for ( int k=0; k< ( int ) thisParticle->numberOfDaughters(); k++ ) {
        if ( !isHadron ( flavour, thisParticle->daughter ( k ) ) ) {
            continue;
        }
        hasDaughter = true;
        break;
    }
    return hasDaughter;
}


/**
* @brief do a recursive search for the mother particles until the b-quark is found or the absolute mother is found
*
* the treatment of b-bar resonances depends on the global parameter noBBbarResonances_
*
* @param[in] thisParticle current particle from which starts the search of the hadron and all its mothers up to proton
* @param[out] hadron the last hadron in the decay chain [that decays weekly]
* @param[out] lepton lepton found in the current decay chain
* @param[out] topDaughterQId Id of the top quark daughter b(c) quark
* @param[out] topBarDaughterQId Id of the antitop quark daughter b(c) quark
* @param[out] hadMothers list of all processed particles ending with proton
* @param[out] hadMothersIndices list of i-vectors containing j-indices representing particles that are mothers of each i-particle from hadMothers
* @param[out] analyzedParticles list of particles analysed in the chain [used for infinite loop detection]
* @param[out] prevPartIndex index of the previous particle in the current chain [used for infinite loop detection]
*
* @returns index of hadron in the hadMothers list [-1 if no hadron found]
*/
int GenHFHadronMatcher::analyzeMothers ( const reco::Candidate* thisParticle, pCRC *hadron, pCRC *lepton, int& topDaughterQId, int& topBarDaughterQId, std::vector<const reco::Candidate*> &hadMothers, std::vector<std::vector<int> > &hadMothersIndices, std::set<const reco::Candidate*> *analyzedParticles, const int prevPartIndex )
{

    int hadronIndex=-1;	// Index of the hadron that is returned by this function
    // Storing the first hadron has been found in the chain when going up from the final particle of the jet
    if ( *hadron == 0 // find only the first b-hadron on the way (the one that decays weekly)
            && isHadron ( flavour_, thisParticle ) // is a hadron
            && !hasHadronDaughter ( flavour_, thisParticle ) // has no hadron daughter (decays weekly)
       ) {
        *hadron = thisParticle;

        int index = isInList ( hadMothers, thisParticle );

        if ( index<0 ) { // If hadron is not in the list of mothers yet
            hadMothers.push_back ( thisParticle );
            hadronIndex=hadMothers.size()-1;
        } else {	    // If hadron is in the list of mothers already
            hadronIndex=index;
        }
    }
    
    // Checking if the particle is a lepton
    if(!*hadron){
        int absPdg = std::abs(thisParticle->pdgId());
        if(absPdg==11 || absPdg==13) {
            const reco::Candidate* mother1 = 0;
            if(thisParticle->numberOfMothers()>0) mother1 = thisParticle->mother(0);
            // If the lepton's mother is a hadron
            if(mother1 && isHadron(flavour_, mother1) ) {
                if(isInList(hadMothers, thisParticle)<0){
                    hadMothers.push_back(thisParticle);
                }
                *lepton = thisParticle;
            }   // If the lepton's mother is a tau lepton
            else if(mother1 && std::abs(mother1->pdgId()) == 15) {
                const reco::Candidate* mother2 = 0;
                if(mother1->numberOfMothers()>0) mother2 = mother1->mother(0);
                if(mother2 && isHadron(flavour_, mother2)) {
                    if(isInList(hadMothers, thisParticle)<0) {
                        hadMothers.push_back(thisParticle);
                    }
                    *lepton = thisParticle;
                }   // If the tau's mother is a hadron
            }   // If the lepton's mother is a tau lepton
        }   // If this is a lepton
    }   // If no hadron found yet

    int partIndex = -1;	  // Index of particle being checked in the list of mothers
    partIndex = isInList ( hadMothers, thisParticle );

    // Checking whether this particle is already in the chain of analyzed particles in order to identify a loop
    bool isLoop = false;
    if ( !analyzedParticles ) {
        analyzedParticles = new std::set<const reco::Candidate*>;
    }
    for ( unsigned int i=0; i<analyzedParticles->size(); i++ ) {
        if ( analyzedParticles->count ( thisParticle ) <=0 ) {
            continue;
        }
        isLoop = true;
        break;
    }
    
    // If a loop has been detected
    if ( isLoop ) {
        if ( prevPartIndex>=0 ) {
            putMotherIndex ( hadMothersIndices, prevPartIndex, -1 );    // Setting mother index of previous particle to -1
        }
        return hadronIndex;		// Stopping further processing of the current chain
    }
    analyzedParticles->insert ( thisParticle );
    
    // Putting the mothers to the list of mothers
    for ( size_t iMother = 0; iMother < thisParticle->numberOfMothers(); ++iMother ) {
        const reco::Candidate* mother = thisParticle->mother ( iMother );
        int mothIndex = isInList ( hadMothers, mother );
        if ( mothIndex == partIndex && partIndex>=0 ) {
            continue;		// Skipping the mother that is its own daughter
        }

    // If this mother isn't yet in the list and hadron or lepton is in the list
        if ( mothIndex<0 && (*hadron || *lepton) ) {
            hadMothers.push_back ( mother );
            mothIndex=hadMothers.size()-1;
        }
    // If hadron has already been found in current chain and the mother isn't a duplicate of the particle being checked
        if ( (*hadron || *lepton) && mothIndex!=partIndex && partIndex>=0 ) {
            putMotherIndex ( hadMothersIndices, partIndex, mothIndex );			// Putting the index of mother for current particle
        }
        int index = analyzeMothers ( mother, hadron, lepton, topDaughterQId, topBarDaughterQId, hadMothers, hadMothersIndices, analyzedParticles, partIndex );
        hadronIndex = index<0?hadronIndex:index;
        // Setting the id of the particle which is a quark from the top decay
        if(*hadron && std::abs(mother->pdgId())==6) {
            int& bId = mother->pdgId() < 0 ? topBarDaughterQId : topDaughterQId;
            int thisFlav = std::abs(thisParticle->pdgId());
            if( bId<0){
                if(thisFlav <= 5) bId = partIndex; 
            } else {
                int bIdFlav = std::abs(hadMothers.at(bId)->pdgId());
                if( bIdFlav != 5 && thisFlav == 5) bId = partIndex;
                else if( thisFlav == 5 && thisParticle->pt() > hadMothers.at(bId)->pt() ) bId = partIndex;
            }           // If daughter quark of the top not found yet
        }           // If the mother is a top quark and hadron ahs been found
    }			// End of loop over mothers

    analyzedParticles->erase ( thisParticle );		// Removing current particle from the current chain that is being analyzed

    if ( partIndex<0 ) {
        return hadronIndex;    // Safety check
    }

    // Adding -1 to the list of mother indices for current particle if it has no mothers (for consistency between numbering of indices and mothers)
    if ( ( int ) thisParticle->numberOfMothers() <=0 && *hadron ) {
        putMotherIndex ( hadMothersIndices, partIndex, -1 );
    }


    return hadronIndex;

}


/**
* @brief puts mother index to the list of mothers of particle, if it isn't there already
*
* @param[in] hadMothersIndices vector of indices of mothers for each particle
* @param[in] partIndex index of the particle for which the mother index should be stored
* @param[in] mothIndex index of mother that should be stored for current particle
* 
* @returns whether the particle index was alreade in the list
*/
bool GenHFHadronMatcher::putMotherIndex ( std::vector<std::vector<int> > &hadMothersIndices, int partIndex, int mothIndex )
{
    // Putting vector of mothers indices for the given particle
    bool inList=false;
    if ( partIndex<0 ) {
        return false;
    }

    while ( ( int ) hadMothersIndices.size() <=partIndex ) { // If there is no list of mothers for current particle yet
        std::vector<int> mothersIndices;
        hadMothersIndices.push_back ( mothersIndices );
    }

    std::vector<int> *hadMotherIndices=&hadMothersIndices.at ( partIndex );
    // Removing other mothers if particle must have no mothers
    if ( mothIndex==-1 ) {
        hadMotherIndices->clear();
    } else {
    // Checking if current mother is already in the list of theParticle's mothers
        for ( int k=0; k< ( int ) hadMotherIndices->size(); k++ ) {
            if ( hadMotherIndices->at ( k ) !=mothIndex && hadMotherIndices->at ( k ) !=-1 ) {
                continue;
            }
            inList=true;
            break;
        }
    }
    // Adding current mother to the list of mothers of this particle
    if ( !inList ) {
        hadMotherIndices->push_back ( mothIndex );
    }

    return inList;
}


/**
* @brief helper function to find indices of particles with particular pdgId and status from the list of mothers of a given particle
*
* @param[in] idx index of particle, mothers of which should be searched
* @param[in] mothChains vector of indices where the found mothers are stored
* @param[in] hadMothersIndices list of indices pointing to mothers of each particle from list of mothers
* @param[in] hadMothers vector of all hadron mother particles of all levels
* @param[in] status status of mother that is being looked for
* @param[in] pdgId pdgId of mother that is being looked for [flavour*111111 used to identify hadrons of particular flavour]
* @param[in] pdgAbs whether the sign of pdgId should be taken into account
* @param[in] stopId id of the particle in the hadMothers array after which the checking should stop
* @param[in] firstLast should all(0), first(1) or last(2) occurances of the searched particle be stored
* @param[in] verbose option to print every particle that is processed during the search
*
* @returns index of the found particle in the hadMothers array [-1 if the specified particle not found]
*/

int GenHFHadronMatcher::findInMothers ( int idx, std::vector<int> &mothChains, std::vector<std::vector<int> > &hadMothersIndices, std::vector<reco::GenParticle> &hadMothers, int status, int pdgId, bool pdgAbs=false, int stopId=-1, int firstLast=0, bool verbose=false	)
{
    int foundStopId = -1;
    int pdg_1 = hadMothers.at ( idx ).pdgId();
    int partCharge = ( hadMothers.at ( idx ).pdgId() >0 ) ?1:-1;
// Inverting charge if mother is a b(c) meson
    if ( abs ( hadMothers.at ( idx ).pdgId() ) /1000 < 1 && ( abs ( hadMothers.at ( idx ).pdgId() ) /100%10 == 4 || abs ( hadMothers.at ( idx ).pdgId() ) /100%10 == 5 ) ) {
        partCharge*=-1;
    }

    if ( ( int ) hadMothersIndices.size() <=idx ) {
        if ( verbose ) {
            printf ( " Stopping checking particle %d. No mothers are stored.\n",idx );
        }
        return -1;     // Skipping if no mothers are stored for this particle
    }
    
    if(std::abs(hadMothers.at( idx ).pdgId()) > 10 &&  std::abs(hadMothers.at( idx ).pdgId()) < 19) printf("Lepton: %d\n", hadMothers.at( idx ).pdgId());

    std::vector<int> mothers = hadMothersIndices.at ( idx );
    unsigned int nMothers = mothers.size();
    bool isCorrect=false;		// Whether current particle is what is being searched
    if ( verbose ) {
        if ( abs ( hadMothers.at ( idx ).pdgId() ) ==2212 ) {
            printf ( "Chk:  %d\tpdg: %d\tstatus: %d",idx, hadMothers.at ( idx ).pdgId(), hadMothers.at ( idx ).status() );
        } else {
            printf ( " Chk:  %d(%d mothers)\tpdg: %d\tstatus: %d\tPt: %.3f\tEta: %.3f",idx, nMothers, hadMothers.at ( idx ).pdgId(), hadMothers.at ( idx ).status(), hadMothers.at ( idx ).pt(),hadMothers.at ( idx ).eta() );
        }
    }
    bool hasCorrectMothers = true;
    if(nMothers<1) hasCorrectMothers=false; else if(mothers.at(0)<0) hasCorrectMothers=false;
    if(!hasCorrectMothers) {
        if(verbose) printf("    NO CORRECT MOTHER\n");
        return -1;
    }
    // Stopping if the particular particle has been found
    if(stopId>=0 && idx == stopId) return idx;
    
    // Stopping if the hadron of particular flavour has been found
    if(pdgId%111111==0 && pdgId!=0) {
        if(isHadronPdgId(pdgId/111111, hadMothers.at(idx).pdgId())) {
            return idx;
        }
    }
    
    // Checking whether current mother satisfies selection criteria
    if ( ( ( hadMothers.at ( idx ).pdgId() == pdgId && pdgAbs==false )
            || ( abs ( hadMothers.at ( idx ).pdgId() ) == abs ( pdgId ) && pdgAbs==true ) )
            && ( hadMothers.at ( idx ).status() == status || status==0 )
            && hasCorrectMothers ) {
        isCorrect=true;
        bool inList=false;
        for ( unsigned int k=0; k<mothChains.size(); k++ ) if ( mothChains[k]==idx ) {
                inList=true;    // Checking whether isn't already in the list
                break;
            }
        if ( !inList && mothers.at ( 0 ) >=0 && ( hadMothers.at ( idx ).pdgId() *pdgId>0 || !pdgAbs ) ) {		// If not in list and mother of this quark has correct charge
            if ( firstLast==0 || firstLast==1 ) {
                mothChains.push_back ( idx );
            }
            if ( verbose ) {
                printf ( "   *" );
            }
        }
        if ( verbose ) {
            printf ( "   +++" );
        }
    }
    if ( verbose ) {
        printf ( "\n" );
    }

    if ( isCorrect && firstLast==1 ) {
        return -1;   // Stopping if only the first particle in the chain is looked for
    }

// Checking next level mothers
    for ( unsigned int i=0; i<nMothers; i++ ) {
        int idx2 = mothers[i];
        if ( idx2<0 ) {
            continue;    // Skipping if mother's id is -1 (no mother), that means current particle is a proton
        }
        if ( idx2==idx ) {
            continue;    // Skipping if particle is stored as its own mother
        }
        int pdg_2 = hadMothers[idx2].pdgId();
        // Inverting the flavour if bb oscillation detected
        if ( isHadronPdgId(pdgId, pdg_1) && isHadronPdgId(pdgId, pdg_2) &&  pdg_1*pdg_2 < 0 ) {
            pdgId*=-1;
            if(verbose) printf("######### Inverting flavour of the hadron\n");
        }
        if ( firstLast==2 && isCorrect && (
                    ( abs ( hadMothers[idx2].pdgId() ) != abs ( pdgId ) && pdgAbs==true ) ||
                    ( hadMothers[idx2].pdgId() != pdgId && pdgAbs==false ) ) ) {		// If only last occurance must be stored and mother has different flavour
            if ( verbose ) {
                printf ( "Checking mother %d out of %d mothers once more to store it as the last quark\n",i,nMothers );
            }
            foundStopId = findInMothers ( idx, mothChains, hadMothersIndices, hadMothers, 0, pdgId, pdgAbs, stopId, 1, verbose );
        }

// Checking next level mother
        if ( verbose ) {
            printf ( "Checking mother %d out of %d mothers, looking for pdgId: %d\n",i,nMothers,pdgId );
        }
        if(firstLast==2 && pdg_1 != pdg_2) continue;    // Requiring the same flavour when finding the last quark
        foundStopId = findInMothers ( idx2, mothChains, hadMothersIndices, hadMothers, status, pdgId, pdgAbs, stopId, firstLast, verbose );
    }
    
    return foundStopId;
}


/**
* @brief Check whether a given pdgId represents neutral particle
*
* @param[in] pdgId
* @param[in] thisParticle - a particle that is to be analysed
*
* @returns if the particle has a hadron among its daughters
*/
bool GenHFHadronMatcher::isNeutralPdg ( int pdgId )
{
    const int max = 5;
    int neutralPdgs[max]= {9,21,22,23,25};
    for ( int i=0; i<max; i++ ) if ( abs ( pdgId ) ==neutralPdgs[i] ) {
            return true;
        }
    return false;
}


/**
 * Finds hadrons that have the same flavour and origin and resolve this ambiguity
 * @method fixExtraSameFlavours
 * @param  hadId                Index of the hadron being checked
 * @param  hadIndex             Vector of indices of each hadron pointing to the hadMothers
 * @param  hadMothers           Vector of gen particles (contain hadrons and all its ancestors)
 * @param  hadMothersIndices    Vector of indices for each particle from hadMothers
 * @param  isFromTopWeakDecay   Vector of values showing whether particle comes from the top weak decay
 * @param  LastQuarkIds         Vector of indices of last quarks for each hadron
 * @param  LastQuarkMotherIds   Vector of indices of mothers for each last quark from LastQuarkIds
 * @param  lastQuakIndices      Vector of indices pointing to particular index from the LastQuarkIds and LastQuarkMotherIds to be used for each hadron
 * @param  lastQuarkIndex       Index from the LastQuarkIds and LastQuarkMotherIds for this particular hadron with index hadId
 * @return Whether other mother with unique association has been found for the hadrons
 */
bool GenHFHadronMatcher::fixExtraSameFlavours(
    const unsigned int hadId, const std::vector<int> &hadIndices, const std::vector<reco::GenParticle> &hadMothers, 
    const std::vector<std::vector<int> > &hadMothersIndices, const std::vector<int> &isFromTopWeakDecay, 
    const std::vector<std::vector<int> > &LastQuarkIds, const std::vector<std::vector<int> > &LastQuarkMotherIds, 
    std::vector<int> &lastQuarkIndices, std::vector<int> &hadronFlavour, 
    std::set<int> &checkedHadronIds, const int lastQuarkIndex)
{
    if(checkedHadronIds.count(hadId) != 0) return false;      // Hadron already checked previously and should be skipped
    checkedHadronIds.insert(hadId);                           // Putting hadron to the list of checked ones in this run

    if(lastQuarkIndex<0) return false;
    if((int)LastQuarkIds.at(hadId).size()<lastQuarkIndex+1) return false;
    int LastQuarkId = LastQuarkIds.at(hadId).at(lastQuarkIndex);
    int LastQuarkMotherId = LastQuarkMotherIds.at( hadId ).at( lastQuarkIndex );
    int qmFlav = hadMothers.at(LastQuarkId).pdgId() < 0 ? -1 : 1;
    int hadFlavour = qmFlav*std::abs( hadMothers.at( LastQuarkMotherId ).pdgId() );
    bool ambiguityResolved = true;
    // If last quark has inconsistent flavour with its mother, setting the hadron flavour to gluon
    if( (hadMothers.at(LastQuarkId).pdgId()*hadMothers.at(LastQuarkMotherId).pdgId() < 0 && !isNeutralPdg(hadMothers.at(LastQuarkMotherId).pdgId())) || 
        // If particle not coming from the Top weak decay has Top flavour
        (std::abs(hadronFlavour.at(hadId))==6 && isFromTopWeakDecay.at(hadId)==0) ) {
        if((int)LastQuarkIds.at(hadId).size()>lastQuarkIndex+1) fixExtraSameFlavours(hadId, hadIndices, hadMothers, hadMothersIndices, isFromTopWeakDecay, LastQuarkIds, LastQuarkMotherIds, lastQuarkIndices, hadronFlavour, checkedHadronIds, lastQuarkIndex+1); 
        else hadronFlavour.at(hadId) = qmFlav*21;
        return true;
    }

    int nSameFlavourHadrons = 0;
    // Looping over all previous hadrons
    for(unsigned int iHad = 0; iHad<hadronFlavour.size(); iHad++) {
        if(iHad==hadId) continue;
        int theLastQuarkIndex = lastQuarkIndices.at(iHad);
        if(theLastQuarkIndex<0) continue;
        if((int)LastQuarkMotherIds.at( iHad ).size() <= theLastQuarkIndex) continue;
        int theLastQuarkMotherId = LastQuarkMotherIds.at( iHad ).at( theLastQuarkIndex );
        int theHadFlavour = hadronFlavour.at(iHad);
        // Skipping hadrons with different flavour
        if(theHadFlavour==0 || std::abs(theHadFlavour)==21) continue;
        if(theHadFlavour != hadFlavour || theLastQuarkMotherId != LastQuarkMotherId) continue;
        ambiguityResolved = false;
        nSameFlavourHadrons++;
        
        // Checking other b-quark mother candidates of this hadron
        if((int)LastQuarkIds.at(hadId).size() > lastQuarkIndex+1) {
            if(fixExtraSameFlavours(hadId, hadIndices, hadMothers, hadMothersIndices, isFromTopWeakDecay, LastQuarkIds, LastQuarkMotherIds, lastQuarkIndices, hadronFlavour, checkedHadronIds, lastQuarkIndex+1) ) {
                ambiguityResolved = true;
                break;
            }
        } else
        // Checking other b-quark mother candidates of the particular previous hadron
        if((int)LastQuarkIds.at(iHad).size() > theLastQuarkIndex+1) {
            if(fixExtraSameFlavours(iHad, hadIndices, hadMothers, hadMothersIndices, isFromTopWeakDecay, LastQuarkIds, LastQuarkMotherIds, lastQuarkIndices, hadronFlavour, checkedHadronIds, theLastQuarkIndex+1) ) {
                ambiguityResolved = true;
                break;
            };
        } 

    }       // End of loop over all previous hadrons

    checkedHadronIds.erase(hadId);      // Removing the hadron from the list of checked hadrons
    if(nSameFlavourHadrons>0 && !ambiguityResolved) {
        hadronFlavour.at(hadId) = qmFlav*21;
        return true;
    }
    lastQuarkIndices.at(hadId) = lastQuarkIndex;
    hadronFlavour.at(hadId) = hadFlavour;
    return true;
}

//define this as a plug-in
DEFINE_FWK_MODULE ( GenHFHadronMatcher );
