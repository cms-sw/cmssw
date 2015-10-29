// -*- C++ -*-
//
// Package:    JetFlavourClustering
// Class:      JetFlavourClustering
//
/**\class JetFlavourClustering JetFlavourClustering.cc PhysicsTools/JetMCAlgos/plugins/JetFlavourClustering.cc
 * \brief Clusters hadrons, partons, and jet contituents to determine the jet flavour
 *
 * This producer clusters hadrons, partons and jet contituents to determine the jet flavour. The jet flavour information
 * is stored in the event as an AssociationVector which associates an object of type JetFlavourInfo to each of the jets.
 *
 * The producer takes as input jets and hadron and partons selected by the HadronAndPartonSelector producer. The hadron
 * and parton four-momenta are rescaled by a very small number (default rescale factor is 10e-18) which turns them into
 * the so-called "ghosts". The "ghost" hadrons and partons are clustered together with all of the jet constituents. It is
 * important to use the same clustering algorithm and jet size as for the original input jet collection. Since the
 * "ghost" hadrons and partons are extremely soft, the resulting jet collection will be practically identical to the
 * original one but now with "ghost" hadrons and partons clustered inside jets. The jet flavour is determined based on
 * the "ghost" hadrons clustered inside a jet:
 *
 * - jet is considered a b jet if there is at least one b "ghost" hadron clustered inside it (hadronFlavour = 5)
 * 
 * - jet is considered a c jet if there is at least one c and no b "ghost" hadrons clustered inside it (hadronFlavour = 4)
 * 
 * - jet is considered a light-flavour jet if there are no b or c "ghost" hadrons clustered inside it (hadronFlavour = 0)
 *
 * To further assign a more specific flavour to light-flavour jets, "ghost" partons are used:
 *
 * - jet is considered a b jet if there is at least one b "ghost" parton clustered inside it (partonFlavour = 5)
 * 
 * - jet is considered a c jet if there is at least one c and no b "ghost" partons clustered inside it (partonFlavour = 4)
 * 
 * - jet is considered a light-flavour jet if there are light-flavour and no b or c "ghost" partons clustered inside it.
 *   The jet is assigned the flavour of the hardest light-flavour "ghost" parton clustered inside it (partonFlavour = 1, 2, 3, or 21)
 * 
 * - jet has an undefined flavour if there are no "ghost" partons clustered inside it (partonFlavour = 0)
 *
 * In rare instances a conflict between the hadron- and parton-based flavours can occur. In such cases it is possible to
 * keep both flavours or to give priority to the hadron-based flavour. This is controlled by the 'hadronFlavourHasPriority'
 * switch. The priority is given to the hadron-based flavour as follows:
 * 
 * - if hadronFlavour==0 && (partonFlavour==4 || partonFlavour==5): partonFlavour is set to the flavour of the hardest
 *   light-flavour parton clustered inside the jet if such parton exists. Otherwise, the parton flavour is left undefined
 * 
 * - if hadronFlavour!=0 && hadronFlavour!=partonFlavour: partonFlavour is set equal to hadronFlavour
 *
 * The producer is also capable of assigning the flavour to subjets of fat jets, in which case it produces an additional
 * AssociationVector providing the flavour information for subjets. In order to assign the flavour to subjets, three input
 * jet collections are required:
 *
 * - jets, in this case represented by fat jets
 * 
 * - groomed jets, which is a collection of fat jets from which the subjets are derived (e.g. pruned, filtered, soft drop, top-tagged, etc. jets)
 * 
 * - subjets, derived from the groomed fat jets
 *
 * The "ghost" hadrons and partons clustered inside a fat jet are assigned to the closest subjet in the rapidity-phi
 * space. Once hadrons and partons have been assigned to subjets, the subjet flavour is determined in the same way as for
 * jets. The reason for requiring three jet collections as input in order to determine the subjet flavour is to avoid
 * possible inconsistencies between the fat jet and subjet flavours (such as a non-b fat jet having a b subjet and vice
 * versa) as well as the fact that re-clustering the constituents of groomed fat jets will generally result in a jet
 * collection different from the input groomed fat jets. Also note that "ghost" particles generally cannot be clustered
 * inside subjets in the same way this is done for fat jets. This is because some of the jet grooming techniques could
 * reject such very soft particle. So instead, the "ghost" particles are assigned to the closest subjet.
 * 
 * Finally, "ghost" leptons can also be clustered inside jets but they are not used in any way to determine the jet
 * flavour. This functionality is optional and is potentially useful to identify jets from hadronic taus.
 * 
 * For more details, please refer to
 * https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideBTagMCTools
 * 
 */
//
// Original Author:  Dinko Ferencek
//         Created:  Wed Nov  6 00:49:55 CET 2013
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/JetCollection.h"
#include "SimDataFormats/JetMatching/interface/JetFlavourInfo.h"
#include "SimDataFormats/JetMatching/interface/JetFlavourInfoMatching.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "PhysicsTools/JetMCUtils/interface/CandMCTag.h"

#include "fastjet/JetDefinition.hh"
#include "fastjet/ClusterSequence.hh"
#include "fastjet/Selector.hh"
#include "fastjet/PseudoJet.hh"

//
// constants, enums and typedefs
//
typedef boost::shared_ptr<fastjet::ClusterSequence>  ClusterSequencePtr;
typedef boost::shared_ptr<fastjet::JetDefinition>    JetDefPtr;

//
// class declaration
//
class GhostInfo : public fastjet::PseudoJet::UserInfoBase{
  public:
    GhostInfo(const bool & isHadron,
              const bool & isbHadron,
              const bool & isParton,
              const bool & isLepton,
              const reco::GenParticleRef & particleRef) :
      m_particleRef(particleRef)
    {
      m_type = 0;
      if( isHadron )  m_type |= ( 1 << 0 );
      if( isbHadron ) m_type |= ( 1 << 1 );
      if( isParton )  m_type |= ( 1 << 2 );
      if( isLepton )  m_type |= ( 1 << 3 );
    }

    const bool isHadron()  const { return ( m_type & (1 << 0) ); }
    const bool isbHadron() const { return ( m_type & (1 << 1) ); }
    const bool isParton()  const { return ( m_type & (1 << 2) ); }
    const bool isLepton()  const { return ( m_type & (1 << 3) ); }
    const reco::GenParticleRef & particleRef() const { return m_particleRef; }

  protected:
    const reco::GenParticleRef m_particleRef;
    int m_type;
};

class JetFlavourClustering : public edm::stream::EDProducer<> {
   public:
      explicit JetFlavourClustering(const edm::ParameterSet&);
      ~JetFlavourClustering();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&);
  
      void insertGhosts(const edm::Handle<reco::GenParticleRefVector>& particles,
                        const double ghostRescaling,
                        const bool isHadron, const bool isbHadron, const bool isParton, const bool isLepton,
                        std::vector<fastjet::PseudoJet>& constituents);

      void matchReclusteredJets(const edm::Handle<edm::View<reco::Jet> >& jets,
                                const std::vector<fastjet::PseudoJet>& matchedJets,
                                std::vector<int>& matchedIndices);
      void matchGroomedJets(const edm::Handle<edm::View<reco::Jet> >& jets,
                            const edm::Handle<edm::View<reco::Jet> >& matchedJets,
                            std::vector<int>& matchedIndices);
      void matchSubjets(const std::vector<int>& groomedIndices,
                        const edm::Handle<edm::View<reco::Jet> >& groomedJets,
                        const edm::Handle<edm::View<reco::Jet> >& subjets,
                        std::vector<std::vector<int> >& matchedIndices);

      void setFlavours(const reco::GenParticleRefVector& clusteredbHadrons,
                       const reco::GenParticleRefVector& clusteredcHadrons,
                       const reco::GenParticleRefVector& clusteredPartons,
                       int&  hadronFlavour,
                       int&  partonFlavour);

      void assignToSubjets(const reco::GenParticleRefVector& clusteredParticles,
                           const edm::Handle<edm::View<reco::Jet> >& subjets,
                           const std::vector<int>& subjetIndices,
                           std::vector<reco::GenParticleRefVector>& assignedParticles);

      // ----------member data ---------------------------
      const edm::EDGetTokenT<edm::View<reco::Jet> >      jetsToken_;        // Input jet collection
      edm::EDGetTokenT<edm::View<reco::Jet> >      groomedJetsToken_; // Input groomed jet collection
      edm::EDGetTokenT<edm::View<reco::Jet> >      subjetsToken_;     // Input subjet collection
      const edm::EDGetTokenT<reco::GenParticleRefVector> bHadronsToken_;    // Input b hadron collection
      const edm::EDGetTokenT<reco::GenParticleRefVector> cHadronsToken_;    // Input c hadron collection
      const edm::EDGetTokenT<reco::GenParticleRefVector> partonsToken_;     // Input parton collection
      edm::EDGetTokenT<reco::GenParticleRefVector> leptonsToken_;     // Input lepton collection

      const std::string   jetAlgorithm_;
      const double        rParam_;
      const double        jetPtMin_;
      const double        ghostRescaling_;
      const double        relPtTolerance_;
      const bool          hadronFlavourHasPriority_;
      const bool          useSubjets_;
      const bool          useLeptons_;

      ClusterSequencePtr  fjClusterSeq_;
      JetDefPtr           fjJetDefinition_;
};

//
// static data member definitions
//

//
// constructors and destructor
//
JetFlavourClustering::JetFlavourClustering(const edm::ParameterSet& iConfig) :

   jetsToken_(consumes<edm::View<reco::Jet> >( iConfig.getParameter<edm::InputTag>("jets")) ),
   bHadronsToken_(consumes<reco::GenParticleRefVector>( iConfig.getParameter<edm::InputTag>("bHadrons") )),
   cHadronsToken_(consumes<reco::GenParticleRefVector>( iConfig.getParameter<edm::InputTag>("cHadrons") )),
   partonsToken_(consumes<reco::GenParticleRefVector>( iConfig.getParameter<edm::InputTag>("partons") )),
   jetAlgorithm_(iConfig.getParameter<std::string>("jetAlgorithm")),
   rParam_(iConfig.getParameter<double>("rParam")),
   jetPtMin_(0.), // hardcoded to 0. since we simply want to recluster all input jets which already had some PtMin applied
   ghostRescaling_(iConfig.exists("ghostRescaling") ? iConfig.getParameter<double>("ghostRescaling") : 1e-18),
   relPtTolerance_(iConfig.exists("relPtTolerance") ? iConfig.getParameter<double>("relPtTolerance") : 1e-03), // 0.1% relative difference in Pt should be sufficient to detect possible misconfigurations
   hadronFlavourHasPriority_(iConfig.getParameter<bool>("hadronFlavourHasPriority")),
   useSubjets_(iConfig.exists("groomedJets") && iConfig.exists("subjets")),
   useLeptons_(iConfig.exists("leptons"))

{
   // register your products
   produces<reco::JetFlavourInfoMatchingCollection>();
   if( useSubjets_ )
     produces<reco::JetFlavourInfoMatchingCollection>("SubJets");

   // set jet algorithm
   if (jetAlgorithm_=="Kt")
     fjJetDefinition_= JetDefPtr( new fastjet::JetDefinition(fastjet::kt_algorithm, rParam_) );
   else if (jetAlgorithm_=="CambridgeAachen")
     fjJetDefinition_= JetDefPtr( new fastjet::JetDefinition(fastjet::cambridge_algorithm, rParam_) );
   else if (jetAlgorithm_=="AntiKt")
     fjJetDefinition_= JetDefPtr( new fastjet::JetDefinition(fastjet::antikt_algorithm, rParam_) );
   else
     throw cms::Exception("InvalidJetAlgorithm") << "Jet clustering algorithm is invalid: " << jetAlgorithm_ << ", use CambridgeAachen | Kt | AntiKt" << std::endl;

   if (useSubjets_) {
     groomedJetsToken_=consumes<edm::View<reco::Jet> >(iConfig.getParameter<edm::InputTag>("groomedJets"));
     subjetsToken_=consumes<edm::View<reco::Jet> >(iConfig.getParameter<edm::InputTag>("subjets"));
   }
   if ( useLeptons_ ) {
     leptonsToken_=consumes<reco::GenParticleRefVector>( iConfig.getParameter<edm::InputTag>("leptons") );
   }

}


JetFlavourClustering::~JetFlavourClustering()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
JetFlavourClustering::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   edm::Handle<edm::View<reco::Jet> > jets;
   iEvent.getByToken(jetsToken_, jets);

   edm::Handle<edm::View<reco::Jet> > groomedJets;
   edm::Handle<edm::View<reco::Jet> > subjets;
   if( useSubjets_ )
   {
     iEvent.getByToken(groomedJetsToken_, groomedJets);
     iEvent.getByToken(subjetsToken_, subjets);
   }

   edm::Handle<reco::GenParticleRefVector> bHadrons;
   iEvent.getByToken(bHadronsToken_, bHadrons);

   edm::Handle<reco::GenParticleRefVector> cHadrons;
   iEvent.getByToken(cHadronsToken_, cHadrons);

   edm::Handle<reco::GenParticleRefVector> partons;
   iEvent.getByToken(partonsToken_, partons);

   edm::Handle<reco::GenParticleRefVector> leptons;
   if( useLeptons_ )
     iEvent.getByToken(leptonsToken_, leptons);

   std::auto_ptr<reco::JetFlavourInfoMatchingCollection> jetFlavourInfos( new reco::JetFlavourInfoMatchingCollection(reco::JetRefBaseProd(jets)) );
   std::auto_ptr<reco::JetFlavourInfoMatchingCollection> subjetFlavourInfos;
   if( useSubjets_ )
     subjetFlavourInfos = std::auto_ptr<reco::JetFlavourInfoMatchingCollection>( new reco::JetFlavourInfoMatchingCollection(reco::JetRefBaseProd(subjets)) );

   // vector of constituents for reclustering jets and "ghosts"
   std::vector<fastjet::PseudoJet> fjInputs;
   // loop over all input jets and collect all their constituents
   for(edm::View<reco::Jet>::const_iterator it = jets->begin(); it != jets->end(); ++it)
   {
     std::vector<edm::Ptr<reco::Candidate> > constituents = it->getJetConstituents();
     std::vector<edm::Ptr<reco::Candidate> >::const_iterator m;
     for( m = constituents.begin(); m != constituents.end(); ++m )
     {
       reco::CandidatePtr constit = *m;
       if(!constit.isNonnull() || !constit.isAvailable()) {
         edm::LogError("MissingJetConstituent") << "Jet constituent required for jet reclustering is missing. Reclustered jets are not guaranteed to reproduce the original jets!";
         continue;
       }
       if(constit->pt() == 0)
       {
         edm::LogWarning("NullTransverseMomentum") << "dropping input candidate with pt=0";
         continue;
       }
       fjInputs.push_back(fastjet::PseudoJet(constit->px(),constit->py(),constit->pz(),constit->energy()));
     }
   }
   // insert "ghost" b hadrons in the vector of constituents
   insertGhosts(bHadrons, ghostRescaling_, true, true, false, false, fjInputs);
   // insert "ghost" c hadrons in the vector of constituents
   insertGhosts(cHadrons, ghostRescaling_, true, false, false, false, fjInputs);
   // insert "ghost" partons in the vector of constituents
   insertGhosts(partons, ghostRescaling_, false, false, true, false, fjInputs);
   // if used, insert "ghost" leptons in the vector of constituents
   if( useLeptons_ )
     insertGhosts(leptons, ghostRescaling_, false, false, false, true, fjInputs);

   // define jet clustering sequence
   fjClusterSeq_ = ClusterSequencePtr( new fastjet::ClusterSequence( fjInputs, *fjJetDefinition_ ) );
   // recluster jet constituents and inserted "ghosts"
   std::vector<fastjet::PseudoJet> inclusiveJets = fastjet::sorted_by_pt( fjClusterSeq_->inclusive_jets(jetPtMin_) );

   if( inclusiveJets.size() < jets->size() )
     edm::LogError("TooFewReclusteredJets") << "There are fewer reclustered (" << inclusiveJets.size() << ") than original jets (" << jets->size() << "). Please check that the jet algorithm and jet size match those used for the original jet collection.";

   // match reclustered and original jets
   std::vector<int> reclusteredIndices;
   matchReclusteredJets(jets,inclusiveJets,reclusteredIndices);

   // match groomed and original jets
   std::vector<int> groomedIndices;
   if( useSubjets_ )
   {
     if( groomedJets->size() > jets->size() )
       edm::LogError("TooManyGroomedJets") << "There are more groomed (" << groomedJets->size() << ") than original jets (" << jets->size() << "). Please check that the two jet collections belong to each other.";

     matchGroomedJets(jets,groomedJets,groomedIndices);
   }

   // match subjets and original jets
   std::vector<std::vector<int> > subjetIndices;
   if( useSubjets_ )
   {
     matchSubjets(groomedIndices,groomedJets,subjets,subjetIndices);
   }

   // determine jet flavour
   for(size_t i=0; i<jets->size(); ++i)
   {
     reco::GenParticleRefVector clusteredbHadrons;
     reco::GenParticleRefVector clusteredcHadrons;
     reco::GenParticleRefVector clusteredPartons;
     reco::GenParticleRefVector clusteredLeptons;

     // if matching reclustered to original jets failed
     if( reclusteredIndices.at(i) < 0 )
     {
       // set an empty JetFlavourInfo for this jet
       (*jetFlavourInfos)[jets->refAt(i)] = reco::JetFlavourInfo(clusteredbHadrons, clusteredcHadrons, clusteredPartons, clusteredLeptons, 0, 0);
     }
     else if( jets->at(i).pt() == 0 )
     {
       edm::LogWarning("NullTransverseMomentum") << "The original jet " << i << " has Pt=0. This is not expected so the jet will be skipped.";

       // set an empty JetFlavourInfo for this jet
       (*jetFlavourInfos)[jets->refAt(i)] = reco::JetFlavourInfo(clusteredbHadrons, clusteredcHadrons, clusteredPartons, clusteredLeptons, 0, 0);

       // if subjets are used
       if( useSubjets_ && subjetIndices.at(i).size()>0 )
       {
         // loop over subjets
         for(size_t sj=0; sj<subjetIndices.at(i).size(); ++sj)
         {
           // set an empty JetFlavourInfo for this subjet
           (*subjetFlavourInfos)[subjets->refAt(subjetIndices.at(i).at(sj))] = reco::JetFlavourInfo(reco::GenParticleRefVector(), reco::GenParticleRefVector(), reco::GenParticleRefVector(), reco::GenParticleRefVector(), 0, 0);
         }
       }
     }
     else
     {
       // since the "ghosts" are extremely soft, the configuration and ordering of the reclustered and original jets should in principle stay the same
       if( ( std::abs( inclusiveJets.at(reclusteredIndices.at(i)).pt() - jets->at(i).pt() ) / jets->at(i).pt() ) > relPtTolerance_ )
       {
         if( jets->at(i).pt() < 10. )  // special handling for low-Pt jets (Pt<10 GeV)
           edm::LogWarning("JetPtMismatchAtLowPt") << "The reclustered and original jet " << i << " have different Pt's (" << inclusiveJets.at(reclusteredIndices.at(i)).pt() << " vs " << jets->at(i).pt() << " GeV, respectively).\n"
                                                   << "Please check that the jet algorithm and jet size match those used for the original jet collection and also make sure the original jets are uncorrected. In addition, make sure you are not using CaloJets which are presently not supported.\n"
                                                   << "Since the mismatch is at low Pt (Pt<10 GeV), it is ignored and only a warning is issued.\n"
                                                   << "\nIn extremely rare instances the mismatch could be caused by a difference in the machine precision in which case make sure the original jet collection is produced and reclustering is performed in the same job.";
         else
           edm::LogError("JetPtMismatch") << "The reclustered and original jet " << i << " have different Pt's (" << inclusiveJets.at(reclusteredIndices.at(i)).pt() << " vs " << jets->at(i).pt() << " GeV, respectively).\n"
                                          << "Please check that the jet algorithm and jet size match those used for the original jet collection and also make sure the original jets are uncorrected. In addition, make sure you are not using CaloJets which are presently not supported.\n"
                                          << "\nIn extremely rare instances the mismatch could be caused by a difference in the machine precision in which case make sure the original jet collection is produced and reclustering is performed in the same job.";
       }

       // get jet constituents (sorted by Pt)
       std::vector<fastjet::PseudoJet> constituents = fastjet::sorted_by_pt( inclusiveJets.at(reclusteredIndices.at(i)).constituents() );

       // loop over jet constituents and try to find "ghosts"
       for(std::vector<fastjet::PseudoJet>::const_iterator it = constituents.begin(); it != constituents.end(); ++it)
       {
         if( !it->has_user_info() ) continue; // skip if not a "ghost"

         // "ghost" hadron
         if( it->user_info<GhostInfo>().isHadron() )
         {
           // "ghost" b hadron
           if( it->user_info<GhostInfo>().isbHadron() )
             clusteredbHadrons.push_back(it->user_info<GhostInfo>().particleRef());
           // "ghost" c hadron
           else
             clusteredcHadrons.push_back(it->user_info<GhostInfo>().particleRef());
         }
         // "ghost" parton
         else if( it->user_info<GhostInfo>().isParton() )
           clusteredPartons.push_back(it->user_info<GhostInfo>().particleRef());
         // "ghost" lepton
         else if( it->user_info<GhostInfo>().isLepton() )
           clusteredLeptons.push_back(it->user_info<GhostInfo>().particleRef());
       }

       int hadronFlavour = 0; // default hadron flavour set to 0 (= undefined)
       int partonFlavour = 0; // default parton flavour set to 0 (= undefined)

       // set hadron- and parton-based flavours
       setFlavours(clusteredbHadrons, clusteredcHadrons, clusteredPartons, hadronFlavour, partonFlavour);

       // set the JetFlavourInfo for this jet
       (*jetFlavourInfos)[jets->refAt(i)] = reco::JetFlavourInfo(clusteredbHadrons, clusteredcHadrons, clusteredPartons, clusteredLeptons, hadronFlavour, partonFlavour);
     }

     // if subjets are used, determine their flavour
     if( useSubjets_ )
     {
       if( subjetIndices.at(i).size()==0 ) continue; // continue if the original jet does not have subjets assigned

       // define vectors of GenParticleRefVectors for hadrons and partons assigned to different subjets
       std::vector<reco::GenParticleRefVector> assignedbHadrons(subjetIndices.at(i).size(),reco::GenParticleRefVector());
       std::vector<reco::GenParticleRefVector> assignedcHadrons(subjetIndices.at(i).size(),reco::GenParticleRefVector());
       std::vector<reco::GenParticleRefVector> assignedPartons(subjetIndices.at(i).size(),reco::GenParticleRefVector());
       std::vector<reco::GenParticleRefVector> assignedLeptons(subjetIndices.at(i).size(),reco::GenParticleRefVector());

       // loop over clustered b hadrons and assign them to different subjets based on smallest dR
       assignToSubjets(clusteredbHadrons, subjets, subjetIndices.at(i), assignedbHadrons);
       // loop over clustered c hadrons and assign them to different subjets based on smallest dR
       assignToSubjets(clusteredcHadrons, subjets, subjetIndices.at(i), assignedcHadrons);
       // loop over clustered partons and assign them to different subjets based on smallest dR
       assignToSubjets(clusteredPartons, subjets, subjetIndices.at(i), assignedPartons);
       // if used, loop over clustered leptons and assign them to different subjets based on smallest dR
       if( useLeptons_ )
         assignToSubjets(clusteredLeptons, subjets, subjetIndices.at(i), assignedLeptons);

       // loop over subjets and determine their flavour
       for(size_t sj=0; sj<subjetIndices.at(i).size(); ++sj)
       {
         int subjetHadronFlavour = 0; // default hadron flavour set to 0 (= undefined)
         int subjetPartonFlavour = 0; // default parton flavour set to 0 (= undefined)

         // set hadron- and parton-based flavours
         setFlavours(assignedbHadrons.at(sj), assignedcHadrons.at(sj), assignedPartons.at(sj), subjetHadronFlavour, subjetPartonFlavour);

         // set the JetFlavourInfo for this subjet
         (*subjetFlavourInfos)[subjets->refAt(subjetIndices.at(i).at(sj))] = reco::JetFlavourInfo(assignedbHadrons.at(sj), assignedcHadrons.at(sj), assignedPartons.at(sj), assignedLeptons.at(sj), subjetHadronFlavour, subjetPartonFlavour);
       }
     }
   }

   // put jet flavour infos in the event
   iEvent.put( jetFlavourInfos );
   // put subjet flavour infos in the event
   if( useSubjets_ )
     iEvent.put( subjetFlavourInfos, "SubJets" );
}

// ------------ method that inserts "ghost" particles in the vector of jet constituents ------------
void
JetFlavourClustering::insertGhosts(const edm::Handle<reco::GenParticleRefVector>& particles,
                                   const double ghostRescaling,
                                   const bool isHadron, const bool isbHadron, const bool isParton, const bool isLepton,
                                   std::vector<fastjet::PseudoJet>& constituents)
{
   // insert "ghost" particles in the vector of jet constituents
   for(reco::GenParticleRefVector::const_iterator it = particles->begin(); it != particles->end(); ++it)
   {
     if((*it)->pt() == 0)
     {
       edm::LogInfo("NullTransverseMomentum") << "dropping input ghost candidate with pt=0";
       continue;
     }
     fastjet::PseudoJet p((*it)->px(),(*it)->py(),(*it)->pz(),(*it)->energy());
     p*=ghostRescaling; // rescale particle momentum
     p.set_user_info(new GhostInfo(isHadron, isbHadron, isParton, isLepton, *it));
     constituents.push_back(p);
   }
}

// ------------ method that matches reclustered and original jets based on minimum dR ------------
void
JetFlavourClustering::matchReclusteredJets(const edm::Handle<edm::View<reco::Jet> >& jets,
                                           const std::vector<fastjet::PseudoJet>& reclusteredJets,
                                           std::vector<int>& matchedIndices)
{
   std::vector<bool> matchedLocks(reclusteredJets.size(),false);

   for(size_t j=0; j<jets->size(); ++j)
   {
     double matchedDR2 = 1e9;
     int matchedIdx = -1;

     for(size_t rj=0; rj<reclusteredJets.size(); ++rj)
     {
       if( matchedLocks.at(rj) ) continue; // skip jets that have already been matched

       double tempDR2 = reco::deltaR2( jets->at(j).rapidity(), jets->at(j).phi(), reclusteredJets.at(rj).rapidity(), reclusteredJets.at(rj).phi_std() );
       if( tempDR2 < matchedDR2 )
       {
         matchedDR2 = tempDR2;
         matchedIdx = rj;
       }
     }

     if( matchedIdx>=0 )
     {
       if ( matchedDR2 > rParam_*rParam_ )
       {
         edm::LogError("JetMatchingFailed") << "Matched reclustered jet " << matchedIdx << " and original jet " << j <<" are separated by dR=" << sqrt(matchedDR2) << " which is greater than the jet size R=" << rParam_ << ".\n"
                                            << "This is not expected so please check that the jet algorithm and jet size match those used for the original jet collection.";
       }
       else
         matchedLocks.at(matchedIdx) = true;
     }
     else
       edm::LogError("JetMatchingFailed") << "Matching reclustered to original jets failed. Please check that the jet algorithm and jet size match those used for the original jet collection.";

     matchedIndices.push_back(matchedIdx);
   }
}

// ------------ method that matches groomed and original jets based on minimum dR ------------
void
JetFlavourClustering::matchGroomedJets(const edm::Handle<edm::View<reco::Jet> >& jets,
                                       const edm::Handle<edm::View<reco::Jet> >& groomedJets,
                                       std::vector<int>& matchedIndices)
{
   std::vector<bool> jetLocks(jets->size(),false);
   std::vector<int>  jetIndices;

   for(size_t gj=0; gj<groomedJets->size(); ++gj)
   {
     double matchedDR2 = 1e9;
     int matchedIdx = -1;

     if( groomedJets->at(gj).pt()>0. ) // skip pathological cases of groomed jets with Pt=0
     {
       for(size_t j=0; j<jets->size(); ++j)
       {
         if( jetLocks.at(j) ) continue; // skip jets that have already been matched

         double tempDR2 = reco::deltaR2( jets->at(j).rapidity(), jets->at(j).phi(), groomedJets->at(gj).rapidity(), groomedJets->at(gj).phi() );
         if( tempDR2 < matchedDR2 )
         {
           matchedDR2 = tempDR2;
           matchedIdx = j;
         }
       }
     }

     if( matchedIdx>=0 )
     {
       if ( matchedDR2 > rParam_*rParam_ )
       {
         edm::LogWarning("MatchedJetsFarApart") << "Matched groomed jet " << gj << " and original jet " << matchedIdx <<" are separated by dR=" << sqrt(matchedDR2) << " which is greater than the jet size R=" << rParam_ << ".\n"
                                                << "This is not expected so the matching of these two jets has been discarded. Please check that the two jet collections belong to each other.";
         matchedIdx = -1;
       }
       else
         jetLocks.at(matchedIdx) = true;
     }
     jetIndices.push_back(matchedIdx);
   }

   for(size_t j=0; j<jets->size(); ++j)
   {
     std::vector<int>::iterator matchedIndex = std::find( jetIndices.begin(), jetIndices.end(), j );

     matchedIndices.push_back( matchedIndex != jetIndices.end() ? std::distance(jetIndices.begin(),matchedIndex) : -1 );
   }
}

// ------------ method that matches subjets and original jets ------------
void
JetFlavourClustering::matchSubjets(const std::vector<int>& groomedIndices,
                                   const edm::Handle<edm::View<reco::Jet> >& groomedJets,
                                   const edm::Handle<edm::View<reco::Jet> >& subjets,
                                   std::vector<std::vector<int> >& matchedIndices)
{
   for(size_t g=0; g<groomedIndices.size(); ++g)
   {
     std::vector<int> subjetIndices;

     if( groomedIndices.at(g)>=0 )
     {
       for(size_t s=0; s<groomedJets->at(groomedIndices.at(g)).numberOfDaughters(); ++s)
       {
         const edm::Ptr<reco::Candidate> & subjet = groomedJets->at(groomedIndices.at(g)).daughterPtr(s);

         for(size_t sj=0; sj<subjets->size(); ++sj)
         {
           if( subjet == edm::Ptr<reco::Candidate>(subjets->ptrAt(sj)) )
           {
             subjetIndices.push_back(sj);
             break;
           }
         }
       }

       if( subjetIndices.size() == 0 )
         edm::LogError("SubjetMatchingFailed") << "Matching subjets to original jets failed. Please check that the groomed jet and subjet collections belong to each other.";

       matchedIndices.push_back(subjetIndices);
     }
     else
       matchedIndices.push_back(subjetIndices);
   }
}

// ------------ method that sets hadron- and parton-based flavours ------------
void
JetFlavourClustering::setFlavours(const reco::GenParticleRefVector& clusteredbHadrons,
                                  const reco::GenParticleRefVector& clusteredcHadrons,
                                  const reco::GenParticleRefVector& clusteredPartons,
                                  int&  hadronFlavour,
                                  int&  partonFlavour)
{
   reco::GenParticleRef hardestParton;
   reco::GenParticleRef hardestLightParton;
   reco::GenParticleRef flavourParton;

   // loop over clustered partons (already sorted by Pt)
   for(reco::GenParticleRefVector::const_iterator it = clusteredPartons.begin(); it != clusteredPartons.end(); ++it)
   {
     // hardest parton
     if( hardestParton.isNull() )
       hardestParton = (*it);
     // hardest light-flavour parton
     if( hardestLightParton.isNull() )
     {
       if( CandMCTagUtils::isLightParton( *(*it) ) )
         hardestLightParton = (*it);
     }
     // c flavour
     if( flavourParton.isNull() && ( std::abs( (*it)->pdgId() ) == 4 ) )
       flavourParton = (*it);
     // b flavour gets priority
     if( std::abs( (*it)->pdgId() ) == 5 )
     {
       if( flavourParton.isNull() )
         flavourParton = (*it);
       else if( std::abs( flavourParton->pdgId() ) != 5 )
         flavourParton = (*it);
     }
   }

   // set hadron-based flavour
   if( clusteredbHadrons.size()>0 )
     hadronFlavour = 5;
   else if( clusteredcHadrons.size()>0 && clusteredbHadrons.size()==0 )
     hadronFlavour = 4;
   // set parton-based flavour
   if( flavourParton.isNull() )
   {
     if( hardestParton.isNonnull() ) partonFlavour = hardestParton->pdgId();
   }
   else
     partonFlavour = flavourParton->pdgId();

   // if enabled, check for conflicts between hadron- and parton-based flavours and give priority to the hadron-based flavour
   if( hadronFlavourHasPriority_ )
   {
     if( hadronFlavour==0 && (std::abs(partonFlavour)==4 || std::abs(partonFlavour)==5) )
       partonFlavour = ( hardestLightParton.isNonnull() ? hardestLightParton->pdgId() : 0 );
     else if( hadronFlavour!=0 && std::abs(partonFlavour)!=hadronFlavour )
       partonFlavour = hadronFlavour;
   }
}

// ------------ method that assigns clustered particles to subjets ------------
void
JetFlavourClustering::assignToSubjets(const reco::GenParticleRefVector& clusteredParticles,
                                      const edm::Handle<edm::View<reco::Jet> >& subjets,
                                      const std::vector<int>& subjetIndices,
                                      std::vector<reco::GenParticleRefVector>& assignedParticles)
{
   // loop over clustered particles and assign them to different subjets based on smallest dR
   for(reco::GenParticleRefVector::const_iterator it = clusteredParticles.begin(); it != clusteredParticles.end(); ++it)
   {
     std::vector<double> dR2toSubjets;

     for(size_t sj=0; sj<subjetIndices.size(); ++sj)
       dR2toSubjets.push_back( reco::deltaR2( (*it)->rapidity(), (*it)->phi(), subjets->at(subjetIndices.at(sj)).rapidity(), subjets->at(subjetIndices.at(sj)).phi() ) );

     // find the closest subjet
     int closestSubjetIdx = std::distance( dR2toSubjets.begin(), std::min_element(dR2toSubjets.begin(), dR2toSubjets.end()) );

     assignedParticles.at(closestSubjetIdx).push_back( *it );
   }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
JetFlavourClustering::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(JetFlavourClustering);
