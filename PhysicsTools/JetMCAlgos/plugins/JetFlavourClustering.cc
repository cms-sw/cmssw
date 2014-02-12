// -*- C++ -*-
//
// Package:    JetFlavourClustering
// Class:      JetFlavourClustering
//
/**\class JetFlavourClustering JetFlavourClustering.cc PhysicsTools/JetMCAlgos/plugins/JetFlavourClustering.cc
 * \brief Clusters hadrons, partons, and jet contituents to determine the jet flavour
 *
 * This producer clusters hadrons, partons and jet contituents to determine the jet flavour. The jet flavour information
 * is stored in the event as an AssociationVector which associates an object of type JetFlavorInfo to each of the jets.
 *
 * The producer takes as input jets and hadron and partons selected by the HadronAndPartonSelector producer. The hadron
 * and parton four-momenta are rescaled by a very small number (default rescale factor is 10e-18) which turns them into
 * the so-called "ghosts". The "ghost" hadrons and partons are clustered together with all of the jet constituents. It is
 * important to use the same clustering algorithm and jet size as for the original input jet collection. Since the
 * "ghost" hadrons and partons are extremely soft, the resulting jet collection will be practically identical to the
 * original one but now with "ghost" hadrons and partons clustered inside jets. The jet flavour is determined based on
 * the "ghost" hadrons clustered inside a jet:
 *
 * - a jet is considered a b jet if there is at least one b "ghost" hadron clustered inside it (hadronFlavour = 5)
 * - a jet is considered a c jet if there is at least one c and no b "ghost" hadrons clustered inside it (hadronFlavour = 4)
 * - a jet is considered a light-flavour jet if there are no b or c "ghost" hadrons clustered inside it (hadronFlavour = 0)
 *
 * To further assign a more specific flavour to light-flavour jets, "ghost" partons are used:
 *
 * - a jet is considered a b jet if there is at least one b "ghost" parton clustered inside it (partonFlavour = 5)
 * - a jet is considered a c jet if there is at least one c and no b "ghost" partons clustered inside it (partonFlavour = 4)
 * - a jet is considered a light-flavour jet if there are light-flavour and no b or c "ghost" partons clustered inside it.
 *   The jet is assigned the flavour of the hardest light-flavour "ghost" parton clustered inside it (partonFlavour = 1,2,3, or 21)
 * - a jet has an undefined flavour if there are no "ghost" partons clustered inside it (partonFlavour = 0)
 *
 * In rare instances a conflict between the hadron- and parton-based flavours can occur. However, it is possible to give
 * priority to the hadron-based by enabling the 'hadronFlavourHasPriority' switch, in which case the parton-based flavour
 * will be reset to the hadron-based flavour when the conflict occurs.
 *
 * The producer is also capable of assigning the flavor to subjets of fat jets, in which case it produces an additional
 * AssociationVector providing the flavour information for subjets. In order to assign the flavor to subjets, three input
 * jet collections are required:
 *
 * - jets, in this case represented by fat jets
 * - groomed jets, which is a collection of fat jets from which the subjets are derived
 * - subjets, derived from the groomed fat jets
 *
 * The "ghost" hadrons and partons clustered inside a fat jet are assigned to the closest subjet in the rapidity-phi
 * space. Once hadrons and partons have been assigned to subjets, the subjet flavor is determined in the same way as for
 * jets. The reason for requiring three jet collections as input in order to determine the subjet flavour is to avoid
 * possible inconsistencies between the fat jet and subjet flavours (such as a non-b fat jet having a b subjet and vice
 * versa) as well as the fact that reclustering the constituents of groomed fat jets will generally result in a jet
 * collection different from the input groomed fat jets.
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
#include "FWCore/Framework/interface/EDProducer.h"

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
    GhostInfo(
               const bool & isHadron,
               const bool & isbHadron,
               const reco::GenParticleRef & particleRef
              ) :
      m_isHadron(isHadron),
      m_isbHadron(isbHadron),
      m_particleRef(particleRef) { }

    const bool isHadron() const { return m_isHadron; }
    const bool isbHadron() const { return m_isbHadron; }
    const reco::GenParticleRef & particleRef() const { return m_particleRef; }

  protected:
    bool m_isHadron;
    bool m_isbHadron;
    reco::GenParticleRef m_particleRef;
};

class JetFlavourClustering : public edm::EDProducer {
   public:
      explicit JetFlavourClustering(const edm::ParameterSet&);
      ~JetFlavourClustering();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      virtual void beginRun(edm::Run&, edm::EventSetup const&);
      virtual void endRun(edm::Run&, edm::EventSetup const&);
      virtual void beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);
      virtual void endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);

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

      // ----------member data ---------------------------
      const edm::EDGetTokenT<edm::View<reco::Jet> >      jetsToken_;        // Input jet collection
      const edm::EDGetTokenT<edm::View<reco::Jet> >      groomedJetsToken_; // Input groomed jet collection
      const edm::EDGetTokenT<edm::View<reco::Jet> >      subjetsToken_;     // Input subjet collection
      const edm::EDGetTokenT<reco::GenParticleRefVector> bHadronsToken_;    // Input b hadron collection
      const edm::EDGetTokenT<reco::GenParticleRefVector> cHadronsToken_;    // Input c hadron collection
      const edm::EDGetTokenT<reco::GenParticleRefVector> partonsToken_;     // Input parton collection

      const std::string   jetAlgorithm_;
      const double        rParam_;
      const double        jetPtMin_;
      const double        ghostRescaling_;
      const bool          hadronFlavourHasPriority_;
      const bool          useSubjets_;

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
   groomedJetsToken_(mayConsume<edm::View<reco::Jet> >( iConfig.exists("groomedJets") ? iConfig.getParameter<edm::InputTag>("groomedJets") : edm::InputTag() )),
   subjetsToken_(mayConsume<edm::View<reco::Jet> >( iConfig.exists("subjets") ? iConfig.getParameter<edm::InputTag>("subjets") : edm::InputTag() )),
   bHadronsToken_(consumes<reco::GenParticleRefVector>( iConfig.getParameter<edm::InputTag>("bHadrons") )),
   cHadronsToken_(consumes<reco::GenParticleRefVector>( iConfig.getParameter<edm::InputTag>("cHadrons") )),
   partonsToken_(consumes<reco::GenParticleRefVector>( iConfig.getParameter<edm::InputTag>("partons") )),
   jetAlgorithm_(iConfig.getParameter<std::string>("jetAlgorithm")),
   rParam_(iConfig.getParameter<double>("rParam")),
   jetPtMin_(0.), // hardcoded to 0. since we simply want to recluster all input jets which already had some PtMin applied
   ghostRescaling_(iConfig.getParameter<double>("ghostRescaling")),
   hadronFlavourHasPriority_(iConfig.getParameter<bool>("hadronFlavourHasPriority")),
   useSubjets_(iConfig.exists("groomedJets") && iConfig.exists("subjets"))

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
     throw cms::Exception("InvalidJetAlgorithm") <<"Jet clustering algorithm is invalid: " << jetAlgorithm_ << ", use CambridgeAachen | Kt | AntiKt" << std::endl;
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
       if(constit->pt() == 0)
       {
         edm::LogWarning("NullTransverseMomentum") << "dropping input candidate with pt=0";
         continue;
       }
       fjInputs.push_back(fastjet::PseudoJet(constit->px(),constit->py(),constit->pz(),constit->energy()));
     }
   }
   // insert "ghost" b hadrons in the vector of constituents
   for(reco::GenParticleRefVector::const_iterator it = bHadrons->begin(); it != bHadrons->end(); ++it)
   {
     fastjet::PseudoJet p((*it)->px(),(*it)->py(),(*it)->pz(),(*it)->energy());
     p*=ghostRescaling_; // rescale hadron momentum
     p.set_user_info(new GhostInfo(true, true, *it));
     fjInputs.push_back(p);
   }
   // insert "ghost" c hadrons in the vector of constituents
   for(reco::GenParticleRefVector::const_iterator it = cHadrons->begin(); it != cHadrons->end(); ++it)
   {
     fastjet::PseudoJet p((*it)->px(),(*it)->py(),(*it)->pz(),(*it)->energy());
     p*=ghostRescaling_; // rescale hadron momentum
     p.set_user_info(new GhostInfo(true, false, *it));
     fjInputs.push_back(p);
   }
   // insert "ghost" partons in the vector of constituents
   for(reco::GenParticleRefVector::const_iterator it = partons->begin(); it != partons->end(); ++it)
   {
     fastjet::PseudoJet p((*it)->px(),(*it)->py(),(*it)->pz(),(*it)->energy());
     p*=ghostRescaling_; // rescale parton momentum
     p.set_user_info(new GhostInfo(false, false, *it));
     fjInputs.push_back(p);
   }

   // define jet clustering sequence
   fjClusterSeq_ = ClusterSequencePtr( new fastjet::ClusterSequence( fjInputs, *fjJetDefinition_ ) );
   // recluster jet constituents and inserted "ghosts"
   std::vector<fastjet::PseudoJet> inclusiveJets = fastjet::sorted_by_pt( fjClusterSeq_->inclusive_jets(jetPtMin_) );

   if( inclusiveJets.size() < jets->size() )
     throw cms::Exception("TooFewReclusteredJets") << "There are fewer reclustered than original jets. Please check that the jet algorithm and jet size match those used for the original jet collection.";

   // match reclustered and original jets
   std::vector<int> reclusteredIndices;
   matchReclusteredJets(jets,inclusiveJets,reclusteredIndices);

   // match groomed and original jets
   std::vector<int> groomedIndices;
   if( useSubjets_ )
   {
     if( groomedJets->size() > jets->size() )
       throw cms::Exception("TooManyGroomedJets") << "There are more groomed than original jets. Please check that the jet algorithm, jet size, and Pt threshold match for the two jet collections.";

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
     // since the "ghosts" are extremely soft, the configuration and ordering of the reclustered and original jets should in principle stay the same
     if( ( fabs( inclusiveJets.at(reclusteredIndices.at(i)).pt() - jets->at(i).pt() ) / jets->at(i).pt() ) > 0.01 ) // 1% difference in Pt should be sufficient to detect possible misconfigurations
       throw cms::Exception("JetPtMismatch") << "The reclustered and original jets have different Pt's. Please check that the jet algorithm and jet size match those used for the original jet collection.";

     reco::GenParticleRefVector clusteredbHadrons;
     reco::GenParticleRefVector clusteredcHadrons;
     reco::GenParticleRefVector clusteredPartons;
     reco::GenParticleRef flavourParton;
     reco::GenParticleRef hardestParton;
     double maxPt = -99.;
     int hadronFlavour = 0; // default hadron flavour set to 0 (= undefined)
     int partonFlavour = 0; // default parton flavour set to 0 (= undefined)

     // get jet constituents
     std::vector<fastjet::PseudoJet> constituents = inclusiveJets.at(reclusteredIndices.at(i)).constituents();
     // loop over jet constituents and try to find "ghosts"
     for(std::vector<fastjet::PseudoJet>::const_iterator it = constituents.begin(); it != constituents.end(); ++it)
     {
       if( !it->has_user_info() ) continue; // skip if not a "ghost"

       // b hadron "ghost"
       if( it->user_info<GhostInfo>().isHadron() && it->user_info<GhostInfo>().isbHadron() )
         clusteredbHadrons.push_back(it->user_info<GhostInfo>().particleRef());
       // c hadron "ghost"
       else if( it->user_info<GhostInfo>().isHadron() && !it->user_info<GhostInfo>().isbHadron() )
         clusteredcHadrons.push_back(it->user_info<GhostInfo>().particleRef());
       // parton "ghost"
       else if( !it->user_info<GhostInfo>().isHadron() )
       {
         const reco::GenParticleRef & tempParton = it->user_info<GhostInfo>().particleRef();
         clusteredPartons.push_back(tempParton);
         // b flavour gets priority
         if( flavourParton.isNull() && ( abs( tempParton->pdgId() ) == 4 )  ) flavourParton = tempParton;
         if( abs( tempParton->pdgId() ) == 5 )
         {
           if( flavourParton.isNull() ) flavourParton = tempParton;
           else if( abs( flavourParton->pdgId() ) != 5 ) flavourParton = tempParton;
         }
         if( tempParton->pt() > maxPt )
         {
           maxPt = tempParton->pt();
           hardestParton = tempParton;
         }
       }
     }
     // set hadron flavour
     if( clusteredbHadrons.size()>0 )
       hadronFlavour = 5;
     else if( clusteredcHadrons.size()>0 && clusteredbHadrons.size()==0 )
       hadronFlavour = 4;
     // set parton flavour
     if( flavourParton.isNull() )
     {
       if( hardestParton.isNonnull() ) partonFlavour = abs( hardestParton->pdgId() );
     }
     else
       partonFlavour = abs( flavourParton->pdgId() );

     // if enabled, check for conflicts between hadron- and parton-based flavours and give priority to the hadron-based flavour
     if( hadronFlavourHasPriority_ )
     {
       if( ( hadronFlavour==0 && (partonFlavour==4 || partonFlavour==5) ) ||
           ( hadronFlavour!=0 && hadronFlavour!=partonFlavour ) )
         partonFlavour = hadronFlavour;
     }

     // set the JetFlavourInfo for this jet
     (*jetFlavourInfos)[jets->refAt(i)] = reco::JetFlavourInfo(clusteredbHadrons,clusteredcHadrons,clusteredPartons,hadronFlavour,partonFlavour);

     // determine subjet flavour
     if( useSubjets_ )
     {
       if( subjetIndices.at(i).size()==0 ) continue; // continue if the original jet does not have subjets assigned

       // define vectors of GenParticleRefVectors for hadrons and partons assigned to different subjets
       std::vector<reco::GenParticleRefVector> assignedbHadrons(subjetIndices.at(i).size(),reco::GenParticleRefVector());
       std::vector<reco::GenParticleRefVector> assignedcHadrons(subjetIndices.at(i).size(),reco::GenParticleRefVector());
       std::vector<reco::GenParticleRefVector> assignedPartons(subjetIndices.at(i).size(),reco::GenParticleRefVector());

       // loop over clustered b hadrons and assign them to different subjets based on smallest dR
       for(reco::GenParticleRefVector::const_iterator it = clusteredbHadrons.begin(); it != clusteredbHadrons.end(); ++it)
       {
         std::vector<double> dRtoSubjets;

         for(size_t sj=0; sj<subjetIndices.at(i).size(); ++sj)
           dRtoSubjets.push_back( reco::deltaR( (*it)->rapidity(), (*it)->phi(), subjets->at(subjetIndices.at(i).at(sj)).rapidity(), subjets->at(subjetIndices.at(i).at(sj)).phi() ) );

         // find the closest subjet
         int closestSubjetIdx = std::distance( dRtoSubjets.begin(), std::min_element(dRtoSubjets.begin(), dRtoSubjets.end()) );

         assignedbHadrons.at(closestSubjetIdx).push_back( *it );
       }
       // loop over clustered c hadrons and assign them to different subjets based on smallest dR
       for(reco::GenParticleRefVector::const_iterator it = clusteredcHadrons.begin(); it != clusteredcHadrons.end(); ++it)
       {
         std::vector<double> dRtoSubjets;

         for(size_t sj=0; sj<subjetIndices.at(i).size(); ++sj)
           dRtoSubjets.push_back( reco::deltaR( (*it)->rapidity(), (*it)->phi(), subjets->at(subjetIndices.at(i).at(sj)).rapidity(), subjets->at(subjetIndices.at(i).at(sj)).phi() ) );

         // find the closest subjet
         int closestSubjetIdx = std::distance( dRtoSubjets.begin(), std::min_element(dRtoSubjets.begin(), dRtoSubjets.end()) );

         assignedcHadrons.at(closestSubjetIdx).push_back( *it );
       }
       // loop over clustered partons and assign them to different subjets based on smallest dR
       for(reco::GenParticleRefVector::const_iterator it = clusteredPartons.begin(); it != clusteredPartons.end(); ++it)
       {
         std::vector<double> dRtoSubjets;

         for(size_t sj=0; sj<subjetIndices.at(i).size(); ++sj)
           dRtoSubjets.push_back( reco::deltaR( (*it)->rapidity(), (*it)->phi(), subjets->at(subjetIndices.at(i).at(sj)).rapidity(), subjets->at(subjetIndices.at(i).at(sj)).phi() ) );

         // find the closest subjet
         int closestSubjetIdx = std::distance( dRtoSubjets.begin(), std::min_element(dRtoSubjets.begin(), dRtoSubjets.end()) );

         assignedPartons.at(closestSubjetIdx).push_back( *it );
       }

       // loop over subjets and determine their flavour
       for(size_t sj=0; sj<subjetIndices.at(i).size(); ++sj)
       {
         reco::GenParticleRef subjetFlavourParton;
         reco::GenParticleRef subjetHardestParton;
         double subjetMaxPt = -99.;
         int subjetHadronFlavour = 0; // default hadron flavour set to 0 (= undefined)
         int subjetPartonFlavour = 0; // default parton flavour set to 0 (= undefined)

         for(reco::GenParticleRefVector::const_iterator it = assignedPartons.at(sj).begin(); it != assignedPartons.at(sj).end(); ++it)
         {
           // b flavour gets priority
           if( subjetFlavourParton.isNull() && ( abs( (*it)->pdgId() ) == 4 )  ) subjetFlavourParton = (*it);
           if( abs( (*it)->pdgId() ) == 5 )
           {
             if( subjetFlavourParton.isNull() ) subjetFlavourParton = (*it);
             else if( abs( subjetFlavourParton->pdgId() ) != 5 ) subjetFlavourParton = (*it);
           }
           if( (*it)->pt() > subjetMaxPt )
           {
             subjetMaxPt = (*it)->pt();
             subjetHardestParton = (*it);
           }
         }
         // set hadron flavour
         if( assignedbHadrons.at(sj).size()>0 )
           subjetHadronFlavour = 5;
         else if( assignedcHadrons.at(sj).size()>0 && assignedbHadrons.at(sj).size()==0 )
           subjetHadronFlavour = 4;
         // set parton flavour
         if( subjetFlavourParton.isNull() )
         {
           if( subjetHardestParton.isNonnull() ) subjetPartonFlavour = abs( subjetHardestParton->pdgId() );
         }
         else
           subjetPartonFlavour = abs( subjetFlavourParton->pdgId() );

         // if enabled, check for conflicts between hadron- and parton-based flavours and give priority to the hadron-based flavour
         if( hadronFlavourHasPriority_ )
         {
           if( ( subjetHadronFlavour==0 && (subjetPartonFlavour==4 || subjetPartonFlavour==5) ) ||
               ( subjetHadronFlavour!=0 && subjetHadronFlavour!=subjetPartonFlavour ) )
             subjetPartonFlavour = subjetHadronFlavour;
         }

         // set the JetFlavourInfo for this subjet
         (*subjetFlavourInfos)[subjets->refAt(subjetIndices.at(i).at(sj))] = reco::JetFlavourInfo(assignedbHadrons.at(sj),assignedcHadrons.at(sj),assignedPartons.at(sj),subjetHadronFlavour,subjetPartonFlavour);
       }
     }
   }

   // put jet flavour infos in the event
   iEvent.put( jetFlavourInfos );
   // put subjet flavour infos in the event
   if( useSubjets_ )
     iEvent.put( subjetFlavourInfos, "SubJets" );
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
     double matchedDR = 1e9;
     int matchedIdx = -1;

     for(size_t rj=0; rj<reclusteredJets.size(); ++rj)
     {
       if( matchedLocks.at(rj) ) continue; // skip jets that have already been matched

       double tempDR = reco::deltaR( jets->at(j).rapidity(), jets->at(j).phi(), reclusteredJets.at(rj).rapidity(), reclusteredJets.at(rj).phi_std() );
       if( tempDR < matchedDR )
       {
         matchedDR = tempDR;
         matchedIdx = rj;
       }
     }

     if( matchedIdx>=0 ) matchedLocks.at(matchedIdx) = true;
     matchedIndices.push_back(matchedIdx);
   }

   if( std::find( matchedIndices.begin(), matchedIndices.end(), -1 ) != matchedIndices.end() )
     throw cms::Exception("JetMatchingFailed") << "Matching reclustered to original jets failed. Please check that the jet algorithm and jet size match those used for the original jet collection.";
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
     double matchedDR = 1e9;
     int matchedIdx = -1;

     for(size_t j=0; j<jets->size(); ++j)
     {
       if( jetLocks.at(j) ) continue; // skip jets that have already been matched

       double tempDR = reco::deltaR( jets->at(j).rapidity(), jets->at(j).phi(), groomedJets->at(gj).rapidity(), groomedJets->at(gj).phi() );
       if( tempDR < matchedDR )
       {
         matchedDR = tempDR;
         matchedIdx = j;
       }
     }

     if( matchedIdx>=0 ) jetLocks.at(matchedIdx) = true;
     jetIndices.push_back(matchedIdx);
   }

   if( std::find( jetIndices.begin(), jetIndices.end(), -1 ) != jetIndices.end() )
     throw cms::Exception("JetMatchingFailed") << "Matching groomed to original jets failed. Please check that the jet algorithm, jet size, and Pt threshold match for the two jet collections.";

   for(size_t j=0; j<jets->size(); ++j)
   {
     std::vector<int>::iterator matchedIndex = std::find( jetIndices.begin(), jetIndices.end(), j );

     matchedIndices.push_back( matchedIndex != jetIndices.end() ? std::distance(jetIndices.begin(),matchedIndex) : -1 );
   }
}

// ------------ method that matches groomed and original jets ------------
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
         throw cms::Exception("SubjetMatchingFailed") << "Matching subjets to original jets failed. Please check that the groomed jet and subjet collections belong to each other.";

       matchedIndices.push_back(subjetIndices);
     }
     else
       matchedIndices.push_back(subjetIndices);
   }
}

// ------------ method called once each job just before starting event loop  ------------
void
JetFlavourClustering::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
JetFlavourClustering::endJob() {
}

// ------------ method called when starting to processes a run  ------------
void
JetFlavourClustering::beginRun(edm::Run&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a run  ------------
void
JetFlavourClustering::endRun(edm::Run&, edm::EventSetup const&)
{
}

// ------------ method called when starting to processes a luminosity block  ------------
void
JetFlavourClustering::beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void
JetFlavourClustering::endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
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
