#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHWriteSpecificDecay_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHWriteSpecificDecay_h

#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHAnalyzerTokenWrapper.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHPlusMinusCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHTrackReference.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/PatCandidates/interface/GenericParticle.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"

#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicParticle.h"
#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicVertex.h"

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <fstream>

class TH1F;
class BPHRecoCandidate;

class BPHWriteSpecificDecay:
      public BPHAnalyzerWrapper<BPHModuleWrapper::one_producer> {

 public:

  explicit BPHWriteSpecificDecay( const edm::ParameterSet& ps );
  virtual ~BPHWriteSpecificDecay();

  static void fillDescriptions( edm::ConfigurationDescriptions& descriptions );

  virtual void beginJob();
  virtual void produce( edm::Event& ev, const edm::EventSetup& es );
  virtual void fill( edm::Event& ev, const edm::EventSetup& es );
  virtual void endJob();

 private:

  std::string pVertexLabel;
  std::string patMuonLabel;
  std::string ccCandsLabel;
  std::string pfCandsLabel;
  std::string pcCandsLabel;
  std::string gpCandsLabel;

  // token wrappers to allow running both on "old" and "new" CMSSW versions
  BPHTokenWrapper< std::vector<reco::Vertex                > > pVertexToken;
  BPHTokenWrapper< pat::MuonCollection                       > patMuonToken;
  BPHTokenWrapper< std::vector<pat::CompositeCandidate     > > ccCandsToken;
  BPHTokenWrapper< std::vector<reco::PFCandidate           > > pfCandsToken;
  BPHTokenWrapper< std::vector<BPHTrackReference::candidate> > pcCandsToken;
  BPHTokenWrapper< std::vector<pat::GenericParticle        > > gpCandsToken;

  bool usePV;
  bool usePM;
  bool useCC;
  bool usePF;
  bool usePC;
  bool useGP;

  std::string oniaName;
  std::string   sdName;
  std::string   ssName;
  std::string   buName;
  std::string   bdName;
  std::string   bsName;

  enum recoType { Onia, Pmm , Psi1, Psi2, Ups , Ups1, Ups2, Ups3,
                  Kx0, Pkk, Bu, Bd, Bs };
  enum  parType { ptMin, etaMax,
                  mPsiMin, mPsiMax, mKx0Min, mKx0Max, mPhiMin, mPhiMax, 
                  massMin, massMax, probMin, mFitMin, mFitMax,
                  constrMass, constrSigma, constrMJPsi, writeCandidate };
  std::map<std::string,recoType> rMap;
  std::map<std::string, parType> pMap;
  std::map<std::string, parType> fMap;
  std::map< recoType, std::map<parType,double> > parMap;

  bool recoOnia;
  bool recoKx0;
  bool recoPkk;
  bool recoBu;
  bool recoBd;
  bool recoBs;

  bool writeOnia;
  bool writeKx0;
  bool writePkk;
  bool writeBu;
  bool writeBd;
  bool writeBs;

  bool writeVertex;
  bool writeMomentum;

  std::vector<BPHPlusMinusConstCandPtr> lFull;
  std::vector<BPHPlusMinusConstCandPtr> lJPsi;
  std::vector<BPHRecoConstCandPtr> lSd;
  std::vector<BPHRecoConstCandPtr> lSs;
  std::vector<BPHRecoConstCandPtr> lBu;
  std::vector<BPHRecoConstCandPtr> lBd;
  std::vector<BPHRecoConstCandPtr> lBs;

  std::map<const BPHRecoCandidate*,const BPHRecoCandidate*> jPsiOMap;
  typedef edm::Ref< std::vector<reco::Vertex> > vertex_ref;
  std::map<const BPHRecoCandidate*,vertex_ref> pvRefMap;
  typedef edm::Ref< pat::CompositeCandidateCollection > compcc_ref;
  std::map<const BPHRecoCandidate*,compcc_ref> ccRefMap;

  void setRecoParameters( const edm::ParameterSet& ps );

  template <class T>
  edm::OrphanHandle<pat::CompositeCandidateCollection> write( edm::Event& ev,
              const std::vector<T>& list, const std::string& name ) {
    pat::CompositeCandidateCollection* ccList =
      new pat::CompositeCandidateCollection;
    int i;
    int n = list.size();
    std::map<const BPHRecoCandidate*,
             const BPHRecoCandidate*>::const_iterator jpoIter;
    std::map<const BPHRecoCandidate*,
             const BPHRecoCandidate*>::const_iterator jpoIend = jPsiOMap.end();
    std::map<const BPHRecoCandidate*,vertex_ref>::const_iterator pvrIter;
    std::map<const BPHRecoCandidate*,vertex_ref>::const_iterator pvrIend =
                                                                 pvRefMap.end();
    std::map<const BPHRecoCandidate*,compcc_ref>::const_iterator ccrIter;
    std::map<const BPHRecoCandidate*,compcc_ref>::const_iterator ccrIend =
                                                                 ccRefMap.end();
    for ( i = 0; i < n; ++i ) {
      const T& ptr = list[i];
      ccList->push_back( ptr->composite() );
      pat::CompositeCandidate& cc = ccList->back();
      if ( ( pvrIter = pvRefMap.find( ptr.get() ) ) != pvrIend )
           cc.addUserData ( "primaryVertex", pvrIter->second );
      const std::vector<std::string>& cNames = ptr->compNames();
      int j = 0;
      int m = cNames.size();
      while ( j < m ) {
        const std::string& compName = cNames[j++];
        const BPHRecoCandidate* cptr = ptr->getComp( compName ).get();
        if ( ( ccrIter = ccRefMap.find( cptr ) ) == ccrIend ) {
          if ( ( jpoIter = jPsiOMap.find( cptr ) ) != jpoIend )
               cptr = jpoIter->second;
          else cptr = 0;
        }
        if ( ( ccrIter = ccRefMap.find( cptr ) ) != ccrIend ) {
          compcc_ref cref = ccrIter->second;
          if ( cref.isNonnull() ) cc.addUserData ( "refTo" + compName, cref );
        }
      }
      const BPHPlusMinusCandidate* pmp =
            dynamic_cast<const BPHPlusMinusCandidate*>( ptr.get() );
      if ( pmp != 0 ) cc.addUserData( "cowboy", pmp->isCowboy() );
      if ( ptr->isEmpty() ) {
        if ( writeVertex ) cc.addUserData( "vertex" , ptr->vertex() );
        continue;
      }
      if ( writeVertex ) cc.addUserData( "fitVertex",
                         reco::Vertex( *ptr->currentDecayVertex() ) );
      if ( ptr->isValidFit() ) {
        const RefCountedKinematicParticle kinPart = ptr->currentParticle();
        const           KinematicState    kinStat = kinPart->currentState();
        cc.addUserFloat( "fitMass", kinStat.mass() );
        if ( writeMomentum ) cc.addUserData ( "fitMomentum",
                             kinStat.kinematicParameters().momentum() );
      }

    }
    typedef std::unique_ptr<pat::CompositeCandidateCollection> ccc_pointer;
    edm::OrphanHandle<pat::CompositeCandidateCollection> ccHandle =
    ev.put( ccc_pointer( ccList ), name );
    for ( i = 0; i < n; ++i ) {
      const BPHRecoCandidate* ptr = list[i].get();
      edm::Ref<pat::CompositeCandidateCollection> ccRef( ccHandle, i );
      ccRefMap[ptr] = ccRef;
    }
    return ccHandle;
  }

};

#endif
