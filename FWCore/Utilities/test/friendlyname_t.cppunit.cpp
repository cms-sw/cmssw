/*
 *  friendlyname_t.cppunit.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 8/8/05.
 *
 */

#include <cppunit/extensions/HelperMacros.h>
#include <iostream>
#include "FWCore/Utilities/interface/FriendlyName.h"

using namespace edm;

class testfriendlyName: public CppUnit::TestFixture
{
   CPPUNIT_TEST_SUITE(testfriendlyName);
   
   CPPUNIT_TEST(test);
   
   CPPUNIT_TEST_SUITE_END();
public:
      void setUp(){}
   void tearDown(){}
   
   void test();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testfriendlyName);


void testfriendlyName::test()
{
  typedef std::pair<std::string,std::string> Values; 
  std::map<std::string, std::string> classToFriendly;
  classToFriendly.insert( Values("Foo","Foo") );
  classToFriendly.insert( Values("bar::Foo","barFoo") );
  classToFriendly.insert( Values("std::vector<Foo>","Foos") );
  classToFriendly.insert( Values("std::vector<bar::Foo>","barFoos") );
  classToFriendly.insert( Values("V<A,B>","ABV") );
  classToFriendly.insert( Values("edm::ExtCollection<std::vector<reco::SuperCluster>,reco::SuperClusterRefProds>","recoSuperClustersrecoSuperClusterRefProdsedmExtCollection") );
  classToFriendly.insert( Values("edm::SortedCollection<EcalUncalibratedRecHit,edm::StrictWeakOrdering<EcalUncalibratedRecHit> >","EcalUncalibratedRecHitsSorted") );
  classToFriendly.insert( Values("edm::OwnVector<aod::Candidate,edm::ClonePolicy<aod::Candidate> >","aodCandidatesOwned") );
  classToFriendly.insert( Values("edm::OwnVector<Foo,edm::ClonePolicy<Foo> >","FoosOwned") );
  classToFriendly.insert( Values("edm::OwnVector<My<int>, edm::ClonePolicy<My<int> > >","intMysOwned") );
  classToFriendly.insert( Values("std::vector<edm::OwnVector<My<int>, edm::ClonePolicy<My<int> > > >","intMysOwneds") );
  classToFriendly.insert( Values("edm::Wrapper<MuonDigiCollection<CSCDetId,CSCALCTDigi> >","CSCDetIdCSCALCTDigiMuonDigiCollection") );
  classToFriendly.insert( Values("edm::AssociationMap<edm::OneToMany<std::vector<CaloJet>,std::vector<reco::Track>,unsigned int> >","CaloJetsToManyrecoTracksAssociation") );
  classToFriendly.insert( Values("edm::AssociationMap<edm::OneToOne<std::vector<reco::Track>,std::vector<reco::TrackInfo>,unsigned int> >","recoTracksToOnerecoTrackInfosAssociation") );
  classToFriendly.insert( Values("edm::AssociationMap<edm::OneToValue<std::vector<reco::Electron>,float,unsigned int> >",
                                 "recoElectronsToValuefloatAssociation"));
  classToFriendly.insert( Values("edm::AssociationMap<edm::OneToManyWithQuality<edm::OwnVector<reco::Candidate,edm::ClonePolicy<reco::Candidate> >,edm::OwnVector<reco::Candidate,edm::ClonePolicy<reco::Candidate> >,double,unsigned int> >",
                                 "recoCandidatesOwnedToManyrecoCandidatesOwnedWithQuantitydoubleAssociation"));
  classToFriendly.insert( Values("edm::AssociationVector<edm::RefProd<std::vector<reco::CaloJet> >,std::vector<int>,edm::Ref<std::vector<reco::CaloJet>,reco::CaloJet,edm::refhelper::FindUsingAdvance<std::vector<reco::CaloJet>,reco::CaloJet> >,unsigned int,edm::helper::AssociationIdenticalKeyReference>",
                                 "recoCaloJetsedmRefProdTointsAssociationVector") );
  classToFriendly.insert( Values("edm::AssociationVector<edm::RefProd<edm::OwnVector<reco::Candidate,edm::ClonePolicy<reco::Candidate> > >,std::vector<double>,edm::Ref<edm::OwnVector<reco::Candidate,edm::ClonePolicy<reco::Candidate> >,reco::Candidate,edm::refhelper::FindUsingAdvance<edm::OwnVector<reco::Candidate,edm::ClonePolicy<reco::Candidate> >,reco::Candidate> >,unsigned int,edm::helper::AssociationIdenticalKeyReference>",
                                 "recoCandidatesOwnededmRefProdTodoublesAssociationVector") );
  classToFriendly.insert( Values("edm::AssociationMap<edm::OneToOne<std::vector<reco::Track>,std::vector<std::pair<double,double> >,unsigned int> >",
                                 "recoTracksToOnedoubledoublestdpairsAssociation"));
  classToFriendly.insert( Values("edm::AssociationMap<edm::OneToOne<std::vector<reco::Track>,std::vector<std::pair<Point3DBase<float,GlobalTag>,GlobalErrorBase<double,ErrorMatrixTag> > >,unsigned int> >",
                                 "recoTracksToOnefloatGlobalTagPoint3DBasedoubleErrorMatrixTagGlobalErrorBasestdpairsAssociation"));
  classToFriendly.insert( Values("A<B<C>, D<E> >","CBEDA"));
  classToFriendly.insert( Values("A<B<C<D> > >","DCBA"));
  classToFriendly.insert( Values("A<B<C,D>, E<F> >","CDBFEA"));
  classToFriendly.insert( Values("Aa<Bb<Cc>, Dd<Ee> >","CcBbEeDdAa"));
  classToFriendly.insert( Values("Aa<Bb<Cc<Dd> > >","DdCcBbAa"));
  classToFriendly.insert( Values("Aa<Bb<Cc,Dd>, Ee<Ff> >","CcDdBbFfEeAa"));
  classToFriendly.insert( Values("Aa<Bb<Cc,Dd>, Ee<Ff,Gg> >","CcDdBbFfGgEeAa"));
  classToFriendly.insert( Values("edm::RangeMap<DetId,edm::OwnVector<SiPixelRecHit,edm::ClonePolicy<SiPixelRecHit> >,edm::ClonePolicy<SiPixelRecHit> >","DetIdSiPixelRecHitsOwnedRangeMap"));
  classToFriendly.insert( Values("std::vector<edm::RangeMap<DetId,edm::OwnVector<SiPixelRecHit,edm::ClonePolicy<SiPixelRecHit> >,edm::ClonePolicy<SiPixelRecHit> > >","DetIdSiPixelRecHitsOwnedRangeMaps"));
  classToFriendly.insert( Values("edm::RefVector< edm::OwnVector<reco::Candidate,edm::ClonePolicy<reco::Candidate> >,reco::Candidate, edm::refhelper::FindUsingAdvance<edm::OwnVector<reco::Candidate,edm::ClonePolicy<reco::Candidate> >, reco::Candidate> >","recoCandidatesOwnedRefs"));
  classToFriendly.insert( Values("edm::RefVector< std::vector<reco::Track>, reco::Track, edm::refhelper::FindUsingAdvance<std::vector<reco::Track>, reco::Track> >","recoTracksRefs"));
  classToFriendly.insert( Values("edm::RefVector<Col, Type, edm::refhelper::FindUsingAdvance<Col, Type> >","ColTypeRefs"));
  classToFriendly.insert( Values("edm::AssociationMap<edm::OneToMany<std::vector<reco::PixelMatchGsfElectron>,edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit> >,unsigned int> >",
                                 "recoPixelMatchGsfElectronsToManyEcalRecHitsSortedAssociation"));
  classToFriendly.insert( Values("edm::AssociationVector<edm::RefToBaseProd<reco::Candidate>,std::vector<double>,edm::RefToBase<reco::Candidate>,unsigned int,edm::helper::AssociationIdenticalKeyReference>",
                                 "recoCandidateedmRefToBaseProdTodoublesAssociationVector"));
  classToFriendly.insert( Values("edm::RefVector<edm::AssociationMap<edm::OneToOne<std::vector<reco::BasicCluster>,std::vector<reco::ClusterShape>,unsigned int> >,edm::helpers::KeyVal<edm::Ref<std::vector<reco::BasicCluster>,reco::BasicCluster,edm::refhelper::FindUsingAdvance<std::vector<reco::BasicCluster>,reco::BasicCluster> >,edm::Ref<std::vector<reco::ClusterShape>,reco::ClusterShape,edm::refhelper::FindUsingAdvance<std::vector<reco::ClusterShape>,reco::ClusterShape> > >,edm::AssociationMap<edm::OneToOne<std::vector<reco::BasicCluster>,std::vector<reco::ClusterShape>,unsigned int> >::Find>",
                                 "recoBasicClustersToOnerecoClusterShapesAssociationRefs"));
                                 
  for(std::map<std::string, std::string>::iterator itInfo = classToFriendly.begin(),
      itInfoEnd = classToFriendly.end();
      itInfo != itInfoEnd;
      ++itInfo) {
    //std::cout <<itInfo->first<<std::endl;
    if( itInfo->second != edm::friendlyname::friendlyName(itInfo->first) ) {
      std::cout <<"class name: '"<<itInfo->first<<"' has wrong friendly name \n"
      <<"expect: '"<<itInfo->second<<"' got: '"<<edm::friendlyname::friendlyName(itInfo->first)<<"'"<<std::endl;
      CPPUNIT_ASSERT(0 && "expected friendly name does not match actual friendly name");
    }
  }
}
