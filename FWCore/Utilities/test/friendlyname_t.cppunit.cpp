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

class testfriendlyName : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testfriendlyName);

  CPPUNIT_TEST(test);

  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {}
  void tearDown() {}

  void test();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testfriendlyName);

void testfriendlyName::test() {
  typedef std::pair<std::string, std::string> Values;
  std::map<std::string, std::string> classToFriendly;
  classToFriendly.insert(Values("Foo", "Foo"));
  classToFriendly.insert(Values("const Foo", "constFoo"));
  classToFriendly.insert(Values("bar::Foo", "barFoo"));
  classToFriendly.insert(Values("std::vector<Foo>", "Foos"));
  classToFriendly.insert(Values("std::vector<bar::Foo>", "barFoos"));
  classToFriendly.insert(Values("std::set<bar::Foo>", "barFoostdset"));
  classToFriendly.insert(Values("std::map<Foo, bar::Bar>", "FoobarBarstdmap"));
  classToFriendly.insert(Values("std::unordered_set<bar::Foo>", "barFoostduset"));
  classToFriendly.insert(Values("std::unordered_set<std::basic_string<char>>", "Stringstduset"));
  classToFriendly.insert(Values("std::unordered_set<bar::Foo, std::hash<bar::Foo>>", "barFoostduset"));
  classToFriendly.insert(
      Values("std::unordered_set<std::basic_string<char>, std::hash<std::basic_string<char>>>", "Stringstduset"));
  classToFriendly.insert(
      Values("std::unordered_set<bar::Foo, std::hash<bar::Foo>, std::equal_to<bar::Foo>>", "barFoostduset"));
  classToFriendly.insert(
      Values("std::unordered_set<std::basic_string<char>, std::hash<std::basic_string<char>>, "
             "std::equal_to<std::basic_string<char>>>",
             "Stringstduset"));
  classToFriendly.insert(
      Values("std::unordered_set<bar::Foo, CustomHash, std::equal_to<bar::Foo>>", "barFooCustomHashstduset"));
  classToFriendly.insert(
      Values("std::unordered_set<std::basic_string<char>, CustomHash, std::equal_to<std::basic_string<char>>>",
             "StringCustomHashstduset"));
  classToFriendly.insert(Values("std::unordered_set<bar::Foo, CustomHash>", "barFooCustomHashstduset"));
  classToFriendly.insert(Values("std::unordered_set<std::basic_string<char>, CustomHash>", "StringCustomHashstduset"));
  classToFriendly.insert(Values("std::unordered_map<Foo, bar::Bar>", "FoobarBarstdumap"));
  classToFriendly.insert(Values("std::unordered_map<std::basic_string<char>, bar::Bar>", "StringbarBarstdumap"));
  classToFriendly.insert(Values("std::unordered_map<Foo, std::basic_string<char>>", "FooStringstdumap"));
  classToFriendly.insert(Values("std::unordered_map<Foo, bar::Bar, std::hash<Foo>>", "FoobarBarstdumap"));
  classToFriendly.insert(
      Values("std::unordered_map<std::basic_string<char> , bar::Bar, std::hash<std::basic_string<char> > >",
             "StringbarBarstdumap"));
  classToFriendly.insert(
      Values("std::unordered_map<Foo, std::basic_string<char> , std::hash<Foo>>", "FooStringstdumap"));
  classToFriendly.insert(
      Values("std::unordered_map<Foo, bar::Bar, std::hash<Foo>, std::equal_to<Foo>>", "FoobarBarstdumap"));
  classToFriendly.insert(
      Values("std::unordered_map<std::basic_string<char>, bar::Bar, std::hash<std::basic_string<char>>, "
             "std::equal_to<std::basic_string<char>>>",
             "StringbarBarstdumap"));
  classToFriendly.insert(Values("std::unordered_map<Foo, std::basic_string<char>, std::hash<Foo>, std::equal_to<Foo>>",
                                "FooStringstdumap"));
  classToFriendly.insert(
      Values("std::unordered_map<Foo, bar::Bar, CustomHash, std::equal_to<Foo>>", "FoobarBarCustomHashstdumap"));
  classToFriendly.insert(Values(
      "std::unordered_map<std::basic_string<char>, bar::Bar, CustomHash, std::equal_to<std::basic_string<char>>>",
      "StringbarBarCustomHashstdumap"));
  classToFriendly.insert(Values("std::unordered_map<Foo, std::basic_string<char>, CustomHash, std::equal_to<Foo>>",
                                "FooStringCustomHashstdumap"));
  classToFriendly.insert(Values("std::unordered_map<Foo, bar::Bar, CustomHash>", "FoobarBarCustomHashstdumap"));
  classToFriendly.insert(
      Values("std::unordered_map<std::basic_string<char>, bar::Bar, CustomHash>", "StringbarBarCustomHashstdumap"));
  classToFriendly.insert(
      Values("std::unordered_map<Foo, std::basic_string<char>, CustomHash>", "FooStringCustomHashstdumap"));
  classToFriendly.insert(Values("std::shared_ptr<Foo>", "FooSharedPtr"));
  classToFriendly.insert(Values("std::shared_ptr<bar::Foo>", "barFooSharedPtr"));
  classToFriendly.insert(Values("std::basic_string<char>", "String"));
  classToFriendly.insert(Values("std::string", "String"));
  classToFriendly.insert(Values("std::__cxx11::basic_string<char>", "String"));
  classToFriendly.insert(Values("std::__cxx11::basic_string<char,std::char_traits<char> >", "String"));
  classToFriendly.insert(Values("std::list<int>", "intstdlist"));
  classToFriendly.insert(Values("std::__cxx11::list<int>", "intstdlist"));
  classToFriendly.insert(Values("std::vector<std::shared_ptr<bar::Foo>>", "barFooSharedPtrs"));
  classToFriendly.insert(Values("std::vector<std::basic_string<char>>", "Strings"));
  classToFriendly.insert(Values("std::__cxx11::vector<std::__cxx11::basic_string<char>>", "Strings"));
  classToFriendly.insert(Values("std::unique_ptr<Foo>", "FooUniquePtr"));
  classToFriendly.insert(Values("std::unique_ptr<bar::Foo>", "barFooUniquePtr"));
  classToFriendly.insert(Values("std::unique_ptr<const Foo>", "constFooUniquePtr"));
  classToFriendly.insert(Values("const std::unique_ptr<Foo>", "FooconstUniquePtr"));
  classToFriendly.insert(Values("std::unique_ptr<Foo,std::default_delete<Foo>>", "FooUniquePtr"));
  classToFriendly.insert(Values("std::unique_ptr<const Foo, std::default_delete<const Foo>>", "constFooUniquePtr"));
  classToFriendly.insert(
      Values("std::unique_ptr<std::unique_ptr<Bar,std::default_delete<Bar>>,std::default_delete<std::unique_ptr<Bar,"
             "std::default_delete<Bar>>>>",
             "BarUniquePtrUniquePtr"));
  classToFriendly.insert(Values("std::vector<std::unique_ptr<bar::Foo>>", "barFooUniquePtrs"));
  classToFriendly.insert(
      Values("std::vector<std::unique_ptr<bar::Foo, std::default_delete<bar::Foo>>>", "barFooUniquePtrs"));
  classToFriendly.insert(Values("std::vector<std::unique_ptr<const Foo>>", "constFooUniquePtrs"));
  classToFriendly.insert(
      Values("std::unique_ptr<std::vector<std::unique_ptr<bar::Foo>>>", "barFooUniquePtrsUniquePtr"));
  classToFriendly.insert(
      Values("std::unique_ptr<std::vector<std::unique_ptr<bar::Foo>>, "
             "std::default_delete<std::vector<std::unique_ptr<bar::Foo>>>>",
             "barFooUniquePtrsUniquePtr"));
  classToFriendly.insert(
      Values("std::unique_ptr<std::vector<std::unique_ptr<bar::Foo, std::default_delete<bar::Foo>>>>",
             "barFooUniquePtrsUniquePtr"));
  classToFriendly.insert(
      Values("std::unique_ptr<std::vector<std::unique_ptr<bar::Foo, std::default_delete<bar::Foo>>>, "
             "std::default_delete<std::vector<std::unique_ptr<bar::Foo, std::default_delete<bar::Foo>>>>>",
             "barFooUniquePtrsUniquePtr"));
  classToFriendly.insert(Values("V<A,B>", "ABV"));
  classToFriendly.insert(Values("edm::ExtCollection<std::vector<reco::SuperCluster>,reco::SuperClusterRefProds>",
                                "recoSuperClustersrecoSuperClusterRefProdsedmExtCollection"));
  classToFriendly.insert(
      Values("edm::SortedCollection<EcalUncalibratedRecHit,edm::StrictWeakOrdering<EcalUncalibratedRecHit> >",
             "EcalUncalibratedRecHitsSorted"));
  classToFriendly.insert(
      Values("edm::OwnVector<aod::Candidate,edm::ClonePolicy<aod::Candidate> >", "aodCandidatesOwned"));
  classToFriendly.insert(Values("edm::OwnVector<Foo,edm::ClonePolicy<Foo> >", "FoosOwned"));
  classToFriendly.insert(Values("edm::OwnVector<My<int>, edm::ClonePolicy<My<int> > >", "intMysOwned"));
  classToFriendly.insert(Values("std::vector<edm::OwnVector<My<int>, edm::ClonePolicy<My<int> > > >", "intMysOwneds"));
  classToFriendly.insert(
      Values("edm::Wrapper<MuonDigiCollection<CSCDetId,CSCALCTDigi> >", "CSCDetIdCSCALCTDigiMuonDigiCollection"));
  classToFriendly.insert(
      Values("edm::AssociationMap<edm::OneToMany<std::vector<CaloJet>,std::vector<reco::Track>,unsigned int> >",
             "CaloJetsToManyrecoTracksAssociation"));
  classToFriendly.insert(
      Values("edm::AssociationMap<edm::OneToOne<std::vector<reco::Track>,std::vector<reco::TrackInfo>,unsigned int> >",
             "recoTracksToOnerecoTrackInfosAssociation"));
  classToFriendly.insert(Values("edm::AssociationMap<edm::OneToValue<std::vector<reco::Electron>,float,unsigned int> >",
                                "recoElectronsToValuefloatAssociation"));
  classToFriendly.insert(
      Values("edm::AssociationMap<edm::OneToManyWithQuality<edm::OwnVector<reco::Candidate,edm::ClonePolicy<reco::"
             "Candidate> >,edm::OwnVector<reco::Candidate,edm::ClonePolicy<reco::Candidate> >,double,unsigned int> >",
             "recoCandidatesOwnedToManyrecoCandidatesOwnedWithQuantitydoubleAssociation"));
  classToFriendly.insert(
      Values("edm::AssociationVector<edm::RefProd<std::vector<reco::CaloJet> "
             ">,std::vector<int>,edm::Ref<std::vector<reco::CaloJet>,reco::CaloJet,edm::refhelper::FindUsingAdvance<"
             "std::vector<reco::CaloJet>,reco::CaloJet> >,unsigned int,edm::helper::AssociationIdenticalKeyReference>",
             "recoCaloJetsedmRefProdTointsAssociationVector"));
  classToFriendly.insert(
      Values("edm::AssociationVector<edm::RefProd<edm::OwnVector<reco::Candidate,edm::ClonePolicy<reco::Candidate> > "
             ">,std::vector<double>,edm::Ref<edm::OwnVector<reco::Candidate,edm::ClonePolicy<reco::Candidate> "
             ">,reco::Candidate,edm::refhelper::FindUsingAdvance<edm::OwnVector<reco::Candidate,edm::ClonePolicy<reco::"
             "Candidate> >,reco::Candidate> >,unsigned int,edm::helper::AssociationIdenticalKeyReference>",
             "recoCandidatesOwnededmRefProdTodoublesAssociationVector"));
  classToFriendly.insert(
      Values("edm::AssociationMap<edm::OneToOne<std::vector<reco::Track>,std::vector<std::pair<double,double> "
             ">,unsigned int> >",
             "recoTracksToOnedoubledoublestdpairsAssociation"));
  classToFriendly.insert(
      Values("edm::AssociationMap<edm::OneToOne<std::vector<reco::Track>,std::vector<std::pair<Point3DBase<float,"
             "GlobalTag>,GlobalErrorBase<double,ErrorMatrixTag> > >,unsigned int> >",
             "recoTracksToOnefloatGlobalTagPoint3DBasedoubleErrorMatrixTagGlobalErrorBasestdpairsAssociation"));
  classToFriendly.insert(Values("A<B<C>, D<E> >", "CBEDA"));
  classToFriendly.insert(Values("A<B<C<D> > >", "DCBA"));
  classToFriendly.insert(Values("A<B<C,D>, E<F> >", "CDBFEA"));
  classToFriendly.insert(Values("Aa<Bb<Cc>, Dd<Ee> >", "CcBbEeDdAa"));
  classToFriendly.insert(Values("Aa<Bb<Cc<Dd> > >", "DdCcBbAa"));
  classToFriendly.insert(Values("Aa<Bb<Cc,Dd>, Ee<Ff> >", "CcDdBbFfEeAa"));
  classToFriendly.insert(Values("Aa<Bb<Cc,Dd>, Ee<Ff,Gg> >", "CcDdBbFfGgEeAa"));
  classToFriendly.insert(
      Values("edm::RangeMap<DetId,edm::OwnVector<SiPixelRecHit,edm::ClonePolicy<SiPixelRecHit> "
             ">,edm::ClonePolicy<SiPixelRecHit> >",
             "DetIdSiPixelRecHitsOwnedRangeMap"));
  classToFriendly.insert(
      Values("std::vector<edm::RangeMap<DetId,edm::OwnVector<SiPixelRecHit,edm::ClonePolicy<SiPixelRecHit> "
             ">,edm::ClonePolicy<SiPixelRecHit> > >",
             "DetIdSiPixelRecHitsOwnedRangeMaps"));
  classToFriendly.insert(
      Values("edm::RefVector< edm::OwnVector<reco::Candidate,edm::ClonePolicy<reco::Candidate> >,reco::Candidate, "
             "edm::refhelper::FindUsingAdvance<edm::OwnVector<reco::Candidate,edm::ClonePolicy<reco::Candidate> >, "
             "reco::Candidate> >",
             "recoCandidatesOwnedRefs"));
  classToFriendly.insert(
      Values("edm::RefVector< std::vector<reco::Track>, reco::Track, "
             "edm::refhelper::FindUsingAdvance<std::vector<reco::Track>, reco::Track> >",
             "recoTracksRefs"));
  classToFriendly.insert(
      Values("edm::RefVector<Col, Type, edm::refhelper::FindUsingAdvance<Col, Type> >", "ColTypeRefs"));
  classToFriendly.insert(
      Values("edm::AssociationMap<edm::OneToMany<std::vector<reco::PixelMatchGsfElectron>,edm::SortedCollection<"
             "EcalRecHit,edm::StrictWeakOrdering<EcalRecHit> >,unsigned int> >",
             "recoPixelMatchGsfElectronsToManyEcalRecHitsSortedAssociation"));
  classToFriendly.insert(
      Values("edm::AssociationVector<edm::RefToBaseProd<reco::Candidate>,std::vector<double>,edm::RefToBase<reco::"
             "Candidate>,unsigned int,edm::helper::AssociationIdenticalKeyReference>",
             "recoCandidateedmRefToBaseProdTodoublesAssociationVector"));
  classToFriendly.insert(
      Values("edm::RefVector<edm::AssociationMap<edm::OneToOne<std::vector<reco::BasicCluster>,std::vector<reco::"
             "ClusterShape>,unsigned int> "
             ">,edm::helpers::KeyVal<edm::Ref<std::vector<reco::BasicCluster>,reco::BasicCluster,edm::refhelper::"
             "FindUsingAdvance<std::vector<reco::BasicCluster>,reco::BasicCluster> "
             ">,edm::Ref<std::vector<reco::ClusterShape>,reco::ClusterShape,edm::refhelper::FindUsingAdvance<std::"
             "vector<reco::ClusterShape>,reco::ClusterShape> > "
             ">,edm::AssociationMap<edm::OneToOne<std::vector<reco::BasicCluster>,std::vector<reco::ClusterShape>,"
             "unsigned int> >::Find>",
             "recoBasicClustersToOnerecoClusterShapesAssociationRefs"));
  classToFriendly.insert(
      Values("edmNew::DetSetVector<edm::Ref<edmNew::DetSetVector<SiPixelCluster>,SiPixelCluster,edmNew::DetSetVector<"
             "SiPixelCluster>::FindForDetSetVector> >",
             "SiPixelClusteredmNewDetSetVectorSiPixelClusterSiPixelClusteredmNewDetSetVectorFindForDetSetVectoredmRefed"
             "mNewDetSetVector"));
  classToFriendly.insert(
      Values("std::vector<std::pair<const pat::Muon *, TLorentzVector>>", "constpatMuonptrTLorentzVectorstdpairs"));
  classToFriendly.insert(Values("int[]", "intAs"));
  classToFriendly.insert(Values("foo<int[]>", "intAsfoo"));
  classToFriendly.insert(Values("bar<foo<int[]>>", "intAsfoobar"));

  // Alpaka types
  classToFriendly.insert(Values("alpaka::DevCpu", "alpakaDevCpu"));
  classToFriendly.insert(Values("alpaka::DevUniformCudaHipRt<alpaka::ApiCudaRt>", "alpakaDevCudaRt"));
  classToFriendly.insert(Values("alpaka::DevUniformCudaHipRt<alpaka::ApiHipRt>", "alpakaDevHipRt"));
  classToFriendly.insert(Values("alpaka::QueueGenericThreadsBlocking<alpaka::DevCpu>", "alpakaQueueCpuBlocking"));
  classToFriendly.insert(Values("alpaka::QueueGenericThreadsNonBlocking<alpaka::DevCpu>", "alpakaQueueCpuNonBlocking"));
  classToFriendly.insert(Values("alpaka::uniform_cuda_hip::detail::QueueUniformCudaHipRt<alpaka::ApiCudaRt,true>",
                                "alpakaQueueCudaRtBlocking"));
  classToFriendly.insert(Values("alpaka::uniform_cuda_hip::detail::QueueUniformCudaHipRt<alpaka::ApiCudaRt,false>",
                                "alpakaQueueCudaRtNonBlocking"));
  classToFriendly.insert(Values("alpaka::uniform_cuda_hip::detail::QueueUniformCudaHipRt<alpaka::ApiHipRt,true>",
                                "alpakaQueueHipRtBlocking"));
  classToFriendly.insert(Values("alpaka::uniform_cuda_hip::detail::QueueUniformCudaHipRt<alpaka::ApiHipRt,false>",
                                "alpakaQueueHipRtNonBlocking"));

  for (std::map<std::string, std::string>::iterator itInfo = classToFriendly.begin(), itInfoEnd = classToFriendly.end();
       itInfo != itInfoEnd;
       ++itInfo) {
    //std::cout <<itInfo->first<<std::endl;
    if (itInfo->second != edm::friendlyname::friendlyName(itInfo->first)) {
      std::cout << "class name: '" << itInfo->first << "' has wrong friendly name \n"
                << "expect: '" << itInfo->second << "' got: '" << edm::friendlyname::friendlyName(itInfo->first) << "'"
                << std::endl;
      CPPUNIT_ASSERT(0 && "expected friendly name does not match actual friendly name");
    }
  }
}
