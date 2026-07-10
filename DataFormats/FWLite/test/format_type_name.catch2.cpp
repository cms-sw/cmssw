/*
 *  format_type_name.cppunit.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 09/12/09.
 *
 */

#include "catch2/catch_all.hpp"
#include <iostream>
#include "DataFormats/FWLite/interface/format_type_name.h"

TEST_CASE("format_type_name", "[FWLite]") {
  SECTION("test") {
    typedef std::pair<std::string, std::string> Values;
    std::vector<Values> classToFriendly;
    classToFriendly.push_back(Values("Foo", "Foo"));
    classToFriendly.push_back(Values("bar::Foo", "bar_1Foo"));
    classToFriendly.push_back(Values("std::vector<Foo>", "std_1vector_9Foo_0"));
    classToFriendly.push_back(Values("std::vector<bar::Foo>", "std_1vector_9bar_1Foo_0"));
    classToFriendly.push_back(Values("V<A,B>", "V_9A_3B_0"));
    classToFriendly.push_back(
        Values("edm::ExtCollection<std::vector<reco::SuperCluster>,reco::SuperClusterRefProds>",
               "edm_1ExtCollection_9std_1vector_9reco_1SuperCluster_0_3reco_1SuperClusterRefProds_0"));
    classToFriendly.push_back(Values("A<B<C>, D<E> >", "A_9B_9C_0_3_4D_9E_0_4_0"));
    classToFriendly.push_back(Values("A<B<C<D> > >", "A_9B_9C_9D_0_4_0_4_0"));
    classToFriendly.push_back(Values("A<B<C,D>, E<F> >", "A_9B_9C_3D_0_3_4E_9F_0_4_0"));
    classToFriendly.push_back(Values("Aa<Bb<Cc>, Dd<Ee> >", "Aa_9Bb_9Cc_0_3_4Dd_9Ee_0_4_0"));
    classToFriendly.push_back(Values("Aa<Bb<Cc<Dd> > >", "Aa_9Bb_9Cc_9Dd_0_4_0_4_0"));
    classToFriendly.push_back(Values("Aa<Bb<Cc,Dd>, Ee<Ff> >", "Aa_9Bb_9Cc_3Dd_0_3_4Ee_9Ff_0_4_0"));
    classToFriendly.push_back(Values("Aa<Bb<Cc,Dd>, Ee<Ff,Gg> >", "Aa_9Bb_9Cc_3Dd_0_3_4Ee_9Ff_3Gg_0_4_0"));

    for (auto const& item : classToFriendly) {
      //std::cout << item.first << std::endl;
      REQUIRE(item.second == fwlite::format_type_to_mangled(item.first));
      REQUIRE(item.first == fwlite::unformat_mangled_to_type(item.second));
    }
  }
}
