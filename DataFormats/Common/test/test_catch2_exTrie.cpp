/*
 */
#include "catch2/catch_all.hpp"

#include "DataFormats/Common/interface/Trie.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>
#include <string>
#include <sstream>

struct Print {
  //  typedef edm::TrieNode<int> const node;
  //void operator()(node& n, std::string const& label) const {
  //  std::cout << label << " " << n.value() << std::endl;
  // }
  void operator()(int v, std::string const& label) const { s_ << label << " " << v << std::endl; }
  mutable std::ostringstream s_;
};

TEST_CASE("test Trie", "[Trie]") {
  /// trie that associates a integer to strings
  /// 0 is the default value I want to receive when there is no match
  /// in trie
  edm::Trie<int> trie(0);
  typedef edm::TrieNode<int> Node;
  typedef Node const* pointer;  // sigh....
  typedef edm::TrieNodeIter<int> node_iterator;

  char tre[] = {'a', 'a', 'a'};
  char quattro[] = {'c', 'a', 'a', 'a'};

  for (int j = 0; j < 3; j++) {
    tre[2] = 'a';
    quattro[3] = 'a';
    for (int i = 0; i < 10; i++) {
      trie.insert(tre, 3, i);
      trie.insert(quattro, 4, i);
      tre[2]++;
      quattro[3]++;
    }
    tre[1]++;
    quattro[2]++;
  }

  SECTION("get") {
    CHECK(trie.find("aac", 3) == 2);
    CHECK(trie.find("caae", 4) == 4);

    trie.setEntry("caag", 4, -2);
    CHECK(trie.find("caag", 4) == -2);

    SECTION("no match") {
      CHECK(trie.find("abcd", 4) == 0);
      CHECK(trie.find("ca", 2) == 0);
    }
  }
  SECTION("display") {
    trie.setEntry("caag", 4, -2);

    std::ostringstream s;
    trie.display(s);
    std::string output = R"(value[0]
  label[a] value[0]
    label[a] value[0]
      label[a] value[0]
      label[b] value[1]
      label[c] value[2]
      label[d] value[3]
      label[e] value[4]
      label[f] value[5]
      label[g] value[6]
      label[h] value[7]
      label[i] value[8]
      label[j] value[9]
    label[b] value[0]
      label[a] value[0]
      label[b] value[1]
      label[c] value[2]
      label[d] value[3]
      label[e] value[4]
      label[f] value[5]
      label[g] value[6]
      label[h] value[7]
      label[i] value[8]
      label[j] value[9]
    label[c] value[0]
      label[a] value[0]
      label[b] value[1]
      label[c] value[2]
      label[d] value[3]
      label[e] value[4]
      label[f] value[5]
      label[g] value[6]
      label[h] value[7]
      label[i] value[8]
      label[j] value[9]
  label[c] value[0]
    label[a] value[0]
      label[a] value[0]
        label[a] value[0]
        label[b] value[1]
        label[c] value[2]
        label[d] value[3]
        label[e] value[4]
        label[f] value[5]
        label[g] value[-2]
        label[h] value[7]
        label[i] value[8]
        label[j] value[9]
      label[b] value[0]
        label[a] value[0]
        label[b] value[1]
        label[c] value[2]
        label[d] value[3]
        label[e] value[4]
        label[f] value[5]
        label[g] value[6]
        label[h] value[7]
        label[i] value[8]
        label[j] value[9]
      label[c] value[0]
        label[a] value[0]
        label[b] value[1]
        label[c] value[2]
        label[d] value[3]
        label[e] value[4]
        label[f] value[5]
        label[g] value[6]
        label[h] value[7]
        label[i] value[8]
        label[j] value[9]
)";
    REQUIRE_THAT(s.str(), Catch::Matchers::Equals(output));

    SECTION("ab display") {
      pointer pn = trie.node("ab", 2);
      REQUIRE(pn != nullptr);
      std::string output = R"(label[ ] value[0]
  label[a] value[0]
  label[b] value[1]
  label[c] value[2]
  label[d] value[3]
  label[e] value[4]
  label[f] value[5]
  label[g] value[6]
  label[h] value[7]
  label[i] value[8]
  label[j] value[9]
label[c] value[0]
  label[a] value[0]
  label[b] value[1]
  label[c] value[2]
  label[d] value[3]
  label[e] value[4]
  label[f] value[5]
  label[g] value[6]
  label[h] value[7]
  label[i] value[8]
  label[j] value[9]
)";
      std::ostringstream s;
      pn->display(s, 0, ' ');
      CHECK(s.str() == output);
    }
  }
  SECTION("iteration") {
    const std::vector<char> labels = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'};

    SECTION("ab") {
      node_iterator e;
      int value = 0;
      int index = 0;
      for (node_iterator p(trie.node("ab", 2)); p != e; p++) {
        CHECK(labels[index] == p.label());
        CHECK(value == p->value());
        ++value;
        ++index;
      }
    }
    SECTION("ab: string") {
      auto pn = trie.node("ab");
      auto e = pn->end();
      int value = 0;
      int index = 0;
      for (Node::const_iterator p = pn->begin(); p != e; p++) {
        CHECK(p.label() == labels[index]);
        CHECK(value == p->value());
        ++value;
        ++index;
      }
    }
    SECTION("top") {
      std::vector<char> labels = {'a', 'c'};
      int index = 0;
      auto e = trie.initialNode()->end();
      for (node_iterator p(trie.initialNode()); p != e; p++) {
        CHECK(p.label() == labels[index]);
        CHECK(p->value() == 0);
        ++index;
      }
    }
  }

  SECTION("full walk") {
    trie.setEntry("caag", 4, -2);
    Print pr;
    edm::walkTrie(pr, *trie.initialNode());
    std::string const output = R"(a 0
aa 0
aaa 0
aab 1
aac 2
aad 3
aae 4
aaf 5
aag 6
aah 7
aai 8
aaj 9
ab 0
aba 0
abb 1
abc 2
abd 3
abe 4
abf 5
abg 6
abh 7
abi 8
abj 9
ac 0
aca 0
acb 1
acc 2
acd 3
ace 4
acf 5
acg 6
ach 7
aci 8
acj 9
c 0
ca 0
caa 0
caaa 0
caab 1
caac 2
caad 3
caae 4
caaf 5
caag -2
caah 7
caai 8
caaj 9
cab 0
caba 0
cabb 1
cabc 2
cabd 3
cabe 4
cabf 5
cabg 6
cabh 7
cabi 8
cabj 9
cac 0
caca 0
cacb 1
cacc 2
cacd 3
cace 4
cacf 5
cacg 6
cach 7
caci 8
cacj 9
)";
    CHECK(output == pr.s_.str());
  }

  SECTION("leaves iteration") {
    trie.setEntry("caag", 4, -2);
    Print pr;
    const std::string output = R"(aaa 0
aab 1
aac 2
aad 3
aae 4
aaf 5
aag 6
aah 7
aai 8
aaj 9
aba 0
abb 1
abc 2
abd 3
abe 4
abf 5
abg 6
abh 7
abi 8
abj 9
aca 0
acb 1
acc 2
acd 3
ace 4
acf 5
acg 6
ach 7
aci 8
acj 9
caaa 0
caab 1
caac 2
caad 3
caae 4
caaf 5
caag -2
caah 7
caai 8
caaj 9
caba 0
cabb 1
cabc 2
cabd 3
cabe 4
cabf 5
cabg 6
cabh 7
cabi 8
cabj 9
caca 0
cacb 1
cacc 2
cacd 3
cace 4
cacf 5
cacg 6
cach 7
caci 8
cacj 9
)";
    edm::iterateTrieLeaves(pr, *trie.initialNode());
  }
}
