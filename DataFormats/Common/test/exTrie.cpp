/*
 */

#include "DataFormats/Common/interface/Trie.h"


#include<string>
#include<iostream>



struct Print {
  typedef edm::TrieNode<int> const node;
  void operator()(node & n, std::string const & label) const {
    std::cout << label << " " << n.getValue() << std::endl;
  }
  
};

int main(int argc, char **argv)
{
  /// trie that associates a integer to strings
  /// 0 is the default value I want to receive when there is no match
  /// in trie
  edm::Trie<int>	trie(0);
  typedef edm::TrieNode<int> Node;
  typedef Node const * pointer; // sigh....
  typedef edm::TrieNodeIter<int> node_iterator;

  char tre[] = {'a','a','a'};
  char quattro[] = {'c','a','a','a'};

  for (int j=0;j<3;j++) {
    tre[2]='a';
    quattro[3]='a';
    for (int i=0;i<10;i++) {
       trie.addEntry(tre,3,i);
       trie.addEntry(quattro,4,i);
       tre[2]++;
       quattro[3]++;
    }
    tre[1]++;
    quattro[2]++;
  }

 
  std::cout << "get [aac] " << trie.getEntry("aac", 3) << std::endl;
  std::cout << "get [caae] = " << trie.getEntry("caae", 4) << std::endl;

  trie.setEntry("caag", 4, -2);
  std::cout << "get [caag] = " << trie.getEntry("caag", 4) << std::endl;

  // no match
  std::cout << "get [abcd] = " << trie.getEntry("abcd", 4) << std::endl;
  // no match
  std::cout << "get [ca] = " << trie.getEntry("ca", 2) << std::endl;

  trie.display(std::cout);
  std::cout << std::endl;

  pointer pn = trie.getNode("ab",2);
  if (pn) pn->display(std::cout,0,' ');
  std::cout << std::endl;

  node_iterator e;
  for (node_iterator p(trie.getNode("ab",2)); p!=e; p++)
    std::cout << "ab" << p.label() << " = " << p->getValue() << std::endl;

  std::cout << std::endl;
  for (node_iterator p(trie.getInitialNode()); p!=e; p++)
    std::cout << p.label() << " = " << p->getValue() << std::endl;
  std::cout << std::endl;
	
  Print pr;
  edm::walkTrie(pr,*trie.getInitialNode());
  std::cout << std::endl;

}
