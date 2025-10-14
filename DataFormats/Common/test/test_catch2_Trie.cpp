/*
** 
** Copyright (C) 2006 Julien Lemoine
** This program is free software; you can redistribute it and/or modify
** it under the terms of the GNU Lesser General Public License as published by
** the Free Software Foundation; either version 2 of the License, or
** (at your option) any later version.
** 
** This program is distributed in the hope that it will be useful,
** but WITHOUT ANY WARRANTY; without even the implied warranty of
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
** GNU Lesser General Public License for more details.
** 
** You should have received a copy of the GNU Lesser General Public License
** along with this program; if not, write to the Free Software
** Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
**
**
**   modified by Vincenzo Innocente on 15/08/2007
*/

#include "catch2/catch_all.hpp"
#include <iostream>
#include <sstream>
#include <list>
#include <string>
#include "DataFormats/Common/interface/Trie.h"

TEST_CASE("edm::Trie", "[Trie]") {
  SECTION("string") {
    try {
      edm::Trie<std::string> strTrie(std::string(""));
      strTrie.insert("Premiere Chaine", 15, std::string("1er"));
      strTrie.insert("Deuxieme Chaine", std::string("2eme"));
      {
        const std::string &s = strTrie.find("unknown", 7);
        REQUIRE(s == "");
      }
      {
        const std::string &s = strTrie.find("test");
        REQUIRE(s == "");
      }
      {
        const std::string &s = strTrie.find("Premiere Chaine", 15);
        REQUIRE(s == "1er");
      }
      {
        const std::string &s = strTrie.find("Premiere Chaine", 14);
        REQUIRE(s == "");
      }
      {
        const std::string &s = strTrie.find("premiere Chaine", 15);
        REQUIRE(s == "");
      }
      {
        const std::string &s = strTrie.find("Premiere Chaine ", 16);
        REQUIRE(s == "");
      }
      {
        const std::string &s = strTrie.find("Deuxieme Chaine");
        REQUIRE(s == "2eme");
      }
    } catch (const edm::Exception &e) {
      std::cerr << e.what() << std::endl;
      REQUIRE(false);
    }
  }

  SECTION("unsigned") {
    try {
      edm::Trie<unsigned> nbTrie(0);
      nbTrie.insert("un", 2, 1);
      nbTrie.insert("deux", 4, 2);
      nbTrie.insert("test", 4, 3);
      nbTrie.insert("tat", 4);
      nbTrie.insert("taa", 4);
      nbTrie.insert("tbp", 5);
      nbTrie.insert("tlp", 3, 6);

      unsigned res = 0;

      res = nbTrie.find("un", 2);
      REQUIRE(res == 1u);

      res = nbTrie.find("Un", 2);
      REQUIRE(res == 0u);

      res = nbTrie.find("UN", 2);
      REQUIRE(res == 0u);

      res = nbTrie.find("", 0);
      REQUIRE(res == 0u);

      res = nbTrie.find("deux");
      REQUIRE(res == 2u);

      res = nbTrie.find(" deux ", 6);
      REQUIRE(res == 0u);
    } catch (const edm::Exception &e) {
      std::cerr << e.what() << std::endl;
      REQUIRE(false);
    }
  }

  SECTION("sort") {
    try {
      //Test if trie is well sorted
      edm::Trie<unsigned> test(0);
      test.insert("acd", 3, 1);
      test.insert("ade", 3, 2);
      test.insert("abc", 3, 3);
      test.insert("ace", 3, 4);
      test.insert("adc", 3, 5);
      test.insert("abe", 3, 6);
      test.insert("acc", 3, 7);
      test.insert("add", 3, 8);
      test.insert("abd", 3, 9);
      const edm::TrieNode<unsigned> *first = test.initialNode(), *last = 0x0;
      REQUIRE(first->value() == 0u);
      REQUIRE(first->brother() == nullptr);
      REQUIRE(first->subNodeLabel() == (unsigned char)'a');
      // Get one first sub node
      first = first->subNode();  //a*
      REQUIRE(first->value() == 0u);
      REQUIRE(first != nullptr);
      // There is no other letter than a
      REQUIRE(first->brother() == nullptr);
      // Get first sub node of 'a'
      REQUIRE(first->subNode() != nullptr);
      REQUIRE(first->subNodeLabel() == (unsigned char)'b');
      first = first->subNode();  //ab*
      REQUIRE(first->value() == 0u);
      REQUIRE(first->subNode() != nullptr);
      REQUIRE(first->subNodeLabel() == (unsigned char)'c');
      last = first->subNode();  //abc
      REQUIRE(last->value() == 3u);
      REQUIRE(last->subNode() == nullptr);
      REQUIRE(last->brother() != nullptr);
      REQUIRE(last->brotherLabel() == (unsigned char)'d');
      last = last->brother();  // abd
      REQUIRE(last->value() == 9u);
      REQUIRE(last->subNode() == nullptr);
      REQUIRE(last->brother() != nullptr);
      REQUIRE(last->brotherLabel() == (unsigned char)'e');
      last = last->brother();  // abe
      REQUIRE(last->value() == 6u);
      REQUIRE(last->subNode() == nullptr);
      REQUIRE(last->brother() == nullptr);
      REQUIRE(first->brother() != nullptr);
      REQUIRE(first->brotherLabel() == (unsigned char)'c');
      first = first->brother();  //ac*
      REQUIRE(first->value() == 0u);
      REQUIRE(first->subNode() != nullptr);
      REQUIRE(first->subNodeLabel() == (unsigned char)'c');
      last = first->subNode();  //acc
      REQUIRE(last->value() == 7u);
      REQUIRE(last->subNode() == nullptr);
      REQUIRE(last->brother() != nullptr);
      REQUIRE(last->brotherLabel() == (unsigned char)'d');
      last = last->brother();  // acd
      REQUIRE(last->value() == 1u);
      REQUIRE(last->subNode() == nullptr);
      REQUIRE(last->brother() != nullptr);
      REQUIRE(last->brotherLabel() == (unsigned char)'e');
      last = last->brother();  // ace
      REQUIRE(last->value() == 4u);
      REQUIRE(last->subNode() == nullptr);
      REQUIRE(last->brother() == nullptr);
      REQUIRE(first->brother() != nullptr);
      REQUIRE(first->brotherLabel() == (unsigned char)'d');
      first = first->brother();  //ad*
      REQUIRE(first->value() == 0u);
      REQUIRE(first->subNode() != nullptr);
      REQUIRE(first->subNodeLabel() == (unsigned char)'c');
      last = first->subNode();  //adc
      REQUIRE(last->value() == 5u);
      REQUIRE(last->subNode() == nullptr);
      REQUIRE(last->brother() != nullptr);
      REQUIRE(last->brotherLabel() == (unsigned char)'d');
      last = last->brother();  // add
      REQUIRE(last->value() == 8u);
      REQUIRE(last->subNode() == nullptr);
      REQUIRE(last->brother() != nullptr);
      REQUIRE(last->brotherLabel() == (unsigned char)'e');
      last = last->brother();  // ade
      REQUIRE(last->value() == 2u);
      REQUIRE(last->subNode() == nullptr);
      REQUIRE(last->brother() == nullptr);
      REQUIRE(first->brother() == nullptr);
    } catch (const edm::Exception &e) {
      std::cerr << e.what() << std::endl;
      REQUIRE(false);
    }
  }
}
