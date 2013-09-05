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

#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

  /**
   *
   * @brief Trie test suite
   *
   * <h2>Try to add/get string from trie</h2>
   *
   * @author Julien Lemoine <speedblue@happycoders.org>
   *
   */

  class TestedmTrie : public CppUnit::TestFixture
    {
      CPPUNIT_TEST_SUITE(TestedmTrie);
      CPPUNIT_TEST(testString);
      CPPUNIT_TEST(testUnsigned);
      CPPUNIT_TEST(testSort);
      CPPUNIT_TEST_SUITE_END();

    public:
      /// run all test tokenizer
      void testString();
      void testUnsigned();
      void testSort();
    };

CPPUNIT_TEST_SUITE_REGISTRATION(TestedmTrie);


#include <iostream>
#include <sstream>
#include <list>

#include "DataFormats/Common/interface/Trie.h"


  void TestedmTrie::testString()
  {
    try
      {

		edm::Trie<std::string> strTrie(std::string(""));
		strTrie.insert("Premiere Chaine", 15, std::string("1er"));
		strTrie.insert("Deuxieme Chaine", std::string("2eme"));
		{
		  const std::string &s = strTrie.find("unknown", 7);
		  CPPUNIT_ASSERT_EQUAL(std::string(""), s);
		}
		{
		  const std::string &s = strTrie.find("test");
		  CPPUNIT_ASSERT_EQUAL(std::string(""), s);
		}
		{
		  const std::string &s = strTrie.find("Premiere Chaine", 15);
		  CPPUNIT_ASSERT_EQUAL(std::string("1er"), s);
		}
		{
		  const std::string &s = strTrie.find("Premiere Chaine", 14);
		  CPPUNIT_ASSERT_EQUAL(std::string(""), s);
		}
		{
		  const std::string &s = strTrie.find("premiere Chaine", 15);
		  CPPUNIT_ASSERT_EQUAL(std::string(""), s);
		}
		{
		  const std::string &s = strTrie.find("Premiere Chaine ", 16);
		  CPPUNIT_ASSERT_EQUAL(std::string(""), s);
		}
		{
		  const std::string &s = strTrie.find("Deuxieme Chaine");
		  CPPUNIT_ASSERT_EQUAL(std::string("2eme"), s);
		}
	  }
    catch (const edm::Exception &e)
      {
	std::cerr << e.what() << std::endl;
	CPPUNIT_ASSERT(false);
      }
  }

  void TestedmTrie::testUnsigned()
  {
	try
	  {
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
		CPPUNIT_ASSERT_EQUAL((unsigned)1, res);

		res = nbTrie.find("Un", 2);
		CPPUNIT_ASSERT_EQUAL((unsigned)0, res);
		
		res = nbTrie.find("UN", 2);
		CPPUNIT_ASSERT_EQUAL((unsigned)0, res);

		res = nbTrie.find("", 0);
		CPPUNIT_ASSERT_EQUAL((unsigned)0, res);
		
		res = nbTrie.find("deux");
		CPPUNIT_ASSERT_EQUAL((unsigned)2, res);
		
		res = nbTrie.find(" deux ", 6);
		CPPUNIT_ASSERT_EQUAL((unsigned)0, res);
	  }
    catch (const edm::Exception &e)
      {
	std::cerr << e.what() << std::endl;
	CPPUNIT_ASSERT(false);
      }
  }

  void TestedmTrie::testSort()
  {
    try
      {
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
		CPPUNIT_ASSERT_EQUAL((unsigned)0, first->value());
		CPPUNIT_ASSERT_EQUAL((void*)0x0, (void*)first->brother());
		CPPUNIT_ASSERT_EQUAL((unsigned char)'a', first->subNodeLabel());
		// Get one first sub node
		first = first->subNode(); //a*
		CPPUNIT_ASSERT_EQUAL((unsigned)0, first->value());
		CPPUNIT_ASSERT(first != 0x0);

		// There is no other letter than a
		CPPUNIT_ASSERT_EQUAL((void*)0x0, (void*)first->brother());

		// Get first sub node of 'a'
		CPPUNIT_ASSERT(first->subNode() != 0x0);
		CPPUNIT_ASSERT_EQUAL((unsigned char)'b', first->subNodeLabel());
		first = first->subNode(); //ab*
		CPPUNIT_ASSERT_EQUAL((unsigned)0, first->value());
		CPPUNIT_ASSERT(first->subNode() != 0x0);
		CPPUNIT_ASSERT_EQUAL((unsigned char)'c', first->subNodeLabel());
		last = first->subNode(); //abc
		CPPUNIT_ASSERT_EQUAL((unsigned)3, last->value());
		CPPUNIT_ASSERT_EQUAL((void*)0x0, (void*)last->subNode());
		CPPUNIT_ASSERT(last->brother() != 0x0);
		CPPUNIT_ASSERT_EQUAL((unsigned char)'d', last->brotherLabel());
		last = last->brother(); // abd
		CPPUNIT_ASSERT_EQUAL((unsigned)9, last->value());
		CPPUNIT_ASSERT_EQUAL((void*)0x0, (void*)last->subNode());
		CPPUNIT_ASSERT(last->brother() != 0x0);
		CPPUNIT_ASSERT_EQUAL((unsigned char)'e', last->brotherLabel());
		last = last->brother(); // abe
		CPPUNIT_ASSERT_EQUAL((unsigned)6, last->value());
		CPPUNIT_ASSERT_EQUAL((void*)0x0, (void*)last->subNode());
		CPPUNIT_ASSERT_EQUAL((void*)0x0, (void*)last->brother());

		CPPUNIT_ASSERT(first->brother() != 0x0);
		CPPUNIT_ASSERT_EQUAL((unsigned char)'c', first->brotherLabel());
		first = first->brother(); //ac*
		CPPUNIT_ASSERT_EQUAL((unsigned)0, first->value());

		CPPUNIT_ASSERT(first->subNode() != 0x0);
		CPPUNIT_ASSERT_EQUAL((unsigned char)'c', first->subNodeLabel());
		last = first->subNode(); //acc
		CPPUNIT_ASSERT_EQUAL((unsigned)7, last->value());
		CPPUNIT_ASSERT_EQUAL((void*)0x0, (void*)last->subNode());
		CPPUNIT_ASSERT(last->brother() != 0x0);
		CPPUNIT_ASSERT_EQUAL((unsigned char)'d', last->brotherLabel());
		last = last->brother(); // acd
		CPPUNIT_ASSERT_EQUAL((unsigned)1, last->value());
		CPPUNIT_ASSERT_EQUAL((void*)0x0, (void*)last->subNode());
		CPPUNIT_ASSERT(last->brother() != 0x0);
		CPPUNIT_ASSERT_EQUAL((unsigned char)'e', last->brotherLabel());
		last = last->brother(); // ace
		CPPUNIT_ASSERT_EQUAL((unsigned)4, last->value());
		CPPUNIT_ASSERT_EQUAL((void*)0x0, (void*)last->subNode());
		CPPUNIT_ASSERT_EQUAL((void*)0x0, (void*)last->brother());

		CPPUNIT_ASSERT(first->brother() != 0x0);
		CPPUNIT_ASSERT_EQUAL((unsigned char)'d', first->brotherLabel());
		first = first->brother(); //ad*
		CPPUNIT_ASSERT_EQUAL((unsigned)0, first->value());

		CPPUNIT_ASSERT(first->subNode() != 0x0);
		CPPUNIT_ASSERT_EQUAL((unsigned char)'c', first->subNodeLabel());
		last = first->subNode(); //adc
		CPPUNIT_ASSERT_EQUAL((unsigned)5, last->value());
		CPPUNIT_ASSERT_EQUAL((void*)0x0, (void*)last->subNode());
		CPPUNIT_ASSERT(last->brother() != 0x0);
		CPPUNIT_ASSERT_EQUAL((unsigned char)'d', last->brotherLabel());
		last = last->brother(); // add
		CPPUNIT_ASSERT_EQUAL((unsigned)8, last->value());
		CPPUNIT_ASSERT_EQUAL((void*)0x0, (void*)last->subNode());
		CPPUNIT_ASSERT(last->brother() != 0x0);
		CPPUNIT_ASSERT_EQUAL((unsigned char)'e', last->brotherLabel());
		last = last->brother(); // ade
		CPPUNIT_ASSERT_EQUAL((unsigned)2, last->value());
		CPPUNIT_ASSERT_EQUAL((void*)0x0, (void*)last->subNode());
		CPPUNIT_ASSERT_EQUAL((void*)0x0, (void*)last->brother());
		CPPUNIT_ASSERT_EQUAL((void*)0x0, (void*)first->brother());
      }
    catch (const edm::Exception &e)
      {
	std::cerr << e.what() << std::endl;
	CPPUNIT_ASSERT(false);
      }
  }
