#ifndef   DataFormat_Common_Trie_H_
# define  DataFormat_Common_Trie_H_
/*
** 
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
**
*/


#include <list>

namespace edm
{
  // fwd declaration
  template <typename T>
  class TrieNode;

  /**
   * The goal of this class is to allocate Trie node by paquet of X
   * element in order to reduce heap-admin size
   */
  template <typename T>
  class TrieFactory
    {
    public:
      TrieFactory(unsigned paquetSize);
      ~TrieFactory();

    private:
      /// avoid default constructor
      TrieFactory();
      /// avoid copy constructor
      TrieFactory(const TrieFactory &e);
      /// avoid affectation operator
      TrieFactory& operator=(const TrieFactory &e);

    public:
      TrieNode<T>* getNewNode(const T &value);
      void clear();

    private:
      unsigned			_paquetSize;
      std::list<TrieNode<T>*>	_allocatedNodes;
      TrieNode<T>		*_lastNodes;
      unsigned			_nbUsedInLastNodes;
    };
}


namespace edm
{
  /**
   * @brief this class represent the node of a trie, it contains a
   * link to a sub node and a link to a brother (node which have the
   * same father)
   */
  template <typename T>
  class TrieNode
  {
  public:
    TrieNode();
    ~TrieNode();

  private:
    /// avoid copy constructor
    TrieNode(const TrieNode &e);
    /// avoid affectation operator
    TrieNode& operator=(const TrieNode &e);

  public:
    /// set value associed to node
    void setValue(const T &val);
    /// get value associed to node
    const T& getValue() const;

    /// get brother (return 0x0 this node has no brother)
    const TrieNode<T>* getBrother() const;
    TrieNode<T>* getBrother();
    /// get brother label
    unsigned char getBrotherLabel() const;

    // get first sub Node
    const TrieNode<T>* getSubNode() const;
    TrieNode<T>* getSubNode();
    unsigned char getSubNodeLabel() const;

    // Looking for a sub node
    const TrieNode<T>* getSubNodeByLabel(unsigned char chr) const;
    TrieNode<T>* getSubNodeByLabel(unsigned char chr);

    // add an edge
    void addSubNode(unsigned char chr, TrieNode<T> *node);

    /// display content of node in output stream
    void display(std::ostream &os, unsigned offset, unsigned char label) const;

    /// clear content of TrieNode
    void clear();

  protected:
    template <typename Node>
    Node _sequentialSearch(Node first, unsigned char label,
			   unsigned char val) const;
    /// set brother (used by sort)
    void _setBrother(TrieNode<T> *brother, unsigned char brotherLabel);
    /// add a new brother
    void _addBrother(unsigned char chr, TrieNode<T> *brother);
    /**
     * @ brief get brother that has the label chr (return 0x0 if brother is
     * not found)
     */
    const TrieNode<T>* _getBrother(unsigned char chr) const;
    /**
     * @ brief get brother that has the label chr (return 0x0 if brother is
     * not found)
     */
    TrieNode<T>* _getBrother(unsigned char chr);

  private:
    /// pointer to brother (node with same father as this one)
    TrieNode<T>		*_brother;
    /// character to go to brother (node with same father as this one)
    unsigned char	_brotherLabel;
    /// pointer to first sub node
    TrieNode<T>		*_firstSubNode;
    /// character to go to first subnode
    unsigned char	_firstSubNodeLabel;
    /// value associed to this node
    T			_value;
  };
}

#include <ostream>

namespace edm
{
  //fwd declaration
  template <typename T>
    class TrieFactory;
  template <typename T>
    class TrieNode;

  /**
   * Implement a trie in memory with the smallest structure as possible
   * (use few RAM as possible)
   */
  template <typename T>
    class Trie
    {
    public:
      /// constuctor, empty is the value returned when no match in found
      /// in trie
      Trie(const T &empty);
      ~Trie();

    private:
      /// avoid default constructor
      Trie();
      /// avoid copy constructor
      Trie(const Trie &e);
      /// avoid affectation operator
      Trie& operator=(const Trie &e);
    
    public:
      /// add an entry in the Trie, if entry already exist an exception
      /// is throw
      void addEntry(const char *str, unsigned strLen, const T &value);
      /// associates a value to a string, if string is already in Trie,
      /// value is overwriten
      void setEntry(const char *str, unsigned strLen, const T &value);
      /// get an entry in the Trie
      const T& getEntry(const char *str, unsigned strLen) const;
      /// get node matching a string
      const TrieNode<T>* getNode(const char *str, unsigned strLen) const;
      ///  get initial TrieNode
      const TrieNode<T>* getInitialNode() const;
      /// display content of trie in output stream
      void display(std::ostream &os);
      /// clear the content of trie
      void clear();

    protected:
      TrieNode<T>* _addEntry(const char *str, unsigned strLen);

    private:
      /// value returned when no match is found in trie
      T			_empty;
      /// factory
      TrieFactory<T>	*_factory;
      /// first node of trie
      TrieNode<T>		*_initialNode;
    };
}



#include <boost/iterator/iterator_facade.hpp>

#include<string>
#include<iostream>

namespace edm{

  template<typename T>
  class TrieNodeIter
    : public boost::iterator_facade<TrieNodeIter<T>,
				    TrieNode<T> const, 
				    boost::forward_traversal_tag >
  {
    
  public:
    typedef TrieNodeIter<T> self;
    typedef TrieNode<T> const node_base;
    TrieNodeIter()
      : m_node(0), m_label(0)
    {}
    
    explicit TrieNodeIter(node_base* p)
      : m_node(p ? p->getSubNode() : 0), 
	m_label(p ? p->getSubNodeLabel() : 0)
    {}
    
    unsigned char label() const { return m_label;}
  private:
    friend class boost::iterator_core_access;
    
    void increment() {
      m_label = m_node->getBrotherLabel();
      m_node = m_node->getBrother(); 
    }
    
    bool equal(self const& other) const
    {
      return this->m_node == other.m_node;
    }
    
    node_base& dereference() const { return *m_node; }
    
    node_base* m_node;
    unsigned char m_label;
  };
  
  
  template<typename V, typename T>
  void walkTrie(V & v,  TrieNode<T> const  & n, std::string const & label="") {
    typedef TrieNode<T> const node_base;
    typedef TrieNodeIter<T> node_iterator;
    node_iterator e;
    for (node_iterator p(&n); p!=e; ++p) {
      v(*p,label+(char)p.label());
      walkTrie(v,*p,label+(char)p.label());
    }
  }
  
} 


//
//----------------------------------------------------------------
//
// implementations


template <typename T>
edm::TrieFactory<T>::TrieFactory(unsigned paquetSize) :
  _paquetSize(paquetSize), _lastNodes(0x0), _nbUsedInLastNodes(0)
{
  _lastNodes = new TrieNode<T>[paquetSize];
}

template <typename T>
edm::TrieFactory<T>::~TrieFactory()
{
  typename std::list<TrieNode<T>*>::const_iterator it;

  for (it = _allocatedNodes.begin(); it != _allocatedNodes.end(); ++it)
    delete[] *it;
  if (_lastNodes)
    delete[] _lastNodes;
}

template <typename T>
edm::TrieNode<T>* edm::TrieFactory<T>::getNewNode(const T &value)
{
  if (_nbUsedInLastNodes == _paquetSize)
    {
      _allocatedNodes.push_back(_lastNodes);
      _nbUsedInLastNodes = 0;
      _lastNodes = new TrieNode<T>[_paquetSize];
    }
  TrieNode<T> *res = &_lastNodes[_nbUsedInLastNodes];
  ++_nbUsedInLastNodes;
  res->setValue(value);
  res->clear();
  return res;
}

template <typename T>
void edm::TrieFactory<T>::clear()
{
  typename std::list<TrieNode<T>*>::const_iterator it;
  for (it = _allocatedNodes.begin(); it != _allocatedNodes.end(); ++it)
    delete[] *it;
  _allocatedNodes.clear();
  _nbUsedInLastNodes = 0;
}


template <typename T>
edm::TrieNode<T>::TrieNode() :
  _brother(0), _brotherLabel(0), _firstSubNode(0), _firstSubNodeLabel(0)
  /// we can not set _value here because type is unknown. assert that
  /// the value is set later with setValue()
{
}

template <typename T>
edm::TrieNode<T>::~TrieNode()
{
  // do not delete _brother and _firstSubNode because they are
  // allocated by factory (TrieFactory) and factory will delete them
}

template <typename T>
void edm::TrieNode<T>::setValue(const T &val)
{
  _value = val;
}

template <typename T>
const T& edm::TrieNode<T>::getValue() const
{
  return _value;
}

template <typename T>
const edm::TrieNode<T>* edm::TrieNode<T>::getBrother() const
{
  return _brother;
}

template <typename T>
edm::TrieNode<T>* edm::TrieNode<T>::getBrother()
{
  return _brother;
}

template <typename T>
const edm::TrieNode<T>* edm::TrieNode<T>::_getBrother(unsigned char chr) const
{
  const TrieNode<T> *brother = _brother;
  return _sequentialSearch(brother, _brotherLabel, chr);
}

template <typename T>
edm::TrieNode<T>* edm::TrieNode<T>::_getBrother(unsigned char chr)
{
  return _sequentialSearch(_brother, _brotherLabel, chr);
}

template <typename T>
void edm::TrieNode<T>::_addBrother(unsigned char chr, TrieNode<T> *brother)
{
  if (!_brother || _brotherLabel > chr)
    {
      brother->_setBrother(_brother, _brotherLabel);
      _brother = brother;
      _brotherLabel = chr;
    }
  else
    _brother->_addBrother(chr, brother);
}

template <typename T>
unsigned char edm::TrieNode<T>::getBrotherLabel() const
{
  return _brotherLabel;
}

template <typename T>
const edm::TrieNode<T>* edm::TrieNode<T>::getSubNode() const
{
  return _firstSubNode;
}

template <typename T>
edm::TrieNode<T>* edm::TrieNode<T>::getSubNode()
{
  return _firstSubNode;
}

template <typename T>
unsigned char edm::TrieNode<T>::getSubNodeLabel() const
{
  return _firstSubNodeLabel;
}

template <typename T>
const edm::TrieNode<T>* edm::TrieNode<T>::getSubNodeByLabel(unsigned char chr) const
{
  const TrieNode<T> *first = _firstSubNode;
  return _sequentialSearch(first, _firstSubNodeLabel, chr);
}

template <typename T>
edm::TrieNode<T>* edm::TrieNode<T>::getSubNodeByLabel(unsigned char chr)
{
  return _sequentialSearch(_firstSubNode, _firstSubNodeLabel, chr);
}

template <typename T>
void edm::TrieNode<T>::addSubNode(unsigned char chr, TrieNode<T> *node)
{
  if (!_firstSubNode || _firstSubNodeLabel > chr)
    {
      node->_setBrother(_firstSubNode, _firstSubNodeLabel);
      _firstSubNode = node;
      _firstSubNodeLabel = chr;
    }
  else
    _firstSubNode->_addBrother(chr, node);
}

template <typename T>
template <typename Node>
inline Node edm::TrieNode<T>::_sequentialSearch(Node first, unsigned char label, unsigned char val) const
{
  if (first && label <= val)
    {
      if (label == val)
	return first;
      return first->_getBrother(val);
    }
  return 0x0;
}

template <typename T>
void edm::TrieNode<T>::_setBrother(TrieNode<T> *brother, unsigned char brotherLabel)
{
  _brother = brother;
  _brotherLabel = brotherLabel;
}

template <typename T>
void edm::TrieNode<T>::display(std::ostream &os, unsigned offset, unsigned char label) const
{
  unsigned int i;
  for (i = 0; i < offset; ++i)
    os << " ";
  if (label)
    os << "label[" << label << "] ";
  os << "value[" << _value << "]" << std::endl;
  if (_firstSubNode)
    _firstSubNode->display(os, offset + 2, _firstSubNodeLabel);
  if (_brother)
    _brother->display(os, offset, _brotherLabel);
}

template <typename T>
void edm::TrieNode<T>::clear()
{
  _brother = 0x0;
  _brotherLabel = 0;
  _firstSubNode = 0x0;
  _firstSubNodeLabel = 0;
}


#include <vector>
#include <algorithm>
#include <string>
#include <cassert>


namespace edm {
  struct VinException {
    explicit VinException(const char * mess) : m_mess(mess){}
    char const * what() const { return m_mess.c_str();}
    std::string m_mess;
  };
}

template <typename T>
edm::Trie<T>::Trie(const T &empty) :
  _empty(empty), _factory(0x0), _initialNode(0x0)
{
  // initialize nodes by paquets of 10000
  _factory = new TrieFactory<T>(10000);
  _initialNode = _factory->getNewNode(_empty);
}

template <typename T>
edm::Trie<T>::~Trie()
{
  if (_factory)
    delete _factory;
}

template <typename T>
void edm::Trie<T>::setEntry(const char *str, unsigned strLen, const T &value)
{
  TrieNode<T>	*node = _addEntry(str, strLen);
  node->setValue(value);
}

template <typename T>
edm::TrieNode<T>* edm::Trie<T>::_addEntry(const char *str, unsigned strLen)
{
  unsigned	pos = 0;
  bool		found = true;
  TrieNode<T>	*node = _initialNode, *previous = 0x0;

  // Look for the part of the word which is in Trie
  while (found && pos < strLen)
    {
      found = false;
      previous = node;
      node = node->getSubNodeByLabel(str[pos]);
      if (node)
	{
	  found = true;
	  ++pos;
	}
    }

  // Add part of the word which is not in Trie
  if (!node || pos != strLen)
    {
      node = previous;
      for (unsigned i = pos; i < strLen; ++i)
	{
	  TrieNode<T> *newNode = _factory->getNewNode(_empty);
	  node->addSubNode(str[pos], newNode);
	  node = newNode;
	  ++pos;
	}
    }
  assert(node != 0x0);
  return node;
}

template <typename T>
void edm::Trie<T>::addEntry(const char *str, unsigned strLen, const T &value)
{
  TrieNode<T>	*node = _addEntry(str, strLen);

  // Set the value on the last node
  if (node && node->getValue() != _empty)
    throw edm::VinException("The word is already in automaton");
  node->setValue(value);
}

template <typename T>
const T& edm::Trie<T>::getEntry(const char *str, unsigned strLen) const
{
  unsigned		pos = 0;
  bool			found = true;
  const TrieNode<T>	*node = _initialNode;
	
  while (found && pos < strLen)
    {
      found = false;
      node = node->getSubNodeByLabel(str[pos]);
      if (node)
	{
	  found = true;
	  ++pos;
	}
    }
  if (node && pos == strLen) // The word is complet in the automaton
    return node->getValue();
  return _empty;
}


template <typename T>
edm::TrieNode<T> const * 
edm::Trie<T>::getNode(const char *str, unsigned strLen) const {
  unsigned		pos = 0;
  bool			found = true;
  const TrieNode<T>	*node = _initialNode;
	
  while (found && pos < strLen)
    {
      found = false;
      node = node->getSubNodeByLabel(str[pos]);
      if (node)
	{
	  found = true;
	  ++pos;
	}
    }
  return node;
}


template <typename T>
const edm::TrieNode<T>* edm::Trie<T>::getInitialNode() const
{
  return _initialNode;
}

template <typename T>
void edm::Trie<T>::clear()
{
  _factory->clear();
  _initialNode = _factory->getNewNode(_empty);
}

template <typename T>
void edm::Trie<T>::display(std::ostream &os)
{
  if (_initialNode)
    _initialNode->display(os, 0, 0);
}

#endif	 //  DataFormat_Common_Trie_H_

