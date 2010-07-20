#include "CondCore/ORA/interface/Exception.h"
#include "MultiIndexDataTrie.h"
//
#include <sstream>
// externals 
#include "CoralBase/AttributeList.h"

ora::MultiIndexDataTrie::MultiIndexDataTrie():
  m_children(),
  m_data(){
  
}

ora::MultiIndexDataTrie::~MultiIndexDataTrie(){
  clear();
}

size_t ora::MultiIndexDataTrie::push( const std::vector<int>& indexes,
                                      const coral::AttributeList& data ){
  size_t s=0;
  MultiIndexDataTrie* trie = this;
  for( size_t i=0;i<indexes.size();i++){
    size_t ts = trie->m_children.size();
    MultiIndexDataTrie* nt = 0;
    if( ts == 0 || indexes[i] > (int)(ts-1) ){
      for(size_t j=0;j<indexes[i]-ts+1;j++){
        ++s;
        nt = new MultiIndexDataTrie;
        trie->m_children.push_back(nt);
      }
    } else {
      nt = trie->m_children[indexes[i]];
      if( !nt ){
        std::stringstream mess;
        mess << "Slot for index["<<i<<"] is empty.";
        throwException( mess.str(),"MultiIndexDataTrie::push" );
      }
    }
    trie = nt;
  }
  trie->m_data.reset( new coral::AttributeList( data ) );
  return s;
}

coral::AttributeList& ora::MultiIndexDataTrie::lookup( const std::vector<int>& indexes ){
  MultiIndexDataTrie* trie = this;
  for( size_t i=0;i<indexes.size();i++){
    if( trie->m_children.size()==0 || indexes[i] > (int)(trie->m_children.size()-1)){
      std::stringstream mess;
      mess << "Index["<<i<<"] is out of bound.";
      throwException( mess.str(),"MultiIndexDataTrie::lookup" );
    }
    trie = trie->m_children[indexes[i]];
    if( !trie ){
      std::stringstream mess;
      mess << "Slot for index["<<i<<"] is empty.";
      throwException( mess.str(),"MultiIndexDataTrie::lookup" );      
    }
  }
  if(!trie->m_data.get()){
    throwException( "No Data for the specified index combination.",
                    "MultiIndexDataTrie::lookup" );
  }
  return *trie->m_data;
}

#include <iostream>
boost::shared_ptr<coral::AttributeList> ora::MultiIndexDataTrie::lookupAndClear( const std::vector<int>& indexes ){
  MultiIndexDataTrie* branch = this;
  MultiIndexDataTrie* trie = 0;
  size_t i=0;
  for( ;i<indexes.size();i++){
    if( branch->m_children.size()==0 || indexes[i] > (int)(branch->m_children.size()-1)){
      std::stringstream mess;
      mess << "Index["<<i<<"] is out of bound.";
      throwException( mess.str(),"MultiIndexDataTrie::lookup" );
    }
    trie = branch;
    branch = branch->m_children[indexes[i]];
    if( !branch ){
      std::stringstream mess;
      mess << "Slot for index["<<i<<"] is empty.";
      throwException( mess.str(),"MultiIndexDataTrie::lookupAndClear" );      
    }
  }
  MultiIndexDataTrie* leaf = trie->m_children[indexes[i-1]];
  if(!leaf->m_data.get()){
    throwException( "No Data for the specified index combination.",
                    "MultiIndexDataTrie::lookupAndClear" );
  }
  boost::shared_ptr<coral::AttributeList> tmp;
  tmp = leaf->m_data;
  delete leaf;
  trie->m_children[indexes[i-1]] = 0;
  return tmp;  
}

size_t ora::MultiIndexDataTrie::size() const {
  return m_children.size();
}

void ora::MultiIndexDataTrie::clear(){
  for(std::vector<MultiIndexDataTrie*>::iterator iT = m_children.begin();
      iT != m_children.end(); iT++){
    if(*iT) delete *iT;
  }
  m_children.clear();
  m_data.reset();
}

size_t ora::MultiIndexDataTrie::branchSize( const std::vector<int>& indexes, size_t depth ) const{
  if( depth > indexes.size() ) depth = indexes.size();
  const MultiIndexDataTrie* trie = this;
  for( size_t i=0;i<depth;i++){
    if( trie->m_children.size()==0 || indexes[i] > (int)(trie->m_children.size()-1)){
      std::stringstream mess;
      mess << "Index["<<i<<"] is out of bound.";
      throwException( mess.str(),"MultiIndexDataTrie::lookup" );
    }
    trie = trie->m_children[indexes[i]];
    if( !trie ){
      std::stringstream mess;
      mess << "Slot for index["<<i<<"] is empty.";
      throwException( mess.str(),"MultiIndexDataTrie::branchSize" );      
    }
  }
  return trie->m_children.size();  
}

size_t ora::MultiIndexDataTrie::totalSize() const {
  size_t sz = 0;
  for(std::vector<MultiIndexDataTrie*>::const_iterator iT = m_children.begin();
      iT != m_children.end(); iT++){
    sz++;
    if(*iT) sz += (*iT)->totalSize();
  }
  return sz;
}


/**
void ora::MultiIndexDataTrie::print() const {
  std::cout << " @@@@ Printing trie ext size="<<m_children.size()<<std::endl;
  std::string tab("");
  recursivePrint( tab );
}

void ora::MultiIndexDataTrie::recursivePrint( const std::string& prev ) const{
  if( m_data ) std::cout << prev << " =>DATA"<<std::endl;
  for(size_t i=0;i<m_children.size();i++){
    std::stringstream ss;
    ss << prev<<"["<<i<<"]";
    m_children[i]->recursivePrint( ss.str() );
  }
}
**/
