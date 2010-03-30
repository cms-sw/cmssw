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

void ora::MultiIndexDataTrie::push( const std::vector<int>& indexes,
                                    const coral::AttributeList& data ){
  MultiIndexDataTrie* trie = this;
  for( size_t i=0;i<indexes.size();i++){
    size_t ts = trie->m_children.size();
    MultiIndexDataTrie* nt = 0;
    if( ts == 0 || indexes[i] > (int)(ts-1) ){
      for(size_t j=0;j<indexes[i]-ts+1;j++){
        nt = new MultiIndexDataTrie;
        trie->m_children.push_back(nt);
      }
    } else {
      nt = trie->m_children[indexes[i]];
    }
    trie = nt;
  }
  trie->m_data.reset( new coral::AttributeList( data ) );
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
  }
  if(!trie->m_data.get()){
    throwException( "No Data for the specified index combination.",
                    "MultiIndexDataTrie::lookup" );
  }
  return *trie->m_data;
}

size_t ora::MultiIndexDataTrie::size() const {
  return m_children.size();
}

void ora::MultiIndexDataTrie::clear(){
  for(std::vector<MultiIndexDataTrie*>::iterator iT = m_children.begin();
      iT != m_children.end(); iT++){
    delete *iT;
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
  }
  return trie->m_children.size();  
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
