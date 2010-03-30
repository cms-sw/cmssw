#ifndef INCLUDE_ORA_MULTIINDEXDATATRIE_H
#define INCLUDE_ORA_MULTIINDEXDATATRIE_H

#include <string>
#include <vector>
#include <memory>

namespace coral {
  class AttributeList;
}

namespace ora {

    // class describing an elementary part of data to be stored 
  class MultiIndexDataTrie {
    public:
    MultiIndexDataTrie();
    virtual ~MultiIndexDataTrie();

    void push( const std::vector<int>& indexes, const coral::AttributeList& data );
    coral::AttributeList& lookup( const std::vector<int>& indexes );

    void clear();
    size_t size() const;
    size_t branchSize( const std::vector<int>& indexes, size_t depth = 0) const;

    //void print() const;

    //private:
    //void recursivePrint( const std::string& prev ) const;
    
    private:

    std::vector<MultiIndexDataTrie*> m_children;
    std::auto_ptr<coral::AttributeList> m_data;
    
  };
  
}

#endif

