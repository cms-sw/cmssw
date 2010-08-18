#ifndef INCLUDE_ORA_MULTIINDEXDATATRIE_H
#define INCLUDE_ORA_MULTIINDEXDATATRIE_H

#include <string>
#include <vector>
//#include <memory>
#include "CondCore/ORA/interface/Record.h"


namespace ora {


    // class describing an elementary part of data to be stored 
  class MultiIndexDataTrie {
    public:
    MultiIndexDataTrie();
    virtual ~MultiIndexDataTrie();

    size_t push( const std::vector<int>& indexes, Record & data );
    //const Record& lookup( const std::vector<int>& indexes ) const;
    boost::shared_ptr<const Record> lookupAndClear( const std::vector<int>& indexes, Record & rec );

    void clear();
    size_t size() const;
    size_t branchSize( const std::vector<int>& indexes, size_t depth = 0) const;

    size_t totalSize() const;

    //void print() const;

    //private:
    //void recursivePrint( const std::string& prev ) const;
    
    private:

    std::vector<MultiIndexDataTrie*> m_children;
    Record m_data;
    
  };
  
}

#endif

