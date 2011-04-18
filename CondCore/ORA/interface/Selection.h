#ifndef INCLUDE_ORA_SELECTION_H
#define INCLUDE_ORA_SELECTION_H

//
#include <string>
#include <vector>
#include <typeinfo>
#include <memory>

namespace coral {
  class AttributeList;
}

namespace ora    {

  typedef enum { EQ, NE, GT, GE, LT, LE } SelectionItemType;

  class Selection{
    public:

    static const int endOfRange = -1;
    static std::string indexVariable();
    static std::vector<std::string>& selectionTypes();
    static std::string variableNameFromUniqueString(const std::string& uniqueString);

    public:
    
    Selection();

    virtual ~Selection();

    Selection( const Selection& rhs );

    Selection& operator=( const Selection& rhs );

    void addIndexItem( int startIndex, int endIndex=endOfRange );
    
    template <typename Prim> void addDataItem(const std::string& dataMemberName, SelectionItemType stype, Prim selectionData);

    void addUntypedDataItem( const std::string& dataMemberName, SelectionItemType stype, const std::type_info& primitiveType, void* data );

    bool isEmpty() const;
    
    const std::vector<std::pair<std::string,std::string> >& items() const;
    
    const coral::AttributeList& data() const;

    private:

    std::string uniqueVariableName(const std::string& varName) const;
    
    private:

    std::vector<std::pair<std::string,std::string> > m_items;
    std::auto_ptr<coral::AttributeList> m_data;
  };
  
}

template <typename Prim> void ora::Selection::addDataItem(const std::string& dataMemberName, ora::SelectionItemType stype, Prim selectionData){
  addUntypedDataItem( dataMemberName, stype, typeid( Prim ), &selectionData );
}

#endif  // 
