#ifndef CondTools_SiPixel_PixelPopConDCSSourceHandler_h
#define CondTools_SiPixel_PixelPopConDCSSourceHandler_h

#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondFormats/SiPixelObjects/interface/PixelDCSObject.h"
#include "CondTools/SiPixel/interface/PixelDCSBase.h"

template <class Type>
class PixelPopConDCSSourceHandler:
  public PixelDCSBase,
  public popcon::PopConSourceHandler< PixelDCSObject<Type> >
{
  public:

  PixelPopConDCSSourceHandler( const edm::ParameterSet& );

  virtual void getNewObjects();

  virtual std::string id() const;

  private:

  virtual coral::AttributeList outputDefn() const;

  virtual void fillObject( coral::ICursor& );
};

#include "CoralBase/Attribute.h"
#include "RelationalAccess/ICursor.h"

template <class Type>
PixelPopConDCSSourceHandler<Type>::PixelPopConDCSSourceHandler(const edm::ParameterSet& cfg):
  PixelDCSBase(cfg)
{
}

template <class Type>
coral::AttributeList PixelPopConDCSSourceHandler<Type>::outputDefn() const
{
  coral::AttributeList output;

  output.extend("value", typeid(Type) );

  return output;
}

template <class Type>
void PixelPopConDCSSourceHandler<Type>::fillObject(coral::ICursor& cursor)
{
  PixelDCSObject<Type>* data = new PixelDCSObject<Type>;

  while ( cursor.next() )
  {
    const coral::AttributeList& row = cursor.currentRow();
row.toOutputStream(std::cout) << '\n';
    typename PixelDCSObject<Type>::Item datum;

    datum.name = row["name"].data<std::string>();
    datum.value = row[0].data<Type>();

    data->items.push_back(datum);
  }
  std::cout << "Number of rows = " << data->items.size() << std::endl;

  this->m_to_transfer.push_back( std::make_pair(data, 1) );
}

template <class Type>
void PixelPopConDCSSourceHandler<Type>::getNewObjects()
{
  getData();
}

template <class Type>
std::string PixelPopConDCSSourceHandler<Type>::id() const
{
  std::string name = "PixelPopCon";

  name += typeid(Type).name();
  name += "SourceHandler";

  return name;
}

#endif
