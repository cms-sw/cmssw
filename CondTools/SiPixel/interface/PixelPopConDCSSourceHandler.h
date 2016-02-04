#ifndef CondTools_SiPixel_PixelPopConDCSSourceHandler_h
#define CondTools_SiPixel_PixelPopConDCSSourceHandler_h

/** \class PixelPopConDCSSourceHandler
 *
 *  Template class for the source handler of DCS data.
 *
 *  Specify the object type via the template parameter Type.
 *
 *  $Date: 2009/10/21 16:22:11 $
 *  $Revision: 1.4 $
 *  \author Chung Khim Lae
 */

#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeList.h"
#include "RelationalAccess/ICursor.h"

#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondFormats/SiPixelObjects/interface/PixelDCSObject.h"
#include "CondTools/SiPixel/interface/PixelDCSBase.h"

template <class Type>
class PixelPopConDCSSourceHandler:
  public PixelDCSBase,
  public popcon::PopConSourceHandler< PixelDCSObject<Type> >
{
  public:

  /// Init PixelDCSBase from cfg file.
  PixelPopConDCSSourceHandler( const edm::ParameterSet& );

  /// Get data from DB by calling PixelDCSBase::getData().
  virtual void getNewObjects();

  /// Name of this source handler.
  virtual std::string id() const;

  private:

  /// Specialise this template to assign values to a non-POD object type from a row in DB.
  static inline void setValue( Type& value, const coral::AttributeList& );

  /// Fill object from all rows in DB.
  virtual void fillObject( coral::ICursor& );
};

template <class Type>
PixelPopConDCSSourceHandler<Type>::PixelPopConDCSSourceHandler(const edm::ParameterSet& cfg):
  PixelDCSBase(cfg)
{
}

template <class Type>
void PixelPopConDCSSourceHandler<Type>::setValue(Type& value, const coral::AttributeList& row)
{
  const size_t nTable = row.size() - 1; // row includes name

  if (1 != nTable)
  {
    throw cms::Exception("PixelPopConDCSSourceHandler")
        << "Found " << nTable << " last value tables instead of 1. "
        << "Check your cfg.\n";
  }

  value = row[0].data<float>();
}

template <>
void PixelPopConDCSSourceHandler<CaenChannel>::setValue(CaenChannel& value, const coral::AttributeList& row)
{
  const size_t nTable = row.size() - 1; // row includes name

  if (3 != nTable)
  {
    throw cms::Exception("PixelPopConDCSSourceHandler<CaenChannel>")
        << "CaenChannel has 3 values (isOn, iMon, vMon) "
        << "but your cfg has " << nTable << " last value tables.\n";
  }

  value.isOn = row[0].data<float>();
  value.iMon = row[1].data<float>();
  value.vMon = row[2].data<float>();
}

template <class Type>
void PixelPopConDCSSourceHandler<Type>::fillObject(coral::ICursor& cursor)
{
  PixelDCSObject<Type>* data = new PixelDCSObject<Type>;

  while ( cursor.next() )
  {
    const coral::AttributeList& row = cursor.currentRow();

    typename PixelDCSObject<Type>::Item datum;

    datum.name = row["name"].data<std::string>();
    setValue(datum.value, row);
    data->items.push_back(datum);
  }

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
