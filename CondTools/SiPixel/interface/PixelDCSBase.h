#ifndef CondTools_SiPixel_PixelDCSBase_h
#define CondTools_SiPixel_PixelDCSBase_h

#include "CoralBase/AttributeList.h"

#include "CondCore/DBCommon/interface/DBSession.h"

namespace coral { class ICursor; }
namespace edm { class ParameterSet; }

class PixelDCSBase
{
  public:

  PixelDCSBase( const edm::ParameterSet& );

  virtual ~PixelDCSBase() {}

  protected:

  virtual coral::AttributeList outputDefn() const = 0;

  virtual void fillObject( coral::ICursor& ) = 0;

  void getData();

  private:

  static inline bool isLVTable( const std::string& table );

  static const std::string theUser;
  static const std::string theOwner;

  cond::DBSession m_dbSession;

  std::string m_connectStr; // connection string to DB
  std::string m_table;      // name of table or view
  std::string m_column;     // column to output
};

bool PixelDCSBase::isLVTable(const std::string& table)
{
  return table.find("DCSLASTVALUE") != std::string::npos;
}

#endif
