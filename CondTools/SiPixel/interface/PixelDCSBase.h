#ifndef CondTools_SiPixel_PixelDCSBase_h
#define CondTools_SiPixel_PixelDCSBase_h

/** \class PixelDCSBase
 *
 *  Base class for PixelPopConDCSSourceHandler to handle database stuff.
 *
 *  $Date: $
 *  $Revision: $
 *  \author Chung Khim Lae
 */

#include "CondCore/DBCommon/interface/DBSession.h"

namespace coral { class ICursor; }
namespace edm { class ParameterSet; }

class PixelDCSBase
{
  typedef std::vector<std::string> Strings;

  public:

  /// Set connection string and tables from cfg file, and init DB session.
  PixelDCSBase( const edm::ParameterSet& );

  virtual ~PixelDCSBase() {}

  protected:

  /// Implemented by PixelPopConDCSSourceHandler for a particular object type.
  virtual void fillObject( coral::ICursor& ) = 0;

  /// Called by PixelPopConDCSSourceHandler::getNewObjects() to get data from DB.
  void getData();

  private:

  /// Check if a table is a last value table.
  static inline bool isLVTable( const std::string& table );

  static const std::string theUser;  // user name for connectng to DB
  static const std::string theOwner; // owner of last value tables

  cond::DBSession m_dbSession;

  std::string m_connectStr; // connection string to DB

  Strings m_tables; // list of last value tables to output
};

bool PixelDCSBase::isLVTable(const std::string& table)
{
  return table.find("DCSLASTVALUE") != std::string::npos;
}

#endif
