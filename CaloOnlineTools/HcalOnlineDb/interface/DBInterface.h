//
// Gena Kukartsev (Brown), Feb. 1, 2008
//
// inspired by Fedor Ratnikov's HcalDbOnline class
//
#ifndef DBInterface_h
#define DBInterface_h

#include <memory>

/**

   \class DBInterface
   \brief Gather data from DB
   \author Gena Kukartsev

*/

namespace oracle {
  namespace occi {
    class Environment;
    class Connection;
    class Statement;
  }
}

class DBInterface{
 public:

  DBInterface (const std::string& fDb, bool fVerbose = false);
  ~DBInterface ();

 protected:
  oracle::occi::Environment* mEnvironment;
  oracle::occi::Connection* mConnect;
  oracle::occi::Statement* mStatement;
  template <class T> bool getObjectGeneric (T* fObject, const std::string& fTag);
  bool mVerbose;
};
#endif
