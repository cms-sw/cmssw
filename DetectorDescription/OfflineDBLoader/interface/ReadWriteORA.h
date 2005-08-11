#ifndef DetectorDescription_ReadWriteORA_H
#define DetectorDescription_ReadWriteORA_H

#include <string>

  /**
     This populates the DDD objects from the XML files.
     Use only one of the read methods at a time if you want to
     use this class because any object with the same name will
     be overwritten by the last source that you read.

     This class is meant only to be used for the copy application.

   **/

class ReadWriteORA {

 public:
  ReadWriteORA ( const std::string& dbConnectString
		 , const std::string& xmlConfiguration 
		 , const std::string& name
		 , const std::string& type = ""
		 , const std::string& userName = ""
		 , const std::string& password = "" );
  ~ReadWriteORA ();

  /// Read from XML and write using POOL Object Relational Access
  bool writeDB ( );


  /// Read from XML
  bool readFromXML ( );

  /// Read from the persistent objects for validation.
  bool readFromDB ( );

 private:
  std::string dbConnectString_;
  std::string xmlConfiguration_;
  std::string name_;
  std::string type_;
  std::string userName_;
  std::string password_;
};
#endif
