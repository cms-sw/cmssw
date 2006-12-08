#ifndef DetectorDescription_ReadWriteORA_H
#define DetectorDescription_ReadWriteORA_H

#include "DetectorDescription/Core/interface/DDCompactView.h"

#include <string>

  /**
     This populates the DDD objects from the XML files.

     This class is meant only to be used for the copy application.

   **/

class ReadWriteORA {

 public:
  ReadWriteORA ( const std::string& dbConnectString
		 , const std::string& metaName
		 , const std::string& userName
		 , const std::string& password 
		 , int rotNumSeed = 0 );
  ~ReadWriteORA ();

  /// write it out...
  bool writeDB ( const DDCompactView& cpv );

 private:
  std::string dbConnectString_;
  std::string metaName_;
  std::string userName_;
  std::string password_;
  int rotNumSeed_;
};
#endif
