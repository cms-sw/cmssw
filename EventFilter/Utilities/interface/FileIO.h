/*
 * FileIO.h
 *
 *  Created on: Sep 25, 2012
 *      Author: aspataru
 */

#ifndef FILEIO_H_
#define FILEIO_H_

#include <string>
#include <vector>

namespace jsoncollector {
class FileIO {
public:
  /**
   * Writes given string to specified file path
   */
  static void writeStringToFile(std::string const& filename, std::string& content);
  /**
   * Reads string from specified path, returns false if file does not exist
   */
  static bool readStringFromFile(std::string const& filename, std::string& content);
  /**
   * Checks if path points to an existing file
   */
  static bool fileExists(std::string const& path);
};
}
#endif /* FILEIO_H_ */

