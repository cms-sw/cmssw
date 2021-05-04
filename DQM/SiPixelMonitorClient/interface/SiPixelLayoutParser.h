#ifndef SiPixelLayoutParser_H
#define SiPixelLayoutParser_H

/** \class SiPixelLayoutParser
 * *
 *  Class that handles the SiPixel Quality Tests
 *
 *  \author Suchandra Dutta
 */

#include <fstream>
#include <map>
#include <string>
#include <vector>

class SiPixelLayoutParser {
public:
  // Constructor
  SiPixelLayoutParser();

  // Get list of Layouts for ME groups
  bool getAllLayouts(std::map<std::string, std::vector<std::string>> &me_names);

private:
};

#endif
