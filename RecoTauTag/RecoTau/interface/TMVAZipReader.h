/*
 * =====================================================================================
 *
 *       Filename:  TMVAZipReader.h
 *
 *    Description:  A stupid function that initializes a TMVA Reader
 *                  using a gzipped or non-zipped xml file.
 *
 *                  The file will be unzipped if it ends in .gz or .gzip
 *                  It will be passed as normal if it ends in .xml
 *
 *                  This file is header-only.
 *
 *          Usage:  loadTMVAWeights(&myReader, "BDT", "path_to_file.xml.gz");
 *
 *
 *         Author:  Evan Friis, evan.friis@cern.ch
 *        Company:  UW Madison
 *
 * =====================================================================================
 */

#ifndef TMVAZIPREADER_7RXIGO70
#define TMVAZIPREADER_7RXIGO70

#include "TMVA/Reader.h"
#include <string>

namespace reco {
  namespace details {
  
    bool hasEnding(std::string const &fullString, std::string const &ending);
    char* readGzipFile(const std::string& weightFile);

    void loadTMVAWeights(TMVA::Reader* reader, const std::string& method,
      const std::string& weightFile, bool verbose=false);


}}
#endif /* end of include guard: TMVAZIPREADER_7RXIGO70 */
