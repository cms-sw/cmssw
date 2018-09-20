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

#ifndef CommonTools_MVAUtils_TMVAZipReader_h
#define CommonTools_MVAUtils_TMVAZipReader_h

#include "TMVA/IMethod.h"
#include "TMVA/Reader.h"
#include <string>

namespace reco::details {

    bool hasEnding(std::string const& fullString, std::string const& ending);
    char* readGzipFile(const std::string& weightFile);

    TMVA::IMethod* loadTMVAWeights(
        TMVA::Reader* reader, const std::string& method, const std::string& weightFile, bool verbose = false);

}

#endif
