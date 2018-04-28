#ifndef DETECTOR_DESCRIPTION_DDL_DOCUMENT_PROVIDER_H
#define DETECTOR_DESCRIPTION_DDL_DOCUMENT_PROVIDER_H

#include <vector>
#include <string>

/// DDLDocumentProvider provides a set of filenames.
/** @class DDLDocumentProvider
 * @author Michael Case
 *
 *  This abstract class defines the interface that is expected by the
 *  DDLParser to obtain its list of files for parsing.
 *
 */
class DDLDocumentProvider {

 public:

  /// Return a list of files as a vector of strings.
  virtual const std::vector < std::string >&  getFileList() const = 0;

  /// Print out the list of files.
  virtual void dumpFileList() const = 0;
};

#endif
