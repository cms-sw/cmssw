#ifndef DDL_DocumentProvider_H
#define DDL_DocumentProvider_H

#include <vector>
#include <string>

/// DDLDocumentProvider provides a set of URLs and filenames.
/** @class DDLDocumentProvider
 * @author Michael Case
 *
 *  DDLDocumentProvider.h  -  description
 *  -------------------
 *  begin: Mon Feb 24 2003
 *  email: case@ucdhep.ucdavis.edu
 *
 *  This abstract class defines the interface that is expected by the
 *  DDLParser to obtain its list of files for parsing.
 *
 */
class DDLDocumentProvider {
public:
  virtual ~DDLDocumentProvider() {}

  /// Return a list of files as a vector of strings.
  virtual const std::vector<std::string>& getFileList(void) const = 0;

  /// Return a list of urls as a vector of strings.
  virtual const std::vector<std::string>& getURLList(void) const = 0;

  /// Return a flag whether to do xml validation or not.
  virtual bool doValidation() const = 0;

  /// Return the Schema Location.
  virtual std::string getSchemaLocation() const = 0;

  /// Print out the list of files.
  virtual void dumpFileList(void) const = 0;

  /// (does not belong here) Read in the configuration file.
  virtual int readConfig(const std::string& filename) = 0;

protected:
private:
};

#endif
