#ifndef CondFormatsCommon_DropBoxMetadata_H
#define CondFormatsCommon_DropBoxMetadata_H

/** \class DropBoxMetadata
 *  No description available.
 *
 *  $Date: 2011/03/02 17:09:28 $
 *  $Revision: 1.2 $
 *  \author G. Cerminara - CERN
 */


#include <map>
#include <string>

class DropBoxMetadata {
public:
  /// Constructor
  DropBoxMetadata();

  /// Destructor
  virtual ~DropBoxMetadata();


  class Parameters {
  public:
    
    void addParameter(const std::string& key, const std::string& vale); 
    
    std::string getParameter(const std::string& key) const;

    const std::map<std::string, std::string> & getParameterMap() const;

  private:

    std::map<std::string, std::string> theParameters;
  };

  // Operations
  void addRecordParameters(const std::string& record, const Parameters& params);
  
  const Parameters& getRecordParameters(const std::string& record) const;

  bool knowsRecord(const std::string& record) const;

protected:

private:
  std::map<std::string, DropBoxMetadata::Parameters> recordSet;

};
#endif

