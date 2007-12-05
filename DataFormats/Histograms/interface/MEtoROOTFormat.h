#ifndef MEtoROOTFormat_h
#define MEtoROOTFormat_h

/** \class MEtoROOT
 *  
 *  DataFormat class to hold the information from a ME tranformed into
 *  ROOT objects as appropriate
 *
 *  $Date: 2007/12/02 03:46:22 $
 *  $Revision: 1.2 $
 *  \author M. Strang SUNY-Buffalo
 */

#include <TObject.h>
#include <TH1F.h>

#include <string>
#include <vector>
#include <memory>
#include <map>

class MEtoROOT
{

 public:
  
  MEtoROOT() {}
  virtual ~MEtoROOT() {}

  struct QValue
  {
    int		code;
    std::string	message;
  };

  typedef std::vector<uint32_t> TagList;
  typedef std::map<std::string, struct QValue> QReports;

  struct MEROOTObject
  {
    uint64_t	version;
    std::string	name;
    TagList 	tags;
    TH1F	object;
    TH1F	refobj;
    QReports	qreports;
    uint32_t	flags;
  };

  typedef std::vector<MEROOTObject> MERootObjectVector;

  void putMERootObject(std::vector<uint64_t> version,
		       std::vector<std::string> name,
		       std::vector<TagList> tags,
		       std::vector<TH1F> object,
		       std::vector<TH1F> refobj,
		       std::vector<QReports> qreports,
		       std::vector<uint32_t> flags);

  MERootObjectVector getMERootObject() const {return MERootObject;}

 private:

  MERootObjectVector MERootObject;

}; // end class declaration

#endif
