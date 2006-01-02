#ifndef CDFRunInfo_hh_included
#define CDFRunInfo_hh_included 1

#include <map>
#include <string>
#include <TFile.h>

/** \brief Per-run or per-file information

    This class allows access to per-run or per-file information as opposed
    to per-event information such as CDFEventInfo.  The information is
    stored in the form of string -> string pairs.  There are utility
    methods for decoding string values as integers and doubles.
*/
class CDFRunInfo {
public: 
  CDFRunInfo();
  CDFRunInfo(TFile* fromFile);

  /// Get some run info by name
  const char* get(const char* key) const;
  /// Get a run info item by name and convert it to an integer
  int getInt(const char* key) const;
  /// Get a run info item by name and convert it to a double
  double getDouble(const char* key) const;
  /// get the number of items
  inline int getSize() const { return m_mapData.size(); }
  /// fill the given array with key name pointers
  int getKeys(const char** buffer, int nbufs);
  /// test for thr presence of given key
  bool hasKey(const char* key) const;

  /// add some information to the run info
  void setInfo(const char* key, const char* value);
  /// print all information to the terminal
  void print() const;

  void store(TFile* toFile);
private:
  bool load(TFile* fromFile);
  static const char* RootVariableName;
  std::map<std::string,std::string> m_mapData;
};

#endif // CDFRunInfo_hh_included
