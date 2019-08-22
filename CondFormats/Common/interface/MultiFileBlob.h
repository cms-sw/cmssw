#ifndef CondFormats_MultiFileBlob_h
#define CondFormats_MultiFileBlob_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
#include <string>
#include <map>
#include <iosfwd>

class MultiFileBlob {
public:
  typedef std::pair<unsigned char const*, unsigned char const*> Range;

  ///
  MultiFileBlob();

  ///
  ~MultiFileBlob();

  ///
  void finalized(bool compress);

  /// read from real file give it name name
  // void read(const std::string& name, const std::string& fname);
  /// write name to real file
  // void write(const std::string& name, const std::string& fname) const;

  /// read from istream
  void read(const std::string& name, std::istream& is);
  /// write to ostream
  void write(const std::string& name, std::ostream& os) const;

  // return blob
  Range rawBlob(const std::string& name) const;

  bool isCompressed() const { return compressed; }

  unsigned long long fullSize() const { return isize; }

  unsigned long long size(const std::string& name) const;

private:
  // expand locally;
  void expand();

  // static unsigned int computeFileSize(const std::string & ifile);
  // static unsigned int computeStreamSize(std::istream & is);

  std::vector<unsigned char> blob;
  typedef std::map<std::string, unsigned long long> Positions;
  Positions positions;
  bool compressed;  // persistent status
  unsigned long long isize;
  bool expanded COND_TRANSIENT;  // transient status

  COND_SERIALIZABLE;
};

#endif
