#ifndef CondFormats_FileBlob_h
#define CondFormats_FileBlob_h

#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
#include <memory>
#endif
#include <vector>
#include <string>
#include <iostream>

class FileBlob{

 public:
  FileBlob() {};
  /// constructor from file to read
  FileBlob(const std::string & fname, bool zip);
  /// constructor from  stream to read
  FileBlob(std::istream & is, bool zip);

  ~FileBlob(){};

  /// read from real file
  void read(const std::string&);
  /// write to real file
  void write(const std::string&) const;
  
  /// read from istream
  void read(std::istream &);
  /// write to ostream
  void write(std::ostream &) const;

  bool isCompressed() const {return compressed;};

  int size() const {return isize;};
  /// i didn't want to do two copies ... hope this works.
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
  std::unique_ptr<std::vector<unsigned char> > getUncompressedBlob() const;
#endif
  void getUncompressedBlob( std::vector<unsigned char>& myblobcopy ) const;
                                 
 private:
  static unsigned int computeFileSize(const std::string &);
  static unsigned int computeStreamSize(std::istream &);

  std::vector<unsigned char> blob;
  bool compressed;
  unsigned int isize;
};

#endif
