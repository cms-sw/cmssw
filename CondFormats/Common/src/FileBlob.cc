#include "CondFormats/Common/interface/FileBlob.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <fstream>
#include <string>
#include <zlib.h>

FileBlob::FileBlob(const std::string & fname, bool zip):isize(0){
  compressed = zip;
  /*  
  std::cout << "isize = " << isize 
	    << "  zip = " << (zip? "true" : "false")
	    << std::endl;
  */
  if (isize==0) isize= computeFileSize(fname);
  // std::cout << "isize = " << isize << std::endl;
  blob.reserve(isize);
  read(fname);
}
FileBlob::FileBlob(std::istream& is, bool zip):isize(0) {
  compressed = zip;
  if (isize==0) isize= computeStreamSize(is);
  blob.reserve(isize);
  read(is);
}

void FileBlob::read(std::istream & is) {
  if(compressed){
    std::vector<unsigned char> in;
    in.reserve(isize);
    char c;
    while (is.get(c))
      in.push_back((unsigned char)c);
    /*
    for(int i=0;i<in.size();i++){
      std::cout<<in[i];
    }
    std::cout<<std::endl;
    */
    blob.resize(isize);
    uLongf destLen = compressBound(in.size());
    int zerr =  compress2(&*blob.begin(), &destLen,
                          &*in.begin(), in.size(),
                          9);
    if (zerr!=0) edm::LogError("FileBlob")<< "Compression error " << zerr;
    blob.resize(destLen);  
  }else{
    //std::cout << "reading uncompressed" << std::endl;
    char c;
    while (is.get(c))
      blob.push_back( (unsigned char)c);
    blob.resize(blob.size());
    isize=blob.size();
  }
}

void FileBlob::write(std::ostream & os) const {
  if(compressed){
    std::vector<unsigned char> out(isize);
    uLongf destLen = out.size();
    int zerr =  uncompress(&*out.begin(),  &destLen,
                           &*blob.begin(), blob.size());
    if (zerr!=0 || out.size()!=destLen) 
      edm::LogError("FileBlob")<< "uncompressing error " << zerr
                                   << " original size was " << isize
                                   << " new size is " << destLen;
    os.write((const char *)(&*out.begin()),out.size());
  }else{
    os.write((char *)&*blob.begin(),blob.size());
  }
}

std::unique_ptr<std::vector<unsigned char> > FileBlob::getUncompressedBlob() const {
  std::unique_ptr<std::vector<unsigned char> >  newblob;
  if(compressed)
  {
    newblob.reset(new std::vector<unsigned char>(isize));
    uLongf destLen = newblob->size();
    //    std::cout<<"Store isize = "<<isize<<"; newblob->size() = "<<newblob->size()<<"; destLen = "<<destLen<<std::endl;
    int zerr =  uncompress(&*(newblob->begin()),  &destLen,
                           &*blob.begin(), blob.size());
    if (zerr!=0 || newblob->size()!=destLen) 
      edm::LogError("FileBlob")<< "uncompressing error " << zerr
                                   << " original size was " << isize
                                   << " new size is " << destLen;
  }else{
    newblob.reset(new std::vector<unsigned char>(blob));
  }
  return newblob;
 }

void FileBlob::getUncompressedBlob( std::vector<unsigned char>& myblobcopy ) const {
  if(compressed)
  {
    myblobcopy.reserve(isize);
    uLongf destLen = isize;
    int zerr =  uncompress(&*myblobcopy.begin(),  &destLen,
			   &*blob.begin(), blob.size());
    if (zerr!=0 || myblobcopy.size()!=destLen) 
      edm::LogError("FileBlob")<< "uncompressing error " << zerr
                                   << " original size was " << isize
                                   << " new size is " << destLen;
  }else{
    myblobcopy = blob;
  }
  
}

void FileBlob::read(const std::string & fname) {
     std::ifstream ifile(fname.c_str());
     if (!ifile) { edm::LogError("FileBlob")<< "file " << fname << " does not exist...";}
     else read(ifile);
     ifile.close();
}
 
void FileBlob::write(const std::string & fname) const {
  std::ofstream ofile(fname.c_str());
  write(ofile);
  ofile.close();
}

unsigned int FileBlob::computeFileSize(const std::string & fname) {
  unsigned int is=0;
  std::ifstream ifile(fname.c_str());
  if (!ifile) { edm::LogError("FileBlob")<< "file " << fname << " does not exist...";}
  else is = computeStreamSize(ifile);
  ifile.close();
  return is;
}

unsigned int FileBlob::computeStreamSize(std::istream & is) {
  unsigned int rs=0;
  char c;
  while (is.get(c)) rs++;
  is.clear();
  is.seekg(0);
  return rs;
}
