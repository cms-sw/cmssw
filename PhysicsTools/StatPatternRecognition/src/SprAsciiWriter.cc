//$Id: SprAsciiWriter.cc,v 1.2 2007/09/21 22:32:09 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAsciiWriter.hh"

#include <stdlib.h>
#include <string>
#include <iostream>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>

using namespace std;


bool SprAsciiWriter::init(const char* filename)
{
  // init
  string fname = filename;
  string cmd;

  // check if file exists, delete and issue a warning
  struct stat buf;
  if( stat(fname.c_str(),&buf) == 0 ) {
    cerr << "Warning: file " << fname.c_str() << " will be deleted." << endl;
    cmd = "rm -f ";
    cmd += fname.c_str();
    if( system(cmd.c_str()) != 0 ) {
      cerr << "Attempt to delete file " << fname.c_str() 
	   << " terminated with error " << errno << endl;
      return false;
    }
  }

  // open output stream
  outfile_.open(fname.c_str());
  if( !outfile_ ) {
    cerr << "Cannot open file " << fname.c_str() << endl;
    return false;
  }

  // exit
  return true;
}


bool SprAsciiWriter::write(int cls, unsigned index, double weight,
			   const std::vector<double>& v, 
			   const std::vector<double>& f)
{
  // sanity check
  unsigned int vdim = v.size();
  unsigned int fdim = f.size();
  if( (vdim+fdim) != axes_.size() ) {
    cerr << "Dimensionality of input vector unequal to dimensionality " 
	 << "of tuple: " << vdim << " " << fdim
	 << " " << axes_.size() << endl;
    return false;
  }

  // write data variables and classifier names
  if( firstCall_ ) {
    firstCall_ = false;
    {
      char s [200];
      sprintf(s," %10s ","index");
      outfile_ << s;
    }
    {
      char s [200];
      sprintf(s," %10s ","i");
      outfile_ << s;
    }
    {
      char s [200];
      sprintf(s," %10s ","w");
      outfile_ << s;
    }
    for( unsigned int i=0;i<axes_.size();i++ ) {
      char s [200];
      sprintf(s," %20s ",axes_[i].c_str());
      outfile_ << s;
    }
    outfile_ << endl;
  }

  // write output
  {
    char s [200];
    sprintf(s," %10i ",index);
    outfile_ << s;
  }
  {
    char s [200];
    sprintf(s," %10i ",cls);
    outfile_ << s;
  }
  {
    char s [200];
    sprintf(s," %10g ",weight);
    outfile_ << s;
  }
  for( unsigned int i=0;i<vdim;i++ ) {
    char s [200];
    sprintf(s,"           %10g ",v[i]);
    outfile_ << s;
  }
  for( unsigned int i=0;i<fdim;i++ ) {
    char s [200];
    sprintf(s,"           %10g ",f[i]);
    outfile_ << s;
  }
  outfile_ << endl;

  // exit
  return true;
}


bool SprAsciiWriter::close()
{
  outfile_.close();
  if( !outfile_ ) {
    cerr << "Unable to close output file." << endl;
    return false;
  }
  return true;
}


