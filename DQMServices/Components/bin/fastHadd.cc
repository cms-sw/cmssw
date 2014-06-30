/** Copyright (c) 2013 "Marco Rovere"

This code is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see
<http://www.gnu.org/licenses/>.

This code is simple: its sole purpose is to either dump or add
ProtocolBuffer-gzipped files that are meant to replace ordinary ROOT
files containing only hierarchies of histograms, arranged in
arbitrarily complex levels of directories. The merging logic is such
that plots present in all files are added, while plots present in some
of the files are anyway tracked and added, if similar ones are found
in other files.

The logic of the merging algorithm is trivial and fully rely on the
ordered nature of the ProtocolBuffer files read in input. An internal
set container of MicroME is used to host the final results. The
relational ordering of the set must be guaranteed to match the one
used to order the ProtocolBuffer files for optimal performance and
correctness.

A dependency on protocolbuffer is needed and should be alrady included
out of the box into any recent CMSSW release.

In case the protoclBuffer package is not avaialble, you need to
install it as an external toolfile. Therefore, in order to be able to
compile and run the code, you need to locally install protocol buffer
2.4.1 and add it as a scram tool to your preferred CMSSW development
area.

The toolfile I used is:

<tool name="protocolbuf" version="2.4.1">
  <client>
    <environment name="PROTOCOLBUF_CLIENT_BASE" default="/afs/cern.ch/work/r/rovere/protocolbuf"/>
    <environment name="LIBDIR"  value="$PROTOCOLBUF_CLIENT_BASE/lib"/>
    <environment name="INCLUDE" value="$PROTOCOLBUF_CLIENT_BASE/include"/>
    <environment name="PATH"    value="$PROTOCOLBUF_CLIENT_BASE/bin"/>
    <lib name="protobuf"/>
    <use name="zlib"/>
  </client>
  <runtime name="PATH"    value="$PROTOCOLBUF_CLIENT_BASE/bin"/>
</tool>

To register it into your development area you can simply do:

scram setup protocolbuf.xml

To verify the correctness of the information, do:

scram tool info protocolbuf. You should see an output similar to the
following:

Tool info as configured in location /afs/cern.ch/work/r/rovere/fastHistoMergingPB/CMSSW_7_0_X_2013-07-08-0200
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Name : protocolbuf
Version : 2.4.1
++++++++++++++++++++
SCRAM_PROJECT=no
PROTOCOLBUF_CLIENT_BASE=/afs/cern.ch/work/r/rovere/protocolbuf
LIB=protobuf
LIBDIR=/afs/cern.ch/work/r/rovere/protocolbuf/lib
INCLUDE=/afs/cern.ch/work/r/rovere/protocolbuf/include
USE=zlib
PATH=/afs/cern.ch/work/r/rovere/protocolbuf/bin

*/

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <vector>
#include <set>
#include <string>
#include <iostream>
#include <memory>
#include "DQMServices/Core/src/ROOTFilePB.pb.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/gzip_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <TROOT.h>
#include <TFile.h>
#include <TBufferFile.h>
#include <TObject.h>
#include <TObjString.h>
#include <TH1.h>
#include <TKey.h>

#define DEBUG(x, msg) if (debug >= x) std::cout << "DEBUG: " << msg << std::flush

int debug = 0;

static bool lessThanMME(const std::string &lhs_dirname,
                        const std::string &lhs_objname,
                        const std::string &rhs_dirname,
                        const std::string &rhs_objname) {
  int diff = lhs_dirname.compare(rhs_dirname);
  return (diff < 0 ? true
          : diff == 0 ? lhs_objname < rhs_objname : false);
};

struct MicroME {
  MicroME(const std::string * full,
          const std::string * dir,
          const std::string * obj,
          uint32_t flags = 0)
      :fullname(full), dirname(dir), objname(obj), flags(flags) {}
  const std::string * fullname;
  const std::string * dirname;
  const std::string * objname;
  mutable TObject * obj;

  uint32_t flags;

  bool operator<(const MicroME &rhs) const {
    return lessThanMME(*this->dirname,
                       *this->objname,
                       *rhs.dirname,
                       *rhs.objname);
  };

  void add(TObject *obj_to_add) const {
      DEBUG(1, "Merging: " << obj->GetName() <<
        " << " << obj_to_add->GetName() << std::endl);

      if (dynamic_cast<TH1 *>(obj) && dynamic_cast<TH1 *>(obj_to_add)) {
        dynamic_cast<TH1 *>(obj)->Add(dynamic_cast<TH1 *>(obj_to_add));
      } else if (dynamic_cast<TObjString *>(obj) && dynamic_cast<TObjString *>(obj_to_add)) {


      } else {
        DEBUG(1, "Cannot merge (different types): " << obj->GetName() <<
          " << " << obj_to_add->GetName() << std::endl);
      }
  };
};

enum TaskType {
  TASK_ADD,
  TASK_DUMP,
  TASK_CONVERT,
  TASK_ENCODE
};

enum ErrType {
  ERR_BADCFG=1,
  ERR_NOFILE
};

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::GzipInputStream;
using google::protobuf::io::GzipOutputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ArrayInputStream;

/** Extract the next serialised ROOT object from @a buf. Returns null
if there are no more objects in the buffer, or a null pointer was
serialised at this location. */
inline TObject * extractNextObject(TBufferFile &buf) {
  if (buf.Length() == buf.BufferSize())
    return 0;
  buf.InitMap();
  return reinterpret_cast<TObject *>(buf.ReadObjectAny(0));
}

static void get_info(const dqmstorepb::ROOTFilePB::Histo &h,
                     std::string &dirname,
                     std::string &objname,
                     TObject ** obj) {

  size_t slash = h.full_pathname().rfind('/');
  size_t dirpos = (slash == std::string::npos ? 0 : slash);
  size_t namepos = (slash == std::string::npos ? 0 : slash+1);
  dirname.assign(h.full_pathname(), 0, dirpos);
  objname.assign(h.full_pathname(), namepos, std::string::npos);
  TBufferFile buf(TBufferFile::kRead, h.size(),
                  (void*)h.streamed_histo().data(),
                  kFALSE);
  buf.Reset();
  *obj = extractNextObject(buf);
  if (!*obj) {
    std::cerr << "Error reading element: " << h.full_pathname() << std::endl;
  }
}

void writeMessage(const dqmstorepb::ROOTFilePB &dqmstore_output_msg,
                  const std::string &output_filename) {

  DEBUG(1, "Writing file" << std::endl);

  int out_fd = ::open(output_filename.c_str(),
                      O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
  FileOutputStream out_stream(out_fd);
  GzipOutputStream::Options options;
  options.format = GzipOutputStream::GZIP;
  options.compression_level = 2;
  GzipOutputStream gzip_stream(&out_stream,
                               options);
  dqmstore_output_msg.SerializeToZeroCopyStream(&gzip_stream);

  google::protobuf::ShutdownProtobufLibrary();
}


void fillMessage(dqmstorepb::ROOTFilePB &dqmstore_output_msg,
                 const std::set<MicroME> & micromes) {
  std::set<MicroME>::iterator mi = micromes.begin();
  std::set<MicroME>::iterator me = micromes.end();

  DEBUG(1, "Streaming ROOT objects" << std::endl);
  for (; mi != me; ++mi) {
    dqmstorepb::ROOTFilePB::Histo* h = dqmstore_output_msg.add_histo();
    DEBUG(2, "Streaming ROOT object " << *(mi->fullname) << "\n");
    h->set_full_pathname(*(mi->fullname));
    TBufferFile buffer(TBufferFile::kWrite);
    buffer.WriteObject(mi->obj);
    h->set_size(buffer.Length());
    h->set_flags(mi->flags);
    h->set_streamed_histo((const void*)buffer.Buffer(),
                          buffer.Length());
    delete mi->obj;
  }
}


void processDirectory(TFile *file,
                      const std::string& curdir,
                      std::set<std::string> &dirs,
                      std::set<std::string> &objs,
                      std::set<std::string> &fullnames,
                      std::set<MicroME>& micromes) {
  DEBUG(1, "Processing directory " << curdir << "\n");
  file->cd(curdir.c_str());
  TKey *key;
  TIter next (gDirectory->GetListOfKeys());
  while ((key = (TKey *) next())) {
    TObject * obj = key->ReadObj();
    if (dynamic_cast<TDirectory *>(obj)) {
      std::string subdir;
      subdir.reserve(curdir.size() + strlen(obj->GetName()) + 2);
      subdir += curdir;
      if (! curdir.empty())
        subdir += '/';
      subdir += obj->GetName();
      processDirectory(file, subdir, dirs, objs, fullnames, micromes);
    } else if ((dynamic_cast<TH1 *>(obj)) && (dynamic_cast<TObjString *>(obj))) {
      if (dynamic_cast<TH1 *>(obj)) {
        dynamic_cast<TH1 *>(obj)->SetDirectory(0);
      }

      DEBUG(2, curdir << "/" << obj->GetName() << "\n");
      MicroME mme(&*(fullnames.insert(curdir
                                      + '/'
                                      + std::string(obj->GetName())).first),
                     &*(dirs.insert(curdir).first),
                     &*(objs.insert(obj->GetName()).first));
      if (obj) {
        mme.obj = obj;
        micromes.insert(mme);
      }
    }
  }
}


int encodeFile(const std::string &output_filename,
               const std::vector<std::string> &filenames) {
  assert(filenames.size() == 1);
  TFile input(filenames[0].c_str());
  DEBUG(0, "Encoding file " << filenames[0] << std::endl);
  std::set<std::string> dirs;
  std::set<std::string> objs;
  std::set<std::string> fullnames;
  std::set<MicroME> micromes;
  dqmstorepb::ROOTFilePB dqmstore_message;

  processDirectory(&input, "", dirs, objs, fullnames, micromes);
  fillMessage(dqmstore_message, micromes);
  writeMessage(dqmstore_message, output_filename);

  return 0;
}

int convertFile(const std::string &output_filename,
                const std::vector<std::string> &filenames) {
  assert(filenames.size() == 1);
  TFile output(output_filename.c_str(), "RECREATE");
  DEBUG(0, "Converting file " << filenames[0] << std::endl);
  dqmstorepb::ROOTFilePB dqmstore_message;

  int filedescriptor = open(filenames[0].c_str(), O_RDONLY);
  FileInputStream fin(filedescriptor);
  GzipInputStream input(&fin);
  CodedInputStream input_coded(&input);
  input_coded.SetTotalBytesLimit(1024*1024*1024, -1);
  if (!dqmstore_message.ParseFromCodedStream(&input_coded)) {
    std::cout << "Fatal Error opening file "
              << filenames[0] << std::endl;
    return ERR_NOFILE;
  }

  for (int i = 0; i < dqmstore_message.histo_size(); i++) {
    const dqmstorepb::ROOTFilePB::Histo& h = dqmstore_message.histo(i);
    DEBUG(1, h.full_pathname() << std::endl);
    DEBUG(1, h.size() << std::endl);
    TBufferFile buf(TBufferFile::kRead, h.size(),
                    (void*)h.streamed_histo().data(),
                    kFALSE);
    buf.Reset();
    TObject *obj = static_cast<TH1*>(extractNextObject(buf));
    std::string path,objname;
    get_info(h, path, objname, &obj);
    gDirectory->cd("/");
    // Find the first path component.
    size_t start = 0;
    size_t end = path.find('/', start);
    if (end == std::string::npos)
    end = path.size();
    while (true)
    {
      std::string part(path, start, end-start);
      if (! gDirectory->Get(part.c_str()))
        gDirectory->mkdir(part.c_str());
      gDirectory->cd(part.c_str());
      // Stop if we reached the end, ignoring any trailing '/'.
      if (end+1 >= path.size())
        break;
      // Find the next path component.
      start = end+1;
      end = path.find('/', start);
      if (end == std::string::npos)
        end = path.size();
    }
    obj->Write();
    DEBUG(1, obj->GetName() << std::endl);
  }
  output.Close();
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}

int dumpFiles(const std::vector<std::string> &filenames) {
  assert(filenames.size() > 0);
  for (int i = 0, e = filenames.size(); i != e; ++i) {
    DEBUG(0, "Dumping file " << filenames[i] << std::endl);
    dqmstorepb::ROOTFilePB dqmstore_message;

    int filedescriptor = open(filenames[0].c_str(), O_RDONLY);
    FileInputStream fin(filedescriptor);
    GzipInputStream input(&fin);
    CodedInputStream input_coded(&input);
    input_coded.SetTotalBytesLimit(1024*1024*1024, -1);
    if (!dqmstore_message.ParseFromCodedStream(&input_coded)) {
      std::cout << "Fatal Error opening file "
                << filenames[0] << std::endl;
      return ERR_NOFILE;
    }

    for (int i = 0; i < dqmstore_message.histo_size(); i++) {
      const dqmstorepb::ROOTFilePB::Histo& h = dqmstore_message.histo(i);
      DEBUG(1, h.full_pathname() << std::endl);
      DEBUG(1, h.size() << std::endl);
      TBufferFile buf(TBufferFile::kRead, h.size(),
                      (void*)h.streamed_histo().data(),
                      kFALSE);
      buf.Reset();
      TObject *obj = extractNextObject(buf);
      DEBUG(1, obj->GetName() << std::endl);
      DEBUG(1, "Flags: " << h.flags() << std::endl);
    }
  }
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}

int addFiles(const std::string &output_filename,
             const std::vector<std::string> &filenames) {
  dqmstorepb::ROOTFilePB dqmstore_outputmessage;
  std::set<std::string> dirs;
  std::set<std::string> objs;
  std::set<std::string> fullnames;
  std::set<MicroME> micromes;

  assert(filenames.size() > 1);
  DEBUG(1, "Adding file " << filenames[0] << std::endl);
  {
    dqmstorepb::ROOTFilePB dqmstore_message;
    int filedescriptor;
    if ((filedescriptor = open(filenames[0].c_str(), O_RDONLY)) == -1) {
      std::cout << "Fatal Error opening file "
                << filenames[0] << std::endl;
      return ERR_NOFILE;
    }

    FileInputStream fin(filedescriptor);
    GzipInputStream input(&fin);
    CodedInputStream input_coded(&input);
    input_coded.SetTotalBytesLimit(1024*1024*1024, -1);
    if (!dqmstore_message.ParseFromCodedStream(&input_coded)) {
      std::cout << "Fatal Error opening file "
                << filenames[0] << std::endl;
      return ERR_NOFILE;
    }
    for (int i = 0; i < dqmstore_message.histo_size(); i++) {
      std::string path;
      std::string objname;
      TObject *obj = NULL;
      const dqmstorepb::ROOTFilePB::Histo &h = dqmstore_message.histo(i);
      get_info(h, path, objname, &obj);
      MicroME * mme = new MicroME(&*(fullnames.insert(h.full_pathname()).first),
                                  &*(dirs.insert(path).first),
                                  &*(objs.insert(objname).first),
                                  h.flags());
      if (obj) {
        mme->obj = obj;
        micromes.insert(*mme);
        DEBUG(2, "Inserting MicroME " << *mme->fullname << std::endl);
      }
    }
  }

  for (int i = 1, e = filenames.size(); i != e; ++i) {
    DEBUG(1, "Adding file " << filenames[i] << std::endl);
    dqmstorepb::ROOTFilePB dqmstore_msg;
    int filedescriptor;
    if ((filedescriptor = open(filenames[i].c_str(), O_RDONLY)) == -1) {
      std::cout << "Fatal Error opening file "
                << filenames[i] << std::endl;
      return ERR_NOFILE;
    }
    FileInputStream fin(filedescriptor);
    GzipInputStream input(&fin);
    CodedInputStream input_coded(&input);
    input_coded.SetTotalBytesLimit(1024*1024*1024, -1);
    if (!dqmstore_msg.ParseFromCodedStream(&input_coded)) {
      std::cout << "Fatal Error opening file "
                << filenames[0] << std::endl;
      return ERR_NOFILE;
    }

    std::set<MicroME>::iterator mi = micromes.begin();
    std::set<MicroME>::iterator me = micromes.end();
    int elem = 0;
    for (; mi != me; ++mi) {
      std::string path;
      std::string objname;
      dqmstorepb::ROOTFilePB::Histo h;
      TObject *obj = NULL;
      if (elem < dqmstore_msg.histo_size()) {
        dqmstorepb::ROOTFilePB::Histo &h =
            const_cast<dqmstorepb::ROOTFilePB::Histo &>(dqmstore_msg.histo(elem));
        get_info(h, path, objname, &obj);

        DEBUG(2, "Comparing " << *(*mi).dirname << "/"
              << *(*mi).objname << " vs "
              << h.full_pathname() << std::endl);
        int diff = (*mi).fullname->compare(h.full_pathname());
        if (diff == 0 && obj != NULL) {
          mi->add(obj);
          delete obj;
          ++elem;
        } else if (! lessThanMME(*(*mi).dirname, *(*mi).objname,
                                 path, objname)) {
          // loop over elem till they are no longer less than iterator.
          bool loop = true;
          while (loop) {
            DEBUG(2, "Adding Missing histogram "
                  << h.full_pathname() << std::endl);
            // That's fine since we add elements to the left of the
            // current node, so we do not screw up the iteration
            // process.
            MicroME * mme = new MicroME(&*(fullnames.insert(h.full_pathname()).first),
                                        &*(dirs.insert(path).first),
                                        &*(objs.insert(objname).first));
            if (obj) {
              mme->obj = obj;
              micromes.insert(*mme);
              ++elem;
            }
            if (elem < dqmstore_msg.histo_size()) {
              h = const_cast<dqmstorepb::ROOTFilePB::Histo &>(dqmstore_msg.histo(elem));
              get_info(h, path, objname, &obj);
              DEBUG(2, "Comparing " << *(*mi).dirname << "/"
                    << *(*mi).objname << " vs "
                    << h.full_pathname() << std::endl);
              loop = ! lessThanMME(*(*mi).dirname, *(*mi).objname,
                                   path, objname);
            } else {
              loop = false;
            }
          }
        }
      }
    }

    // Transfer whatever else is left pending in the new file.
    while (elem < dqmstore_msg.histo_size()) {
      std::string path;
      std::string objname;
      TObject *obj = NULL;

      const dqmstorepb::ROOTFilePB::Histo &h = dqmstore_msg.histo(elem);
      get_info(h, path, objname, &obj);
      DEBUG(2, "Adding Missing histogram " << h.full_pathname() << std::endl);
      MicroME * mme = new MicroME(&*(fullnames.insert(h.full_pathname()).first),
                                  &*(dirs.insert(path).first),
                                  &*(objs.insert(objname).first));
      if (obj) {
        mme->obj = obj;
        micromes.insert(*mme);
        ++elem;
      }
    }
  }

  dqmstorepb::ROOTFilePB dqmstore_output_msg;
  fillMessage(dqmstore_output_msg, micromes);
  writeMessage(dqmstore_output_msg, output_filename);

  return 0;
}

static int
showusage(void)
{
  static const std::string app_name("fasthadd");

  std::cerr << "Usage: " << app_name
            << " [--[no-]debug] TASK OPTIONS\n\n  "
            << app_name << " [OPTIONS] add -o OUTPUT_FILE [DAT FILE...]\n  "
            << app_name << " [OPTIONS] convert -o ROOT_FILE DAT_FILE\n  "
            << app_name << " [OPTIONS] encode -o DAT_FILE ROOT_FILE\n  "
            << app_name << " [OPTIONS] dump [DAT FILE...]\n  ";
  return ERR_BADCFG;
}

int main(int argc, char * argv[]) {
  int arg;
  int ret = 0;
  std::string output_file;
  std::vector<std::string> filenames;
  TaskType task;

  filenames.reserve(argc);

  for (arg = 1; arg < argc; ++arg) {
    if (! strcmp(argv[arg], "--no-debug"))
      debug = 0;
    else if (! strcmp(argv[arg], "--debug")
             || ! strcmp(argv[arg], "-d"))
      debug++;
    else
      break;
  }

  if (arg < argc) {
    if (! strcmp(argv[arg], "add")) {
      ++arg;
      task = TASK_ADD;
    } else if (! strcmp(argv[arg], "dump")) {
      ++arg;
      task = TASK_DUMP;
    } else if (! strcmp(argv[arg], "convert")) {
      ++arg;
      task = TASK_CONVERT;
    } else if (! strcmp(argv[arg], "encode")) {
      ++arg;
      task = TASK_ENCODE;
    } else {
      std::cerr << "Unknown action: " << argv[arg] << std::endl;
      return showusage();
    }
  } else {
    std::cerr << "Not enough arguments\n";
    return showusage();
  }

  if (task == TASK_ADD || task == TASK_CONVERT || task == TASK_ENCODE) {
    if (arg == argc) {
      std::cerr << "add|convert|encode actions requires a -o option to be set\n";
      return showusage();
    }
    if (! strcmp(argv[arg], "-o")) {
      if (arg < argc-1) {
        output_file = argv[++arg];
      } else {
        std::cerr << " -o option requires a value\n";
        return showusage();
      }
    }
  } else if (task == TASK_DUMP) {
    if (arg == argc) {
      std::cerr << "Missing input file(s)\n";
      return showusage();
    }
    for (; arg < argc; ++arg) {
      filenames.push_back(argv[arg]);
    }
  }

  if (task == TASK_ADD || task == TASK_CONVERT || task == TASK_ENCODE) {
    if (++arg == argc) {
      std::cerr << "Missing input file(s)\n";
      return showusage();
    }
    for (; arg < argc; ++arg) {
      filenames.push_back(argv[arg]);
    }
  }

  if (task == TASK_ADD)
    ret = addFiles(output_file, filenames);
  else if (task == TASK_DUMP)
    ret = dumpFiles(filenames);
  else if (task == TASK_CONVERT)
    ret = convertFile(output_file, filenames);
  else if (task == TASK_ENCODE)
    ret = encodeFile(output_file, filenames);

  return ret;
}
