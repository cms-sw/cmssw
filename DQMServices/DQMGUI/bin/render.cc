#define DEBUG(n, x)

#include <iostream>

#define logme() std::cout

#include "DQMServices/DQMGUI/interface/Objects.h"
#include "DQMServices/DQMGUI/interface/DQMRenderPlugin.h"

#include <string>
#include <cstdio>
#include <chrono>

#include "TASImage.h"
#include "TAxis.h"
#include "TBufferFile.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TDirectory.h"
#include "TError.h"
#include "TFile.h"
#include "TGraphAsymmErrors.h"
#include "TGraphErrors.h"
#include "TH1D.h"
#include "TH2.h"
#include "TH3.h"
#include "THStack.h"
#include "TImageDump.h"
#include "TList.h"
#include "TObjString.h"
#include "TPaveStats.h"
#include "TProfile.h"
#include "TProfile2D.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TText.h"
#include "TBufferJSON.h"
#include "boost/algorithm/string.hpp"
#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <sys/mman.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>

#include <png.h>

static int colors[] = {kGray, kAzure, kOrange + 7, kRed + 1, kMagenta + 2, kGreen - 3};
using DataBlob = std::vector<unsigned char>;
struct Bucket {
  DataBlob data;
};

// ROOT uses global state in many places, but not to report errors...
// We set a custom error handler to know if any errors happend, but then call
// the default handler. The flag that is then reported to the client depends
// on the flags that we set in this method after parsing the error.
ErrorHandlerFunc_t defaulterrorhandler;
bool haderror = false;
bool hadmissingstreamers = false;
void RootErrorHandler(int level, Bool_t abort, const char *location, const char *msg) {
  // These errors happen whenever displacement is used with TBufferFile, because
  // null pointers are no longer recognized as null. (ROOT bug?)
  // The last one is cause by the others.
  // Ignore them.
  if (std::string(msg).find("reference to object of unavailable class THashList, offset=-") != std::string::npos ||
      std::string(msg).find("reference to object of unavailable class TList, offset=-") != std::string::npos ||
      std::string(msg).find("reference to object of unavailable class TObject, offset=-") != std::string::npos ||
      std::string(location).find("TExMap::Remove") != std::string::npos) {
    return;
  }

  haderror = true;

  // Check for a specific kind of error. Other errors could be identified here as well
  if (std::string(msg).find("Could not find the StreamerInfo") != std::string::npos) {
    hadmissingstreamers = true;
  }

  defaulterrorhandler(level, abort, location, msg);
}

// This method parses the flags set by RootErrorHandler and creates a single
// status code. Currently used codes:
// 0 - OK
// 1 - Streamers are missing
// 2 - Some other ROOT error
int getReturnCode() {
  int returnCode = 0;
  if (hadmissingstreamers) {
    returnCode = 1;
  } else if (haderror) {
    returnCode = 2;
  }
  return returnCode;
}

// Any new error flags must be reset here
void resetErrorFlags() {
  haderror = false;
  hadmissingstreamers = false;
}

//----------------------------------------------------------------------
/** Extract the next serialised ROOT object from @a buf.  Returns null
    if there are no more objects in the buffer, or a null pointer was
    serialised at this location. */
inline TObject *extractNextObject(TBufferFile &buf) {
  if (buf.Length() == buf.BufferSize())
    return nullptr;
  buf.InitMap();
  return (TObject *)buf.ReadObjectAny(nullptr);
}

// ----------------------------------------------------------------------
static void stripNameToLabel(std::string &label, const std::string &name) {
  size_t slash = name.rfind('/');
  size_t namepart = (slash == std::string::npos ? 0 : slash + 1);
  size_t len = name.size() - namepart;
  label.reserve(len + 1);
  if (len < 20)
    label.append(name, namepart, len);
  else {
    label.append(name, namepart, 12);
    label.append("...");
    label.append(name, name.size() - 5, 5);
  }
}

static void PngWriteCallback(png_structp png_ptr, png_bytep data, png_size_t length) {
  auto *p = (std::vector<unsigned char> *)png_get_io_ptr(png_ptr);
  p->insert(p->end(), data, data + length);
}

// ----------------------------------------------------------------------
// Convert the canvas into a PNG image.
class DQMImageDump : public TImageDump {
public:
  DQMImageDump(TVirtualPad *pad, DataBlob &pngdata) {
    fImage = new TASImage;
    fType = 114;
    SetName(pad->GetName());
    SetBit(BIT(11));

    pad->Update();
    pad->Paint();

    DataBlob imgdata;
    UInt_t width = fImage->GetWidth();
    UInt_t height = fImage->GetHeight();
    imgdata.resize(width * height * sizeof(uint8_t) * 3);
    uint32_t *argb = (uint32_t *)fImage->GetArgbArray();
    uint8_t *rgb = (uint8_t *)&imgdata[0];
    for (UInt_t row = 0, idx = 0; row < height; ++row) {
      for (UInt_t col = 0; col < width; ++col, ++idx) {
        uint32_t val = argb[idx];
        rgb[idx * 3 + 0] = (val >> 16) & 0xff;
        rgb[idx * 3 + 1] = (val >> 8) & 0xff;
        rgb[idx * 3 + 2] = (val >> 0) & 0xff;
      }
    }
    compress(&pngdata, imgdata, width, height);
  }

  void compress(DataBlob *out, DataBlob &imgbytes, unsigned int w, unsigned int h) {
    out->clear();
    png_structp p = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    assert(p || !"png_create_write_struct() failed");
    png_infop info_ptr = png_create_info_struct(p);
    assert(info_ptr || !"png_create_info_struct() failed");
    auto ok = setjmp(png_jmpbuf(p));
    assert(ok == 0 || !"setjmp(png_jmpbuf(p) failed");
    png_set_IHDR(p,
                 info_ptr,
                 w,
                 h,
                 8,
                 PNG_COLOR_TYPE_RGB,
                 PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT,
                 PNG_FILTER_TYPE_DEFAULT);
    png_set_compression_level(p, 6);
    std::vector<unsigned char *> rows(h);
    for (size_t y = 0; y < h; ++y)
      rows[y] = &imgbytes[y * w * 3];
    png_set_rows(p, info_ptr, &rows[0]);
    png_set_write_fn(p, out, PngWriteCallback, nullptr);
    png_write_png(p, info_ptr, PNG_TRANSFORM_IDENTITY, nullptr);
    png_destroy_write_struct(&p, nullptr);
  }
};

// ----------------------------------------------------------------------
// Parse an integer parameter in an image spec.  If @ap starts with
// the parameter prefix @a name of length @a len, extracts the number
// that follows the prefix into @a value and advances @a p to the next
// character after the extracted value.  Returns true if the parameter
// was parsed, false otherwise.
static bool parseInt(const char *&p, const char *name, size_t len, int &value) {
  if (!strncmp(p, name, len)) {
    value = strtol(p + len, (char **)&p, 10);
    return true;
  }
  return false;
}

// Parse a double parameter in an image spec, like parseInt().
static bool parseDouble(const char *&p, const char *name, size_t len, double &value) {
  if (!strncmp(p, name, len)) {
    value = strtod(p + len, (char **)&p);
    return true;
  }
  return false;
}

// Parse reference type parameter in an image spec, like parseInt().
static bool parseReference(const char *&p, const char *name, size_t len, VisDQMReference &value) {
  if (!strncmp(p, name, len)) {
    p += len;
    if (!strncmp(p, "object", 6)) {
      value = DQM_REF_OBJECT;
      p += 6;
      return true;
    } else if (!strncmp(p, "overlay", 7)) {
      value = DQM_REF_OVERLAY;
      p += 7;
      return true;
    } else if (!strncmp(p, "ratiooverlay", 12)) {
      value = DQM_REF_OVERLAY_RATIO;
      p += 12;
      return true;
    } else if (!strncmp(p, "stacked", 7)) {
      value = DQM_REF_STACKED;
      p += 7;
      return true;
    }
  }
  return false;
}

// Parse an axis type parameter in an image spec, like parseInt().
static bool parseAxisType(const char *&p, const char *name, size_t len, std::string &value) {
  if (!strncmp(p, name, len)) {
    p += len;
    if (!strncmp(p, "def", 3) || !strncmp(p, "lin", 3) || !strncmp(p, "log", 3)) {
      value.clear();
      value.insert(value.end(), p, p + 3);
      p += 3;
      return true;
    }
  }
  return false;
}

// Parse ROOT draw options parameter in an image spec, like parseInt().
static bool parseOption(const char *&p, const char *name, size_t len, std::string &value) {
  static const char SAFE[] = {
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "abcdefghijklmnopqrstuvwxyz"
      "01234567890 "};
  if (!strncmp(p, name, len)) {
    p += len;
    len = 0;
    while (p[len] && strchr(SAFE, p[len]))
      ++len;
    value.insert(value.end(), p, p + len);
    p += len;
    return true;
  }
  return false;
}

// Build a parsed image specification for an object image.  Returns
// already parsed image data if it exists, otherwise breaks the data
// into its components and adds a new empty image into the object.
// The image specification is expected to be a series of "key=value"
// pairs separated by semicolons.
static bool parseImageSpec(VisDQMImgInfo &i, const std::string &spec, const char *&error) {
  error = nullptr;
  i.imgspec = spec;
  i.width = -1;
  i.height = -1;
  i.reference = DQM_REF_OBJECT;
  i.showstats = 1;
  i.showerrbars = 0;
  i.trend = DQM_TREND_OBJECT;
  i.ktest = NAN;
  i.xaxis.type = i.yaxis.type = i.zaxis.type = "def";
  i.xaxis.min = i.yaxis.min = i.zaxis.min = NAN;
  i.xaxis.max = i.yaxis.max = i.zaxis.max = NAN;

  const char *p = spec.c_str();
  while (*p) {
    // Skip separators.
    while (*p == ';')
      ++p;

    // Handle various keywords.
    error = "unexpected image parameter";
    bool ok = false;
    ok |= parseInt(p, "w=", 2, i.width);
    ok |= parseInt(p, "h=", 2, i.height);
    ok |= parseReference(p, "ref=", 4, i.reference);
    ok |= parseInt(p, "showstats=", 10, i.showstats);
    ok |= parseInt(p, "showerrbars=", 12, i.showerrbars);
    ok |= parseDouble(p, "xmin=", 5, i.xaxis.min);
    ok |= parseDouble(p, "xmax=", 5, i.xaxis.max);
    ok |= parseAxisType(p, "xtype=", 6, i.xaxis.type);
    ok |= parseDouble(p, "ymin=", 5, i.yaxis.min);
    ok |= parseDouble(p, "ymax=", 5, i.yaxis.max);
    ok |= parseAxisType(p, "ytype=", 6, i.yaxis.type);
    ok |= parseDouble(p, "zmin=", 5, i.zaxis.min);
    ok |= parseDouble(p, "zmax=", 5, i.zaxis.max);
    ok |= parseDouble(p, "ktest=", 6, i.ktest);
    ok |= parseAxisType(p, "ztype=", 6, i.zaxis.type);
    ok |= parseOption(p, "drawopts=", 9, i.drawOptions);
    ok |= parseOption(p, "norm=", 5, i.refnorm);
    ok |= parseOption(p, "json=", 5, i.json);
    char refoptionname[] = "reflabel0=";
    std::string reflabel;
    for (int k = 0; k < 10; k++) {
      bool gotit = parseOption(p, refoptionname, 10, reflabel);
      if (gotit) {
        ok = true;
        i.reflabels.resize(k + 1);
        i.reflabels[k] = reflabel;
      }
      refoptionname[8] += 1;  // 8 is the position of the digit
    }
    if (!ok)
      return false;

    if (*p && *p != ';') {
      error = "unexpected characters in parameter value";
      return false;
    }
  }

  if (i.width < 0 || i.height < 0) {
    error = "image dimensions missing, zero or negative";
    return false;
  }

  return true;
}

typedef std::string Filename;
bool fileExists(Filename const &f) { return std::filesystem::exists(f); }

// no idea what the bools mean, that is from classlib...
void removeFile(Filename const &f, bool, bool) { std::filesystem::remove(f); }

// ----------------------------------------------------------------------
class VisDQMImageServer {
public:
  static const uint32_t SUMMARY_PROP_EFFICIENCY_PLOT = 0x00200000;

  typedef std::vector<DQMRenderPlugin *> RenderPlugins;
  typedef std::map<std::string, uint64_t> BlackList;
  struct ServerInfo {
    int argc;
    char **argv;
    bool debug;
    Filename dynobj;
    Filename statedir;
    Filename sockpath;
    Filename stopfile;
    BlackList blacklist;
  };

private:
  RenderPlugins plugins_;
  std::vector<bool> applied_;
  TStyle *standardStyle_;
  ServerInfo info_;

public:
  VisDQMImageServer(const ServerInfo &info) : standardStyle_(nullptr), info_(info) {
    // Prevent histograms from getting attached to the global directory.
    TH1::AddDirectory(kFALSE);

    // Capture unchangeable standard ROOT style.
    gStyle->Reset();
    gStyle->SetFillColor(10);
    gStyle->SetCanvasColor(10);
    gStyle->SetPadColor(10);
    gStyle->SetTitleFillColor(10);
    gStyle->SetTitleBorderSize(0);
    gStyle->SetOptStat(1111);
    gStyle->SetStatColor(10);
    gStyle->SetStatBorderSize(1);
    gStyle->SetNumberContours(100);
    gStyle->SetFrameFillColor(10);
    gStyle->SetLabelSize(0.04);
    standardStyle_ = (TStyle *)gStyle->Clone();
    gROOT->ForceStyle();

    // Indicate we've started.
    logme() << "INFO: DQM image server started in path " << info_.statedir << std::endl;

    for (BlackList::const_iterator i = info.blacklist.begin(), e = info.blacklist.end(); i != e; ++i)
      logme() << "INFO: black-listed '" << i->first << ")\n";

    // Initialise plugins now.
    initPlugins();
  }

  virtual ~VisDQMImageServer(void) { shutPlugins(); }

  void run() {
    // TODO
    // stolen here: https://github.com/zappala/socket-programming-examples-c/blob/master/unix-echo/unix-server.cc
    struct sockaddr_un server_addr;

    // setup socket address structure
    bzero(&server_addr, sizeof(server_addr));
    server_addr.sun_family = AF_UNIX;
    strncpy(server_addr.sun_path, info_.sockpath.c_str(), sizeof(server_addr.sun_path) - 1);

    // create socket
    auto server_ = socket(PF_UNIX, SOCK_STREAM, 0);
    if (!server_) {
      perror("socket");
      exit(-1);
    }

    // call bind to associate the socket with the UNIX file system
    if (bind(server_, (const struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
      perror("bind");
      exit(-1);
    }

    // convert the socket to listen for incoming connections
    if (listen(server_, SOMAXCONN) < 0) {
      perror("listen");
      exit(-1);
    }

    int client;
    struct sockaddr_un client_addr;
    socklen_t clientlen = sizeof(client_addr);

    // accept clients
    while ((client = accept(server_, (struct sockaddr *)&client_addr, &clientlen)) > 0) {
      while (true) {
        // get a request
        bool done = false;
        char buf[1024];
        std::vector<unsigned char> msg;
        int len = -1;
        while (!done) {
          int nread = recv(client, buf, sizeof(buf), 0);
          if (nread < 0) {
            if (errno == EINTR)
              // the socket call was interrupted -- try again
              continue;
            else
              // an error occurred, so break out
              break;
          } else if (nread == 0) {
            // the socket is closed
            break;
          }
          if (len < 0) {
            memcpy(&len, buf, 4);
          }
          msg.insert(msg.end(), buf, buf + nread);
          if (msg.size() >= (unsigned int)(len)) {
            done = true;
          }
        }
        if (done) {
          Bucket out;
          onMessage(&out, &msg[0], len);

          const unsigned char *ptr = &out.data[0];
          int nleft = out.data.size();
          int nwritten;
          // loop to be sure it is all sent
          while (nleft) {
            if ((nwritten = send(client, ptr, nleft, 0)) < 0) {
              if (errno == EINTR) {
                // the socket call was interrupted -- try again
                continue;
              } else {
                // an error occurred, so break out
                perror("write");
                break;
              }
            } else if (nwritten == 0) {
              // the socket is closed
              break;
            }
            nleft -= nwritten;
            ptr += nwritten;
          }
        } else {
          break;
        }
      }
      close(client);
    }
  }

protected:
  // Check if the server should stop.
  virtual bool shouldStop(void) { return fileExists(info_.stopfile); }

  void copydata(Bucket *msg, const void *data, size_t len) {
    msg->data.insert(msg->data.end(), (const unsigned char *)data, (const unsigned char *)data + len);
  }

  void setErrorAndEmptyResponse(Bucket* msg) {
    msg->data.reserve(8);
    uint32_t errorcode = 2;
    uint32_t length = 0;
    copydata(msg, &errorcode, 4);
    copydata(msg, &length, 4);
  }

  // Web server facing message handling.  Respond to image requests.
  virtual bool onMessage(Bucket *msg, unsigned char *data, size_t len) {
    // Decode and process this message.
    uint32_t type;
    uint32_t nwords = 2;
    uint32_t flags;
    uint32_t vlow;
    uint32_t vhigh;
    uint32_t numobjs;
    uint32_t filelen;
    uint32_t namelen;
    uint32_t speclen;
    uint32_t datalen;
    uint32_t qlen;
    uint32_t gotlen;

    memcpy(&type, data + sizeof(uint32_t), sizeof(type));
    auto start = std::chrono::steady_clock::now();
    if (len < 9 * sizeof(uint32_t)) {
      logme() << "ERROR: corrupt 'GET IMAGE DATA' message of length " << len << "\n";
      setErrorAndEmptyResponse(msg);
      return false;
    }

    memcpy(&flags, data + nwords++ * sizeof(uint32_t), sizeof(uint32_t));
    memcpy(&vlow, data + nwords++ * sizeof(uint32_t), sizeof(uint32_t));
    memcpy(&vhigh, data + nwords++ * sizeof(uint32_t), sizeof(uint32_t));
    memcpy(&numobjs, data + nwords++ * sizeof(uint32_t), sizeof(uint32_t));
    memcpy(&filelen, data + nwords++ * sizeof(uint32_t), sizeof(uint32_t));
    memcpy(&namelen, data + nwords++ * sizeof(uint32_t), sizeof(uint32_t));
    memcpy(&speclen, data + nwords++ * sizeof(uint32_t), sizeof(uint32_t));
    memcpy(&datalen, data + nwords++ * sizeof(uint32_t), sizeof(uint32_t));
    memcpy(&qlen, data + nwords++ * sizeof(uint32_t), sizeof(uint32_t));
    gotlen = nwords * sizeof(uint32_t) + filelen + namelen + speclen + datalen + qlen;

    if (len != gotlen) {
      logme() << "ERROR: corrupt 'GET IMAGE DATA' message of length " << len << ", expected length "
              << (nwords * sizeof(uint32_t)) << " + " << filelen << " + " << namelen << " + " << speclen << " + "
              << datalen << " + " << qlen << " = " << gotlen << std::endl;
      setErrorAndEmptyResponse(msg);
      return false;
    }

    if (numobjs < 1 || numobjs > 10) {
      logme() << "ERROR: 'GET IMAGE DATA' with <1 or >10 parts\n";
      setErrorAndEmptyResponse(msg);
      return false;
    }

    const char *part = (const char *)data + nwords * sizeof(uint32_t);
    std::string file(part, filelen);
    part += filelen;
    std::string name(part, namelen);
    part += namelen;
    std::string spec(part, speclen);
    part += speclen;
    std::string odata(part, datalen);
    part += datalen;
    std::string qdata(part, qlen);
    part += qlen;

    size_t slash = name.rfind('/');
    size_t dirpos = (slash == std::string::npos ? 0 : slash);
    size_t namepos = (slash == std::string::npos ? 0 : slash + 1);
    std::string dirpart(name, 0, dirpos);

    // reset global error flags, then check in the end if there were errors.
    ::resetErrorFlags();

    // if a file name/URL was passed, open this file first to get its streamers.
    // the data in the file is not actually used.
    if (filelen > 0) {
      TFile *streamerfile = TFile::Open(file.c_str());
      if (streamerfile) {
        streamerfile->ReadStreamerInfo();  // redundant, should happen automatically
        streamerfile->Close();
        delete streamerfile;
      }
    }

    // Validate the image request.
    VisDQMImgInfo info;
    const char *error = nullptr;

    if (!parseImageSpec(info, spec, error)) {
      logme() << "ERROR: invalid image specification '" << spec << "' for object '" << name << "': " << error
              << std::endl;
      setErrorAndEmptyResponse(msg);
      return false;
    }

    // Reconstruct the object, defaulting to missing in action.
    std::vector<VisDQMObject> &objs = info.objects;
    objs.resize(numobjs);
    for (uint32_t i = 0; i < numobjs; ++i) {
      objs[i].flags = flags;
      objs[i].version = (((uint64_t)vhigh) << 32) | vlow;
      objs[i].dirname = dirpart;
      objs[i].objname.append(name, namepos, std::string::npos);
      //unpackQualityData(objs[i].qreports, objs[i].flags, qdata.c_str());
      objs[i].name = name;
      objs[i].object = nullptr;
    }

    if (!readRequest(info, odata, objs, numobjs)) {
      setErrorAndEmptyResponse(msg);
      return false;
    }

    // Now render and build response.
    bool blacklisted = false;

    if (info.json == "True") {
      // Produce a JSON representation of an object
      TString json;
      blacklisted = doRenderJson(info, &objs[0], numobjs, json);
      uint32_t returncode = ::getReturnCode();
      uint32_t jsonsize = strlen(json.Data());
      msg->data.reserve(jsonsize + sizeof(returncode) + sizeof(jsonsize));
      copydata(msg, &returncode, sizeof(returncode));
      copydata(msg, &jsonsize, sizeof(jsonsize));
      if (jsonsize) {
        copydata(msg, json.Data(), jsonsize);
      }
    } else {
      // Render a PNG
      DataBlob imgdata;
      blacklisted = doRender(info, &objs[0], numobjs, imgdata);
      uint32_t returncode = ::getReturnCode();
      uint32_t imgsize = imgdata.size();
      msg->data.reserve(imgsize + sizeof(returncode) + sizeof(imgsize));
      copydata(msg, &returncode, sizeof(returncode));
      copydata(msg, &imgsize, sizeof(imgsize));
      if (imgsize)
        copydata(msg, &imgdata[0], imgdata.size());
    }

    // Release the objects.
    for (uint32_t i = 0; i < numobjs; ++i) {
      delete objs[i].object;
    }

    // Report how long it took.
    //Time end = Time::current();
    auto end = std::chrono::steady_clock::now();
    auto time_span = std::chrono::duration<double>(end - start);
    logme() << "INFO: rendered '" << objs[0].name << "' version " << objs[0].version << " as '" << spec
            << (blacklisted ? "', black-listed" : "'") << ", " << numobjs << " objects"
            << " in " << (int)(time_span.count() * 1e6) / 1e3 << "ms" << (::haderror ? " (ERROR)" : " (OK)") << "\n";

    ::resetErrorFlags();
    return true;
  }

private:
  void initPlugins(void) {
    // Scan the render plug-in catalogue and set up plug-ins.
    if (!info_.dynobj.empty()) {
      DQMRenderPlugin::master(&plugins_);
      std::string name = info_.dynobj;
      if (!(::dlopen(name.c_str(), RTLD_LAZY | RTLD_GLOBAL))) {
        const char *msg = ::dlerror();
        msg = msg ? msg : "dynamic linker error message lost!";
        logme() << "dlopen() error: " << msg << "\n";
      }

      for (size_t i = 0, e = plugins_.size(); i != e; ++i) {
        if (info_.debug)
          logme() << "initialising plugin #" << i << ": " << typeid(*plugins_[i]).name() << "\n";

        plugins_[i]->initialise(info_.argc, info_.argv);
      }

      applied_.resize(plugins_.size(), false);
    }
  }

  void shutPlugins(void) {
    // There's nothing to do here, plug-ins are statically allocated.
    // We just detach ourselves.
    plugins_.clear();
  }

  // ----------------------------------------------------------------------
  bool readRequest(VisDQMImgInfo &info, std::string &odata, std::vector<VisDQMObject> &objs, uint32_t &numobjs) {
    uint32_t lenbits;
    int32_t displacement;
    uint32_t firstobj = 0;
    uint32_t pos = 0;

    // Read remaining objects, either scalar or real ROOT objects.
    // Draw a scalar only if the string is non-empty; empty means
    // "missing in action" and is handled in render code.  Real ROOT
    // objects come as len, object pairs for
    // many objects as we were sent.
    if ((objs[0].flags & DQM_PROP_TYPE_MASK) <= DQM_PROP_TYPE_SCALAR) {
      if (!odata.empty())
        objs[0].object = new TObjString(odata.c_str());
    } else {
      for (uint32_t i = firstobj; i < numobjs; ++i) {
        if (odata.size() - pos < sizeof(lenbits) + sizeof(displacement)) {
          logme() << "ERROR: 'GET IMAGE DATA' missing object length"
                  << ", expected " << sizeof(lenbits) << " but " << (odata.size() - pos) << " bytes left of "
                  << odata.size() << " for image #" << i << " of " << numobjs << std::endl;
          return false;
        }

        memcpy(&lenbits, &odata[pos], sizeof(lenbits));
        pos += sizeof(lenbits);
        memcpy(&displacement, &odata[pos], sizeof(displacement));
        pos += sizeof(displacement);

        if (odata.size() - pos < lenbits) {
          logme() << "ERROR: 'GET IMAGE DATA' missing object data"
                  << ", expected " << (lenbits) << " but " << (odata.size() - pos) << " bytes left of " << odata.size()
                  << " for image #" << i << " of " << numobjs << std::endl;
          return false;
        }

        TBufferFile buf(TBufferFile::kRead, lenbits, &odata[pos], kFALSE);
        buf.Reset();
        buf.SetBufferDisplacement(displacement);
        objs[i].object = extractNextObject(buf);
        pos += lenbits;
      }

      if (pos != odata.size()) {
        logme() << "ERROR: 'GET IMAGE DATA' request with excess"
                << " object data, expected " << pos << " bytes but found " << odata.size() << std::endl;
        return false;
      }
    }

    return true;
  }

  void applyTPadCustomizations(TPad *p, VisDQMImgInfo &i) {
    if (i.xaxis.type == "lin")
      p->SetLogx(0);
    else if (i.xaxis.type == "log")
      p->SetLogx(1);

    if (i.yaxis.type == "lin")
      p->SetLogy(0);
    else if (i.yaxis.type == "log")
      p->SetLogy(1);

    if (i.zaxis.type == "lin")
      p->SetLogz(0);
    else if (i.zaxis.type == "log")
      p->SetLogz(1);

    if (gStyle)
      gStyle->SetOptStat(i.showstats);
  }

  void drawReferenceStatBox(
      VisDQMImgInfo &i, size_t order, TH1 *ref, int color, const std::string &name, std::vector<TObject *> &nukem) {
    // Draw stats box for every additional ovelayed reference
    // object, appending latest at the bottom of the stats box
    // drawn last. FIXME: stats' coordinates are fixed, which
    // is ugly, but apparently we cannot have them back from
    // ROOT unless we use some public variable (gPad) which I
    // do not know if it is thread safe but which I know is
    // causing us problems.
    TPaveStats *currentStat = new TPaveStats(0.78, 0.835 - (order + 1) * 0.16, 0.98, 0.835 - order * 0.16, "brNDC");
    if (currentStat) {
      currentStat->SetBorderSize(1);
      nukem.push_back(currentStat);
      std::stringstream ss;
      // The numbering logic here is misleading. Numbers do **not** refer
      // directly to the text-field numbering used in the Javascript,
      // rather they refer to the actual objects that have been asked and
      // found in the index. If an object/sample has been asked but not
      // found, it is skipped but the counter is not incremented. All the
      // correct handling is done on the server side while searching for
      // samples/objects and packing information. At this stage we are mere
      // clients that rely on that information.
      if (i.reflabels.size() > order)
        ss << i.reflabels[order];
      currentStat->AddText(ss.str().c_str())->SetTextColor(color);
      ss.str("");
      ss << "Entries = " << ref->GetEntries();
      currentStat->AddText(ss.str().c_str())->SetTextColor(color);
      ss.str("");
      ss << "Mean  = " << ref->GetMean();
      currentStat->AddText(ss.str().c_str())->SetTextColor(color);
      ss.str("");
      ss << "RMS   = " << ref->GetRMS();
      currentStat->AddText(ss.str().c_str())->SetTextColor(color);
      ss.str("");
      currentStat->SetOptStat(1111);
      currentStat->SetOptFit(0);
      currentStat->Draw();
    }
  }

  double calculateScalingFactor(VisDQMObject *objs, size_t numobjs) {
    double data_area = 0;
    double total_stack_area = 0;

    if (TH1 *h = dynamic_cast<TH1 *>(objs[0].object))
      data_area = h->Integral();

    for (size_t n = 1; n < numobjs; n++) {
      TH1 *histogram = dynamic_cast<TH1 *>(objs[n].object);
      assert(histogram);
      double histogram_area = histogram->Integral();
      assert(histogram_area >= 0);
      total_stack_area += histogram_area;
    }

    return data_area > 0 ? (total_stack_area > 0.0 ? data_area / total_stack_area : 1.0) : 1.0;
  }

  // ----------------------------------------------------------------------
  // Render arbitraty text message into the TCanvas.
  void doRenderMsg(std::string histogram_name, const char *msg, const Color_t &c, std::vector<TObject *> &nukem) {
    std::string label;
    stripNameToLabel(label, histogram_name);
    TText *t1 = new TText(.5, .58, label.c_str());
    TText *t2 = new TText(.5, .42, msg);
    t1->SetNDC(kTRUE);
    t2->SetNDC(kTRUE);
    t1->SetTextSize(0.10);
    t2->SetTextSize(0.08);
    t1->SetTextAlign(22);
    t2->SetTextAlign(22);
    t1->SetTextColor(c);
    t2->SetTextColor(c);
    t1->Draw();
    t2->Draw();
    nukem.push_back(t1);
    nukem.push_back(t2);
  }

  // ----------------------------------------------------------------------
  // Extract the minimum and maximum range of a histogram, excluding
  // bin with 0 content or with 0 value.

  void computeMinAndMaxForEfficiency(
      const TH1 *h, double ratio_min, double ratio_max, double &minimum, double &maximum) {
    int bin, binx, biny, binz;
    int xfirst = h->GetXaxis()->GetFirst();
    int xlast = h->GetXaxis()->GetLast();
    int yfirst = h->GetYaxis()->GetFirst();
    int ylast = h->GetYaxis()->GetLast();
    int zfirst = h->GetZaxis()->GetFirst();
    int zlast = h->GetZaxis()->GetLast();
    double value = 0;
    for (binz = zfirst; binz <= zlast; binz++) {
      for (biny = yfirst; biny <= ylast; biny++) {
        for (binx = xfirst; binx <= xlast; binx++) {
          bin = h->GetBin(binx, biny, binz);
          value = h->GetBinContent(bin);
          if (value == 0)
            continue;
          if (value < minimum && value > ratio_min) {
            minimum = value;
          }
          if (value > maximum && value < ratio_max) {
            maximum = value;
          }
        }
      }
    }
  }

  // ----------------------------------------------------------------------
  // Set user-defined ranges for histograms
  void applyUserRange(TH1 *obj, VisDQMImgInfo &i) {
    double xmin = NAN, xmax = NAN;
    double ymin = NAN, ymax = NAN;
    double zmin = NAN, zmax = NAN;
    // If we have an object that has axes, apply the requested params.
    // Set only the bounds that were specified, and leave the rest to
    // the natural range.  Unset bounds have NaN as their value.
    if (TAxis *a = obj->GetXaxis()) {
      if (!(isnan(i.xaxis.min) && isnan(i.xaxis.max))) {
        xmin = a->GetXmin();
        xmax = a->GetXmax();
        a->SetRangeUser(isnan(i.xaxis.min) ? xmin : i.xaxis.min, isnan(i.xaxis.max) ? xmax : i.xaxis.max);
      }
    }

    if (TAxis *a = obj->GetYaxis()) {
      if (!(isnan(i.yaxis.min) && isnan(i.yaxis.max))) {
        ymin = a->GetXmin();
        ymax = a->GetXmax();
        a->SetRangeUser(isnan(i.yaxis.min) ? ymin : i.yaxis.min, isnan(i.yaxis.max) ? ymax : i.yaxis.max);
      }
    }

    if (TAxis *a = obj->GetZaxis()) {
      if (!(isnan(i.zaxis.min) && isnan(i.zaxis.max))) {
        zmin = a->GetXmin();
        zmax = a->GetXmax();
        a->SetRangeUser(isnan(i.zaxis.min) ? zmin : i.zaxis.min, isnan(i.zaxis.max) ? zmax : i.zaxis.max);
      }
    }
  }

  // ----------------------------------------------------------------------
  // Render ovelaid ROOT objects and their ratio at the bottom.
  void doRenderOverlayAndRatio(TCanvas &c,
                               VisDQMImgInfo &i,
                               VisDQMObject *objs,
                               size_t numobjs,
                               const VisDQMRenderInfo &ri,
                               std::vector<TObject *> &nukem) {
    TObject *ob = objs[0].object;
    //Do normal rendering for everything but TH1*
    if (dynamic_cast<TH2 *>(ob) || dynamic_cast<TProfile *>(ob) || dynamic_cast<TObjString *>(ob)) {
      doRenderOrdinary(c, i, objs, numobjs, ri, nukem);
      return;
    }

    int colorIndex = sizeof(colors) / sizeof(int);
    float ratio_label_font_size = 0.12;
    // Prepare the TPads.  We do not locally delete the TPad nor mark
    // them to be deleted since they will be take care of by the
    // TCanvas in which they are rendered.
    TPad *pad1 = new TPad("main", "main", 0, 0.3, 1, 1);
    pad1->SetBottomMargin(0.05);
    applyTPadCustomizations(pad1, i);
    pad1->Draw();
    pad1->cd();

    doRenderOrdinary(c, i, objs, numobjs, ri, nukem);
    c.cd();
    TPad *pad2 = new TPad("ratio", "ratio", 0, 0.05, 1, 0.3);
    pad2->SetTopMargin(0.05);
    if (i.xaxis.type == "lin")
      pad2->SetLogx(0);
    else if (i.xaxis.type == "log")
      pad2->SetLogx(1);
    pad2->Draw();
    pad2->cd();

    TH1 *h = nullptr;
    if ((h = dynamic_cast<TH1 *>(ob)) && (h->GetEntries() != 0)) {
      double norm = h->GetSumOfWeights();
      static const double RATIO_MIN = 0.001;
      static const double RATIO_MAX = 10.0;
      double ratio_min = RATIO_MAX;
      double ratio_max = RATIO_MIN;
      size_t histograms_to_be_drawn = nukem.size();
      for (size_t n = 1; n < numobjs; ++n) {
        if (TH1 *den = dynamic_cast<TH1 *>(objs[n].object)) {
          // No point in making the ratio if we do have an empty
          // denominator. The numerator has been already checked.
          if (den->GetEntries() == 0 or den->GetSumOfWeights() == 0)
            continue;

          TH1 *num = dynamic_cast<TH1 *>(h->Clone());
          nukem.push_back(num);
          double sum = den->GetSumOfWeights();
          if (sum) {
            if (objs[0].flags & SUMMARY_PROP_EFFICIENCY_PLOT)
              norm = sum = 1.;
            den->Scale(norm / sum);
            num->SetStats(false);
            if (!num->GetSumw2N())
              num->Sumw2();
            num->Divide(den);
            num->SetLineColor(colors[n % colorIndex]);
            num->SetTitle("");

            num->GetXaxis()->SetLabelFont(42);
            num->GetXaxis()->SetLabelSize(ratio_label_font_size);
            num->GetXaxis()->SetTitleFont(42);
            num->GetXaxis()->SetTitleSize(ratio_label_font_size);
            num->GetXaxis()->SetTitleOffset(-1.2);

            computeMinAndMaxForEfficiency(num, RATIO_MIN, RATIO_MAX, ratio_min, ratio_max);
            num->GetYaxis()->SetTitle("");
            num->GetYaxis()->SetLabelFont(42);
            num->GetYaxis()->SetLabelSize(ratio_label_font_size);

            TH1 *ratio_shadow_error_range = dynamic_cast<TH1 *>(num->Clone());
            nukem.push_back(ratio_shadow_error_range);
            ratio_shadow_error_range->SetFillStyle(3001);
            ratio_shadow_error_range->SetFillColor(TColor::GetColorTransparent(colors[n % colorIndex], 0.3));
            ratio_shadow_error_range->SetLineColor(TColor::GetColorTransparent(colors[n % colorIndex], 0.3));
            ratio_shadow_error_range->SetMarkerColor(TColor::GetColorTransparent(colors[n % colorIndex], 0.3));
            ratio_shadow_error_range->SetMarkerSize(0);
            for (int bin = 1; bin < ratio_shadow_error_range->GetNbinsX(); ++bin)
              ratio_shadow_error_range->SetBinContent(bin, 1.0);
          }
        }
      }

      // Finally do the actual drawing, not that all limits have been
      // globally computed across all references.

      // We need some further checking on the limits here. If the
      // ratio-plot is empty, the minimum value is set to RATIO_MAX
      // and the maximum value is correctly set to 0. We arbirarily
      // set the minimum to RATIO_MIN and the maximum to be
      // RATIO_MAX. We generically check this condition by requiring
      // that the maximum found is lower than the minimum.
      if (ratio_max <= ratio_min) {
        if (ratio_min == 1) {
          ratio_min = 0.2;
          ratio_max = 1.8;
        } else {
          ratio_min = RATIO_MIN;
          ratio_max = RATIO_MAX;
        }
      }
      for (size_t histo = histograms_to_be_drawn; histo < nukem.size(); ++histo) {
        TH1 *to_draw = dynamic_cast<TH1 *>(nukem[histo]);
        to_draw->GetYaxis()->SetRangeUser(ratio_min, ratio_max);
        if (histo == histograms_to_be_drawn) {
          to_draw->Draw("epl");
        } else {
          if (to_draw->GetMarkerSize() == 0) {
            to_draw->Draw("same e2");
          } else {
            to_draw->Draw("same epl");
          }
        }
      }
    }
    c.cd();
  }

  // ----------------------------------------------------------------------
  // Render stacked ROOT objects.
  void doRenderStacked(TCanvas &c,
                       VisDQMImgInfo &i,
                       VisDQMObject *objs,
                       size_t numobjs,
                       const VisDQMRenderInfo & /* ri */,
                       std::vector<TObject *> &nukem) {
    int colorIndex = sizeof(colors) / sizeof(int);
    uint32_t draw_me_first_index = 0;
    double absolute_y_max = 0;
    double y_max = 0;

    applyTPadCustomizations(dynamic_cast<TPad *>(&c), i);

    TObject *ob = objs[0].object;
    // Do not even try if we are offered TH2 or TProfile objects
    if (dynamic_cast<TH2 *>(ob) || dynamic_cast<TProfile *>(ob) || dynamic_cast<TObjString *>(ob)) {
      Color_t c = TColor::GetColor(64, 64, 64);
      doRenderMsg(objs[0].name, "Stacking not supported", c, nukem);
      return;
    }

    if (TH1 *dataHistogram = dynamic_cast<TH1 *>(objs[0].object)) {
      absolute_y_max = dataHistogram->GetMaximum();
      // Calculate what each the histograms that are to be stacked should be scaled by
      // to match the data histogram
      double scalingFactor = calculateScalingFactor(objs, numobjs);
      // Build the histogram stack object (THStack)
      THStack *histogramStack = new THStack();
      nukem.push_back(histogramStack);
      for (size_t n = 1; n < numobjs; ++n) {
        if (TH1 *h = dynamic_cast<TH1 *>(objs[n].object)) {
          histogramStack->Add(h);
          h->Scale(scalingFactor);
          h->SetLineColor(colors[n % colorIndex]);
          h->SetFillColor(colors[n % colorIndex]);
          h->SetFillStyle(3002);
          h->SetLineWidth(1);
          h->GetListOfFunctions()->Delete();
          y_max = h->GetMaximum();
          if (y_max > absolute_y_max) {
            absolute_y_max = y_max;
            draw_me_first_index = n;
          }
        }
      }

      // Draw the histogram stack and then the histogram on top
      dataHistogram->SetLineColor(kBlack);
      dataHistogram->SetLineWidth(2);
      if (draw_me_first_index == 0)
        dataHistogram->Draw();
      histogramStack->Draw((draw_me_first_index == 0 ? "same" : ""));
      dataHistogram->Draw("same");
    }

    if (i.showstats)
      for (size_t n = 1; n < numobjs; ++n)
        drawReferenceStatBox(i, n, dynamic_cast<TH1 *>(objs[n].object), colors[n % colorIndex], objs[n].name, nukem);
  }

  // ----------------------------------------------------------------------
  // Ordinary rendering of a ROOT object.
  void doRenderOrdinary(TCanvas &c,
                        VisDQMImgInfo &i,
                        VisDQMObject *objs,
                        size_t numobjs,
                        const VisDQMRenderInfo &ri,
                        std::vector<TObject *> &nukem) {
    std::string samePlotOptions("same p");
    TObject *ob = objs[0].object;
    TH1 *h = dynamic_cast<TH1 *>(ob);
    bool isMultiDimensional = dynamic_cast<TH2 *>(ob) ? true : false;
    isMultiDimensional |= dynamic_cast<TH3 *>(ob) ? 1 : 0;
    float max_value_in_Y = std::numeric_limits<float>::lowest();

    // Handle text.
    if (TObjString *os = dynamic_cast<TObjString *>(ob)) {
      // Using TPaveText here because it supports multiline strings.
      // We split a string on new line character and add each line separately.
      TPaveText *t = new TPaveText(0.0, 0.0, 1.0, 1.0, "NB");
      std::string delim("\n");
      char *token = strtok(strdup(os->GetName()), delim.c_str());
      while (token) {
        t->AddText(token);
        token = strtok(NULL, delim.c_str());
      }

      t->SetTextFont(63);
      t->SetTextAlign(22);
      t->SetTextSize(16);
      nukem.push_back(t);
      ob = t;
    }

    // The logic to account for possible user-supplied ranges
    // (especially in Y) and the automatic selection of the best Y
    // range in case of overlaid histograms is rather tricky. As a
    // final solution, the values supplied by the user (if any) are
    // applied *last*, so that they will prevail over the automatic
    // algorithm. Plots that have been labelled as efficiency plots
    // do not even enter this rescaling algorithm

    // Maybe draw overlay from reference and other objects.
    if (h) {
      max_value_in_Y = h->GetMaximum();
      float norm = h->GetSumOfWeights();
      if (i.refnorm != "False" && norm > 0 && !(objs[0].flags & SUMMARY_PROP_EFFICIENCY_PLOT) && !isMultiDimensional) {
        for (size_t n = 1; n < numobjs; ++n) {
          TObject *refobj = objs[n].object;
          TH1F *ref1 = dynamic_cast<TH1F *>(refobj);
          if (ref1) {
            float den = ref1->GetSumOfWeights();
            if (den > 0)
              max_value_in_Y =
                  max_value_in_Y > ref1->GetMaximum() * norm / den ? max_value_in_Y : ref1->GetMaximum() * norm / den;
          }
        }
        if (max_value_in_Y > 0 && numobjs > 1) {
          h->SetMaximum(max_value_in_Y * 1.05);
        }
      }
      applyUserRange(h, i);
      // Increase lineWidth in case there are other objects to
      // draw on top of the main one.
      if (numobjs > 1)
        h->SetLineWidth(2);
    }

    applyTPadCustomizations(dynamic_cast<TPad *>(&c), i);

    // Draw the main object on top.
    ob->Draw(ri.drawOptions.c_str());

    // Maybe draw overlay from reference and other objects.
    for (size_t n = 1; n < numobjs; ++n) {
      TObject *refobj = nullptr;
      // Compute colors array size on the fly and use it to loop
      // over defined colors in case the number of objects to
      // overlay is greater than the available colors
      // (n%colorIndex).
      int colorIndex = sizeof(colors) / sizeof(int);
      int color = colors[n % colorIndex];
      refobj = objs[n].object;

      TH1 *ref1 = dynamic_cast<TH1 *>(refobj);
      TProfile *refp = dynamic_cast<TProfile *>(refobj);
      TH2 *ref2d = dynamic_cast<TH2 *>(refobj);
      TH3 *ref3d = dynamic_cast<TH3 *>(refobj);
      if (refp) {
        refp->SetLineColor(color);
        refp->SetLineWidth(2);
        refp->GetListOfFunctions()->Delete();
        refp->Draw("same hist");

        // No point in even trying to overlay 2D or 3D histograms in the
        // same canvas. Still draw the stat box, in case they are requested.
      } else if (!ref2d && !ref3d && ref1) {
        // Perform KS statistical test only on the first available reference
        double norm = 1.;
        if (h)
          norm = h->GetSumOfWeights();

        if (n == 1 && (!isnan(i.ktest)) && h && norm && ref1 && ref1->GetSumOfWeights()) {
          double prob = h->KolmogorovTest(ref1);
          color = prob < i.ktest ? kRed - 4 : kGreen - 3;
          char buffer[14];
          snprintf(buffer, 14, "%6.5f", prob);
          TText t;
          t.SetTextColor(color);
          t.DrawTextNDC(0.45, 0.9, buffer);
        }

        ref1->SetLineColor(color);
        ref1->SetMarkerColor(color);
        ref1->SetMarkerStyle(kStar);
        ref1->SetMarkerSize(0.85);
        ref1->GetListOfFunctions()->Delete();
        if (i.showerrbars)
          samePlotOptions += " e1 x0";

        // Check if the original plot has been flagged as an
        // efficieny plot at production time: if this is the case,
        // then avoid any kind of normalization that introduces
        // fake effects.
        // Also, if the option norm=False has been supplied, avoid
        // doing any normalization.
        if (norm && !(objs[0].flags & SUMMARY_PROP_EFFICIENCY_PLOT) && i.refnorm != "False") {
          if (ref1->GetSumOfWeights())
            nukem.push_back(ref1->DrawNormalized(samePlotOptions.c_str(), norm));
        } else {
          ref1->Draw(samePlotOptions.c_str());
        }
        if (i.showstats) {
          // Apply user-defined ranges also to reference histograms, so
          // that the shown statistics is consistent with the one of the
          // main plot.
          applyUserRange(ref1, i);
          drawReferenceStatBox(i, n, ref1, color, objs[n].name, nukem);
        }
      }
    }  // End of loop over all reference sample
  }

  // ----------------------------------------------------------------------
  // Render a ROOT object.
  bool doRender(VisDQMImgInfo &i, VisDQMObject *objs, size_t numobjs, DataBlob &imgdata) {
    VisDQMObject &o = objs[0];
    VisDQMRenderInfo ri = {i.drawOptions, (info_.blacklist.count(o.name) > 0)};

    // Give me 30 seconds to render the histogram, then die.
    // TODO: POSIX but obsolete?
    struct itimerval timer;
    timer.it_value.tv_sec = 30;
    timer.it_value.tv_usec = 0;
    timer.it_interval.tv_sec = 0;
    timer.it_interval.tv_usec = 0;
    setitimer(ITIMER_VIRTUAL, &timer, nullptr);

    // Prepare render defaults and the canvas.
    delete gStyle;
    gStyle = (TStyle *)standardStyle_->Clone();
    gROOT->ForceStyle();
    for (size_t n = 0; n < numobjs; ++n) {
      if (objs[n].object)
        objs[n].object->UseCurrentStyle();
    }

    TObject *ob = o.object;

    TCanvas c("", "", i.width + 4, i.height + 28);
    c.SetFillColor(TColor::GetColor(255, 255, 255));
    c.SetBorderMode(0);

    // Record list of temporary objects we should clear out.
    std::vector<TObject *> nukem;

    // Force histogram bounds reset now.  Plug-ins can change them.
    // We need this for the axis bounds code to work.
    bool unsafe = false;
    if (TH1 *h = dynamic_cast<TH1 *>(ob)) {
      h->SetMinimum();
      h->SetMaximum();
      if (isnan(h->Integral()) || isnan(h->GetRMS()) || isnan(h->GetSumOfWeights()))
        unsafe = true;
      
      // This is a workaround to a ROOT bug where a trailing underscore 
      // in the histogram title causes drawing to crash.
      // This code removes that trailing underscore.
      TString title = h->GetTitle();
      if (title.EndsWith("_")) {
        title.Remove(title.Length()-1);
        h->SetTitle(title.Data());
      }
    }

    // Invoke pre-draw hook on plug-ins that want to apply.
    for (size_t p = 0; p < plugins_.size(); ++p) {
      applied_[p] = false;
      if (!ri.blacklisted) {
        if (info_.debug)
          logme() << "offering to plugin #" << p << ": " << typeid(*plugins_[p]).name() << "\n";

        if (plugins_[p]->applies(o, i)) {
          if (info_.debug)
            logme() << "start predraw on plugin #" << p << ": " << typeid(*plugins_[p]).name() << "\n";

          plugins_[p]->preDraw(&c, o, i, ri);
          applied_[p] = true;

          if (info_.debug)
            logme() << "end predraw on plugin #" << p << ": " << typeid(*plugins_[p]).name() << "\n";
        }
      }
    }

    // Actually draw something.  If there's no object, inform user.
    if (!ob) {
      Color_t c = TColor::GetColor(64, 64, 64);
      doRenderMsg(o.name, "is not available now", c, nukem);
    }

    // If it exists but was black-listed, inform user.
    else if (ri.blacklisted) {
      Color_t c = TColor::GetColor(178, 32, 32);
      doRenderMsg(o.name, "blacklisted for crashing ROOT", c, nukem);
    }

    // It's there and wasn't black-listed, paint it.
    else {
      try {
        // Real drawing
        switch (i.reference) {
          case DQM_REF_STACKED:
            doRenderStacked(c, i, objs, numobjs, ri, nukem);
            break;
          case DQM_REF_OVERLAY_RATIO:
            doRenderOverlayAndRatio(c, i, objs, numobjs, ri, nukem);
            break;
          default:
            doRenderOrdinary(c, i, objs, numobjs, ri, nukem);
        }
      } catch (const std::exception &e) {
        Color_t c = TColor::GetColor(178, 32, 32);
        doRenderMsg(o.name, e.what(), c, nukem);
      }

      // If the stats are bad draw alert on top.
      if (unsafe) {
        Color_t c = TColor::GetColor(178, 32, 32);
        doRenderMsg(o.name, "NaNs", c, nukem);
      }
    }

    // Invoke post-draw hook on plug-ins that applied.
    for (size_t p = 0; p < plugins_.size(); ++p)
      if (applied_[p]) {
        if (info_.debug)
          logme() << "start postdraw on plugin #" << p << ": " << typeid(*plugins_[p]).name() << "\n";

        plugins_[p]->postDraw(&c, o, i);

        if (info_.debug)
          logme() << "end postdraw on plugin #" << p << ": " << typeid(*plugins_[p]).name() << "\n";
      }

    // Grab image data and remember what version this object was.
    DQMImageDump idump(&c, imgdata);

    // Clear out temporary objects we created to avoid leaks.
    for (size_t n = 0, e = nukem.size(); n < e; ++n)
      delete nukem[n];

    // Let timer go now, we made it.
    timer.it_value.tv_sec = 0;
    timer.it_value.tv_usec = 0;
    timer.it_interval.tv_sec = 0;
    timer.it_interval.tv_usec = 0;
    setitimer(ITIMER_VIRTUAL, &timer, nullptr);

    // Indicate whether it was blacklisted.
    return ri.blacklisted;
  }

  // Render ROOT object(s) to JSON.
  bool doRenderJson(VisDQMImgInfo &i, VisDQMObject *objs, size_t numobjs, TString &json) {
    if (numobjs > 1) {
      // If there are multiple objects, we will return a JSON array of ROOT objects
      json += "{ \"objects\": [";
    }

    for (size_t n = 0; n < numobjs; ++n) {
      if (objs[n].object) {
        json += TBufferJSON::ToJSON(objs[n].object);

        if (n != numobjs - 1) {
          json += ",";
        }
      }
    }

    if (numobjs > 1) {
      json += "]}";
    }

    return false;
  }
};

// ----------------------------------------------------------------------
int main(int argc, char **argv) {
  // Make all output line buffered. We need this because multiple
  // concurrent processes are feeding output to a single log daemon.
  std::cout.flush();
  std::cerr.flush();
  std::ios_base::sync_with_stdio(true);
  setvbuf(stdout, nullptr, _IOLBF, 1024);
  setvbuf(stderr, nullptr, _IOLBF, 1024);

  // Check and process arguments.
  VisDQMImageServer::ServerInfo params;
  params.argc = argc;
  params.argv = argv;
  params.debug = false;
  for (int i = 1; i < argc; ++i)
    if (i < argc - 1 && !strcmp(argv[i], "--state-directory"))
      params.statedir = argv[++i];
    else if (i < argc - 1 && !strcmp(argv[i], "--load"))
      params.dynobj = argv[++i];
    else if (!strcmp(argv[i], "--no-debug"))
      params.debug = false;
    else if (!strcmp(argv[i], "--debug"))
      params.debug = true;
    else
      break;

  if (params.statedir.empty() || !fileExists(params.statedir)) {
    std::cerr << "Usage: " << argv[0] << " --state-directory STATE-DIRECTORY"
              << " [--load DYNAMIC-OBJECT]"
              << " [--[no-]debug]"
              << " [render-plugin-options]\n";
    return 1;
  }

  // Remove any existing socket.
  params.sockpath = params.statedir + "socket";
  removeFile(params.sockpath, false, true);

  // Remove any existing stop file.
  params.stopfile = params.statedir + "stop";
  removeFile(params.stopfile, false, true);

  // Create pid file.
  auto pidfile = params.statedir + "pid";
  FILE *f = fopen(pidfile.c_str(), "w");
  fprintf(f, "%ld", (long)getpid());
  fclose(f);

  // Read black list.
  Filename topdir = params.statedir + "../";
  Filename blacklistFile = topdir + "blacklist.txt";
  if (fileExists(blacklistFile)) {
    std::string line;
    std::ifstream f(blacklistFile);
    do {
      getline(f, line);
      if (line.empty())
        continue;

      params.blacklist[line] = 1;  // TODO: store time here.
    } while (f);
  }

  // Re-capture signals from ROOT after ROOT has initialised.
  ROOT::GetROOT();
  // TODO, maybe?

  // set up custom error handling.
  ::defaulterrorhandler = ::SetErrorHandler(&::RootErrorHandler);

  // Start serving.
  VisDQMImageServer server(params);
  server.run();
  return 0;
}
