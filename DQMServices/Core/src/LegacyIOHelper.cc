#include "DQMServices/Core/interface/LegacyIOHelper.h"

#include <cstdio>
#include <cfloat>
#include <vector>
#include <string>

#include "TString.h"
#include "TSystem.h"
#include "TFile.h"
#include <sys/stat.h>

void LegacyIOHelper::save(std::string const &filename,
                          std::string const &path /* = "" */,
                          uint32_t const run /* = 0 */,
                          bool saveall /* = true */,
                          std::string const &fileupdate /* = "RECREATE" */) {
  // TFile flushes to disk with fsync() on every TDirectory written to
  // the file.  This makes DQM file saving painfully slow, and
  // ironically makes it _more_ likely the file saving gets
  // interrupted and corrupts the file.  The utility class below
  // simply ignores the flush synchronisation.
  class TFileNoSync : public TFile {
  public:
    TFileNoSync(char const *file, char const *opt) : TFile{file, opt} {}
    Int_t SysSync(Int_t) override { return 0; }
  };

  std::cout << "DQMFileSaver::globalEndRun()" << std::endl;

  char suffix[64];
  sprintf(suffix, "R%09d", run);
  TFileNoSync *file = new TFileNoSync(filename.c_str(), fileupdate.c_str());  // open file

  // Traverse all MEs
  std::vector<MonitorElement *> mes;
  if (saveall) {
    // this is typically used, at endJob there will only be JOB histos here
    mes = dbe_->getAllContents(path);
  } else {
    // at endRun it might make sense to use this, to not save JOB histos yet.
    mes = dbe_->getAllContents(path, run, 0);
  }

  for (auto me : mes) {
    // Modify dirname to comply with DQM GUI format. Change:
    // A/B/C/plot
    // into:
    // DQMData/Run X/A/Run summary/B/C/plot
    std::string dirName = me->getPathname();
    uint64_t firstSlashPos = dirName.find("/");
    if (firstSlashPos == std::string::npos) {
      firstSlashPos = dirName.length();
    }

    if (run) {
      // Rewrite paths to "Run Summary" format when given a run number.
      // Else, write a simple, flat TDirectory for local usage.
      dirName = dirName.substr(0, firstSlashPos) + "/Run summary" + dirName.substr(firstSlashPos, dirName.size());
      dirName = "DQMData/Run " + std::to_string(run) + "/" + dirName;
    }

    std::string objectName = me->getName();

    // Create dir if it doesn't exist and cd into it
    createDirectoryIfNeededAndCd(dirName);

    // INTs are saved as strings in this format: <objectName>i=value</objectName>
    // REALs are saved as strings in this format: <objectName>f=value</objectName>
    // STRINGs are saved as strings in this format: <objectName>s="value"</objectName>
    if (me->kind() == MonitorElement::Kind::INT) {
      int value = me->getIntValue();
      std::string content = "<" + objectName + ">i=" + std::to_string(value) + "</" + objectName + ">";
      TObjString str(content.c_str());
      str.Write();
    } else if (me->kind() == MonitorElement::Kind::REAL) {
      double value = me->getFloatValue();
      char buf[64];
      // use printf here to preserve exactly the classic formatting.
      std::snprintf(buf, sizeof(buf), "%.*g", DBL_DIG + 2, value);
      std::string content = "<" + objectName + ">f=" + buf + "</" + objectName + ">";
      TObjString str(content.c_str());
      str.Write();
    } else if (me->kind() == MonitorElement::Kind::STRING) {
      const std::string &value = me->getStringValue();
      std::string content = "<" + objectName + ">s=" + value + "</" + objectName + ">";
      TObjString str(content.c_str());
      str.Write();
    } else {
      // Write a histogram
      TH1 *value = me->getTH1();
      value->Write();

      if (me->getEfficiencyFlag()) {
        std::string content = "<" + objectName + ">e=1</" + objectName + ">";
        TObjString str(content.c_str());
        str.Write();
      }

      for (QReport *qr : me->getQReports()) {
        std::string result;
        // TODO: 64 is likely too short; memory corruption in the old code?
        char buf[64];
        std::snprintf(buf, sizeof(buf), "qr=st:%d:%.*g:", qr->getStatus(), DBL_DIG + 2, qr->getQTresult());
        result = '<' + objectName + '.' + qr->getQRName() + '>';
        result += buf;
        result += qr->getAlgorithm() + ':' + qr->getMessage();
        result += "</" + objectName + '.' + qr->getQRName() + '>';
        TObjString str(result.c_str());
        str.Write();
      }
    }

    // Go back to the root directory
    gDirectory->cd("/");
  }

  file->Close();
}

// Use this for saving monitoring objects in ROOT files with dir structure;
// cds into directory (creates it first if it doesn't exist);
// returns a success flag
bool LegacyIOHelper::createDirectoryIfNeededAndCd(const std::string &path) {
  assert(!path.empty());

  // Find the first path component.
  size_t start = 0;
  size_t end = path.find('/', start);
  if (end == std::string::npos)
    end = path.size();

  while (true) {
    // Check if this subdirectory component exists.  If yes, make sure
    // it is actually a subdirectory.  Otherwise create or cd into it.
    std::string part(path, start, end - start);
    TObject *o = gDirectory->Get(part.c_str());
    if (o && !dynamic_cast<TDirectory *>(o))
      throw cms::Exception("DQMFileSaver") << "Attempt to create directory '" << path
                                           << "' in a file"
                                              " fails because the part '"
                                           << part
                                           << "' already exists and is not"
                                              " directory";
    else if (!o)
      gDirectory->mkdir(part.c_str());

    if (!gDirectory->cd(part.c_str()))
      throw cms::Exception("DQMFileSaver") << "Attempt to create directory '" << path
                                           << "' in a file"
                                              " fails because could not cd into subdirectory '"
                                           << part << "'";

    // Stop if we reached the end, ignoring any trailing '/'.
    if (end + 1 >= path.size())
      break;

    // Find the next path component.
    start = end + 1;
    end = path.find('/', start);
    if (end == std::string::npos)
      end = path.size();
  }

  return true;
}
