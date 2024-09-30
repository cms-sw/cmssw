//  .
// ..: P. Chang, philip@physics.ucsd.edu

#include "fileutil.h"

TChain* RooUtil::FileUtil::createTChain(TString name, TString inputs) {
  using namespace std;

  // hadoopmap to bypass some of the broken files
  ifstream infile("hadoopmap.txt");
  std::map<TString, TString> _map;
  if (infile.good()) {
    ifstream mapfile;
    mapfile.open("hadoopmap.txt");
    std::string line, oldpath, newpath;
    while (std::getline(mapfile, line)) {
      mapfile >> oldpath >> newpath;
      TString oldpath_tstr = oldpath.c_str();
      TString newpath_tstr = newpath.c_str();
      _map[oldpath_tstr] = newpath_tstr;
    }
  }

  // globbing if the provided path is only a directory
  // It will check via looking at the last character == "/"
  if (inputs.EndsWith("/")) {
    std::string pattern = TString::Format("%s/*.root", inputs.Data()).Data();
    inputs = RooUtil::StringUtil::join(glob(pattern));
  }

  TChain* chain = new TChain(name);
  inputs = inputs.ReplaceAll("\"", "");  // In case some rogue " or ' is left over
  inputs = inputs.ReplaceAll("\'", "");  // In case some rogue " or ' is left over
  char hostnamestupid[100];
  gethostname(hostnamestupid, 100);
  TString hostname(hostnamestupid);
  std::cout << ">>> Hostname is " << hostname << std::endl;
  bool useXrootd = inputs.BeginsWith("/store/");
  // if (useXrootd and hostname.Contains("t2.ucsd.edu"))
  // {
  //     if (inputs.Contains("/hadoop/cms"))
  //         inputs.ReplaceAll("/hadoop/cms", "root://redirector.t2.ucsd.edu/");
  //     else
  //         inputs.ReplaceAll("/store", "root://redirector.t2.ucsd.edu//store");
  // }
  // else
  if (useXrootd) {
    inputs.ReplaceAll("/store", "root://cmsxrootd.fnal.gov//store");
  }
  std::cout << "inputs : " << inputs.Data() << std::endl;
  for (auto& ff : RooUtil::StringUtil::split(inputs, ",")) {
    TString filepath = ff;
    if (_map.find(ff) != _map.end())
      filepath = _map[ff];
    RooUtil::print(Form("Adding %s", filepath.Data()));
    chain->Add(filepath);
  }
  return chain;
}

TH1* RooUtil::FileUtil::get(TString name) { return (TH1*)gDirectory->Get(name); }

std::map<TString, TH1*> RooUtil::FileUtil::getAllHistograms(TFile* f) {
  std::map<TString, TH1*> hists;
  for (int ikey = 0; ikey < f->GetListOfKeys()->GetEntries(); ++ikey) {
    TString histname = f->GetListOfKeys()->At(ikey)->GetName();
    hists[histname] = (TH1*)f->Get(histname);
  }
  return hists;
}

void RooUtil::FileUtil::saveAllHistograms(std::map<TString, TH1*> allhists, TFile* ofile) {
  ofile->cd();
  for (auto& hist : allhists)
    if (hist.second)
      hist.second->Write();
}

std::vector<TString> RooUtil::FileUtil::getFilePathsInDirectory(TString dirpath) {
  std::vector<TString> rtn;
  DIR* dir;
  struct dirent* ent;
  if ((dir = opendir(dirpath.Data())) != NULL) {
    /* print all the files and directories within directory */
    while ((ent = readdir(dir)) != NULL) {
      if (!TString(ent->d_name).EqualTo(".") && !TString(ent->d_name).EqualTo(".."))
        rtn.push_back(ent->d_name);
    }
    closedir(dir);
    return rtn;
  } else {
    /* could not open directory */
    error(TString::Format("Could not open directory = %s", dirpath.Data()));
    return rtn;
  }
}

//https://stackoverflow.com/questions/8401777/simple-glob-in-c-on-unix-system

std::vector<TString> RooUtil::FileUtil::glob(const std::string& pattern) {
  using namespace std;

  // glob struct resides on the stack
  glob_t glob_result;
  memset(&glob_result, 0, sizeof(glob_result));

  // do the glob operation
  int return_value = glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
  if (return_value != 0) {
    globfree(&glob_result);
    stringstream ss;
    ss << "glob() failed with return_value " << return_value << endl;
    throw std::runtime_error(ss.str());
  }

  // collect all the filenames into a std::list<std::string>
  vector<TString> filenames;
  for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
    filenames.push_back(string(glob_result.gl_pathv[i]));
  }

  // cleanup
  globfree(&glob_result);

  // done
  return filenames;
}
