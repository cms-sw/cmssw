/***********************************************************************************
 * Macro was updated for the purpose of TrackerAlignment group "All-In-One tool"
 * Reference to original macro by Samantha Hewamanage <samanthaATSPAMMENOTfnal.gov>: 
 * https://github.com/hkaushalya/haddws
 * Changes are mostly of informative character and dealing with inconsistent binning
 * @Author: Tomas Kello <tomas DOT kello AT cern.ch>
 ***********************************************************************************/

#include "Riostream.h"
#include "TChain.h"
#include "TFile.h"
#include "TH1.h"
#include "TKey.h"
#include "TProfile.h"
#include "TTree.h"
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <utility>
#include <vector>

using namespace std;

TFile *Target;
typedef vector<pair<TFile *, double> > vec_pair;
typedef vector<pair<TFile *, double> >::const_iterator vec_pair_it;

void MergeRootfile(TDirectory *target, const vector<pair<TFile *, double> > &vFileList);
void loginfo(char log_type, std::string text);

void loginfo(char log_type = 'i', std::string text = "") {
  //**************************************************************************************************************************
  //Logger:
  //    INFO    = Informative text
  //    WARNING = Notify user about unpredictable changes or missing files which do not result in abort
  //    ERROR   = Error in logic results in abort. Can be fixed by user (missing input, settings clash ...)
  //    FATAL   = Fatal error results in abort. Cannot be fixed by user (the way how input is produced has changed or bug ...)
  //**************************************************************************************************************************

  std::string source = "haddws:                    ";
  if (log_type == 'i') {
    std::cout << source << "[INFO]     " << text << std::endl;
  } else if (log_type == 'w') {
    std::cout << source << "[WARNING]  " << text << std::endl;
  } else if (log_type == 'e') {
    std::cout << source << "[ERROR]    " << text << std::endl;
  } else if (log_type == 'f') {
    std::cout << source << "[FATAL]    " << text << std::endl;
  }
  return;
}

/* entry point for commandline executable */
int main(int argc, char *argv[]) {
  if (argc < 2) {
    cout << "\n======================== HADDWS USAGE INFO ===============================================" << endl;
    cout << "       ./haddws file1.root file2.root w1 w2\n" << endl;
    cout << "       w1, w2: are doubleing point numbers." << endl;
    cout << "          Each input file should have a corresponding weight (>0)." << endl;
    cout << "          A weight of 1 will leave those histograms/trees unchanged." << endl;
    cout << "       Files names and weights can be given in any order." << endl;
    cout << "          But the first weight found will be assigned to the first file listed," << endl;
    cout << "          second weight found will be assigned to the second file listed and so on." << endl;
    cout << "       Output file will be named results.root and will be recreated with subseqent" << endl;
    cout << "          executions." << endl;
    cout << "       Return value:" << endl;
    cout << "          0 in any error, 1 upon successful completion." << endl;
    cout << "======================== END USAGE INFO ==================================================\n" << endl;
    return 0;
  }

  vec_pair inputList;
  vector<string> inputFileNames;
  vector<double> inputWeights;

  try {
    for (int i = 1; i < argc; ++i) {
      //const char *argi = argv[i];
      string sargi(argv[i]);

      //check if this a number
      if (sargi.find_first_not_of("1234567890.") == string::npos) {
        const double w = atof(argv[i]);
        inputWeights.push_back(w);
      } else {
        inputFileNames.push_back(sargi);
      }
    }
  } catch (exception &e) {
    cout << e.what() << endl;
  }

  //do some basic sanity checks
  //if no weights are given, just add them. else use the weights.
  if (!inputWeights.empty()) {
    if (inputFileNames.size() != inputWeights.size()) {
      loginfo('e', "Every input root file must have a corresponding weight! please check!");
      return 0;
    }
  }
  //check if all files exists and readable
  for (vector<string>::const_iterator it = inputFileNames.begin(); it != inputFileNames.end(); ++it) {
    TFile *f = new TFile(it->c_str());
    if (f->IsZombie()) {
      cout << "File: " << (*it) << " not found or readable!" << endl;
      return 0;
    }
    f->Close();
    delete f;
  }

  vec_pair vFileList;

  for (unsigned i = 0; i < inputFileNames.size(); ++i) {
    double w = 1.0;
    if (!inputWeights.empty())
      w = inputWeights.at(i);
    vFileList.push_back(make_pair(TFile::Open(inputFileNames.at(i).c_str()), w));

    string msg("");
    msg += "File";
    if (!inputWeights.empty())
      msg += "/Weight";
    msg += " = ";
    msg += vFileList.at(i).first->GetName();
    if (!inputWeights.empty()) {
      stringstream swgt;
      swgt << " <- " << vFileList.at(i).second;
      msg += swgt.str();
    }
    loginfo('i', msg);
  }

  Target = TFile::Open("result.root", "RECREATE");
  MergeRootfile(Target, vFileList);

  //now cleanup. This makes valgrind happy :)
  Target->Close();
  for (vec_pair_it it = vFileList.begin(); it != vFileList.end(); ++it) {
    delete (it->first);
  }

  return 1;
}

void haddws() {
  /**********************************************************
   * in an interactive ROOT session, edit the file names, 
	* corresponding weights, and target name. Then
   * root:> .x haddws.C+
	**********************************************************/

  Target = TFile::Open("result.root", "RECREATE");

  vec_pair vFileList;
  vFileList.push_back(make_pair(TFile::Open("simple1.root"), 1.0));
  vFileList.push_back(make_pair(TFile::Open("simple2.root"), 0.5));

  for (vec_pair_it it = vFileList.begin(); it != vFileList.end(); ++it) {
    cout << "File/weight = " << it->first->GetName() << "/" << it->second << endl;
  }

  MergeRootfile(Target, vFileList);
  Target->Close();

  for (vec_pair_it it = vFileList.begin(); it != vFileList.end(); ++it) {
    delete (it->first);
  }
}

void MergeRootfile(TDirectory *target, const vector<pair<TFile *, double> > &vFileList) {
  //cout << "Target path: " << target->GetPath() << endl;
  TString path((char *)strstr(target->GetPath(), ":"));
  path.Remove(0, 2);

  vec_pair_it it = vFileList.begin();
  TFile *first_source = (*it).first;
  const double first_weight = (*it).second;
  first_source->cd(path);
  TDirectory *current_sourcedir = gDirectory;
  //gain time, do not add the objects in the list in memory
  Bool_t status = TH1::AddDirectoryStatus();
  TH1::AddDirectory(kFALSE);

  // loop over all keys in this directory
  TChain *globChain = nullptr;
  TIter nextkey(current_sourcedir->GetListOfKeys());
  TKey *key, *oldkey = nullptr;
  while ((key = (TKey *)nextkey())) {
    //keep only the highest cycle number for each key
    if (oldkey && !strcmp(oldkey->GetName(), key->GetName()))
      continue;

    // read object from first source file
    first_source->cd(path);
    TObject *obj = key->ReadObj();

    if (obj->IsA()->InheritsFrom(TH1::Class())) {
      // descendant of TH1 -> merge it

      //cout << "Merging histogram " << obj->GetName() << endl;
      TH1 *h1 = (TH1 *)obj;
      if (first_weight > 0) {
        h1->Scale(first_weight);
      }

      // loop over all source files and add the content of the
      // correspondant histogram to the one pointed to by "h1"
      for (vec_pair_it nextsrc = vFileList.begin() + 1; nextsrc != vFileList.end(); ++nextsrc) {
        // make sure we are at the correct directory level by cd'ing to path
        (*nextsrc).first->cd(path);
        const double next_weight = (*nextsrc).second;
        TKey *key2 = (TKey *)gDirectory->GetListOfKeys()->FindObject(h1->GetName());

        //check for the same number of bins and same bin labels, then scale and add
        if (key2) {
          TH1 *h2 = (TH1 *)key2->ReadObj();
          if (next_weight > 0) {
            h2->Scale(next_weight);
          }
          if (h1->GetNbinsX() == h2->GetNbinsX()) {
            h1->Add(h2);
          } else {
            loginfo('w',
                    "Inconsistent binning detected:. Merge will possibly fail for histogram: " +
                        std::string(h2->GetName()));
            h1->Add(h2);
          }
          delete h2;
        }
      }

    } else if (obj->IsA()->InheritsFrom(TTree::Class())) {
      // loop over all source files create a chain of Trees "globChain"
      const char *obj_name = obj->GetName();

      globChain = new TChain(obj_name);
      globChain->Add(first_source->GetName());
      for (vec_pair_it nextsrc = vFileList.begin() + 1; nextsrc != vFileList.end(); ++nextsrc) {
        globChain->Add(nextsrc->first->GetName());
      }

    } else if (obj->IsA()->InheritsFrom(TDirectory::Class())) {
      // it's a subdirectory
      loginfo('i', "Found subdirectory " + std::string(obj->GetName()));

      // create a new subdir of same name and title in the target file
      target->cd();
      TDirectory *newdir = target->mkdir(obj->GetName(), obj->GetTitle());

      // newdir is now the starting point of another round of merging
      // newdir still knows its depth within the target file via
      // GetPath(), so we can still figure out where we are in the recursion
      MergeRootfile(newdir, vFileList);

    } else {
      // object is of no type that we know or can handle
      cout << "Unknown object type, name: " << obj->GetName() << " title: " << obj->GetTitle() << endl;
    }

    // now write the merged histogram (which is "in" obj) to the target file
    // note that this will just store obj in the current directory level,
    // which is not persistent until the complete directory itself is stored
    // by "target->Write()" below
    if (obj) {
      target->cd();

      //!!if the object is a tree, it is stored in globChain...
      if (obj->IsA()->InheritsFrom(TTree::Class()))
        globChain->Merge(target->GetFile(), 0, "keep");
      else
        obj->Write(key->GetName());
    }

  }  // while ( ( TKey *key = (TKey*)nextkey() ) )

  // save modifications to target file
  target->SaveSelf(kTRUE);
  TH1::AddDirectory(status);
}
