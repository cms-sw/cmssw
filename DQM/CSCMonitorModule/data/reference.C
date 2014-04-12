
#include <vector>
#include <iostream>
#include <string>

bool chamberExists(const unsigned int x, const unsigned int y) {

  if (y == 1) return false;

  if (y == 3 || y == 5 || (y >= 7 && y <= 12) || y == 14 || y == 16) {
    if (x >= 1 && x <= 36) return true;
    return false;
  }

  if (y == 2 || y == 4 || y == 6 || y == 13 || y == 15 || y == 17) {
    if (x >= 1 && x <= 18) return true;
    return false;
  }

  if (y == 18) {
    if (x >= 9 && x <= 13) return true;
    return false;
  }

  return false;

}

void splitString(const std::string& str, const std::string& delim, std::vector<std::string>& results) {
  std::string lstr = str;
  unsigned int cutAt;
  while ((cutAt = lstr.find_first_of(delim)) != lstr.npos) {
    if(cutAt > 0) {
      results.push_back(lstr.substr(0, cutAt));
    }
    lstr = lstr.substr(cutAt + 1);
  }
  if(lstr.length() > 0) {
    results.push_back(lstr);
  }
}

void fillCellsWithRowNeighbourAverage(TH2* h2) {
  double lastNotZeroValue = 0.0;
  std::vector<unsigned int> zeros;

  // Looping through Y-coordinate
  for (unsigned int y = 1; y <= 18; y++) {

    // Set Zeros (we consider only rows!)
    lastNotZeroValue = 0.0;
    zeros.clear();

    // Looping through X-coordinate
    for (unsigned int x = 1; x <= 36; x++) {
      if (chamberExists(x, y)) {
        double v = h2->GetBinContent(x, y);
      
        if (v == 0.0) {
          // Put another element to zeros vector
          zeros.push_back(x);
        } else {
          // If there already were empty cells - fill 'em with averages
          double aver = (lastNotZeroValue > 0.0 ? (lastNotZeroValue  + v) / 2.0 : v);
          for (unsigned int i = 0; i < zeros.size(); i++) {
            h2->SetBinContent(zeros.at(i), y, aver);
          }
          // Clean vector
          zeros.clear();
          // Save as last non zero
          lastNotZeroValue = v;
        }
      }
    }

    // If there were empty cells - fill 'em with last non-zero
    if (lastNotZeroValue != 0.0) {
      for (unsigned int i = 0; i < zeros.size(); i++) {
        h2->SetBinContent(zeros.at(i), y, lastNotZeroValue);
      }
    }

  }
}

void fillCellsWithRowAverage(TH2* h2) {

  std::vector<double> averages;
  double items = 0;

  // Looping through Y-coordinate
  for (unsigned int y = 1; y <= 18; y++) {

    averages.push_back(0.0);
    double items = 0;

    // Looping through X-coordinate
    for (unsigned int x = 1; x <= 36; x++) {
      if (chamberExists(x, y)) {
        double v = h2->GetBinContent(x, y);
        if (v != 0.0) {
          averages[y - 1] += v;
          items += 1;
        }
      }
    }
    averages[y - 1] /= items;
  }

  // Write row averages to histogram

  for (unsigned int y = 1; y <= 18; y++) {
    for (unsigned int x = 1; x <= 36; x++) {
      if (chamberExists(x, y)) {
        h2->SetBinContent(x, y, averages[y - 1]);
      }
    }
  }

}

void copy(const std::string& src_file, const std::string& src_obj, const std::string& des_file, const std::string& des_path, const std::string& des_obj) {

  gStyle->SetPalette(1,0);

  TFile *old_file = new TFile(src_file.c_str());
  if (!old_file) {
    std::cerr << "Can not open file: " << src_file << std::endl;
    return;
  }

  TObject* old_obj = old_file->Get(src_obj.c_str());
  if (!old_obj) {
    std::cerr << "Can not open object: " << src_obj << " in file " << src_file << std::endl;
    return;
  }

  TFile *new_file = new TFile(des_file.c_str(), "RECREATE");
  TDirectory *cdto = 0;

  std::vector<std::string> tokens;
  splitString(des_path, "/", tokens);
  for (unsigned int i = 0; i < tokens.size(); i++) {
    if (cdto == 0) {
      cdto = new_file->mkdir(tokens.at(i).c_str());
    } else {
      cdto = cdto->mkdir(tokens.at(i).c_str());
    }
    cdto->cd();
  }

  TObject *new_obj = dynamic_cast<TObject*> (old_obj->Clone(des_obj.c_str()));

  if (new_obj->IsA()->InheritsFrom("TH2")) {
    TH2* h2 = (TH2*) new_obj;
    if (h2) {
      if(h2->GetXaxis()->GetXmin() <= 1 && h2->GetXaxis()->GetXmax() >= 36 &&
         h2->GetYaxis()->GetXmin() <= 1 && h2->GetYaxis()->GetXmax() >= 18) {

        //fillCellsWithRowNeighbourAverage(h2);
        fillCellsWithRowAverage(h2);
        h2->Draw("colz");

      }
    }
  }

  new_file->Write();

}

void reference() {

  copy("DQM_V0001_CSC_R000132601.root", 
       "DQMData/Run 132601/CSC/Run summary/Summary/CSC_Reporting", 
       "csc_reference.root", 
       "DQMData/Run 132601/CSC/Run summary/Summary/CSC_Reporting", 
       "CSC_Reporting");

}
