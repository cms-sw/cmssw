using namespace ROOT;
using ROOT::RDF::RNode;
using floats = ROOT::VecOps::RVec<float>;
using ints = ROOT::VecOps::RVec<int>;
using bools = ROOT::VecOps::RVec<bool>;
using chars = ROOT::VecOps::RVec<UChar_t>;
using doubles = ROOT::VecOps::RVec<double>;

vector<float> HitResolutionVector;
vector<float> DoubleDifferenceVector;
vector<float> HitDXVector;
vector<float> TrackDXVector;
vector<float> TrackDXEVector;

std::string InputFileString;
std::string HitResoFileName;
std::string GaussianFitsFileName;

void ResolutionsCalculator(const string& region, const int& Unit_Int, const int& UL) {
  std::string CutFlowReportString;
  std::string DoubleDiffString;
  std::string HitDXString;
  std::string TrackDXString;
  std::string TrackDXEString;
  std::string ClusterW1String = "clusterW1";
  std::string ClusterW2String = "clusterW2";

  switch (UL) {
    case 0:
      switch (Unit_Int) {
        case 0:
          GaussianFitsFileName = "GaussianFits_PitchUnits_ALCARECO.root";
          HitResoFileName = "HitResolutionValues_PitchUnits_ALCARECO.txt";
          CutFlowReportString = "CutFlowReport_" + region + "_PitchUnits_ALCARECO.txt";
          DoubleDiffString = "hitDX_OverPitch-trackDX_OverPitch";
          HitDXString = "hitDX_OverPitch";
          TrackDXString = "trackDX_OverPitch";
          TrackDXEString = "trackDXE_OverPitch";
          break;

        case 1:
          GaussianFitsFileName = "GaussianFits_Centimetres_ALCARECO.root";
          HitResoFileName = "HitResolutionValues_Centimetres_ALCARECO.txt";
          CutFlowReportString = "CutFlowReport_" + region + "_Centimetres_ALCARECO.txt";
          DoubleDiffString = "hitDX-trackDX";
          HitDXString = "hitDX";
          TrackDXString = "trackDX";
          TrackDXEString = "trackDXE";
          break;

        default:
          std::cout << "ERROR: UnitInt must be 0 or 1." << std::endl;
          break;
      }

      InputFileString = "hitresol_ALCARECO.root";
      break;

    case 1:
      switch (Unit_Int) {
        case 0:
          GaussianFitsFileName = "GaussianFits_PitchUnits_ALCARECO_UL.root";
          HitResoFileName = "HitResolutionValues_PitchUnits_ALCARECO_UL.txt";
          CutFlowReportString = "CutFlowReport_" + region + "_PitchUnits_ALCARECO_UL.txt";
          DoubleDiffString = "hitDX_OverPitch-trackDX_OverPitch";
          HitDXString = "hitDX_OverPitch";
          TrackDXString = "trackDX_OverPitch";
          TrackDXEString = "trackDXE_OverPitch";
          break;

        case 1:
          GaussianFitsFileName = "GaussianFits_Centimetres_ALCARECO_UL.root";
          HitResoFileName = "HitResolutionValues_Centimetres_ALCARECO_UL.txt";
          CutFlowReportString = "CutFlowReport_" + region + "_Centimetres_ALCARECO_UL.txt";
          DoubleDiffString = "hitDX-trackDX";
          HitDXString = "hitDX";
          TrackDXString = "trackDX";
          TrackDXEString = "trackDXE";
          break;

        default:
          std::cout << "ERROR: UnitInt must be 0 or 1." << std::endl;
          break;
      }

      InputFileString = "hitresol_ALCARECO_UL.root";
      break;
    default:
      std::cout << "The UL input parameter must be set to 0 (for ALCARECO) or 1 (for UL ALCARECO)." << std::endl;
      break;
  }

  //opening the root file
  ROOT::RDataFrame d("anResol/reso", InputFileString);

  int RegionInt = 0;

  if (region == "TIB_L1") {
    RegionInt = 1;
  } else if (region == "TIB_L2") {
    RegionInt = 2;
  } else if (region == "TIB_L3") {
    RegionInt = 3;
  } else if (region == "TIB_L4") {
    RegionInt = 4;
  } else if (region == "Side_TID") {
    RegionInt = 5;
  } else if (region == "Wheel_TID") {
    RegionInt = 6;
  } else if (region == "Ring_TID") {
    RegionInt = 7;
  } else if (region == "TOB_L1") {
    RegionInt = 8;
  } else if (region == "TOB_L2") {
    RegionInt = 9;
  } else if (region == "TOB_L3") {
    RegionInt = 10;
  } else if (region == "TOB_L4") {
    RegionInt = 11;
  } else if (region == "TOB_L5") {
    RegionInt = 12;
  } else if (region == "TOB_L6") {
    RegionInt = 13;
  } else if (region == "Side_TEC") {
    RegionInt = 14;
  } else if (region == "Wheel_TEC") {
    RegionInt = 15;
  } else if (region == "Ring_TEC") {
    RegionInt = 16;
  } else if (region == "TIB_All") {
    RegionInt = 17;
  } else if (region == "TOB_All") {
    RegionInt = 18;
  } else if (region == "TID_All") {
    RegionInt = 19;
  } else if (region == "TEC_All") {
    RegionInt = 20;
  } else if (region == "Pixel_Barrel") {
    RegionInt = 21;
  } else if (region == "Pixel_EndcapDisk") {
    RegionInt = 22;
  } else {
    std::cout << "Error: The tracker region " << region
              << " was chosen. Please choose a region out of: TIB L1, TIB L2, TIB L3, TIB L4, Side TID, Wheel TID, "
                 "Ring TID, TOB L1, TOB L2, TOB L3, TOB L4, TOB L5, TOB L6, Side TEC, Wheel TEC or Ring TEC."
              << std::endl;
    return 0;
  }

  //Lambda function to filter the detID for different layers
  auto SubDet_Function{[&RegionInt](const int& detID1_input, const int& detID2_input) {
    bool OutputBool = 0;

    switch (RegionInt) {
      case 1: {
        OutputBool = (((detID1_input >> 25) & 0x7) == 3) && ((detID1_input >> 14) & 0x7) == 1 &&
                     (((detID2_input >> 25) & 0x7) == 3) && ((detID2_input >> 14) & 0x7) == 1;  //TIB L1
        break;
      }

      case 2: {
        OutputBool = (((detID1_input >> 25) & 0x7) == 3) && (((detID1_input >> 14) & 0x7) == 2) &&
                     (((detID2_input >> 25) & 0x7) == 3) && (((detID2_input >> 14) & 0x7) == 2);  //TIB L2
        break;
      }

      case 3: {
        OutputBool = (((detID1_input >> 25) & 0x7) == 3) && (((detID1_input >> 14) & 0x7) == 3) &&
                     (((detID2_input >> 25) & 0x7) == 3) && (((detID2_input >> 14) & 0x7) == 3);  //TIB L3
        break;
      }

      case 4: {
        OutputBool = (((detID1_input >> 25) & 0x7) == 3) && (((detID1_input >> 14) & 0x7) == 4) &&
                     (((detID2_input >> 25) & 0x7) == 3) && (((detID2_input >> 14) & 0x7) == 4);  //TIB L4
        break;
      }

      case 5: {
        OutputBool = ((((detID1_input >> 13) & 0x3) == 1) && (((detID2_input >> 13) & 0x3) == 1)) ||
                     ((((detID1_input >> 13) & 0x3) == 2) &&
                      (((detID2_input >> 13) & 0x3) == 2));  //TID Side (1 -> TID-, 2 -> TID+)

        break;
      }

      case 6: {
        OutputBool = (((detID1_input >> 11) & 0x3) == 2) && (((detID2_input >> 11) & 0x3) == 2);  //TID Wheel

        break;
      }

      case 7: {
        OutputBool = ((((detID1_input >> 9) & 0x3) == 2) && (((detID2_input >> 9) & 0x3) == 2));  //TID Ring

        break;
      }

      case 8: {
        OutputBool = (((detID1_input >> 25) & 0x7) == 5) && (((detID1_input >> 14) & 0x7) == 1) &&
                     (((detID2_input >> 25) & 0x7) == 5) && (((detID2_input >> 14) & 0x7) == 1);  //TOB L1
        break;
      }

      case 9: {
        OutputBool = (((detID1_input >> 25) & 0x7) == 5) && (((detID1_input >> 14) & 0x7) == 2) &&
                     (((detID2_input >> 25) & 0x7) == 5) && (((detID2_input >> 14) & 0x7) == 2);  //TOB L2
        break;
      }

      case 10: {
        OutputBool = (((detID1_input >> 25) & 0x7) == 5) && (((detID1_input >> 14) & 0x7) == 3) &&
                     (((detID2_input >> 25) & 0x7) == 5) && (((detID2_input >> 14) & 0x7) == 3);  //TOB L3
        break;
      }

      case 11: {
        OutputBool = (((detID1_input >> 25) & 0x7) == 5) && (((detID1_input >> 14) & 0x7) == 4) &&
                     (((detID2_input >> 25) & 0x7) == 5) && (((detID2_input >> 14) & 0x7) == 4);  //TOB L4
        break;
      }

      case 12: {
        OutputBool = (((detID1_input >> 25) & 0x7) == 5) && (((detID1_input >> 14) & 0x7) == 5) &&
                     (((detID2_input >> 25) & 0x7) == 5) && (((detID2_input >> 14) & 0x7) == 5);  //TOB L5
        break;
      }

      case 13: {
        OutputBool = (((detID1_input >> 25) & 0x7) == 5) && (((detID1_input >> 14) & 0x7) == 6) &&
                     (((detID2_input >> 25) & 0x7) == 5) && (((detID2_input >> 14) & 0x7) == 6);  //TOB L6
        break;
      }

      case 14: {
        OutputBool = ((((detID1_input >> 18) & 0x3) == 1) && (((detID2_input >> 18) & 0x3) == 1)) ||
                     ((((detID1_input >> 18) & 0x3) == 2) &&
                      (((detID2_input >> 18) & 0x3) == 2));  //Side TEC (1 -> back, 2 -> front)
        break;
      }

      case 15: {
        OutputBool = (((detID1_input >> 14) & 0xF) == 4) && (((detID2_input >> 14) & 0xF) == 4);  //Wheel TEC
        break;
      }

      case 16: {
        OutputBool = (((detID1_input >> 5) & 0x7) == 3) && (((detID2_input >> 5) & 0x7) == 3);  //Ring TEC

        break;
      }

      case 17: {
        OutputBool = ((((detID1_input >> 25) & 0x7) == 3) && (((detID2_input >> 25) & 0x7) == 3));  //All TIB

        break;
      }

      case 18: {
        OutputBool = ((((detID1_input >> 25) & 0x7) == 5) && (((detID2_input >> 25) & 0x7) == 5));  //All TOB

        break;
      }

      case 19: {
        OutputBool = ((((detID1_input >> 13) & 0x3) == 1) && (((detID2_input >> 13) & 0x7) == 1)) ||
                     ((((detID1_input >> 13) & 0x3) == 2) && (((detID2_input >> 13) & 0x7) == 2)) ||
                     ((((detID1_input >> 11) & 0x3) == 2) && (((detID2_input >> 11) & 0x3) == 2)) ||
                     ((((detID1_input >> 9) & 0x3) == 2) && (((detID2_input >> 9) & 0x3) == 2)) ||
                     ((((detID1_input >> 7) & 0x3) == 1) && (((detID2_input >> 7) & 0x3) == 1)) ||
                     ((((detID1_input >> 7) & 0x3) == 2) && (((detID2_input >> 7) & 0x3) == 2)) ||
                     ((((detID1_input >> 2) & 0x1F) == 5) && (((detID2_input >> 2) & 0x1F) == 5)) ||
                     ((((detID1_input >> 0) & 0x3) == 0) && (((detID2_input >> 0) & 0x3) == 0)) ||
                     ((((detID1_input >> 0) & 0x3) == 1) && (((detID2_input >> 0) & 0x3) == 1)) ||
                     ((((detID1_input >> 0) & 0x3) == 2) && (((detID2_input >> 0) & 0x3) == 2));  //All TID

        break;
      }

      case 20: {
        OutputBool = ((((detID1_input >> 18) & 0x3) == 1) && (((detID2_input >> 18) & 0x3) == 1)) ||
                     ((((detID1_input >> 18) & 0x3) == 2) && (((detID2_input >> 18) & 0x3) == 2)) ||
                     ((((detID1_input >> 14) & 0xF) == 4) && (((detID2_input >> 14) & 0xF) == 4)) ||
                     ((((detID1_input >> 12) & 0x3) == 1) && (((detID2_input >> 12) & 0x3) == 1)) ||
                     ((((detID1_input >> 12) & 0x3) == 2) && (((detID2_input >> 12) & 0x3) == 2)) ||
                     ((((detID1_input >> 8) & 0xF) == 4) && (((detID2_input >> 8) & 0xF) == 4)) ||
                     ((((detID1_input >> 5) & 0x7) == 3) && (((detID2_input >> 5) & 0x7) == 3)) ||
                     ((((detID1_input >> 2) & 0x7) == 3) && (((detID2_input >> 2) & 0x7) == 3)) ||
                     ((((detID1_input >> 0) & 0x3) == 1) && (((detID2_input >> 0) & 0x3) == 1)) ||
                     ((((detID1_input >> 0) & 0x3) == 2) && (((detID2_input >> 0) & 0x3) == 2)) ||
                     ((((detID1_input >> 0) & 0x3) == 3) && (((detID2_input >> 0) & 0x3) == 3));  //All TEC

        break;
      }

      case 21: {
        OutputBool =
            (((detID1_input >> 20) & 0xF) == 4) && (((detID2_input >> 20) & 0xF) == 4);  //pixel barrel (phase 1)
        break;
      }

      case 22: {
        OutputBool =
            (((detID1_input >> 18) & 0xF) == 4) && (((detID2_input >> 18) & 0xF) == 4);  //pixel endcap disk (phase 1)
        break;
      }
    }

    return OutputBool;
  }};

  //Function for expressing the hit resolution in either micrometres or pitch units.
  auto Pitch_Function{[&Unit_Int](const float& pitch, const float& input) {
    float InputOverPitch = input / pitch;
    return InputOverPitch;
  }};

  //Defining columns needed for the unit conversion into pitch units, and applying the filter for the subdetector
  auto dataframe = d.Define("hitDX_OverPitch", Pitch_Function, {"pitch1", "hitDX"})
                       .Define("trackDX_OverPitch", Pitch_Function, {"pitch1", "trackDX"})
                       .Define("trackDXE_OverPitch", Pitch_Function, {"pitch1", "trackDXE"})
                       .Filter(SubDet_Function, {"detID1", "detID2"}, "Subdetector filter");

  //Implementing selection criteria that were not implemented in HitResol.cc
  auto PairPathCriteriaFunction{[&RegionInt](const float& pairPath_input) {
    if ((RegionInt > 0 && RegionInt < 5) || (RegionInt > 7 || RegionInt < 13) || (RegionInt == 17) ||
        (RegionInt == 18)) {
      return abs(pairPath_input) < 7;
    }  //for TIB and TOB
    else if (RegionInt == 21 || RegionInt == 22) {
      return abs(pairPath_input) < 2;
    }  //for pixels
    else {
      return abs(pairPath_input) < 20;
    }  //for everything else (max value is 15cm so this will return all events anyway)
  }};

  auto MomentaFunction{[&RegionInt](const float& momentum_input) {
    if (RegionInt == 21 || RegionInt == 22) {
      return momentum_input > 5;
    }  //pixels
    else {
      return momentum_input > 15;
    }  //strips
  }};

  auto dataframe_filtered =
      dataframe.Filter(PairPathCriteriaFunction, {"pairPath"}, "Pair path criterion filter")
          .Filter(MomentaFunction, {"momentum"}, "Momentum criterion filter")
          .Filter("trackChi2 > 0.001", "chi2 criterion filter")
          .Filter("numHits > 6", "numHits filter")
          .Filter("trackDXE < 0.0025", "trackDXE filter")
          .Filter("(clusterW1 == clusterW2) && clusterW1 <= 4 && clusterW2 <= 4", "cluster filter");

  //Creating histograms for the difference between the two hit positions, the difference between the two predicted positions and for the double difference
  //hitDX = the difference in the hit positions for the pair
  //trackDX =  the difference in the track positions for the pair

  auto HistoName_DoubleDiff = "DoubleDifference_" + region;
  auto HistoName_HitDX = "HitDX_" + region;
  auto HistoName_TrackDX = "TrackDX_" + region;
  auto HistoName_TrackDXE = "TrackDXE_" + region;
  auto HistoName_ClusterW1 = "ClusterW1_" + region;
  auto HistoName_ClusterW2 = "ClusterW2_" + region;

  auto h_DoubleDifference =
      dataframe_filtered.Define(HistoName_DoubleDiff, DoubleDiffString)
          .Histo1D({HistoName_DoubleDiff.c_str(), HistoName_DoubleDiff.c_str(), 40, -0.5, 0.5}, HistoName_DoubleDiff);
  auto h_hitDX = dataframe_filtered.Define(HistoName_HitDX, HitDXString).Histo1D(HistoName_HitDX);
  auto h_trackDX = dataframe_filtered.Define(HistoName_TrackDX, TrackDXString).Histo1D(HistoName_TrackDX);
  auto h_trackDXE = dataframe_filtered.Define(HistoName_TrackDXE, TrackDXEString).Histo1D(HistoName_TrackDXE);

  auto h_clusterW1 = dataframe_filtered.Define(HistoName_ClusterW1, ClusterW1String).Histo1D(HistoName_ClusterW1);
  auto h_clusterW2 = dataframe_filtered.Define(HistoName_ClusterW2, ClusterW2String).Histo1D(HistoName_ClusterW2);

  //Applying gaussian fits, taking the resolutions and squaring them
  h_DoubleDifference->Fit("gaus");

  auto double_diff_StdDev = h_DoubleDifference->GetStdDev();
  auto hitDX_StdDev = h_hitDX->GetStdDev();
  auto trackDX_StdDev = h_trackDX->GetStdDev();
  auto trackDXE_Mean = h_trackDXE->GetMean();

  auto sigma2_MeasMinusPred = pow(double_diff_StdDev, 2);
  auto sigma2_Meas = pow(hitDX_StdDev, 2);
  auto sigma2_Pred = pow(trackDX_StdDev, 2);
  auto sigma2_PredError = pow(trackDXE_Mean, 2);

  DoubleDifferenceVector.push_back(sigma2_MeasMinusPred);
  HitDXVector.push_back(sigma2_Meas);
  TrackDXVector.push_back(sigma2_Pred);
  TrackDXEVector.push_back(sigma2_PredError);

  //Saving the histograms with gaussian fits applied to an output root file
  TFile* output = new TFile(GaussianFitsFileName.c_str(), "UPDATE");

  h_DoubleDifference->Write();
  h_hitDX->Write();
  h_trackDX->Write();
  h_trackDXE->Write();
  h_clusterW1->Write();
  h_clusterW2->Write();

  output->Close();

  //Calculating the hit resolution;
  auto numerator = sigma2_MeasMinusPred - sigma2_PredError;

  auto HitResolution = sqrt(numerator / 2);
  HitResolutionVector.push_back(HitResolution);

  //Printing the resolution
  std::cout << '\n' << std::endl;
  std::cout << "The hit resolution for tracker region " << region << " is: " << HitResolution << std::endl;
  std::cout << '\n' << std::endl;

  //Cut flow report
  auto allCutsReport = d.Report();
  std::ofstream CutFlowReport;

  CutFlowReport.open(CutFlowReportString.c_str());

  for (auto&& cutInfo : allCutsReport) {
    CutFlowReport << cutInfo.GetName() << '\t' << cutInfo.GetAll() << '\t' << cutInfo.GetPass() << '\t'
                  << cutInfo.GetEff() << " %" << std::endl;
  }
}

void Resolutions() {
  int UnitInteger = 0;
  int ULInteger = 0;

  vector<std::string> LayerNames = {"TIB_L1",   "TIB_L2",   "TIB_L3",       "TIB_L4",          "Side_TID", "Wheel_TID",
                                    "Ring_TID", "TOB_L1",   "TOB_L2",       "TOB_L3",          "TOB_L4",   "TOB_L5",
                                    "TOB_L6",   "Side_TEC", "Wheel_TEC",    "Ring_TEC",        "TIB_All",  "TOB_All",
                                    "TID_All",  "TEC_All",  "Pixel_Barrel", "Pixel_EndcapDisk"};

  for (int i = 0; i < LayerNames.size(); i++) {
    ResolutionsCalculator(LayerNames.at(i), UnitInteger, ULInteger);
  }

  std::ofstream HitResoTextFile;
  HitResoTextFile.open(HitResoFileName);

  auto Width = 28;

  HitResoTextFile << std::right << "Layer " << std::setw(Width) << " Resolution " << std::setw(Width)
                  << " sigma2_HitDX " << std::setw(Width) << " sigma2_trackDX " << std::setw(Width)
                  << " sigma2_trackDXE " << std::setw(Width) << " sigma2_DoubleDifference " << std::endl;

  for (int i = 0; i < HitResolutionVector.size(); i++) {
    HitResoTextFile << std::right << LayerNames.at(i) << std::setw(Width) << HitResolutionVector.at(i)
                    << std::setw(Width) << HitDXVector.at(i) << std::setw(Width) << TrackDXVector.at(i)
                    << std::setw(Width) << TrackDXEVector.at(i) << std::setw(Width) << DoubleDifferenceVector.at(i)
                    << std::endl;
  }

  system(
      "mkdir HitResolutionValues; mkdir GaussianFits; mkdir CutFlowReports; mv CutFlowReport_* CutFlowReports/; mv "
      "HitResolutionValues_* HitResolutionValues/; mv GaussianFits_* GaussianFits/;");
}
