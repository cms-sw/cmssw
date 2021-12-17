#include "plotting/PlotValidation.cpp+"

void runValidation(const TString& test = "",
                   const Bool_t cmsswComp = false,
                   const int algo = 0,
                   const Bool_t mvInput = true,
                   const Bool_t rmSuffix = true,
                   const Bool_t saveAs = false,
                   const TString& image = "pdf") {
  // PlotValidation arguments
  // First is additional input name of root file
  // Second is name of output directory
  // First boolean argument is to do special CMSSW validation
  // The second boolean argument == true to move input root file to output directory, false to keep input file where it is.
  // Third Bool is saving the image files
  // Last argument is output type of plots

  PlotValidation Val(Form("valtree%s.root", test.Data()),
                     Form("validation%s", test.Data()),
                     cmsswComp,
                     algo,
                     mvInput,
                     rmSuffix,
                     saveAs,
                     image);
  Val.Validation(algo);
}
