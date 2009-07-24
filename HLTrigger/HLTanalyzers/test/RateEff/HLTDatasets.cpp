/**
	@file					HLTDatasets.cpp
	@author				Sue Ann Koay (sakoay@cern.ch)
*/

#include <iostream>
#include <iomanip>

#include <TMath.h>
#include <TAxis.h>
#include <TSystem.h>
#include <TH1.h>
#include <TFile.h>

#include "HLTDatasets.h"




//=============================================================================
// Miscellaneous Functions
//=============================================================================

/// Computes the trigger rate, given the per-event trigger efficiency.
Double_t toRate ( Double_t    collisionRate       ///< Rate of bunch crossings, for onverting numbers of events into rates.
                , Double_t    mu                  ///< bunchCrossingTime * crossSection * instantaneousLuminosity * maxFilledBunches / nFilledBunches
                , UInt_t      numPassedEvents     ///< Number of events that passed, to be used as the numerator in the efficiency calculation.
                , UInt_t      numProcessedEvents  ///< Total number of events processed, to be used as the denominator in the efficiency calculation.
                )
{
  if (numProcessedEvents < 1) return 0;
  ////std::cout << std::setprecision(15)                                               << std::endl;
  ////std::cout << "  ---  numPassedEvents    = " << numPassedEvents                   << std::endl;
  ////std::cout << "  ---  numProcessedEvents = " << numProcessedEvents                << std::endl;
  ////std::cout << "  ---  exp(-mu eff)       = " << TMath::Exp(-mu * numPassedEvents / numProcessedEvents)      << std::endl;
  ////std::cout << "  ---  rate               = " << collisionRate * (1 - TMath::Exp(- mu * numPassedEvents / numProcessedEvents)) << std::endl;
  return collisionRate * (1 - TMath::Exp(- mu * numPassedEvents / numProcessedEvents));
}


/**
  Computes the statistical uncertainty (squared) on the trigger rate, given the per-event 
  trigger efficiency. The rate uncertainty is assumed to be purely due to the binomial 
  uncertainty on the evaluation of the trigger efficiency.
*/
Double_t toRateUncertainty2 ( Double_t    collisionRate       ///< Rate of bunch crossings, for onverting numbers of events into rates.
                            , Double_t    mu                  ///< bunchCrossingTime * crossSection * instantaneousLuminosity * maxFilledBunches / nFilledBunches
                            , UInt_t      numPassedEvents     ///< Number of events that passed, to be used as the numerator in the efficiency calculation.
                            , UInt_t      numProcessedEvents  ///< Total number of events processed, to be used as the denominator in the efficiency calculation.
                            )
{
  if (numProcessedEvents < 1)       return 0;
  Double_t    efficiency      = 1.0 * numPassedEvents / numProcessedEvents;
  Double_t    binomialError2  = efficiency * (1 - efficiency) / numProcessedEvents;
  Double_t    factor          = collisionRate * mu;
  ////std::cout << std::setprecision(15)                                               << std::endl;
  ////std::cout << "  ---  numPassedEvents    = " << numPassedEvents                   << std::endl;
  ////std::cout << "  ---  numProcessedEvents = " << numProcessedEvents                << std::endl;
  ////std::cout << "  ---  efficiency         = " << efficiency                        << std::endl;
  ////std::cout << "  ---  binomialError2     = " << binomialError2                    << std::endl;
  ////std::cout << "  ---  factor             = " << factor                            << std::endl;
  ////std::cout << "  ---  exp(-2 mu eff)     = " << TMath::Exp(- 2 * mu * efficiency) << std::endl;
  return factor * factor * binomialError2 * TMath::Exp(- 2 * mu * efficiency);
}

/**
  Computes the uncertainty on a ratio numerator / denominator.
*/
Double_t ratioError ( Double_t    numerator           ///< The numerator of the ratio.
                    , Double_t    numeratorError2     ///< The squared error on the numerator.
                    , Double_t    denominator         ///< The denominator of the ratio.
                    , Double_t    denominatorError2   ///< The squared error on the denominator.
                    )
{
  if (denominator < 1e-300)         return 0;
  Double_t    ratio   = numerator / denominator;
  return TMath::Sqrt(numeratorError2 + ratio*ratio*denominatorError2) / denominator;
}

/// I don't know why TString::Strip() sometimes doesn't work right for me...
TString& strip(TString& string) 
{
  Int_t   last  = string.Length();  while (--last  > 0    && isspace(string[last]))  ;
  Int_t   first = -1;               while (++first < last && isspace(string[first])) ;
  string        = string(first, last - first + 1);
  return string;
}

/// Finds the index of an item in the given list.
template<typename Type1, typename Type2>
Int_t indexOf(const std::vector<Type1>& whereToLook, const Type2& whatToFind)
{
  Int_t   index = static_cast<Int_t>(whereToLook.size());
  while (--index >= 0 && whereToLook[index] != whatToFind)  ;
  return index;
}

/// Some escapes to convert to good latex. Mostly for trigger names.
TString latexEscape(const Char_t string[])
{
  TString   latex = string;
  latex.ReplaceAll("_","\\_");
  return latex;
}

/**
  Gets the number of decimal places in order to report a number with the given
  number of non-zero digits.
*/
Int_t decimalPlaces(Double_t number, Int_t nonzeroDigits = 2) 
{
  if (number <  0)    return decimalPlaces(-number, nonzeroDigits);
  if (number == 0)    return nonzeroDigits;

  Int_t     decimals  = nonzeroDigits - 1;
  while (number < 1) {
    ++decimals;
    number *= 10;
  }
  return decimals;
}



//=============================================================================
// Dataset
//=============================================================================

Dataset& Dataset::operator+=(const Dataset& addend)
{
  rate                     += addend.rate;
  rateUncertainty2         += addend.rateUncertainty2;
  const UInt_t              numDatasets  = datasetIndices.size();
  if (numDatasets) {
    for (UInt_t iSet = 0; iSet < numDatasets; ++iSet) {
      addedRate         [iSet]      += addend.addedRate         [iSet];
      addedUncertainty2 [iSet]      += addend.addedUncertainty2 [iSet];
    } // end loop over compared datasets
  }
  else {                    // Nothing yet, so this is a copy
    datasetIndices          = addend.datasetIndices       ;
    addedRate               = addend.addedRate            ;
    addedUncertainty2       = addend.addedUncertainty2    ;
  }

  return *this;
}

Bool_t Dataset::checkEvent(const std::vector<Int_t>& triggerBit)
{
  pass      = kFALSE;
  const UInt_t    numTriggers = size();
  for (UInt_t iTrig = 0; iTrig < numTriggers; ++iTrig) {
    ////std::cout << operator[](iTrig).index << ":" << triggerBit[operator[](iTrig).index] << " ";
    if (triggerBit[operator[](iTrig).index] == 1) {
      pass  = kTRUE;
      ++numEventsPassed;
      break;
    }
  } // end loop over trigger bits
  return pass;
}

void Dataset::checkAddition(const std::vector<Dataset>& datasets)
{
  if (pass) {       // Only adds something if it passed!
    const UInt_t    numDatasets = datasetIndices.size();
    for (UInt_t iSet = 0; iSet < numDatasets; ++iSet)
      if (!datasets[datasetIndices[iSet]].pass)   ++numEventsAdded[iSet];
  }
}

void Dataset::setup(const std::vector<Dataset>& datasets)
{
  const UInt_t      numDatasets = datasets.size();
  for (UInt_t iSet = 0; iSet < numDatasets; ++iSet) {
    if (datasets[iSet].isNewTrigger)  continue;
    if (datasets[iSet].name == name)  continue;     // Don't count self
    numEventsAdded    .push_back(0);
    addedRate         .push_back(0);
    addedUncertainty2 .push_back(0);
    datasetIndices    .push_back(iSet);
  } // end loop over datasets
}


void Dataset::computeRate(Double_t collisionRate, Double_t mu, UInt_t numProcessedEvents)
{
  rate                = toRate            (collisionRate, mu, numEventsPassed, numProcessedEvents);
  rateUncertainty2    = toRateUncertainty2(collisionRate, mu, numEventsPassed, numProcessedEvents);
  const UInt_t        numDatasets  = datasetIndices.size();
  for (UInt_t iSet = 0; iSet < numDatasets; ++iSet) {
    addedRate         [iSet]       = toRate            (collisionRate, mu, numEventsAdded[iSet], numProcessedEvents);
    addedUncertainty2 [iSet]       = toRateUncertainty2(collisionRate, mu, numEventsAdded[iSet], numProcessedEvents);
    ////std::cout << "  +   " << numEventsAdded[iSet] << " -> " << addedRate[iSet] << " +/- " << TMath::Sqrt(addedUncertainty2[iSet]) << std::endl;
  } // end loop over compared datasets
  ////std::cout << "  +   " << name << " : " << numEventsPassed << " -> " << rate << std::endl;
}


void Dataset::report(std::ofstream& output, const std::vector<Dataset>& datasets, const Char_t* errata, 
                     const Int_t significantDigits) const
{
  Double_t        rateErr     = TMath::Sqrt(rateUncertainty2);
  Int_t           decimals    = decimalPlaces(rateErr);

  //.. Table formatting .......................................................
  output << "\\begin{longtable}{|l|c|c|c|}" << std::endl;

  output << "\\multicolumn{4}{l}{";
  if (errata)     output << errata;
  output << "Stand-alone rate of " << latexEscape(name) << " is ";
  output << TString::Format("%.*f", decimals, rate)     << "~$\\pm$~";
  output << TString::Format("%.*f", decimals, rateErr)  << " Hz" << std::endl;
  output << "} \\\\" << std::endl;
  output << "\\hline " << std::endl;
  output << "{\\bf Primary Dataset} & ";
  output << "{\\bf Orig. Dataset Rate (Hz)} & ";
  output << "{\\bf Rate Added (Hz)} & ";
  output << "{\\bf Contribution (\\%)} \\\\ " << std::endl;
  output << "\\hline" << std::endl;
  output << "\\endfirsthead " << std::endl << std::endl;

  output << "\\multicolumn{4}{l}{\\bf \\bfseries -- continued from previous page --} \\\\" << std::endl;
  output << "\\hline " << std::endl;
  output << "{\\bf Primary Dataset} & ";
  output << "{\\bf Dataset Rate (Hz)} & ";
  output << "{\\bf Rate Added (Hz)} & ";
  output << "{\\bf Contribution (\\%)} \\\\ " << std::endl;
  output << "\\hline" << std::endl;  
  output << "\\endhead " << std::endl << std::endl;

  output << "\\hline \\multicolumn{4}{|r|}{{Continued on next page}} \\\\ \\hline " << std::endl;
  output << "\\endfoot " << std::endl;

  output << "\\hline " << std::endl;
  output << "\\endlastfoot " << std::endl;
  //...........................................................................

  const UInt_t      numDatasets     = datasetIndices.size();
  for (UInt_t iSet = 0; iSet < numDatasets; ++iSet) {
    const Dataset&  dataset         = datasets[datasetIndices[iSet]];
    const Double_t  addedUncertainty= TMath::Sqrt(addedUncertainty2[iSet]);

    output << "\\color{blue}" << latexEscape(dataset.name)                                    << " & ";
    if (dataset.rate > 0)
      output  << TString::Format("%.*f", decimals, dataset.rate)                              << " $\\pm$ " 
              << TString::Format("%.*f", decimals, TMath::Sqrt(dataset.rateUncertainty2))     << " & ";
    else  output << " (n/a) & ";
    output    << TString::Format("%.*f", decimals, addedRate[iSet])                           << " $\\pm$ "
              << TString::Format("%.*f", decimals, addedUncertainty)                          << " & ";
    if (dataset.rate > 0)
      output  << TString::Format("%.*g", significantDigits, 100*addedRate[iSet]/dataset.rate) << " $\\pm$ " 
              << TString::Format("%.*g", significantDigits, 100*addedUncertainty/dataset.rate);
    else  output << " (n/a) ";
    output << " \\\\ ";
    output << std::endl;
  } // end loop over compared datasets

  output << "\\hline " << std::endl;
  output << "\\end{longtable}" << std::endl << std::endl;
}



//=============================================================================
//	SampleDiagnostics
//=============================================================================

void SampleDiagnostics::setup()
{
  const UInt_t        numDatasets = size();
  commonEvents                    .resize(numDatasets);
  commonRates                     .resize(numDatasets);
  commonRateUncertainties2        .resize(numDatasets);
  for (UInt_t iSet = 0; iSet < numDatasets; ++iSet) {
    commonEvents            [iSet].resize(numDatasets + 1, 0);
    commonRates             [iSet].resize(numDatasets + 1, 0);
    commonRateUncertainties2[iSet].resize(numDatasets + 1, 0);
    at(iSet).setup(*this);

    if (firstNewTrigger < 0 && operator[](iSet).isNewTrigger)
      firstNewTrigger = iSet;
  } // end loop over datasets
}

SampleDiagnostics& SampleDiagnostics::operator+=(const SampleDiagnostics& addend)
{
  const UInt_t                  numDatasets  = size();
  for (UInt_t iSet = 0; iSet < numDatasets; ++iSet) {
    at(iSet)  += addend[iSet];
    for (UInt_t jSet = 0; jSet <= numDatasets; ++jSet) {
      commonRates             [iSet][jSet]  += addend.commonRates             [iSet][jSet];
      commonRateUncertainties2[iSet][jSet]  += addend.commonRateUncertainties2[iSet][jSet];
    } // end loop over other datasets
  } // end loop over datasets
  passedRate              += addend.passedRate;
  passedRateUncertainty2  += addend.passedRateUncertainty2;
  numProcessedEvents      += addend.numProcessedEvents;
  numConstituentSamples   += addend.numConstituentSamples;

  return *this;
}

void SampleDiagnostics::fill( const std::vector<Int_t>& triggerBit )
{
  if (timer)            timer->Start(kFALSE);
  const UInt_t          numDatasets = size();

  // First loop to record the decision of each dataset
  for (UInt_t iSet = 0; iSet < numDatasets; ++iSet) 
    operator[](iSet).checkEvent(triggerBit);

  // Second loop to record the correlations
  Int_t                 numPasses   = 0;
  for (UInt_t iSet = 0; iSet < numDatasets; ++iSet) {
    Dataset&            dataset     = operator[](iSet);
    if (!dataset.pass)              continue;
    dataset.checkAddition(*this);
    if (!dataset.isNewTrigger)  ++numPasses;

    Int_t               numOthers   = 0;
    for (UInt_t jSet = 0; jSet < numDatasets; ++jSet) {
      const Dataset&    other       = operator[](jSet);
      if (other.pass) {
        ++commonEvents[iSet][jSet];
        if (!other.isNewTrigger && iSet != jSet)
          ++numOthers;
      }
    } // end loop over other datasets
    if (numOthers > 0)  ++commonEvents[iSet].back();
  } // end loop over datasets

  if (numPasses)        ++numPassedEvents;
  ++numProcessedEvents;
  if (timer)            timer->Stop();
}

void SampleDiagnostics::computeRate(Double_t collisionRate, Double_t mu)
{
  passedRate                  = toRate            (collisionRate, mu, numPassedEvents, numProcessedEvents);
  passedRateUncertainty2      = toRateUncertainty2(collisionRate, mu, numPassedEvents, numProcessedEvents);
  std::cout << "computeRate(" << name << ") : collisionRate = " << collisionRate << " ; mu = " << mu << std::endl;

  const UInt_t                numDatasets   = size();
  for (UInt_t iSet = 0; iSet < numDatasets; ++iSet) {
    at(iSet).computeRate(collisionRate, mu, numProcessedEvents);
    for (UInt_t jSet = 0; jSet <= numDatasets; ++jSet) {
      commonRates             [iSet][jSet]  = toRate            (collisionRate, mu, commonEvents[iSet][jSet], numProcessedEvents);
      commonRateUncertainties2[iSet][jSet]  = toRateUncertainty2(collisionRate, mu, commonEvents[iSet][jSet], numProcessedEvents);
    } // end loop over other datasets
  } // end loop over datasets
}

void SampleDiagnostics::write() const
{
  if (passedRate == 0) {
    std::cerr << "WARNING :  No accumulated rate for " << name << ". Maybe you just didn't run over it/them?" << std::endl;
    return;
  }


  //.. Histogram output .......................................................
  const Int_t       numDatasets   = static_cast<Int_t>(size());
  const UInt_t      numBins       = numDatasets + 3;
  TH2*              hCorrelation  = new TH2F("h_correlation_" + name, name, numBins, 0, numBins, numBins, 0, numBins);
  TH2*              hSharedRate   = new TH2F("h_shared_rate_" + name, name, numBins, 0, numBins, numBins, 0, numBins);
  //...........................................................................


  Double_t          overhead      = 0;
  Double_t          overheadErr   = 0;
  for (Int_t iSet = 0, xBin = 1; iSet < numDatasets; ++iSet, ++xBin) {
    const Dataset&  dataset       = at(iSet);
    if (!dataset.isNewTrigger) {
      overhead     += dataset.rate;
      overheadErr  += dataset.rateUncertainty2;      // I think this is over-estimating it because the values are NOT uncorrelated, but oh well
    }

    if (iSet == firstNewTrigger)    ++xBin;
    if (dataset.rate == 0)          continue;
    for (Int_t jSet = 0, yBin = 1; jSet <= numDatasets; ++jSet, ++yBin) {
      if (jSet == firstNewTrigger)  ++yBin;
      if (jSet == numDatasets)      ++yBin;

      hCorrelation->SetBinContent (xBin, yBin, commonRates[iSet][jSet] / dataset.rate);
      hCorrelation->SetBinError   (xBin, yBin, ratioError(commonRates[iSet][jSet], commonRateUncertainties2[iSet][jSet], dataset.rate, dataset.rateUncertainty2));
      hSharedRate ->SetBinContent (xBin, yBin, commonRates[iSet][jSet]);
      hSharedRate ->SetBinError   (xBin, yBin, TMath::Sqrt(commonRateUncertainties2[iSet][jSet]));
    } // end loop over other datasets

    // Rightmost column is the fraction of rate out of the total
    hCorrelation->SetBinContent   (numBins, xBin, dataset.rate / passedRate);
    hCorrelation->SetBinError     (numBins, xBin, ratioError(dataset.rate, dataset.rateUncertainty2, passedRate, passedRateUncertainty2));
    hSharedRate ->SetBinContent   (numBins, xBin, dataset.rate);
    hSharedRate ->SetBinError     (numBins, xBin, TMath::Sqrt(dataset.rateUncertainty2));
  } // end loop over datasets

  // Top-right cell is the total overhead for the _current_ datasets (not including new triggers)
  hSharedRate ->SetBinContent     (numBins, numBins, overhead);
  hSharedRate ->SetBinError       (numBins, numBins, TMath::Sqrt(overheadErr));
  overheadErr       = ratioError  (overhead, overheadErr, passedRate, passedRateUncertainty2);
  overhead         /= passedRate; // Can only do this after error is computed
  overhead         -= 1;
  hCorrelation->SetBinContent     (numBins, numBins, overhead);
  hCorrelation->SetBinError       (numBins, numBins, overheadErr);



  //...........................................................................
  // Histogram format
  hCorrelation->SetTitle  (TString::Format("%s (overhead = %.3g%% #pm %.3g%%)" , hCorrelation->GetTitle(), 100*overhead, 100*overheadErr));
  hSharedRate ->SetTitle  (TString::Format("%s (total rate = %.3g #pm %.3g Hz)", hSharedRate ->GetTitle(), passedRate, TMath::Sqrt(passedRateUncertainty2)));
  hCorrelation->SetZTitle ("(X #cap Y) / X");   hSharedRate->SetZTitle ("X #cap Y");
  hCorrelation->SetOption ("colz");             hSharedRate->SetOption ("colz");
  hCorrelation->SetStats  (kFALSE);             hSharedRate->SetStats  (kFALSE);
  hCorrelation->SetMinimum(0);                  hSharedRate->SetMinimum(0);
  hCorrelation->SetMaximum(1);

  std::vector<TAxis*>     axes;
  axes.push_back(hCorrelation->GetXaxis());     axes.push_back(hCorrelation->GetYaxis());
  axes.push_back(hSharedRate ->GetXaxis());     axes.push_back(hSharedRate ->GetYaxis());
  const UInt_t            numAxes   = axes.size();
  for (UInt_t iAxis = 0; iAxis < numAxes; ++iAxis) {
    TAxis*                axis      = axes[iAxis];
    for (Int_t iSet = 0, bin = 1; iSet < numDatasets; ++iSet, ++bin) {
      if (iSet == firstNewTrigger)  ++bin;
      axis->SetBinLabel(bin, at(iSet).name);
    } // end loop over datasets
    axis->SetLabelSize    (0.04f);
    axis->LabelsOption    ("v");
    axis->SetTitle        (iAxis % 2 == 0 ? "X" : "Y");
  } // end loop over axes to format
  hCorrelation->GetXaxis()->SetBinLabel(numBins, "rate / total");
  hCorrelation->GetYaxis()->SetBinLabel(numBins, "overlap / rate");
  hSharedRate ->GetXaxis()->SetBinLabel(numBins, "rate");
  hSharedRate ->GetYaxis()->SetBinLabel(numBins, "overlap");

  if (gDirectory->GetDirectory("unnormalized") == 0)
    gDirectory->mkdir("unnormalized");
  gDirectory->cd("unnormalized");     hSharedRate ->Write();
  gDirectory->cd("/");                hCorrelation->Write();
  //...........................................................................
}


void SampleDiagnostics::report(TString tablesPrefix, const Char_t* errata) const
{
  if (passedRate == 0) {
    std::cerr << "WARNING :  No accumulated rate for " << name << ". Maybe you just didn't run over it/them?" << std::endl;
    return;
  }


  TString           blurb;        blurb.Form("These results are for %s. %s", name.Data(), errata ? errata : "");
  std::ofstream     tablesFile    (tablesPrefix + ".tex");
  tablesFile << "\\documentclass[amsmath,amssymb]{revtex4}" << std::endl;
  tablesFile << "\\usepackage{longtable}" << std::endl;
  tablesFile << "\\usepackage{color}" << std::endl;
  tablesFile << "\\begin{document}" << std::endl;
  tablesFile << "\\tableofcontents" << std::endl;
  tablesFile << "\\clearpage" << std::endl << std::endl << std::endl;

  const UInt_t      numDatasets   = size();
  for (UInt_t iSet = 0; iSet < numDatasets; ++iSet)
    if (at(iSet).isNewTrigger)    at(iSet).report(tablesFile, *this, blurb);

  tablesFile << std::endl << std::endl << "\\end{document}" << std::endl;
  tablesFile.close();
  TString     pdfIt;
  pdfIt.Form("latex %s.tex ; latex %s.tex ; latex %s.tex ; dvipdf %s.dvi %s.pdf", tablesPrefix.Data(), tablesPrefix.Data(), tablesPrefix.Data(), tablesPrefix.Data(), tablesPrefix.Data());
  //pdfIt       += TString::Format(" & (rm %s.aux %s.dvi %s.tex %s.log %s.toc)", tablesPrefix.Data(), tablesPrefix.Data(), tablesPrefix.Data(), tablesPrefix.Data(), tablesPrefix.Data(), tablesPrefix.Data());
  pdfIt       += TString::Format(" ; rm %s.aux %s.dvi %s.log %s.toc", tablesPrefix.Data(), tablesPrefix.Data(), tablesPrefix.Data(), tablesPrefix.Data(), tablesPrefix.Data());
	//RR commenting the latex compilation for now
//   if (gSystem->Exec(pdfIt) == 0)   std::clog << "  +  " << tablesPrefix << ".pdf" << std::endl;
//   else  std::clog << "  -  " << tablesPrefix << ".pdf  ---  FAILED to compile tex file!" << std::endl;
}


//=============================================================================
//	HLTDatasets
//=============================================================================

HLTDatasets::HLTDatasets(const std::vector<TString>& triggerNames, const Char_t* datasetDefinitionFile,
                         Bool_t preferEmulatedTriggers, TString emulationPrefix)
  :  datasetsConfig("", &timer)
{
  //...........................................................................
  // Parse input file and mark all the triggers that belong to datasets
  const UInt_t              numTriggers   = triggerNames.size();
  std::vector<Bool_t>       notInDataset  (numTriggers, kTRUE);
  if (datasetDefinitionFile && datasetDefinitionFile[0]) {
    std::ifstream           input(datasetDefinitionFile);
    if (!input.good())
      std::cerr << "ERROR : Cannot open dataset definitions file " << datasetDefinitionFile << std::endl;
    else {
      scenarioName          = gSystem->BaseName(datasetDefinitionFile);
      Int_t                 iExtension    = scenarioName.Last('.');
      if (iExtension != kNPOS)            scenarioName  = scenarioName(0, iExtension);


      // Parse dataset definition blocks
      TString               line;
      int                   lineNumber    = 0;
      while (input.good() && !input.eof()) {  ++lineNumber;

        // Get a line, strip comments and whitespace
        line.ReadLine(input);       strip(line);
        const Int_t         commentIndex  = line.First('#');
        if (commentIndex != kNPOS) {line  = line(0, commentIndex);      strip(line);}
        if (line.Length() < 1)      continue;

        // If the line ends with a semi-colon, it is the start of a dataset definition block
        if (line.EndsWith(":")) {
          TString           name          = line(0, line.Length()-1);   strip(name);
          if (indexOf(datasetsConfig, name) >= 0)
            std::cerr << "WARNING : One or more datasets have the same name '" << name 
                      << "'. Are you sure this is acceptable?" << std::endl;
          datasetsConfig.push_back(Dataset(name));
        }

        // Otherwise it is a trigger in the current dataset
        else {
          if (datasetsConfig.empty())
            std::cerr << "ERROR : Skipping unexpected line before declaration of first dataset : " << std::endl
                      << " [" << std::setw(3) << lineNumber << "]  " << line << std::endl;
          else {
            const Int_t     triggerIndex  = indexOf(triggerNames, line);
            // Also check in case there is an emulated version -- don't use both!
            TString         emulationName = emulationPrefix + line;
            const Int_t     emulationIndex= indexOf(triggerNames, emulationName);

            if (triggerIndex < 0 && emulationIndex < 0) {
              std::cerr << "WARNING : Skipping un-available trigger : " << std::endl
                        << " [" << std::setw(3) << lineNumber << "]  " << line << std::endl;
            } else {
              if (triggerIndex < 0 || (emulationIndex >= 0 && preferEmulatedTriggers))
                datasetsConfig.back().push_back(Trigger(emulationName, emulationIndex));
              else datasetsConfig.back().push_back(Trigger(line, triggerIndex));
              if (triggerIndex >= 0)      notInDataset[triggerIndex]    = kFALSE;
              if (emulationIndex >= 0)    notInDataset[emulationIndex]  = kFALSE;
            }
          }
        }
      } // end loop over lines in file
    }
  }
  // If there are no datasets defined, there's no point in marking all triggers as new
  // since the extra diagnostics will show nothing
  else {
    scenarioName  = "pertrigger";
    for (UInt_t iTrigger = 0; iTrigger < numTriggers; ++iTrigger) {
      // Prefer either the original or emulated version (but not both)
      Bool_t      isEmulated  = triggerNames[iTrigger].BeginsWith(emulationPrefix);
      if          ( preferEmulatedTriggers && !isEmulated) {
        if (indexOf(triggerNames, emulationPrefix + triggerNames[iTrigger]) >= 0)  continue;
      }
      else if     (!preferEmulatedTriggers &&  isEmulated) {
        Int_t     prefix      = emulationPrefix.Length();
        if (indexOf(triggerNames, triggerNames[iTrigger](prefix, triggerNames[iTrigger].Length()-prefix)) >= 0)  continue;
      }
      datasetsConfig.push_back(Dataset(triggerNames[iTrigger]));
      datasetsConfig.back().push_back(Trigger(triggerNames[iTrigger], iTrigger));
      notInDataset[iTrigger]  = kFALSE;
    } // end loop over triggers
  }


  //...........................................................................

  const UInt_t    numDatasets     = datasetsConfig.size();
  std::clog << "-- HLTDatasets -----------------------------------------------------------" << std::endl;
  std::clog << "There are " << numDatasets << " datasets:"            << std::endl;
  for (UInt_t iSet = 0; iSet < numDatasets; ++iSet)
    std::clog << "  " << std::left << std::setw(40) << datasetsConfig[iSet].name << " (x " 
              << std::right << std::setw(2) << datasetsConfig[iSet].size() << " triggers)"      << std::endl;

  //...........................................................................
  // Register all triggers _not_ in any dataset as new triggers
  UInt_t          numNewTriggers  = 0;
  for (UInt_t iTrigger = 0; iTrigger < numTriggers; ++iTrigger) {
    if (notInDataset[iTrigger] && 
				(!triggerNames[iTrigger].Contains("AlCa_Ecal"      ,TString::kIgnoreCase) &&
				 !triggerNames[iTrigger].Contains("AlCa_IsoTrack"  ,TString::kIgnoreCase) &&
				 !triggerNames[iTrigger].Contains("AlCa_HcalPhiSym",TString::kIgnoreCase) &&
				 !triggerNames[iTrigger].Contains("AlCa_RPC"       ,TString::kIgnoreCase))) {
      datasetsConfig.push_back(Dataset(triggerNames[iTrigger], kTRUE));
      datasetsConfig.back().push_back(Trigger(triggerNames[iTrigger], iTrigger));
      ++numNewTriggers;
    }
  } // end loop over triggers

  if (numNewTriggers) {
    std::clog << "and " << numNewTriggers << " new triggers:"                               << std::endl;
    for (UInt_t iTrigger = 0; iTrigger < numNewTriggers; ++iTrigger)
      std::clog << "  +  " << datasetsConfig[numDatasets + iTrigger].name                   << std::endl;
  }
  else                  std::clog << "There are no new triggers."                           << std::endl;
  std::clog << "==========================================================================" << std::endl << std::endl;
 
  
  datasetsConfig.setup();   // Make sure to allocate space at the end after everything is registered
}

HLTDatasets::~HLTDatasets()
{
  std::clog << std::endl;
  std::clog << "================================================================================" << std::endl;
  std::clog << "  HLTDatasets -- Done!  The event loop took:  "                                   << std::endl;
  timer.Print();
  std::clog << "================================================================================" << std::endl;
  std::clog << std::endl;
}

void HLTDatasets::addSample(const Char_t sampleName[], SampleCategory typeOfSample)
{
  if (datasetsConfig.empty())
    std::cerr << "WARNING : There are no datasets defined -- are you sure this HLTDatasets has been set up properly?" << std::endl;

  push_back(datasetsConfig);
  SampleDiagnostics&  sample  = back();
  sample.name         = sampleName;
  sample.typeOfSample = typeOfSample;
}


void HLTDatasets::write(const Char_t* outputPrefix, Option_t* writeOptions) const
{
  TString           outputPath;
  outputPath.Form("%s%s_", outputPrefix ? outputPrefix : "", scenarioName.Data());
  if (outputPrefix) {
    TString         outputDir     = gSystem->DirName(outputPath);
    if (outputDir.Length() && outputDir != "." && gSystem->mkdir(outputDir, kTRUE) != 0) {
      std::cerr << "ERROR : Could not create output directory " << outputDir << std::endl;
      return;
    }
  }
  // Make additional compilations of samples
// 	printf("HLTDatasets::write. About to call compileSamples(diagnostics)\n"); //RR
  std::vector<SampleDiagnostics>  diagnostics;    compileSamples(diagnostics);
  // Create output file
  TFile             correlationsFile(outputPath + "correlations.root", writeOptions, scenarioName + " : Correlation Plots");

  // Store corerlation plots
  const UInt_t      numDiagnostics = diagnostics.size();
// 	printf("HLTDatasets::write. About to call diagnostics[iSample].write\n"); //RR
  for (UInt_t iSample = 0; iSample < numDiagnostics; ++iSample)
    diagnostics[iSample].write();
  correlationsFile.Close();
}



void HLTDatasets::report(const Char_t* luminosity, const Char_t* outputPrefix, const Int_t significantDigits) const
{
  TString           outputPath;
  outputPath.Form("%s%s_", outputPrefix ? outputPrefix : "", scenarioName.Data());
  if (outputPrefix) {
    TString         outputDir     = gSystem->DirName(outputPath);
    if (outputDir.Length() && outputDir != "." && gSystem->mkdir(outputDir, kTRUE) != 0) {
      std::cerr << "ERROR : Could not create output directory " << outputDir << std::endl;
      return;
    }
  }

  // Make additional compilations of samples
  std::vector<SampleDiagnostics>  diagnostics;
  const UInt_t                    numDiagnostics  = compileSamples(diagnostics);
  const UInt_t                    numSamples      = size();


  //.. Create output file .....................................................
  TString           tablesPrefix  = outputPath + "newtriggers";
  std::ofstream     tablesFile    (tablesPrefix + ".tex");
  tablesFile << "\\documentclass[amsmath,amssymb]{revtex4}" << std::endl;
  tablesFile << "\\usepackage{longtable}" << std::endl;
  tablesFile << "\\usepackage{color}" << std::endl;
  tablesFile << "\\begin{document}" << std::endl;
  tablesFile << "\\begin{center}" << std::endl;
  tablesFile << "{\\large Instantaneous Luminosity " << luminosity << "~cm$^{-2}$~s$^{-1}$}" << std::endl;
  tablesFile << "\\end{center}" << std::endl;
  tablesFile << "\\tableofcontents" << std::endl;
  tablesFile << "\\clearpage" << std::endl << std::endl << std::endl;
  //...........................................................................



  // Store dataset definition
  const UInt_t      numDatasets   = datasetsConfig.size();
  if (diagnostics[1 + RATE_SAMPLE].numConstituentSamples > 0) {
  tablesFile        << "\\section{Primary Datasets}\\label{primaryDatasets}"  << std::endl;
    tablesFile      << "The ``rate'' samples are:"                            << std::endl;
    tablesFile      << "\\begin{itemize}"                                     << std::endl;
    for (UInt_t iSample = 0; iSample < numSamples; ++iSample)                  
      if (at(iSample).typeOfSample == RATE_SAMPLE)
        tablesFile  << "\\item " << latexEscape(at(iSample).name)             << std::endl;
    tablesFile      << "\\end{itemize}"                                       << std::endl;
    tablesFile      << std::endl;
  }
  if (diagnostics[1 + PHYSICS_SAMPLE].numConstituentSamples > 0) {
    tablesFile      << "The ``physics'' samples are:"                         << std::endl;
    tablesFile      << "\\begin{itemize}"                                     << std::endl;
    for (UInt_t iSample = 0; iSample < numSamples; ++iSample)                  
      if (at(iSample).typeOfSample == PHYSICS_SAMPLE)
        tablesFile  << "\\item " << latexEscape(at(iSample).name)             << std::endl;
    tablesFile      << "\\end{itemize}"                                       << std::endl;
    tablesFile      << std::endl;
  }
  tablesFile        << "\\begin{longtable}{|c|c|l|}"                                    << std::endl;
//  tablesFile        << "\\begin{tabular}{|c|c|l|}"                            << std::endl;
  tablesFile        << "\\hline"                                              << std::endl;
  tablesFile        << "{\\bf Primary Dataset} & {\\bf Rate in Hz ("
                    << TString(diagnostics[0].name).ReplaceAll("samples", " samples")
                    << ")} & {\\bf Triggers (OR-ed)} \\\\" << std::endl;
  tablesFile        << "\\hline"                                              << std::endl;
  tablesFile        << "\\endfirsthead"  << std::endl;
  tablesFile        << "\\hline"                                              << std::endl;
  tablesFile        << "{\\bf Primary Dataset} & {\\bf Rate in Hz ("
                    << TString(diagnostics[0].name).ReplaceAll("samples", " samples")
                    << ")} & {\\bf Triggers (OR-ed)} \\\\" << std::endl;
  tablesFile        << "\\hline"   << std::endl;
  tablesFile        << "\\endhead"  << std::endl;
  tablesFile        << "\\hline"  << std::endl;
  tablesFile        << "\\endfoot"  << std::endl;
  tablesFile        << "\\hline"    << std::endl;
  tablesFile        << "\\endlastfoot"  << std::endl;

  Double_t          overheadRate  = 0;
  for (UInt_t iSet = 0; iSet < numDatasets; ++iSet) {
    const Dataset&  dataset       = diagnostics[0][iSet];
    if (dataset.isNewTrigger)     continue;
    overheadRate   += dataset.rate;
    Double_t        rateErr       = TMath::Sqrt(dataset.rateUncertainty2);
    Int_t           decimals      = decimalPlaces(rateErr);
    tablesFile      << latexEscape(dataset.name.Data())                       << " & "
                    << TString::Format("%.*f", decimals, dataset.rate)        << " $\\pm$ "
                    << TString::Format("%.*f", decimals, rateErr)             << " & "
                    ;
    const UInt_t    numTriggers   = dataset.size();
    for (UInt_t iTrig = 0; iTrig < numTriggers; ++iTrig)
      tablesFile    << (iTrig ? "          &      & " : "")
                    << latexEscape(dataset[iTrig].name) << " \\\\"            << std::endl;
    tablesFile      << "\\hline"                                              << std::endl;
  } // end loop over datasets
  overheadRate     -= diagnostics[0].passedRate;
  Double_t          passedRateErr = TMath::Sqrt(diagnostics[0].passedRateUncertainty2);
  Int_t             decimals      = decimalPlaces(passedRateErr);
  tablesFile        << "\\multicolumn{3}{l}{Total rate is (";
  tablesFile        << TString::Format("%.*f", decimals, diagnostics[0].passedRate) << " $\\pm$ "
                    << TString::Format("%.*f", decimals, passedRateErr)       << ") Hz, plus "
                    << TString::Format("%.*f", decimals, overheadRate)        
                    << " Hz of datasets storage overhead"
                    ;
  tablesFile        << "} \\\\ \\hline"                                       << std::endl;
//  tablesFile        << "\\end{tabular}"                                       << std::endl;
  tablesFile        << "\\end{longtable}"                                         << std::endl;
  tablesFile        << std::endl;
  tablesFile        << "\\clearpage"                                          << std::endl;
  tablesFile        << "%================================================================================" << std::endl;
  tablesFile        << std::endl << std::endl;


  // Store one page per new trigger
  for (UInt_t iSet = 0; iSet < numDatasets; ++iSet) {
    const Dataset&  dataset       = datasetsConfig[iSet];
    if (! dataset.isNewTrigger)   continue;

    // Start of section for this trigger
    TString         title         = latexEscape(dataset.name);
    const UInt_t    numTriggers   = dataset.size();

    tablesFile      << "\\section{Contribution of " << title << "}";
    tablesFile      << "\\label{" << TString(dataset.name).ReplaceAll("_","") << "Contribution}" << std::endl;
    tablesFile      << "The rate that would be added by ";
    if (numTriggers > 1) {
      tablesFile    << title << "~$\\equiv$~(";
      for (UInt_t iTrig = 0; iTrig < numTriggers; ++iTrig)
        tablesFile  << (iTrig ? "~$\\vee$~" : "") << latexEscape(dataset[iTrig].name);
      tablesFile    << ")";
    }
    else tablesFile << title;
    tablesFile      << " to the various primary datasets. " << std::endl;
    tablesFile      << "The ``contribution'' is the (new) rate added by " << title 
                    << " as a percent of the original dataset rate. ";
    tablesFile      << std::endl << std::endl;


    for (UInt_t iSample = 0; iSample < numDiagnostics; ++iSample) {
      const SampleDiagnostics&      sample          = diagnostics[iSample];
      ////std::cout << " +++   " << sample.name << " = " << sample.numProcessedEvents << std::endl;
      if (sample.typeOfSample == PHYSICS_SAMPLE && sample.numProcessedEvents > 0) {
        TString     errata;
        errata.Form("{\\bf %s}~:~", TString(sample.name).ReplaceAll("samples", " samples").Data());
        sample[iSet].report(tablesFile, sample, errata, significantDigits);
      }
    } // end loop over samples


    tablesFile      << "\\clearpage" << std::endl;
    tablesFile      << "%================================================================================" << std::endl;
    tablesFile      << std::endl << std::endl;
  } // end loop over datasets



  //.. Latex it ...............................................................
  tablesFile << std::endl << std::endl << "\\end{document}" << std::endl;
  tablesFile.close();
  TString     pdfIt;
  pdfIt.Form("latex %s.tex ; latex %s.tex ; latex %s.tex ; dvipdf %s.dvi %s.pdf", tablesPrefix.Data(), tablesPrefix.Data(), tablesPrefix.Data(), tablesPrefix.Data(), tablesPrefix.Data());
  //pdfIt       += TString::Format(" & (rm %s.aux %s.dvi %s.tex %s.log %s.toc)", tablesPrefix.Data(), tablesPrefix.Data(), tablesPrefix.Data(), tablesPrefix.Data(), tablesPrefix.Data(), tablesPrefix.Data());
  pdfIt       += TString::Format(" ; rm %s.aux %s.dvi %s.log %s.toc", tablesPrefix.Data(), tablesPrefix.Data(), tablesPrefix.Data(), tablesPrefix.Data(), tablesPrefix.Data());
	// RR commenting the latex xompilation for now
//   if (gSystem->Exec(pdfIt) == 0)   std::clog << "  +  " << tablesPrefix << ".pdf" << std::endl;
//   else  std::clog << "  -  " << tablesPrefix << ".pdf  ---  FAILED to compile tex file!" << std::endl;
  //...........................................................................
}


  
UInt_t HLTDatasets::compileSamples( std::vector<SampleDiagnostics>& compiled ) const
{
  compiled.clear();
  compiled.push_back(datasetsConfig);  
  compiled.back().name = "allsamples";      compiled.back().numConstituentSamples = 0;
  compiled.push_back(datasetsConfig);  
  compiled.back().name = "ratesamples";     compiled.back().numConstituentSamples = 0;
  compiled.push_back(datasetsConfig);  
  compiled.back().name = "physicssamples";  compiled.back().numConstituentSamples = 0;

  const UInt_t                              numSamples  = size();
// 	printf("HLTDatasets::compileSamples. About to loop over %d samples\n",numSamples); //RR
  for (UInt_t iSample = 0; iSample < numSamples; ++iSample) {
    const SampleDiagnostics&                sample      = at(iSample);
    compiled.front()                        += sample;
    compiled[1 + sample.typeOfSample]       += sample;
  } // end loop over samples
  compiled.insert(compiled.end(), begin(), end());
  return compiled.size();
}
