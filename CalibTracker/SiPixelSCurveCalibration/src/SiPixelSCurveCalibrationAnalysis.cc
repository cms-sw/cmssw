#include "CalibTracker/SiPixelSCurveCalibration/interface/SiPixelSCurveCalibrationAnalysis.h"
#include "TMath.h"

#include <iostream>
#include <fstream>

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/SiPixelDigi/interface/SiPixelCalibDigiError.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFrameConverter.h"
#include "CondFormats/SiPixelObjects/interface/ElectronicIndex.h"
#include "CondFormats/SiPixelObjects/interface/DetectorIndex.h"
#include "CondFormats/SiPixelObjects/interface/LocalPixel.h"
#include <sstream>

//initialize static members
std::vector<float> SiPixelSCurveCalibrationAnalysis::efficiencies_(0);
std::vector<float> SiPixelSCurveCalibrationAnalysis::effErrors_(0);


void SiPixelSCurveCalibrationAnalysis::calibrationEnd(){
  if(printoutthresholds_)
    makeThresholdSummary();
}

void SiPixelSCurveCalibrationAnalysis::makeThresholdSummary(void){
  ofstream myfile;
  myfile.open (thresholdfilename_.c_str());
  for(detIDHistogramMap::iterator  thisDetIdHistoGrams= histograms_.begin();  thisDetIdHistoGrams != histograms_.end(); ++thisDetIdHistoGrams){
   // loop over det id (det id = number (unsigned int) of pixel module 
    const MonitorElement *sigmahist = (*thisDetIdHistoGrams).second[kSigmas];
    const MonitorElement *thresholdhist = (*thisDetIdHistoGrams).second[kThresholds];  
    uint32_t detid = (*thisDetIdHistoGrams).first;
    std::string name = sigmahist->getTitle();
    std::string rocname = name.substr(0,name.size()-7);
    rocname+="_ROC";
    int total_rows = sigmahist ->getNbinsY();
    int total_columns = sigmahist->getNbinsX();
    //loop over all rows on columns on all ROCs 
    for (int irow=0; irow<total_rows; ++irow){
      for (int icol=0; icol<total_columns; ++icol){
	float threshold_error = sigmahist->getBinContent(icol+1,irow+1); // +1 because root bins start at 1
       	if(writeZeroes_ ||(!writeZeroes_ && threshold_error>0)){	     
	  //changing from offline to online numbers
	  int realfedID=-1;
	  for(int fedid=0; fedid<=40; ++fedid){
	    SiPixelFrameConverter converter(theCablingMap_.product(),fedid);
	    if(converter.hasDetUnit(detid)){
	      realfedID=fedid;
	      break;   
	    }
	  }
	  if (realfedID==-1){
	    std::cout<<"error: could not obtain real fed ID"<<std::endl;
	  }
	  sipixelobjects::DetectorIndex detector ={detid,irow,icol};
	  sipixelobjects::ElectronicIndex cabling; 
	  SiPixelFrameConverter formatter(theCablingMap_.product(),realfedID);
	  formatter.toCabling(cabling,detector);
	  // cabling should now contain cabling.roc and cabling.dcol  and cabling.pxid
	  // however, the coordinates now need to be converted from dcl,pxid to the row,col coordinates used in the calibration info 
	  sipixelobjects::LocalPixel::DcolPxid loc;
	  loc.dcol = cabling.dcol;
	  loc.pxid = cabling.pxid;
	  // FIX to adhere to new cabling map. To be replaced with CalibTracker/SiPixelTools detid - > hardware id classes ASAP.
	  //        const sipixelobjects::PixelFEDCabling *theFed= theCablingMap.product()->fed(realfedID);
	  //        const sipixelobjects::PixelFEDLink * link = theFed->link(cabling.link);
	  //        const sipixelobjects::PixelROC *theRoc = link->roc(cabling.roc);
	  sipixelobjects::LocalPixel locpixel(loc);
	  sipixelobjects::CablingPathToDetUnit path = {static_cast<unsigned int>(realfedID),
                                                       static_cast<unsigned int>(cabling.link),
                                                       static_cast<unsigned int>(cabling.roc)};
	  const sipixelobjects::PixelROC *theRoc = theCablingMap_->findItem(path);
	  // END of FIX
	  int newrow= locpixel.rocRow();
	  int newcol = locpixel.rocCol();
	  myfile<<rocname<<theRoc->idInDetUnit()<<" "<<newcol<<" "<<newrow<<" "<<thresholdhist->getBinContent(icol+1, irow+1)<<" "<<threshold_error;  // +1 because root bins start at 1
	  myfile<<"\n";
	}
      }
    }
  }
  myfile.close();
}

//used for TMinuit fitting
void chi2toMinimize(int &npar, double* grad, double &fcnval, double* xval, int iflag)
{
   TF1 * theFormula = SiPixelSCurveCalibrationAnalysis::fitFunction_;
   //setup function parameters
   for (int i = 0; i < npar; i++)
      theFormula->SetParameter(i, xval[i]);
   fcnval = 0;
   //compute Chi2 of all points
   const std::vector<short>* theVCalValues = SiPixelSCurveCalibrationAnalysis::getVcalValues();
   for (uint32_t i = 0; i < theVCalValues->size(); i++)
   {
      float chi = (SiPixelSCurveCalibrationAnalysis::efficiencies_[i] - theFormula->Eval((*theVCalValues)[i]) );
      chi       /= SiPixelSCurveCalibrationAnalysis::effErrors_[i];
      fcnval += chi*chi;
   }
}

void
SiPixelSCurveCalibrationAnalysis::doSetup(const edm::ParameterSet& iConfig)
{
   edm::LogInfo("SiPixelSCurveCalibrationAnalysis") << "Setting up calibration paramters.";
   std::vector<uint32_t>        anEmptyDefaultVectorOfUInts;
   std::vector<uint32_t>        detIDsToSaveVector_;
   useDetectorHierarchyFolders_ = iConfig.getUntrackedParameter<bool>("useDetectorHierarchyFolders", true);
   saveCurvesThatFlaggedBad_    = iConfig.getUntrackedParameter<bool>("saveCurvesThatFlaggedBad", false);
   detIDsToSaveVector_          = iConfig.getUntrackedParameter<std::vector<uint32_t> >("detIDsToSave", anEmptyDefaultVectorOfUInts);
   maxCurvesToSave_             = iConfig.getUntrackedParameter<uint32_t>("maxCurvesToSave", 1000);
   write2dHistograms_           = iConfig.getUntrackedParameter<bool>("write2dHistograms", true);
   write2dFitResult_            = iConfig.getUntrackedParameter<bool>("write2dFitResult", true);
   printoutthresholds_          = iConfig.getUntrackedParameter<bool>("writeOutThresholdSummary",true);
   thresholdfilename_           = iConfig.getUntrackedParameter<std::string>("thresholdOutputFileName","thresholds.txt");  
   minimumChi2prob_             = iConfig.getUntrackedParameter<double>("minimumChi2prob", 0);
   minimumThreshold_            = iConfig.getUntrackedParameter<double>("minimumThreshold", -10);
   maximumThreshold_            = iConfig.getUntrackedParameter<double>("maximumThreshold", 300);
   minimumSigma_                = iConfig.getUntrackedParameter<double>("minimumSigma", 0);
   maximumSigma_                = iConfig.getUntrackedParameter<double>("maximumSigma", 100);
   minimumEffAsymptote_         = iConfig.getUntrackedParameter<double>("minimumEffAsymptote", 0);
   maximumEffAsymptote_         = iConfig.getUntrackedParameter<double>("maximumEffAsymptote", 1000);
   maximumSigmaBin_             = iConfig.getUntrackedParameter<double>("maximumSigmaBin", 10);
   maximumThresholdBin_         = iConfig.getUntrackedParameter<double>("maximumThresholdBin", 255);

   writeZeroes_= iConfig.getUntrackedParameter<bool>("alsoWriteZeroThresholds", false);

   // convert the vector into a map for quicker lookups.
   for(unsigned int i = 0; i < detIDsToSaveVector_.size(); i++)
      detIDsToSave_.insert( std::make_pair(detIDsToSaveVector_[i], true) );
}

SiPixelSCurveCalibrationAnalysis::~SiPixelSCurveCalibrationAnalysis()
{
   //do nothing
}

void SiPixelSCurveCalibrationAnalysis::buildACurveHistogram(const uint32_t& detid, const uint32_t& row, const uint32_t& col, sCurveErrorFlag errorFlag, const std::vector<float>& efficiencies, const std::vector<float>& errors)
{
   if (curvesSavedCounter_ > maxCurvesToSave_)
   {
      edm::LogWarning("SiPixelSCurveCalibrationAnalysis") << "WARNING: Request to save curve for [detid](col/row):  [" << detid << "](" << col << "/" << row << ") denied. Maximum number of saved curves (defined in .cfi) exceeded.";
      return;
   }
   std::ostringstream rootName;
   rootName << "SCurve_row_" << row << "_col_" << col;
   std::ostringstream humanName;
   humanName << translateDetIdToString(detid) << "_" << rootName.str() << "_ErrorFlag_" << (int)errorFlag;

   unsigned int numberOfVCalPoints = vCalPointsAsFloats_.size()-1; //minus one is necessary since the lower edge of the last bin must be added
   if (efficiencies.size() != numberOfVCalPoints || errors.size() != numberOfVCalPoints)
   {
      edm::LogError("SiPixelSCurveCalibrationAnalysis") << "Error saving single curve histogram!  Number of Vcal values (" << numberOfVCalPoints << ") does not match number of efficiency points or error points!";
      return;
   }
   setDQMDirectory(detid);
   float * vcalValuesToPassToCrappyRoot = &vCalPointsAsFloats_[0];
   MonitorElement * aBadHisto = bookDQMHistogram1D(detid, rootName.str(), humanName.str(), numberOfVCalPoints, vcalValuesToPassToCrappyRoot);  //ROOT only takes an input as array. :(  HOORAY FOR CINT!
   curvesSavedCounter_++;
   for(unsigned int iBin = 0; iBin < numberOfVCalPoints; ++iBin)
   {
      int rootBin = iBin + 1;  //root bins start at 1
      aBadHisto->setBinContent(rootBin, efficiencies[iBin]);
      aBadHisto->setBinError(rootBin, errors[iBin]);
   }
}

void SiPixelSCurveCalibrationAnalysis::calibrationSetup(const edm::EventSetup& iSetup)
{
   edm::LogInfo("SiPixelSCurveCalibrationAnalysis") << "Calibration Settings: VCalLow: " << vCalValues_[0] << "  VCalHigh: " << vCalValues_[vCalValues_.size()-1] << " nVCal: " << vCalValues_.size() << "  nTriggers: " << nTriggers_;
   curvesSavedCounter_ = 0;
   if (saveCurvesThatFlaggedBad_)
   {
      //build the vCal values as a vector of floats if we want to save single curves
      const std::vector<short>* theVCalValues = this->getVcalValues();
      unsigned int numberOfVCalPoints = theVCalValues->size();
      edm::LogWarning("SiPixelSCurveCalibrationAnalysis") << "WARNING: Option set to save indiviual S-Curves - max number: " 
                                                          << maxCurvesToSave_ << " This can lead to large memory consumption! (Got " << numberOfVCalPoints << " VCal Points";
      for(unsigned int i = 0; i < numberOfVCalPoints; i++)
      {
         vCalPointsAsFloats_.push_back( static_cast<float>((*theVCalValues)[i]) );
         edm::LogInfo("SiPixelSCurveCalibrationAnalysis") << "Adding calibration Vcal: " << (*theVCalValues)[i];
      }
      // must add lower edge of last bin to the vector
      vCalPointsAsFloats_.push_back( vCalPointsAsFloats_[numberOfVCalPoints-1] + 1 );
   }

   fitFunction_ = new TF1("sCurve", "0.5*[2]*(1+TMath::Erf( (x-[0]) / ([1]*sqrt(2)) ) )", vCalValues_[0], vCalValues_[vCalValues_.size()-1]);
}

bool
SiPixelSCurveCalibrationAnalysis::checkCorrectCalibrationType()
{
  if(calibrationMode_=="SCurve")
    return true;
  else if(calibrationMode_=="unknown"){
    edm::LogInfo("SiPixelSCurveCalibrationAnalysis") <<  "calibration mode is: " << calibrationMode_ << ", continuing anyway..." ;
    return true;
  }
  else{
    //    edm::LogDebug("SiPixelSCurveCalibrationAnalysis") << "unknown calibration mode for SCurves, should be \"SCurve\" and is \"" << calibrationMode_ << "\"";
  }
  return false;
}

sCurveErrorFlag SiPixelSCurveCalibrationAnalysis::estimateSCurveParameters(const std::vector<float>& eff, float& threshold, float& sigma)
{
   sCurveErrorFlag output = errAllZeros;
   bool allZeroSoFar    = true;
   int turnOnBin        = -1;
   int saturationBin    = -1;
   for (uint32_t iVcalPt = 0; iVcalPt < eff.size(); iVcalPt++)
   {
      if (allZeroSoFar && eff[iVcalPt] != 0 ) {
         turnOnBin = iVcalPt;
         allZeroSoFar = false;
         output = errNoTurnOn;
      } else if (eff[iVcalPt] > 0.90)
      {
         saturationBin  = iVcalPt;
         short turnOnVcal       = vCalValues_[turnOnBin];
         short saturationVcal   = vCalValues_[saturationBin];
         short delta            = saturationVcal - turnOnVcal;
         sigma                  = delta * 0.682;
         if (sigma < 1)         //check to make sure sigma guess is larger than our X resolution.  Hopefully prevents Minuit from getting stuck at boundary
            sigma = 1;
         threshold              = turnOnVcal + (0.5 * delta);
         return errOK;
      }
   }
   return output;
}

sCurveErrorFlag SiPixelSCurveCalibrationAnalysis::fittedSCurveSanityCheck(float threshold, float sigma, float amplitude)
{
   //check if nonsensical
   if (threshold > vCalValues_[vCalValues_.size()-1] || threshold < vCalValues_[0] ||
         sigma > vCalValues_[vCalValues_.size()-1] - vCalValues_[0] )
      return errFitNonPhysical;

   if (threshold < minimumThreshold_ || threshold > maximumThreshold_ ||
         sigma < minimumSigma_ || sigma > maximumSigma_ ||
         amplitude < minimumEffAsymptote_ || amplitude > maximumEffAsymptote_)
      return errFlaggedBadByUser;

   return errOK;
}

void calculateEffAndError(int nADCResponse, int nTriggers, float& eff, float& error)
{
   eff = (float)nADCResponse / (float)nTriggers;
   double effForErrorCalculation = eff;
   if (eff <= 0 || eff >= 1)
      effForErrorCalculation = 0.5 / (double)nTriggers;
   error = TMath::Sqrt(effForErrorCalculation*(1-effForErrorCalculation) / (double)nTriggers);
}

//book histograms when new DetID is encountered in Event Record
void SiPixelSCurveCalibrationAnalysis::newDetID(uint32_t detid)
{
   edm::LogInfo("SiPixelSCurveCalibrationAnalysis") << "Found a new DetID (" << detid << ")!  Checking to make sure it has not been added.";
   //ensure that this DetID has not been added yet
   sCurveHistogramHolder tempMap;
   std::pair<detIDHistogramMap::iterator, bool> insertResult; 
   insertResult = histograms_.insert(std::make_pair(detid, tempMap));
   if (insertResult.second)     //indicates successful insertion
   {
      edm::LogInfo("SiPixelSCurveCalibrationAnalysisHistogramReport") << "Histogram Map.insert() returned true!  Booking new histogrames for detID: " << detid;
      // use detector hierarchy folders if desired
      if (useDetectorHierarchyFolders_)
         setDQMDirectory(detid);

      std::string detIdName = translateDetIdToString(detid);
      if (write2dHistograms_){
	MonitorElement * D2sigma       = bookDQMHistoPlaquetteSummary2D(detid,"ScurveSigmas", detIdName + " Sigmas");
	MonitorElement * D2thresh      = bookDQMHistoPlaquetteSummary2D(detid,"ScurveThresholds", detIdName + " Thresholds");
	MonitorElement * D2chi2        = bookDQMHistoPlaquetteSummary2D(detid,"ScurveChi2Prob",detIdName + " Chi2Prob");
         insertResult.first->second.insert(std::make_pair(kSigmas, D2sigma));
         insertResult.first->second.insert(std::make_pair(kThresholds, D2thresh));
         insertResult.first->second.insert(std::make_pair(kChi2s, D2chi2));
      }
      if (write2dFitResult_){
	MonitorElement * D2FitResult = bookDQMHistoPlaquetteSummary2D(detid,"ScurveFitResult", detIdName + " Fit Result");
         insertResult.first->second.insert(std::make_pair(kFitResults, D2FitResult));
      }
      MonitorElement * D1sigma       = bookDQMHistogram1D(detid,"ScurveSigmasSummary", detIdName + " Sigmas Summary", 100, 0, maximumSigmaBin_);
      MonitorElement * D1thresh      = bookDQMHistogram1D(detid,"ScurveThresholdSummary", detIdName + " Thresholds Summary", 255, 0, maximumThresholdBin_);
      MonitorElement * D1chi2        = bookDQMHistogram1D(detid,"ScurveChi2ProbSummary", detIdName + " Chi2Prob Summary", 101, 0, 1.01);
      MonitorElement * D1FitResult   = bookDQMHistogram1D(detid,"ScurveFitResultSummary", detIdName + " Fit Result Summary", 10, -0.5, 9.5);
      insertResult.first->second.insert(std::make_pair(kSigmaSummary, D1sigma));
      insertResult.first->second.insert(std::make_pair(kThresholdSummary, D1thresh));
      insertResult.first->second.insert(std::make_pair(kChi2Summary, D1chi2));
      insertResult.first->second.insert(std::make_pair(kFitResultSummary, D1FitResult));
   }
}

bool SiPixelSCurveCalibrationAnalysis::doFits(uint32_t detid, std::vector<SiPixelCalibDigi>::const_iterator calibDigi)
{
   sCurveErrorFlag errorFlag = errOK;
   uint32_t nVCalPts = calibDigi->getnpoints();
   //reset and fill static datamembers with vector of points and errors
   efficiencies_.resize(0);
   effErrors_.resize(0);
   for (uint32_t iVcalPt = 0; iVcalPt < nVCalPts; iVcalPt++)
   {
      float eff;
      float error;
      calculateEffAndError(calibDigi->getnentries(iVcalPt), nTriggers_, eff, error);
      edm::LogInfo("SiPixelSCurveCalibrationAnalysis") << "Eff: " << eff << " Error:  " << error << "  nEntries: " << calibDigi->getnentries(iVcalPt) << "  nTriggers: " << nTriggers_ << " VCalPt " << vCalValues_[iVcalPt];
      efficiencies_.push_back(eff);
      effErrors_.push_back(error);
   } 

   //estimate the S-Curve parameters
   float thresholdGuess = -1.0;
   float sigmaGuess = -1.0;
   errorFlag = estimateSCurveParameters(efficiencies_, thresholdGuess, sigmaGuess);

   // these -1.0 default values will only be filled if the curve is all zeroes, or doesn't turn on, WHICH INDICATES A SERIOUS PROBLEM
   Double_t sigma		        = -1.0;
   Double_t sigmaError		        = -1.0;
   Double_t threshold		        = -1.0;
   Double_t thresholdError		= -1.0;
   Double_t amplitude		        = -1.0;
   Double_t amplitudeError		= -1.0;
   Double_t chi2		        = -1.0;
   //calculate NDF
   Int_t nDOF                           = vCalValues_.size() - 3;
   Double_t chi2probability	        = 0;

   if (errorFlag == errOK)          //only do fit if curve is fittable
   {
      //set up minuit fit
      TMinuit *gMinuit = new TMinuit(3);
      gMinuit->SetPrintLevel(-1);  //save ourselves from gigabytes of stdout
      gMinuit->SetFCN(chi2toMinimize);

      //define threshold parameters - choose step size 1, max 300, min -50
      gMinuit->DefineParameter(0, "Threshold", (Double_t)thresholdGuess, 1, -50, 300);
      //sigma
      gMinuit->DefineParameter(1, "Sigma", (Double_t)sigmaGuess, 0.1, 0, 255); 
      //amplitude
      gMinuit->DefineParameter(2, "Amplitude", 1, 0.1, -0.001, 200);

      //Do Chi2 minimazation
      gMinuit->Migrad();
      gMinuit->GetParameter(0, threshold, thresholdError);
      gMinuit->GetParameter(1, sigma, sigmaError);
      gMinuit->GetParameter(2, amplitude, amplitudeError);

      //get Chi2
      Double_t params[3]   = {threshold, sigma, amplitude};
      gMinuit->Eval(3, NULL, chi2, params, 0);
      //calculate Chi2 proability
      if (nDOF <= 0)
         chi2probability = 0;
      else
         chi2probability = TMath::Prob(chi2, nDOF);
      
      //check to make sure output makes sense (i.e. threshold > 0)
      if (chi2probability > minimumChi2prob_)
         errorFlag = fittedSCurveSanityCheck(threshold, sigma, amplitude);
      else
         errorFlag = errBadChi2Prob;

      edm::LogInfo("SiPixelSCurveCalibrationAnalysis") << "Fit finished with errorFlag: " << errorFlag << " - threshold: " << threshold << "  sigma: " << sigma << "  chi2: " << chi2 << "  nDOF: " << nDOF << " chi2Prob: " << chi2probability << " chi2MinUser: " << minimumChi2prob_;

      delete gMinuit;
   }
   //get row and column for this pixel
   uint32_t row = calibDigi->row();
   uint32_t col = calibDigi->col();

   //get iterator to histogram holder for this detid
   detIDHistogramMap::iterator thisDetIdHistoGrams;
   thisDetIdHistoGrams = histograms_.find(detid);
   if (thisDetIdHistoGrams != histograms_.end())
   {
      edm::LogInfo("SiPixelSCurveCalibrationAnalysisHistogramReport") << "Filling histograms for [detid](col/row):  [" << detid << "](" << col << "/" << row << ") ErrorFlag: " << errorFlag;
      //always fill fit result
      (*thisDetIdHistoGrams).second[kFitResultSummary]->Fill(errorFlag);
      if (write2dFitResult_)
         (*thisDetIdHistoGrams).second[kFitResults]->setBinContent(col+1, row+1, errorFlag); // +1 because root bins start at 1

      // fill sigma/threshold result
      (*thisDetIdHistoGrams).second[kSigmaSummary]->Fill(sigma);
      (*thisDetIdHistoGrams).second[kThresholdSummary]->Fill(threshold);
      if (write2dHistograms_)
      {
         (*thisDetIdHistoGrams).second[kSigmas]->setBinContent(col+1, row+1, sigma); // +1 because root bins start at 1
         (*thisDetIdHistoGrams).second[kThresholds]->setBinContent(col+1, row+1, threshold); // +1 because root bins start at 1
      }
      // fill chi2
      (*thisDetIdHistoGrams).second[kChi2Summary]->Fill(chi2probability);
      if (write2dHistograms_)
         (*thisDetIdHistoGrams).second[kChi2s]->Fill(col, row, chi2probability);
   }
   // save individual curves, if requested
   if (saveCurvesThatFlaggedBad_)
   {
      bool thisDetIDinList = false;
      if (detIDsToSave_.find(detid) != detIDsToSave_.end()) //see if we want to save this histogram
         thisDetIDinList = true;

      if (errorFlag != errOK || thisDetIDinList)
      {
         edm::LogError("SiPixelSCurveCalibrationAnalysis") << "Saving error histogram for [detid](col/row):  [" << detid << "](" << col << "/" << row << ") ErrorFlag: " << errorFlag;
         buildACurveHistogram(detid, row, col, errorFlag, efficiencies_, effErrors_);
      }
   }

   return true;
   
}
