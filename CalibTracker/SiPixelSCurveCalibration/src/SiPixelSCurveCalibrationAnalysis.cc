#include "CalibTracker/SiPixelSCurveCalibration/interface/SiPixelSCurveCalibrationAnalysis.h"
#include "TMath.h"

//initialize static members
std::vector<float> SiPixelSCurveCalibrationAnalysis::efficiencies_(0);
std::vector<float> SiPixelSCurveCalibrationAnalysis::effErrors_(0);

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
   std::vector<std::string>     anEmptyDefaultVectorOfStrings;
   plaquettesToSave_            = iConfig.getUntrackedParameter<std::vector<std::string> >("plaquettesToSave", anEmptyDefaultVectorOfStrings);
   saveCurvesThatFlaggedBad_    = iConfig.getUntrackedParameter<bool>("saveCurvesThatFlaggedBad", false);
   write2dHistograms_           = iConfig.getUntrackedParameter<bool>("write2dHistograms", true);
   write2dFitResult_            = iConfig.getUntrackedParameter<bool>("write2dFitResult", true);
   minimumChi2prob_             = iConfig.getUntrackedParameter<double>("minimumChi2prob", 0);
   minimumThreshold_            = iConfig.getUntrackedParameter<double>("minimumThreshold", -10);
   maximumThreshold_            = iConfig.getUntrackedParameter<double>("maximumThreshold", 300);
   minimumSigma_                = iConfig.getUntrackedParameter<double>("minimumSigma", 0);
   maximumSigma_                = iConfig.getUntrackedParameter<double>("maximumSigma", 100);
   minimumEffAsymptote_         = iConfig.getUntrackedParameter<double>("minimumEffAsymptote", 0);
   maximumEffAsymptote_         = iConfig.getUntrackedParameter<double>("maximumEffAsymptote", 1000);
}

SiPixelSCurveCalibrationAnalysis::~SiPixelSCurveCalibrationAnalysis()
{
   //do nothing
}

void SiPixelSCurveCalibrationAnalysis::calibrationSetup(const edm::EventSetup& iSetup)
{
   edm::LogInfo("SiPixelSCurveCalibrationAnalysis") << "Calibration Settings: VCalLow: " << vCalValues_[0] << "  VCalHigh: " << vCalValues_[vCalValues_.size()-1] << " nVCal: " << vCalValues_.size() << "  nTriggers: " << nTriggers_;
   fitFunction_ = new TF1("sCurve", "0.5*[2]*(1+TMath::Erf( (x-[0]) / ([1]*sqrt(2)) ) )", vCalValues_[0], vCalValues_[vCalValues_.size()-1]);
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
      //set DQM directory eventually...
      std::string detIdName = translateDetIdToString(detid);
      if (write2dHistograms_){
         MonitorElement * D2sigma       = bookDQMHistoPlaquetteSummary2D(detIdName + "_sigmas", detIdName + " Sigmas", detid);
         MonitorElement * D2thresh      = bookDQMHistoPlaquetteSummary2D(detIdName + "_thresholds", detIdName + " Thresholds",detid);
         MonitorElement * D2chi2        = bookDQMHistoPlaquetteSummary2D(detIdName + "_Chi2NDF",detIdName + " Chi2NDF", detid);
         insertResult.first->second.insert(std::make_pair(kSigmas, D2sigma));
         insertResult.first->second.insert(std::make_pair(kThresholds, D2thresh));
         insertResult.first->second.insert(std::make_pair(kChi2s, D2chi2));
      }
      if (write2dFitResult_){
         MonitorElement * D2FitResult = bookDQMHistoPlaquetteSummary2D(detIdName + "_fitresult", detIdName + " Fit Result", detid);
         insertResult.first->second.insert(std::make_pair(kFitResults, D2FitResult));
      }
      MonitorElement * D1sigma       = bookDQMHistogram1D(detIdName + "_sigmas_summary", detIdName + " Sigmas Summary", 100, 0, maximumSigma_);
      MonitorElement * D1thresh      = bookDQMHistogram1D(detIdName + "_thresholds_summary", detIdName + " Thresholds Summary", 255, 0, maximumThreshold_);
      MonitorElement * D1chi2        = bookDQMHistogram1D(detIdName + "_Chi2NDF_summary", detIdName + " Chi2NDF Summary", 100, 0, 50);
      MonitorElement * D1FitResult   = bookDQMHistogram1D(detIdName + "_fitresult_summary", detIdName + " Fit Result Summary", 10, -0.5, 9.5);
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
   float thresholdGuess;
   float sigmaGuess;
   errorFlag = estimateSCurveParameters(efficiencies_, thresholdGuess, sigmaGuess);

   Double_t sigma		        = 0;
   Double_t sigmaError		        = 0;
   Double_t threshold		        = 0;
   Double_t thresholdError		= 0;
   Double_t amplitude		        = 0;
   Double_t amplitudeError		= 0;
   Double_t chi2		        = 0;
   //calculate NDF
   Int_t nDOF                           = vCalValues_.size() - 3;
   Double_t chi2probability	        = 0;

   if (errorFlag == errOK)          //only do fit if curve is fittable
   {
      //set up minuit fit
      TMinuit *gMinuit = new TMinuit(3);
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
         (*thisDetIdHistoGrams).second[kFitResults]->setBinContent(col, row, errorFlag);

      //fill sigmas and thresholds and chi2 only if fit made sense
      if (errorFlag == errOK || errorFlag == errFlaggedBadByUser) 
      {
         (*thisDetIdHistoGrams).second[kSigmaSummary]->Fill(sigma);
         (*thisDetIdHistoGrams).second[kThresholdSummary]->Fill(threshold);
         if (write2dHistograms_)
         {
            (*thisDetIdHistoGrams).second[kSigmas]->setBinContent(col, row, sigma);
            (*thisDetIdHistoGrams).second[kThresholds]->setBinContent(col, row, threshold);
         }
      } 
      //only fill chi2's if a fit was performed
      if(errorFlag != errAllZeros && errorFlag != errNoTurnOn) 
      {
         (*thisDetIdHistoGrams).second[kChi2Summary]->Fill(chi2probability);
         if (write2dHistograms_)
            (*thisDetIdHistoGrams).second[kChi2s]->Fill(col, row, chi2probability);
      }
   }

   return true;
   
}
