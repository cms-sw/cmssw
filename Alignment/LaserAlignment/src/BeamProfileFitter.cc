/** \file BeamProfileFitter.cc
*
	*  $Date: 2008/03/04 07:52:33 $
	*  $Revision: 1.16 $
	*  \author Maarten Thomas
*/

#include "Alignment/LaserAlignment/interface/BeamProfileFitter.h"

// Framework headers
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Geometry headers
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h"

// Topology

#include "TF1.h"
#include "TH1.h"
#include "TMath.h"

// function to return an angle in radian between 0 and 2 Pi
double BeamProfileFitter::angle(double theAngle)
{
	if ( theAngle >= 0.0 )
		{ return theAngle; }
	else if ( theAngle < 0.0 )
		{ return theAngle + 2.0 * M_PI; }
	else 
	{ 
		edm::LogError("BeamProfileFitter") << "BeamProfileFitter::Angle(..): Warning! Something wrong with this Angle = " 
			<< theAngle << "! Please check!!!";
		return -666;
	}
}

// default constructor
BeamProfileFitter::BeamProfileFitter(edm::ParameterSet const& theConf, const edm::EventSetup* aSetup ) : 
  theClearHistoAfterFit(theConf.getUntrackedParameter<bool>("ClearHistogramAfterFit",true)),
  theScaleHisto(theConf.getUntrackedParameter<bool>("ScaleHistogramBeforeFit",true)),
  theMinSignalHeight(theConf.getUntrackedParameter<double>("MinimalSignalHeight",0.0)),
  theCorrectBSkink(theConf.getUntrackedParameter<bool>("CorrectBeamSplitterKink",true)),
  theBSAnglesSystematic(theConf.getUntrackedParameter<double>("BSAnglesSystematic",0.0007)) {

  theSetup = aSetup;

}


// default destructor
BeamProfileFitter::~BeamProfileFitter() {}



// the fitting routine
std::vector<LASBeamProfileFit> BeamProfileFitter::doFit(
	DetId theDetUnitId, 
	TH1D * theHistogram,
	bool theSaveHistograms,
	int ScalingFactor, 
	int theBeam, 
	int theDisc, 
	int theRing, 
	int theSide,
	bool isTEC2TEC, 
	bool & isGoodResult) {


	double theScalingFactor = (double)ScalingFactor;

	// access the Tracker
	edm::ESHandle<TrackerGeometry> theTrackerGeometry;
	theSetup->get<TrackerDigiGeometryRecord>().get(theTrackerGeometry);
	const TrackerGeometry& theTracker(*theTrackerGeometry);

	// return the result of the fit as a vector of LASBeamProfileFits. They are later on stored into the Event data
	std::vector<LASBeamProfileFit> theResult;

	if ( theHistogram )
	  {
	    if ( theHistogram->GetEntries() > 0 )
	      { 
		// get the name of the histogram
		const char * theHistoName = theHistogram->GetName();
		
		if (theSaveHistograms)
		  {
		    // save a copy of the histogram before scaling and clearing, so it is still available in the root file
		    // first get the name
		    std::string theSavedHistoName = theHistoName;
		    // append Saved to the histogram name
		    theSavedHistoName += "Saved";
		    // clone the original histogram and give it a new name
		    TH1D *theSavedHist = (TH1D*)theHistogram->Clone(theSavedHistoName.c_str());
		    // set the directory for the copy of the histogram
		    theSavedHist->SetDirectory(theHistogram->GetDirectory());
		  }
		
		// clone the histogram and scale it
		TH1D *theCloneHist = (TH1D*)theHistogram->Clone("CloneHist");
		if (theScaleHisto) theCloneHist->Scale(theScalingFactor);
		
		// find peaks and fit the beam profile
		std::vector<double> thePeakPosition = findPeakGaus(theCloneHist,theDisc, theRing);


		// finally if the fit was ok, calculate the fitted beam coordinates in the CMS detector
		if ( thePeakPosition.at(0) > 0.0 && thePeakPosition.at(1) > 0.0 &&
		     thePeakPosition.at(2) > 0.0 && thePeakPosition.at(3) > 0.0) 
		  {
		    // get the mean, sigma and the errors
		    double theMean = thePeakPosition.at(0);
		    double theSigma = thePeakPosition.at(1);
		    double theMeanError = thePeakPosition.at(2);
		    double theSigmaError = thePeakPosition.at(3);
		    
		    // correct for the BeamSplitter kink
		    
		    // the known kinks for TEC+
		    std::vector<double> bsKinkPosTEC;
		    bsKinkPosTEC.push_back(-0.001399 + theBSAnglesSystematic); // kink of beam 0 in ring 4 (sector 1)
		    bsKinkPosTEC.push_back(-0.000796 + theBSAnglesSystematic); // kink of beam 1 in ring 4 (sector 2)
		    bsKinkPosTEC.push_back(0.000397 + theBSAnglesSystematic); // kink of beam 2 in ring 4 (sector 3)
		    bsKinkPosTEC.push_back(-0.001260 + theBSAnglesSystematic); // kink of beam 3 in ring 4 (sector 4)
		    bsKinkPosTEC.push_back(0.000160 + theBSAnglesSystematic); // kink of beam 4 in ring 4 (sector 5)
		    bsKinkPosTEC.push_back(0.000068 + theBSAnglesSystematic); // kink of beam 5 in ring 4 (sector 6)
		    bsKinkPosTEC.push_back(-0.000632 + theBSAnglesSystematic); // kink of beam 6 in ring 4 (sector 7)
		    bsKinkPosTEC.push_back(0.000562 + theBSAnglesSystematic); // kink of beam 7 in ring 4 (sector 8)
		    bsKinkPosTEC.push_back(-0.002532 + theBSAnglesSystematic); // kink of beam 0 (8) in ring 6 (sector 1)
		    bsKinkPosTEC.push_back(-0.000274 + theBSAnglesSystematic); // kink of beam 1 (9) in ring 6 (sector 2)
		    bsKinkPosTEC.push_back(-0.002072 + theBSAnglesSystematic); // kink of beam 2 (10) in ring 6 (sector 3)
		    bsKinkPosTEC.push_back(-0.001195 + theBSAnglesSystematic); // kink of beam 3 (11) in ring 6 (sector 4)
		    bsKinkPosTEC.push_back(-0.001983 + theBSAnglesSystematic); // kink of beam 4 (12) in ring 6 (sector 5)
		    bsKinkPosTEC.push_back(0.000816 + theBSAnglesSystematic); // kink of beam 5 (13) in ring 6 (sector 6)
		    bsKinkPosTEC.push_back(0.000693 + theBSAnglesSystematic); // kink of beam 6 (14) in ring 6 (sector 7)
		    bsKinkPosTEC.push_back(0.000009 + theBSAnglesSystematic); // kink of beam 7 (15) in ring 6 (sector 8)

		    // the known kinks for TEC-
		    std::vector<double> bsKinkNegTEC;
		    bsKinkNegTEC.push_back(0.001010 + theBSAnglesSystematic); // kink of beam 0 in ring 4 (sector 1)
		    bsKinkNegTEC.push_back(0.000346 + theBSAnglesSystematic); // kink of beam 1 in ring 4 (sector 2)
		    bsKinkNegTEC.push_back(-0.002120 + theBSAnglesSystematic); // kink of beam 2 in ring 4 (sector 3)
		    bsKinkNegTEC.push_back(0.000151 + theBSAnglesSystematic); // kink of beam 3 in ring 4 (sector 4)
		    bsKinkNegTEC.push_back(0.001206 + theBSAnglesSystematic); // kink of beam 4 in ring 4 (sector 5)
		    bsKinkNegTEC.push_back(-0.002780 + theBSAnglesSystematic); // kink of beam 5 in ring 4 (sector 6)
		    bsKinkNegTEC.push_back(0.000313 + theBSAnglesSystematic); // kink of beam 6 in ring 4 (sector 7)
		    bsKinkNegTEC.push_back(-0.001397 + theBSAnglesSystematic); // kink of beam 7 in ring 4 (sector 8)
		    bsKinkNegTEC.push_back(-0.000386 + theBSAnglesSystematic); // kink of beam 0 (8) in ring 6 (sector 1)
		    bsKinkNegTEC.push_back(0.000356 + theBSAnglesSystematic); // kink of beam 1 (9) in ring 6 (sector 2)
		    bsKinkNegTEC.push_back(-0.002350 + theBSAnglesSystematic); // kink of beam 2 (10) in ring 6 (sector 3)
		    bsKinkNegTEC.push_back(-0.000432 + theBSAnglesSystematic); // kink of beam 3 (11) in ring 6 (sector 4)
		    bsKinkNegTEC.push_back(0.000251 + theBSAnglesSystematic); // kink of beam 4 (12) in ring 6 (sector 5)
		    bsKinkNegTEC.push_back(-0.001587 + theBSAnglesSystematic); // kink of beam 5 (13) in ring 6 (sector 6)
		    bsKinkNegTEC.push_back(-0.002577 + theBSAnglesSystematic); // kink of beam 6 (14) in ring 6 (sector 7)
		    bsKinkNegTEC.push_back(-0.000478 + theBSAnglesSystematic); // kink of beam 7 (15) in ring 6 (sector 8)

		    // known kinks for BS in the Alignment Tubes???

		    const double theBSPosition = 70.5;

		    // calculate the coordinates of the fitted beam profile
		    if ( theMean > 0.0 && theMean < 512.0 )
		      {
			// get the DetUnit via the DetUnitId and cast it to a StripGeomDetUnit
			const StripGeomDetUnit* const theStripDet = dynamic_cast<const StripGeomDetUnit*>(theTracker.idToDet(theDetUnitId));

			// store the uncorrected mean
			double theUncorrectedMean = theMean;

			if (theCorrectBSkink)
			  {
			    if ( (theDisc == 5) || (theDisc == 6) || (theDisc == 7) || (theDisc == 8) )
			      {
				if (theSide == 1)
				  {
				    // correction for TEC-
				    // correct the fitted mean for the BS kink
				    theMean -= (TMath::Abs(theStripDet->position().z()) - theBSPosition) * tan(bsKinkNegTEC.at(theBeam));
				  }
				else if (theSide == 2)
				  {
				    // correction for TEC+
				    // correct the fitted mean for the BS kink
				    theMean -= (TMath::Abs(theStripDet->position().z()) - theBSPosition) * tan(bsKinkPosTEC.at(theBeam));
				  }
			      }
			  }

			// global position of the LaserProfile
			TVector3 GlobalPos(theStripDet->surface().toGlobal(theStripDet->specificTopology().localPosition(theMean)).x(),
					   theStripDet->surface().toGlobal(theStripDet->specificTopology().localPosition(theMean)).y(),
					   theStripDet->surface().toGlobal(theStripDet->specificTopology().localPosition(theMean)).z());

			// use the error on the mean from the fit to calculate the error on phi
			ErrorFrameTransformer theErrorTransformer;
			GlobalError theGlobalPositionError = theErrorTransformer.transform(theStripDet->specificTopology().localError(theMean,pow(theMeanError,2)),
											   theStripDet->surface());

			Float_t gerrors[9] = { theGlobalPositionError.matrix()(1,1), theGlobalPositionError.matrix()(1,2), theGlobalPositionError.matrix()(1,3),
					       theGlobalPositionError.matrix()(2,1), theGlobalPositionError.matrix()(2,2), theGlobalPositionError.matrix()(2,3),
					       theGlobalPositionError.matrix()(3,1), theGlobalPositionError.matrix()(3,2), theGlobalPositionError.matrix()(3,3) };
			TMatrix GlobalError(3,3,gerrors,"");


			// errors of the x,y and z coordinate are given as the square root of the (i,i) element of the Covariance Matrix
			TVector3 GlobalPosError(TMath::Sqrt(GlobalError(0,0)),TMath::Sqrt(GlobalError(1,1)),TMath::Sqrt(GlobalError(2,2)));

			// global phi and error of the LaserProfile
			Double_t GlobalPhi = angle(GlobalPos.Phi());
			Double_t GlobalPhiError = phiError(GlobalPos, GlobalError);

			// create a new LASBeamProfileFit from the result and return it
			theResult.push_back(LASBeamProfileFit(theHistoName, theMean, theMeanError, theUncorrectedMean, 
							      theSigma, theSigmaError, theStripDet->specificTopology().localPitch(theStripDet->specificTopology().localPosition(theMean)), GlobalPhi, GlobalPhiError));
			// this is a good fit!?
			isGoodResult = true;

			// some final output about the result of the fit
			LogDebug("BeamProfileFitter:doFit") << " **** Results of the Beam Profile Fit for " << theHistoName << " **** "
							    << "\n Mean            = " << theMean << " +/- " << theMeanError
							    << "\n Sigma           = " << theSigma << " +/- " << theSigmaError
							    << "\n Pitch           = " << theStripDet->specificTopology().localPitch(theStripDet->specificTopology().localPosition(theMean))
							    << "\n Fitted Global Position = (" << GlobalPos.X() << "," << GlobalPos.Y() << "," << GlobalPos.Z() << ")"
							    << "\n Global Position Error  = (" << GlobalPosError.X() << "," << GlobalPosError.Y() << "," << GlobalPosError.Z() << ")"
							    << "\n Fitted Global Phi      = " << GlobalPhi << " +/- " << GlobalPhiError
							    << "\n ******************************************************************************* ";
		      }
		    else
		      {
			edm::LogWarning("BeamProfileFitter::Fit ERROR") << " Error! Result of the fit for detId: " << theDetUnitId.rawId()
									<< " is not ok! Mean: " << theMean << "  is not a value between 0 and 512 ... ";

			// return an empty LASBeamProfileFit and set isGoodResult to false	      
			theResult.push_back(LASBeamProfileFit(theHistoName, 0.0, 0.0, 0.0, 0.0));
			// this is not a good fit
			isGoodResult = false;
		      }

		  }
		else
		  { 
		    edm::LogWarning("BeamProfileFitter:Fit ERROR") << " Error! Result of the fit for detId: " << theDetUnitId.rawId()
								   << " is not ok! No position information available ... "; 

		    // return an empty LASBeamProfileFit and set isGoodResult to false	      
		    theResult.push_back(LASBeamProfileFit(theHistoName, 0.0, 0.0, 0.0, 0.0));
		    // this is not a good fit
		    isGoodResult = false;
		  }
	      }
	    else
	      { 
		edm::LogWarning("BeamProfileFitter:Fit ERROR") << "<BeamProfileFitter::DoFit(...)>: Histogram for detId: " << theDetUnitId.rawId() 
							       << " is empty!!! Skipping the fit :-( ... ";

		// return an empty LASBeamProfileFit and set isGoodResult to false	      
		theResult.push_back(LASBeamProfileFit(theHistogram->GetName(), 0.0, 0.0, 0.0, 0.0));
		// this is not a good fit
		isGoodResult = false;
	      }
	  }
	else 
	  { 
	    edm::LogWarning("BeamProfileFitter:Fit ERROR") << "<BeamProfileFitter::DoFit(...)>: Histogram for detId: " << theDetUnitId.rawId()
							   << " does not exist!???? Nothing to fit :-( ... "; 

	    // return an empty LASBeamProfileFit and set isGoodResult to false	      
	    theResult.push_back(LASBeamProfileFit("no histogram found", 0.0, 0.0, 0.0, 0.0));
	    // this is not a good fit
	    isGoodResult = false;
	  }

	// clear the histogram
	if (theClearHistoAfterFit) 
	  {
	    theHistogram->Reset("");
	  }

	// return the result of the fit
	return theResult;
}




std::vector<double> BeamProfileFitter::findPeakGaus(TH1D * hist, int theDisc, int theRing)
{
	double position = -1.0;
	double positionError = -1.0;
	double sigmaError = -1.0;

	TF1 *fitSignal   = new TF1("fitSignal", "gaus"  ,0.0,50000.0);
	TF1 *fitFun;

	hist->SetLineColor(kBlue);
	int    iMax    = hist->GetMaximumBin();
	double mu      = hist->GetBinCenter(iMax);
	double sigma   = 200.0;
	double sMax    = hist->GetBinContent(iMax);
	hist->SetMaximum(1.3*sMax);
	hist->SetMinimum(0.0);

	if (sMax < theMinSignalHeight) 
	{
		LogDebug("BeamProfileFitter:findPeakGaus") << "R" << theRing 
			<< ": Laser signal below threshold for Disc " 
			<< theDisc+1;
		std::vector<double> result;
		for (int i = 0; i < 4; i++)
			{ result.push_back(-2.0); }

		return result; // no signal
	}


	fitSignal->SetParameters(sMax,mu,sigma);
	hist->Fit("fitSignal","EQ","",mu-4*sigma,mu+4*sigma);
	fitFun = hist->GetFunction("fitSignal");
	sMax       = fitFun->GetParameter(0);
	mu         = fitFun->GetParameter(1);
	sigma      = fabs(fitFun->GetParameter(2));
	fitSignal->SetParameters(sMax,mu,sigma);
	hist->Fit("fitSignal","EQ","",mu-2*sigma,mu+2*sigma); //was 3*sigma
	fitFun     = hist->GetFunction("fitSignal");
	position = fitFun->GetParameter(1);

	// only needed for Ring 6; setting sMax,mu and sigma from previous fit
	// makes fit result worse? therefore only refitting with smaller range?
	if ( (theDisc == 0 || theDisc == 1)  &&  (theRing == 6) )
	{
		fitSignal->SetParameters(sMax,mu,sigma);
		hist->Fit("fitSignal","EQ","",mu-1*sigma,mu+1*sigma);
		fitFun     = hist->GetFunction("fitSignal");
		position = fitFun->GetParameter(1);

	}

	int iBin0 = hist->FindBin(position-6*sigma);
	int iBin1 = hist->FindBin(position+6*sigma);
	bool refit = false;
	for (int i=iBin0; i<=iBin1; i++) 
	{
		if (hist->GetBinContent(i)<=0) 
		{
			hist->SetBinError(i,100000.0);
			refit = true;
		}
	}
	if (refit) 
	{
		if (theDisc != 0 && theDisc != 1)
		{
			fitSignal->SetParameters(hist->GetBinContent(iMax),hist->GetBinCenter(iMax),200.0);
			hist->Fit("fitSignal","EQ","",mu-4*sigma,mu+4*sigma);
			fitFun = hist->GetFunction("fitSignal");
			position = fitFun->GetParameter(1);
		}
		else 
		{
			sigma = 100.0;
			fitSignal->SetParameters(hist->GetBinContent(iMax),hist->GetBinCenter(iMax),200.0);
			hist->Fit("fitSignal","EQ","",mu-4*sigma,mu+4*sigma);
			fitFun = hist->GetFunction("fitSignal");
			sMax       = fitFun->GetParameter(0);
			mu         = fitFun->GetParameter(1);
			sigma      = fabs(fitFun->GetParameter(2));
			fitSignal->SetParameters(sMax,mu,sigma);
			hist->Fit("fitSignal","EQ","",mu-2*sigma,mu+2*sigma); //was 3*sigma
			fitFun = hist->GetFunction("fitSignal");
			position = fitFun->GetParameter(1);
			sigma = fitFun->GetParameter(2);
		}
	}

	position      = fitFun->GetParameter(1);
	positionError = fitFun->GetParError(1);
	sigma         = fitFun->GetParameter(2);
	sigmaError    = fitFun->GetParError(2);


	std::vector<double> result;
	result.push_back(position);
	result.push_back(sigma);
	result.push_back(positionError);
	result.push_back(sigmaError);

	return result;
}  

Double_t BeamProfileFitter::phiError(TVector3 thePosition, TMatrix theCovarianceMatrix)
{
	// function to calculate the error on phi, using the position and covariance matrix of the LaserProfile
	Double_t aX = thePosition.X();
	Double_t aY = thePosition.Y();
	Double_t aVarX = theCovarianceMatrix(0,0);
	Double_t aVarY = theCovarianceMatrix(1,1);
	Double_t aCovXY = theCovarianceMatrix(0,1);

	Double_t thePhiError = 0.0; // error to calculate

	thePhiError = TMath::Sqrt( TMath::Abs( pow(aY, 2)/( pow(aX, 4) + 2 * pow(aX,2) * pow(aY,2) + pow(aY, 4) ) * aVarX          // first term in the error propagation for sigma x
		+ pow(aX, 2)/( pow(aX, 4) + 2 * pow(aX,2) * pow(aY,2) + pow(aY, 4) ) * aVarY        // second term in the error propagation for sigma y
		- 2.0 * aX*aY/( pow(aX, 4) + 2 * pow(aX,2) * pow(aY,2) + pow(aY, 4) ) *aCovXY       // third term in the error propagation for cov(x,y)
		));

	return thePhiError;
}

