#include <fstream>

#include "TFile.h"
#include "TTree.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"  
#include "Alignment/CommonAlignment/interface/AlignableNavigator.h"  
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"  
#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"
#include "Alignment/CommonAlignment/interface/SurveyResidual.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterStore.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterSelector.h"
#include "Alignment/HIPAlignmentAlgorithm/interface/HIPUserVariables.h"
#include "Alignment/HIPAlignmentAlgorithm/interface/HIPUserVariablesIORoot.h"
#include "Alignment/MuonAlignment/interface/AlignableMuon.h"
#include <DataFormats/GeometrySurface/interface/LocalError.h> 
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/CommonAlignment/interface/AlignableExtras.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "FWCore/Framework/interface/ValidityInterval.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"


#include "Alignment/HIPAlignmentAlgorithm/interface/HIPAlignmentAlgorithm.h"

// using namespace std;

// Constructor ----------------------------------------------------------------

HIPAlignmentAlgorithm::HIPAlignmentAlgorithm(const edm::ParameterSet& cfg):
  AlignmentAlgorithmBase( cfg )
{
  
  // parse parameters
  
  verbose = cfg.getParameter<bool>("verbosity");
  
  outpath = cfg.getParameter<std::string>("outpath");
  outfile = cfg.getParameter<std::string>("outfile");
  outfile2 = cfg.getParameter<std::string>("outfile2");
  struefile = cfg.getParameter<std::string>("trueFile");
  smisalignedfile = cfg.getParameter<std::string>("misalignedFile");
  salignedfile = cfg.getParameter<std::string>("alignedFile");
  siterationfile = cfg.getParameter<std::string>("iterationFile");
  suvarfile = cfg.getParameter<std::string>("uvarFile");
  sparameterfile = cfg.getParameter<std::string>("parameterFile");
  ssurveyfile = cfg.getParameter<std::string>("surveyFile");
	
  outfile        =outpath+outfile;//Eventwise tree
  outfile2       =outpath+outfile2;//Alignablewise tree
  struefile      =outpath+struefile;
  smisalignedfile=outpath+smisalignedfile;
  salignedfile   =outpath+salignedfile;
  siterationfile =outpath+siterationfile;
  suvarfile      =outpath+suvarfile;
  sparameterfile =outpath+sparameterfile;
  ssurveyfile    =outpath+ssurveyfile;
	
  // parameters for APE
  theApplyAPE = cfg.getParameter<bool>("applyAPE");
  theAPEParameterSet = cfg.getParameter<std::vector<edm::ParameterSet> >("apeParam");
	
  theMaxAllowedHitPull = cfg.getParameter<double>("maxAllowedHitPull");
  theMinimumNumberOfHits = cfg.getParameter<int>("minimumNumberOfHits");
  theMaxRelParameterError = cfg.getParameter<double>("maxRelParameterError");
	
  // for collector mode (parallel processing)
  isCollector=cfg.getParameter<bool>("collectorActive");
  theCollectorNJobs=cfg.getParameter<int>("collectorNJobs");
  theCollectorPath=cfg.getParameter<std::string>("collectorPath");
  theFillTrackMonitoring=cfg.getUntrackedParameter<bool>("fillTrackMonitoring");
	
  if (isCollector) edm::LogWarning("Alignment") << "[HIPAlignmentAlgorithm] Collector mode";
	
  theEventPrescale = cfg.getParameter<int>("eventPrescale");
  theCurrentPrescale = theEventPrescale;
	
  for (std::string &s : cfg.getUntrackedParameter<std::vector<std::string> >("surveyResiduals")) {
    theLevels.push_back(AlignableObjectId::stringToId(s) );
  }
	
  edm::LogWarning("Alignment") << "[HIPAlignmentAlgorithm] constructed.";
	
}

// Call at beginning of job ---------------------------------------------------

void 
HIPAlignmentAlgorithm::initialize( const edm::EventSetup& setup, 
				   AlignableTracker* tracker, AlignableMuon* muon, AlignableExtras* extras,
				   AlignmentParameterStore* store )
{
  edm::LogWarning("Alignment") << "[HIPAlignmentAlgorithm] Initializing...";

  edm::ESHandle<Alignments> globalPositionRcd;
  // FIXME! temporary solution to get highest possible run number
  const unsigned int MAX_VAL(std::numeric_limits<unsigned int>::max());
  edm::ValidityInterval iov(setup.get<GlobalPositionRcd>().validityInterval() );
  if (iov.first().eventID().run()!=1 || iov.last().eventID().run()!=MAX_VAL) {
    throw cms::Exception("DatabaseError")
      << "@SUB=AlignmentProducer::applyDB"
      << "\nTrying to apply "<< setup.get<GlobalPositionRcd>().key().name()
      << " with multiple IOVs in tag.\n"
      << "Validity range is "
      << iov.first().eventID().run() << " - " << iov.last().eventID().run();
  }
	
  // accessor Det->AlignableDet
  if ( !muon )
    theAlignableDetAccessor = new AlignableNavigator(tracker);
  else if ( !tracker )
    theAlignableDetAccessor = new AlignableNavigator(muon);
  else 
    theAlignableDetAccessor = new AlignableNavigator(tracker, muon);
  
  // set alignmentParameterStore
  theAlignmentParameterStore=store;
	
  // get alignables
  theAlignables = theAlignmentParameterStore->alignables();
	
  // clear theAPEParameters, if necessary
  theAPEParameters.clear();
	
  // get APE parameters
  if(theApplyAPE){
    AlignmentParameterSelector selector(tracker, muon);
    for (std::vector<edm::ParameterSet>::const_iterator setiter = theAPEParameterSet.begin();
	 setiter != theAPEParameterSet.end();
	 ++setiter) {
      std::vector<Alignable*> alignables;
      
      selector.clear();
      edm::ParameterSet selectorPSet = setiter->getParameter<edm::ParameterSet>("Selector");
      std::vector<std::string> alignParams = selectorPSet.getParameter<std::vector<std::string> >("alignParams");
      if (alignParams.size() == 1  &&  alignParams[0] == std::string("selected")) {
	alignables = theAlignables;
      }
      else {
	selector.addSelections(selectorPSet);
	alignables = selector.selectedAlignables();
      }
			
      std::vector<double> apeSPar = setiter->getParameter<std::vector<double> >("apeSPar");
      std::vector<double> apeRPar = setiter->getParameter<std::vector<double> >("apeRPar");
      std::string function = setiter->getParameter<std::string>("function");
			
      if (apeSPar.size() != 3  ||  apeRPar.size() != 3)
	throw cms::Exception("BadConfig") << "apeSPar and apeRPar must have 3 values each" << std::endl;
			
      for (std::vector<double>::const_iterator i = apeRPar.begin();  i != apeRPar.end();  ++i) {
	apeSPar.push_back(*i);
      }
			
      if (function == std::string("linear")) {
	apeSPar.push_back(0.); // c.f. note in calcAPE
      }
      else if (function == std::string("exponential")) {
	apeSPar.push_back(1.); // c.f. note in calcAPE
      }
      else if (function == std::string("step")) {
	apeSPar.push_back(2.); // c.f. note in calcAPE
      }
      else {
	throw cms::Exception("BadConfig") << "APE function must be \"linear\" or \"exponential\"." << std::endl;
      }

      theAPEParameters.push_back(std::pair<std::vector<Alignable*>, std::vector<double> >(alignables, apeSPar));
    }
  }
}

// Call at new loop -------------------------------------------------------------
void HIPAlignmentAlgorithm::startNewLoop( void )
{
	
  // iterate over all alignables and attach user variables
  for( std::vector<Alignable*>::const_iterator it=theAlignables.begin(); 
       it!=theAlignables.end(); it++ )
    {
      AlignmentParameters* ap = (*it)->alignmentParameters();
      int npar=ap->numSelected();
      HIPUserVariables* userpar = new HIPUserVariables(npar);
      ap->setUserVariables(userpar);
    }
	
  // try to read in alignment parameters from a previous iteration
  AlignablePositions theAlignablePositionsFromFile =
    theIO.readAlignableAbsolutePositions(theAlignables,
					 salignedfile.c_str(),-1,ioerr);
	
  int numAlignablesFromFile = theAlignablePositionsFromFile.size();
	
  if (numAlignablesFromFile==0) { // file not there: first iteration 
		
    // set iteration number to 1
    if (isCollector) theIteration=0;
    else theIteration=1;
    edm::LogWarning("Alignment") << "[HIPAlignmentAlgorithm] File not found => iteration "<<theIteration;
		
    // get true (de-misaligned positions) and write to root file
    // hardcoded iteration=1
    theIO.writeAlignableOriginalPositions(theAlignables,
					  struefile.c_str(),1,false,ioerr);
		
    // get misaligned positions and write to root file
    // hardcoded iteration=1
    theIO.writeAlignableAbsolutePositions(theAlignables,
					  smisalignedfile.c_str(),1,false,ioerr);
		
  }
	
  else { // there have been previous iterations
		
    edm::LogWarning("Alignment") << "[HIPAlignmentAlgorithm] Alignables Read " 
				 << numAlignablesFromFile;
		
    // get iteration number from file     
    theIteration = readIterationFile(siterationfile);
		
    // increase iteration
    theIteration++;
    edm::LogWarning("Alignment") <<"[HIPAlignmentAlgorithm] Iteration increased by one!";
		
    // now apply psotions of file from prev iteration
    edm::LogWarning("Alignment") <<"[HIPAlignmentAlgorithm] Apply positions from file ...";
    theAlignmentParameterStore->applyAlignableAbsolutePositions(theAlignables, 
								theAlignablePositionsFromFile,ioerr);
		
  }
	
  edm::LogWarning("Alignment") <<"[HIPAlignmentAlgorithm] Current Iteration number: " 
			       << theIteration;
	
	
  // book root trees
  bookRoot();
	
	
  /*---------------------moved to terminate------------------------------
    if (theLevels.size() > 0)
    {
    edm::LogWarning("Alignment") << "[HIPAlignmentAlgorithm] Using survey constraint";
	 
    unsigned int nAlignable = theAlignables.size();
	 
    for (unsigned int i = 0; i < nAlignable; ++i)
    {
    const Alignable* ali = theAlignables[i];
	 
    AlignmentParameters* ap = ali->alignmentParameters();
	 
    HIPUserVariables* uservar =
    dynamic_cast<HIPUserVariables*>(ap->userVariables());
	 
    for (unsigned int l = 0; l < theLevels.size(); ++l)
    {
    SurveyResidual res(*ali, theLevels[l], true);
	 
    if ( res.valid() )
    {
    AlgebraicSymMatrix invCov = res.inverseCovariance();
	 
    // variable for tree
    AlgebraicVector sensResid = res.sensorResidual();
    m3_Id = ali->id();
    m3_ObjId = theLevels[l];
    m3_par[0] = sensResid[0]; m3_par[1] = sensResid[1]; m3_par[2] = sensResid[2];
    m3_par[3] = sensResid[3]; m3_par[4] = sensResid[4]; m3_par[5] = sensResid[5];
	 
    uservar->jtvj += invCov;
    uservar->jtve += invCov * sensResid;
	 
    theTree3->Fill();
    }
    }
	 
    // 	align::LocalVectors residuals = res1.pointsResidual();
	 
    // 	unsigned int nPoints = residuals.size();
	 
    // 	for (unsigned int k = 0; k < nPoints; ++k)
    // 	{
    // 	  AlgebraicMatrix J = term->survey()->derivatives(k);
    // 	  AlgebraicVector e(3); // local residual
	 
    // 	  const align::LocalVector& lr = residuals[k];
	 
    // 	  e(1) = lr.x(); e(2) = lr.y(); e(3) = lr.z();
	 
    // 	  uservar->jtvj += invCov1.similarity(J);
    // 	  uservar->jtve += J * (invCov1 * e);
    // 	}
	 
    }
    }
    //------------------------------------------------------*/
	
  // set alignment position error 
  setAlignmentPositionError();
	
  // run collector job if we are in parallel mode
  if (isCollector) collector();
	
}

// Call at end of job ---------------------------------------------------------

void HIPAlignmentAlgorithm::terminate(const edm::EventSetup& iSetup)
{
	
  edm::LogWarning("Alignment") << "[HIPAlignmentAlgorithm] Terminating";
	
  // calculating survey residuals
  if (theLevels.size() > 0)
    {
      edm::LogWarning("Alignment") << "[HIPAlignmentAlgorithm] Using survey constraint";
		
      unsigned int nAlignable = theAlignables.size();
		
      for (unsigned int i = 0; i < nAlignable; ++i)
	{
	  const Alignable* ali = theAlignables[i];
			
	  AlignmentParameters* ap = ali->alignmentParameters();
			
	  HIPUserVariables* uservar =
	    dynamic_cast<HIPUserVariables*>(ap->userVariables());
			
	  for (unsigned int l = 0; l < theLevels.size(); ++l)
	    {
	      SurveyResidual res(*ali, theLevels[l], true);
				
	      if ( res.valid() )
		{
		  AlgebraicSymMatrix invCov = res.inverseCovariance();
					
		  // variable for tree
		  AlgebraicVector sensResid = res.sensorResidual();
		  m3_Id = ali->id();
		  m3_ObjId = theLevels[l];
		  m3_par[0] = sensResid[0]; m3_par[1] = sensResid[1]; m3_par[2] = sensResid[2];
		  m3_par[3] = sensResid[3]; m3_par[4] = sensResid[4]; m3_par[5] = sensResid[5];
					
		  uservar->jtvj += invCov;
		  uservar->jtve += invCov * sensResid;
					
		  theTree3->Fill();
		}
	    }
			
	  // 	align::LocalVectors residuals = res1.pointsResidual();
			
	  // 	unsigned int nPoints = residuals.size();
			
	  // 	for (unsigned int k = 0; k < nPoints; ++k)
	  // 	{
	  // 	  AlgebraicMatrix J = term->survey()->derivatives(k);
	  // 	  AlgebraicVector e(3); // local residual
			
	  // 	  const align::LocalVector& lr = residuals[k];
			
	  // 	  e(1) = lr.x(); e(2) = lr.y(); e(3) = lr.z();
			
	  // 	  uservar->jtvj += invCov1.similarity(J);
	  // 	  uservar->jtve += J * (invCov1 * e);
	  // 	}
			
	}
    }
	
  // write user variables
  HIPUserVariablesIORoot HIPIO;
  HIPIO.writeHIPUserVariables (theAlignables,suvarfile.c_str(),
			       theIteration,false,ioerr);
	
  // now calculate alignment corrections ...
  int ialigned=0;
  // iterate over alignment parameters
  for(std::vector<Alignable*>::const_iterator
	it=theAlignables.begin(); it!=theAlignables.end(); it++) {
    Alignable* ali=(*it);
    // Alignment parameters
    AlignmentParameters* par = ali->alignmentParameters();

    // try to calculate parameters
    bool test = calcParameters(ali);

    // if successful, apply parameters
    if (test) { 
      edm::LogInfo("Alignment") << "now apply params";
      theAlignmentParameterStore->applyParameters(ali);
      // set these parameters 'valid'
      ali->alignmentParameters()->setValid(true);
      // increase counter
      ialigned++;
    }
    else par->setValid(false);
  }
  edm::LogWarning("Alignment") << "[HIPAlignmentAlgorithm::terminate] Aligned units: " << ialigned;
	
  // fill alignable wise root tree
  fillRoot(iSetup);
	
  edm::LogWarning("Alignment") << "[HIPAlignmentAlgorithm] Writing aligned parameters to file: " << theAlignables.size();
	
  // write new absolute positions to disk
  theIO.writeAlignableAbsolutePositions(theAlignables,
					salignedfile.c_str(),theIteration,false,ioerr);
	
  // write alignment parameters to disk
  theIO.writeAlignmentParameters(theAlignables, 
				 sparameterfile.c_str(),theIteration,false,ioerr);
	
  // write iteration number to file
  writeIterationFile(siterationfile,theIteration);
	
  // write out trees and close root file
	
  // eventwise tree
  theFile->cd();
  theTree->Write();
  delete theFile;
	
  if (theLevels.size() > 0){
    theFile3->cd();
    theTree3->Write();
    delete theFile3;
  }
	
  // alignable-wise tree is only filled once
  if (theIteration==1) { // only for 1st iteration
    theFile2->cd();
    theTree2->Write(); 
    delete theFile2;
  }  
	
}

bool HIPAlignmentAlgorithm::processHit1D(const AlignableDetOrUnitPtr& alidet,
					 const Alignable* ali,
					 const TrajectoryStateOnSurface & tsos,
					 const TransientTrackingRecHit* hit)
{
  static const unsigned int hitDim = 1;

  // get trajectory impact point
  LocalPoint alvec = tsos.localPosition();
  AlgebraicVector pos(hitDim);
  pos[0] = alvec.x();

  // get impact point covariance
  AlgebraicSymMatrix ipcovmat(hitDim);
  ipcovmat[0][0] = tsos.localError().positionError().xx();

  // get hit local position and covariance
  AlgebraicVector coor(hitDim);
  coor[0] = hit->localPosition().x();

  AlgebraicSymMatrix covmat(hitDim);
  covmat[0][0] = hit->localPositionError().xx();
  
  // add hit and impact point covariance matrices
  covmat = covmat + ipcovmat;

  // calculate the x pull of this hit
  double xpull = 0.;
  if (covmat[0][0] != 0.) xpull = (pos[0] - coor[0])/sqrt(fabs(covmat[0][0]));

  // get Alignment Parameters
  AlignmentParameters* params = ali->alignmentParameters();
  // get derivatives
  AlgebraicMatrix derivs2D = params->selectedDerivatives(tsos, alidet);
  // calculate user parameters
  int npar = derivs2D.num_row();
  // std::cout << "Dumping derivsSEL: \n" << derivs2D <<std::endl;
  AlgebraicMatrix derivs(npar, hitDim, 0);
  
  for (int ipar=0;ipar<npar;ipar++) {
    for (unsigned int idim=0;idim<hitDim;idim++) {
      derivs[ipar][idim] = derivs2D[ipar][idim];
    }
  }
  // std::cout << "Dumping derivs: \n" << derivs << std::endl;
  // invert covariance matrix
  int ierr;
  // if (covmat[1][1]>4.0) nhitDim = hitDim;
				  
  covmat.invert(ierr);
  if (ierr != 0) {
    edm::LogError("Alignment") << "Matrix inversion failed!"; 
    return false; 
  }

  bool useThisHit = (theMaxAllowedHitPull <= 0.);
  
  useThisHit |= (fabs(xpull) < theMaxAllowedHitPull);
  
  // bailing out
  if (!useThisHit) return false;

  // std::cout << "We use this hit in " << subDet << std::endl;
  // std::cout << "Npars= " << npar << "  NhitDim=1 Size of derivs: "
  //           << derivs.num_row() << " x " << derivs.num_col() << std::endl;

  // AlgebraicMatrix jtvjtmp();
  AlgebraicMatrix covtmp(covmat);
  // std::cout << "Preapring JTVJTMP -> " << std::flush;
  AlgebraicMatrix jtvjtmp(derivs * covtmp *derivs.T());
  // std::cout << "TMP JTVJ= \n" << jtvjtmp << std::endl;
  AlgebraicSymMatrix thisjtvj(npar);
  AlgebraicVector thisjtve(npar);
  // std::cout << "Preparing ThisJtVJ" << std::flush;
  // thisjtvj = covmat.similarity(derivs);
  thisjtvj.assign(jtvjtmp);
  // std::cout<<" ThisJtVE"<<std::endl;
  thisjtve=derivs * covmat * (pos-coor);
  
  AlgebraicVector hitresidual(hitDim);
  hitresidual[0] = (pos[0] - coor[0]);
  
  AlgebraicMatrix hitresidualT;
  hitresidualT = hitresidual.T();
  // std::cout << "HitResidualT = \n" << hitresidualT << std::endl;

  // access user variables (via AlignmentParameters)
  HIPUserVariables* uservar = dynamic_cast<HIPUserVariables*>(params->userVariables());
  uservar->jtvj += thisjtvj;
  uservar->jtve += thisjtve;
  uservar->nhit++;
  // The following variable is needed for the extended 1D/2D hit fix using
  // matrix shrinkage and expansion
  // uservar->hitdim = hitDim;

  //for alignable chi squared
  float thischi2;
  thischi2 = (hitresidualT *covmat *hitresidual)[0]; 
					
  if ( verbose && ((thischi2/ static_cast<float>(uservar->nhit)) >10.0) ) {
    edm::LogWarning("Alignment") << "Added to Chi2 the number " << thischi2 <<" having "
				 << uservar->nhit << "  dof  " << std::endl << "X-resid " 
				 << hitresidual[0] << "  Y-resid " 
				 << hitresidual[1] << std::endl << "  Cov^-1 matr (covmat): [0][0]= "
				 << covmat[0][0] << " [0][1]= "
				 << covmat[0][1] << " [1][0]= "
				 << covmat[1][0] << " [1][1]= "
				 << covmat[1][1] << std::endl;
  }
					
  uservar->alichi2 += thischi2; // a bit weird, but vector.transposed * matrix * vector doesn't give a double in CMSSW's opinion
  uservar->alindof += hitDim;   // 2D hits contribute twice to the ndofs

  return true;
}

bool HIPAlignmentAlgorithm::processHit2D(const AlignableDetOrUnitPtr& alidet,
					 const Alignable* ali,
					 const TrajectoryStateOnSurface & tsos,
					 const TransientTrackingRecHit* hit)
{
  static const unsigned int hitDim = 2;

  // get trajectory impact point
  LocalPoint alvec = tsos.localPosition();
  AlgebraicVector pos(hitDim);
  pos[0] = alvec.x();
  pos[1] = alvec.y();
  
  // get impact point covariance
  AlgebraicSymMatrix ipcovmat(hitDim);
  ipcovmat[0][0] = tsos.localError().positionError().xx();
  ipcovmat[1][1] = tsos.localError().positionError().yy();
  ipcovmat[0][1] = tsos.localError().positionError().xy();
  
  // get hit local position and covariance
  AlgebraicVector coor(hitDim);
  coor[0] = hit->localPosition().x();
  coor[1] = hit->localPosition().y();

  AlgebraicSymMatrix covmat(hitDim);
  covmat[0][0] = hit->localPositionError().xx();
  covmat[1][1] = hit->localPositionError().yy();
  covmat[0][1] = hit->localPositionError().xy();

  // add hit and impact point covariance matrices
  covmat = covmat + ipcovmat;

  // calculate the x pull and y pull of this hit
  double xpull = 0.;
  double ypull = 0.;
  if (covmat[0][0] != 0.) xpull = (pos[0] - coor[0])/sqrt(fabs(covmat[0][0]));
  if (covmat[1][1] != 0.) ypull = (pos[1] - coor[1])/sqrt(fabs(covmat[1][1]));

  // get Alignment Parameters
  AlignmentParameters* params = ali->alignmentParameters();
  // get derivatives
  AlgebraicMatrix derivs2D = params->selectedDerivatives(tsos, alidet);
  // calculate user parameters
  int npar = derivs2D.num_row();
  // std::cout << "Dumping derivsSEL: \n"<< derivs2D <<std::endl;
  AlgebraicMatrix derivs(npar, hitDim, 0);
  
  for (int ipar=0;ipar<npar;ipar++) {
    for (unsigned int idim=0;idim<hitDim;idim++) {
      derivs[ipar][idim] = derivs2D[ipar][idim];
    }
  }
  // std::cout << "Dumping derivs: \n" << derivs << std::endl;
  // invert covariance matrix
  int ierr;
  // if (covmat[1][1]>4.0) nhitDim = hitDim;
				  
  covmat.invert(ierr);
  if (ierr != 0) { 
    edm::LogError("Alignment") << "Matrix inversion failed!"; 
    return false; 
  }

  bool useThisHit = (theMaxAllowedHitPull <= 0.);
  
  useThisHit |= (fabs(xpull) < theMaxAllowedHitPull  &&  fabs(ypull) < theMaxAllowedHitPull);
  
  // bailing out
  if (!useThisHit) return false;
  
  // std::cout << "We use this hit in " << subDet << std::endl;
  // std::cout << "Npars= " << npar << "  NhitDim=2 Size of derivs: "
  //           << derivs.num_row() << " x " << derivs.num_col() << std::endl;

  // AlgebraicMatrix jtvjtmp();
  AlgebraicMatrix covtmp(covmat);
  // std::cout << "Preapring JTVJTMP -> " << std::flush;
  AlgebraicMatrix jtvjtmp(derivs * covtmp *derivs.T());
  // std::cout << "TMP JTVJ= \n" << jtvjtmp << std::endl;
  AlgebraicSymMatrix thisjtvj(npar);
  AlgebraicVector thisjtve(npar);
  // std::cout << "Preparing ThisJtVJ" << std::flush;
  // thisjtvj = covmat.similarity(derivs);
  thisjtvj.assign(jtvjtmp);
  // std::cout<<" ThisJtVE"<<std::endl;
  thisjtve=derivs * covmat * (pos-coor);
  
  AlgebraicVector hitresidual(hitDim);
  hitresidual[0] = (pos[0] - coor[0]);
  hitresidual[1] = (pos[1] - coor[1]);
  // if(nhitDim>1)  {
  //  hitresidual[1] =0.0;
  //  nhitDim=1;
  // }
	 
  AlgebraicMatrix hitresidualT;
  hitresidualT = hitresidual.T();
  // std::cout << "HitResidualT = \n" << hitresidualT << std::endl;
  // access user variables (via AlignmentParameters)
  HIPUserVariables* uservar = dynamic_cast<HIPUserVariables*>(params->userVariables());
  uservar->jtvj += thisjtvj;
  uservar->jtve += thisjtve;
  uservar->nhit++;
  // The following variable is needed for the extended 1D/2D hit fix using
  // matrix shrinkage and expansion
  // uservar->hitdim = hitDim;

  //for alignable chi squared
  float thischi2;
  thischi2 = (hitresidualT *covmat *hitresidual)[0]; 
					
  if ( verbose && ((thischi2/ static_cast<float>(uservar->nhit)) >10.0) ) {
    edm::LogWarning("Alignment") << "Added to Chi2 the number " << thischi2 <<" having "
				 << uservar->nhit << "  dof  " << std::endl << "X-resid " 
				 << hitresidual[0] << "  Y-resid " 
				 << hitresidual[1] << std::endl << "  Cov^-1 matr (covmat): [0][0]= "
				 << covmat[0][0] << " [0][1]= "
				 << covmat[0][1] << " [1][0]= "
				 << covmat[1][0] << " [1][1]= "
				 << covmat[1][1] << std::endl;
  }
					
  uservar->alichi2 += thischi2; // a bit weird, but vector.transposed * matrix * vector doesn't give a double in CMSSW's opinion
  uservar->alindof += hitDim;   // 2D hits contribute twice to the ndofs

  return true;
}

// Run the algorithm on trajectories and tracks -------------------------------
void HIPAlignmentAlgorithm::run(const edm::EventSetup& setup, const EventInfo &eventInfo)
{
  if (isCollector) return;
	
  TrajectoryStateCombiner tsoscomb;
  
  // AM: not really needed
  // AM: m_Ntracks = 0 should be sufficient
  int itr=0;
  m_Ntracks=0;
  for(itr=0;itr<MAXREC;++itr){
    m_Nhits[itr]=0;
    m_Pt[itr]=-5.0;
    m_P[itr]=-5.0;
    m_nhPXB[itr]=0;
    m_nhPXF[itr]=0;
    m_nhTIB[itr]=0;
    m_nhTOB[itr]=0;
    m_nhTIB[itr]=0;
    m_nhTEC[itr]=0;
    m_Eta[itr]=-99.0;
    m_Phi[itr]=-4.0;
    m_Chi2n[itr]=-11.0;
    m_d0[itr]=-999;
    m_dz[itr]=-999;
  }
  itr=0;
	
  // AM: what is this needed for?
  theFile->cd();
	
  // loop over tracks  
  const ConstTrajTrackPairCollection &tracks = eventInfo.trajTrackPairs();
  for (ConstTrajTrackPairCollection::const_iterator it=tracks.begin();
       it!=tracks.end();
       ++it) {
		
    const Trajectory* traj = (*it).first;
    const reco::Track* track = (*it).second;
		
    float pt    = track->pt();
    float eta   = track->eta();
    float phi   = track->phi();
    float p     = track->p();
    float chi2n = track->normalizedChi2();
    int   nhit  = track->numberOfValidHits();
    float d0    = track->d0();
    float dz    = track->dz();

    int nhpxb   = track->hitPattern().numberOfValidPixelBarrelHits();
    int nhpxf   = track->hitPattern().numberOfValidPixelEndcapHits();
    int nhtib   = track->hitPattern().numberOfValidStripTIBHits();
    int nhtob   = track->hitPattern().numberOfValidStripTOBHits();
    int nhtid   = track->hitPattern().numberOfValidStripTIDHits();
    int nhtec   = track->hitPattern().numberOfValidStripTECHits();

    if (verbose) edm::LogInfo("Alignment") << "New track pt,eta,phi,chi2n,hits: "
					   << pt << ","
					   << eta << ","
					   << phi << ","
					   << chi2n << ","
					   << nhit;
    // edm::LogWarning("Alignment") << "New track pt,eta,phi,chi2n,hits: " 
    //	  		        << pt << ","
    //			        << eta << ","
    //			        << phi << ","
    //			        << chi2n << ","
    //			        << nhit;
		
    // fill track parameters in root tree
    if (itr<MAXREC) {
      m_Nhits[itr]=nhit;
      m_Pt[itr]=pt;
      m_P[itr]=p;
      m_Eta[itr]=eta;
      m_Phi[itr]=phi;
      m_Chi2n[itr]=chi2n;
      m_nhPXB[itr]=nhpxb;
      m_nhPXF[itr]=nhpxf;
      m_nhTIB[itr]=nhtib;
      m_nhTOB[itr]=nhtob;
      m_nhTID[itr]=nhtid;
      m_nhTEC[itr]=nhtec;
      m_d0[itr]=d0;
      m_dz[itr]=dz;
      itr++;
      m_Ntracks=itr;
    }
    // AM: Can be simplified
		
    std::vector<const TransientTrackingRecHit*> hitvec;
    std::vector<TrajectoryStateOnSurface> tsosvec;
		
    // loop over measurements	
    std::vector<TrajectoryMeasurement> measurements = traj->measurements();
    for (std::vector<TrajectoryMeasurement>::iterator im=measurements.begin();
	 im!=measurements.end();
	 ++im) {
   
      TrajectoryMeasurement meas = *im;

      // const TransientTrackingRecHit* ttrhit = &(*meas.recHit());
      // const TrackingRecHit *hit = ttrhit->hit();
      const TransientTrackingRecHit* hit = &(*meas.recHit());

      if (hit->isValid() && theAlignableDetAccessor->detAndSubdetInMap( hit->geographicalId() )) {

	// this is the updated state (including the current hit)
	// TrajectoryStateOnSurface tsos=meas.updatedState();
	// combine fwd and bwd predicted state to get state 
	// which excludes current hit
	
	//////////Hit prescaling part 	 
	bool skiphit = false; 	 
	if (eventInfo.clusterValueMap()) { 	 
	  // check from the PrescalingMap if the hit was taken. 	 
	  // If not skip to the next TM 	 
	  // bool hitTaken=false; 	 
	  AlignmentClusterFlag myflag; 	 
	  
	  int subDet = hit->geographicalId().subdetId();
	  //take the actual RecHit out of the Transient one
	  const TrackingRecHit *rechit=hit->hit();
	  if (subDet>2) { // AM: if possible use enum instead of hard-coded value	 
	    const std::type_info &type = typeid(*rechit); 	 
	    
	    if (type == typeid(SiStripRecHit1D)) { 	 
	      
	      const SiStripRecHit1D* stripHit1D = dynamic_cast<const SiStripRecHit1D*>(rechit); 	 
	      if (stripHit1D) { 	 
		SiStripRecHit1D::ClusterRef stripclust(stripHit1D->cluster()); 	 
		// myflag=PrescMap[stripclust]; 	 
		myflag = (*eventInfo.clusterValueMap())[stripclust]; 	 
	      } else { 	 
		edm::LogError("HIPAlignmentAlgorithm") 
		  << "ERROR in <HIPAlignmentAlgorithm::run>: Dynamic cast of Strip RecHit failed! "
		  << "TypeId of the RecHit: " << className(*hit) <<std::endl; 	 
	      } 	 
	      
	    }//end if type = SiStripRecHit1D 	 
	    else if(type == typeid(SiStripRecHit2D)){ 	 
	      
	      const SiStripRecHit2D* stripHit2D = dynamic_cast<const SiStripRecHit2D*>(rechit); 	 
	      if (stripHit2D) { 	 
		SiStripRecHit2D::ClusterRef stripclust(stripHit2D->cluster()); 	 
		// myflag=PrescMap[stripclust]; 	 
		myflag = (*eventInfo.clusterValueMap())[stripclust]; 	 
	      } else { 	 
		edm::LogError("HIPAlignmentAlgorithm") 
		  << "ERROR in <HIPAlignmentAlgorithm::run>: Dynamic cast of Strip RecHit failed! "
		  // << "TypeId of the TTRH: " << className(*ttrhit) << std::endl; 	 
		  << "TypeId of the TTRH: " << className(*hit) << std::endl; 	 
	      } 
	    } //end if type == SiStripRecHit2D 	 
	  } //end if hit from strips 	 
	  else { 	 
	    const SiPixelRecHit* pixelhit= dynamic_cast<const SiPixelRecHit*>(rechit); 	 
	    if (pixelhit) { 	 
	      SiPixelClusterRefNew  pixelclust(pixelhit->cluster()); 	 
	      // myflag=PrescMap[pixelclust]; 	 
	      myflag = (*eventInfo.clusterValueMap())[pixelclust]; 	 
	    }
	    else { 	 
	      edm::LogError("HIPAlignmentAlgorithm")
		<< "ERROR in <HIPAlignmentAlgorithm::run>: Dynamic cast of Pixel RecHit failed! "
		// << "TypeId of the TTRH: " << className(*ttrhit) << std::endl; 	 
		<< "TypeId of the TTRH: " << className(*hit) << std::endl; 	 
	    }
	  } //end 'else' it is a pixel hit 	 
	    // bool hitTaken=myflag.isTaken(); 	 
	  if (!myflag.isTaken()) { 	 
	    skiphit=true;
	    //std::cout<<"Hit from subdet "<<rechit->geographicalId().subdetId()<<" prescaled out."<<std::endl;
	    continue;
	  }
	}//end if Prescaled Hits 	 
	//////////////////////////////// 	 
	if (skiphit) { 	 
	  throw cms::Exception("LogicError")
	    << "ERROR  in <HIPAlignmentAlgorithm::run>: this hit should have been skipped!"
	    << std::endl;	 
	}
	
	TrajectoryStateOnSurface tsos = tsoscomb.combine(meas.forwardPredictedState(),
							 meas.backwardPredictedState());
	
	if(tsos.isValid()){
	  // hitvec.push_back(ttrhit);
	  hitvec.push_back(hit);
	  tsosvec.push_back(tsos);
	}

      } //hit valid
    }
    
    // transform RecHit vector to AlignableDet vector
    std::vector <AlignableDetOrUnitPtr> alidetvec = theAlignableDetAccessor->alignablesFromHits(hitvec);
		
    // get concatenated alignment parameters for list of alignables
    CompositeAlignmentParameters aap = theAlignmentParameterStore->selectParameters(alidetvec);
		
    std::vector<TrajectoryStateOnSurface>::const_iterator itsos=tsosvec.begin();
    std::vector<const TransientTrackingRecHit*>::const_iterator ihit=hitvec.begin();
    
    // loop over vectors(hit,tsos)
    while (itsos != tsosvec.end()) {

      // get AlignableDet for this hit
      const GeomDet* det = (*ihit)->det();
      // int subDet= (*ihit)->geographicalId().subdetId();
      uint32_t nhitDim = (*ihit)->dimension();
      
      AlignableDetOrUnitPtr alidet = theAlignableDetAccessor->alignableFromGeomDet(det);
			
      // get relevant Alignable
      Alignable* ali = aap.alignableFromAlignableDet(alidet);
      
      if (ali!=0) {
	if (nhitDim==1) {
	  processHit1D(alidet, ali, *itsos, *ihit);
	} else if (nhitDim==2) {
	  processHit2D(alidet, ali, *itsos, *ihit);
	}
      }
			
      itsos++;
      ihit++;
    } 
  } // end of track loop
	
  // fill eventwise root tree (with prescale defined in pset)
  if (theFillTrackMonitoring) {
    theCurrentPrescale--;
    if (theCurrentPrescale<=0) {
      theTree->Fill();
      theCurrentPrescale = theEventPrescale;
    }
  }
}

// ----------------------------------------------------------------------------

int HIPAlignmentAlgorithm::readIterationFile(std::string filename)
{
  int result;
  
  std::ifstream inIterFile(filename.c_str(), std::ios::in);
  if (!inIterFile) {
    edm::LogError("Alignment") << "[HIPAlignmentAlgorithm::readIterationFile] ERROR! "
			       << "Unable to open Iteration file";
    result = -1;
  }
  else {
    inIterFile >> result;
    edm::LogWarning("Alignment") << "[HIPAlignmentAlgorithm::readIterationFile] "
				 << "Read last iteration number from file: " << result;
  }
  inIterFile.close();
	
  return result;
}

// ----------------------------------------------------------------------------

void HIPAlignmentAlgorithm::writeIterationFile(std::string filename,int iter)
{
  std::ofstream outIterFile((filename.c_str()), std::ios::out);
  if (!outIterFile) {
    edm::LogError("Alignment") << "[HIPAlignmentAlgorithm::writeIterationFile] ERROR: Unable to write Iteration file";
  }
  else {
    outIterFile << iter;
    edm::LogWarning("Alignment") <<"[HIPAlignmentAlgorithm::writeIterationFile] writing iteration number to file: " << iter;
  }
  outIterFile.close();
}


// ----------------------------------------------------------------------------
// set alignment position error

void HIPAlignmentAlgorithm::setAlignmentPositionError(void)
{
	
	
  // Check if user wants to override APE
  if ( !theApplyAPE )
    {
      edm::LogWarning("Alignment") <<"[HIPAlignmentAlgorithm::setAlignmentPositionError] No APE applied";
      return; // NO APE APPLIED
    }
	
	
  edm::LogWarning("Alignment") <<"[HIPAlignmentAlgorithm::setAlignmentPositionError] Apply APE!";
	
  double apeSPar[3], apeRPar[3];
  for (std::vector<std::pair<std::vector<Alignable*>, std::vector<double> > >::const_iterator alipars = theAPEParameters.begin();
       alipars != theAPEParameters.end();
       ++alipars) {
    const std::vector<Alignable*> &alignables = alipars->first;
    const std::vector<double> &pars = alipars->second;
		
    apeSPar[0] = pars[0];
    apeSPar[1] = pars[1];
    apeSPar[2] = pars[2];
    apeRPar[0] = pars[3];
    apeRPar[1] = pars[4];
    apeRPar[2] = pars[5];
		
    double function = pars[6];
		
    // Printout for debug
    printf("APE: %u alignables\n", (unsigned int)alignables.size());
    for ( int i=0; i<21; ++i ) {
      double apelinstest=calcAPE(apeSPar,i,0.);
      double apeexpstest=calcAPE(apeSPar,i,1.);
      double apelinrtest=calcAPE(apeRPar,i,0.);
      double apeexprtest=calcAPE(apeRPar,i,1.);
      printf("APE: iter slin sexp rlin rexp: %5d %12.5f %12.5f %12.5f %12.5f\n",
	     i,apelinstest,apeexpstest,apelinrtest,apeexprtest);
    }
		
    // set APE
    double apeshift=calcAPE(apeSPar,theIteration,function);
    double aperot  =calcAPE(apeRPar,theIteration,function);
    theAlignmentParameterStore->setAlignmentPositionError( alignables, apeshift, aperot );
  }
	
}

// ----------------------------------------------------------------------------
// calculate APE

double 
HIPAlignmentAlgorithm::calcAPE(double* par, int iter, double function)
{
  double diter=(double)iter;
	
  // The following floating-point equality check is safe because this
  // "0." and this "1." are generated by the compiler, in the very
  // same file.  Whatever approximization scheme it uses to turn "1."
  // into 0.9999999999998 in the HIPAlignmentAlgorithm::initialize is
  // also used here.  If I'm wrong, you'll get an assertion.
  if (function == 0.) {
    return std::max(par[1],par[0]+((par[1]-par[0])/par[2])*diter);
  }
  else if (function == 1.) {
    return std::max(0.,par[0]*(exp(-pow(diter,par[1])/par[2])));
  }
  else if (function == 2.) {
    int ipar2 = (int) par[2];
    int step = iter/ipar2;
    double dstep = (double) step;
    return std::max(0.0, par[0] - par[1]*dstep);
  }
  else assert(false);  // should have been caught in the constructor
}


// ----------------------------------------------------------------------------
// book root trees

void HIPAlignmentAlgorithm::bookRoot(void)
{
  // create ROOT files
  theFile = new TFile(outfile.c_str(),"update");
  theFile->cd();
	
  // book event-wise ROOT Tree
	
  TString tname="T1";
  char iterString[5];
  snprintf(iterString, sizeof(iterString), "%i",theIteration);
  tname.Append("_");
  tname.Append(iterString);
	
  theTree  = new TTree(tname,"Eventwise tree");
	
  //theTree->Branch("Run",     &m_Run,     "Run/I");
  //theTree->Branch("Event",   &m_Event,   "Event/I");
  theTree->Branch("Ntracks", &m_Ntracks, "Ntracks/I");
  theTree->Branch("Nhits",    m_Nhits,   "Nhits[Ntracks]/I");       
  theTree->Branch("nhPXB",    m_nhPXB,   "nhPXB[Ntracks]/I");       
  theTree->Branch("nhPXF",    m_nhPXF,   "nhPXF[Ntracks]/I");    
  theTree->Branch("nhTIB",    m_nhTIB,   "nhTIB[Ntracks]/I");       
  theTree->Branch("nhTOB",    m_nhTOB,   "nhTOB[Ntracks]/I");    
  theTree->Branch("nhTID",    m_nhTID,   "nhTID[Ntracks]/I");       
  theTree->Branch("nhTEC",    m_nhTEC,   "nhTEC[Ntracks]/I");  
  theTree->Branch("Pt",       m_Pt,      "Pt[Ntracks]/F");
  theTree->Branch("P",        m_P,       "P[Ntracks]/F");
  theTree->Branch("Eta",      m_Eta,     "Eta[Ntracks]/F");
  theTree->Branch("Phi",      m_Phi,     "Phi[Ntracks]/F");
  theTree->Branch("Chi2n",    m_Chi2n,   "Chi2n[Ntracks]/F");
  theTree->Branch("d0",       m_d0,      "d0[Ntracks]/F");
  theTree->Branch("dz",       m_dz,      "dz[Ntracks]/F");
  
  // book Alignable-wise ROOT Tree
	
  theFile2 = new TFile(outfile2.c_str(),"update");
  theFile2->cd();
	
  theTree2 = new TTree("T2","Alignablewise tree");
	
  theTree2->Branch("Nhit",   &m2_Nhit,    "Nhit/I");
  theTree2->Branch("Type",   &m2_Type,    "Type/I");
  theTree2->Branch("Layer",  &m2_Layer,   "Layer/I");
  theTree2->Branch("Xpos",   &m2_Xpos,    "Xpos/F");
  theTree2->Branch("Ypos",   &m2_Ypos,    "Ypos/F");
  theTree2->Branch("Zpos",   &m2_Zpos,    "Zpos/F");
  theTree2->Branch("Eta",    &m2_Eta,     "Eta/F");
  theTree2->Branch("Phi",    &m2_Phi,     "Phi/F");
  theTree2->Branch("Id",     &m2_Id,      "Id/i");
  theTree2->Branch("ObjId",  &m2_ObjId,   "ObjId/I");
	
  // book survey-wise ROOT Tree only if survey is enabled
  if (theLevels.size() > 0){
		
    edm::LogWarning("Alignment") << "[HIPAlignmentAlgorithm::bookRoot] Survey trees booked.";
    theFile3 = new TFile(ssurveyfile.c_str(),"update");
    theFile3->cd();
    theTree3 = new TTree(tname, "Survey Tree");
    theTree3->Branch("Id", &m3_Id, "Id/i");
    theTree3->Branch("ObjId", &m3_ObjId, "ObjId/I");
    theTree3->Branch("Par", &m3_par, "Par[6]/F");
  }
	
  edm::LogWarning("Alignment") << "[HIPAlignmentAlgorithm::bookRoot] Root trees booked.";
	
}

// ----------------------------------------------------------------------------
// fill alignable-wise root tree

void HIPAlignmentAlgorithm::fillRoot(const edm::EventSetup& iSetup)
{
  using std::setw;
  theFile2->cd();
	
  int naligned=0;

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<IdealGeometryRecord>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();
	
  for (std::vector<Alignable*>::const_iterator it=theAlignables.begin();
       it!=theAlignables.end();
       ++it) {
    Alignable* ali = (*it);
    AlignmentParameters* dap = ali->alignmentParameters();
		
    // consider only those parameters classified as 'valid'
    if (dap->isValid()) {
			
      // get number of hits from user variable
      HIPUserVariables* uservar = dynamic_cast<HIPUserVariables*>(dap->userVariables());
      m2_Nhit = uservar->nhit;
			
      // get type/layer
      std::pair<int,int> tl = theAlignmentParameterStore->typeAndLayer(ali, tTopo);
      m2_Type = tl.first;
      m2_Layer = tl.second;
			
      // get identifier (as for IO)
      m2_Id    = ali->id();
      m2_ObjId = ali->alignableObjectId();
			
      // get position
      GlobalPoint pos = ali->surface().position();
      m2_Xpos = pos.x();
      m2_Ypos = pos.y();
      m2_Zpos = pos.z();
      m2_Eta = pos.eta();
      m2_Phi = pos.phi();
			
      AlgebraicVector pars = dap->parameters();
			
      if (verbose) {
	edm::LogVerbatim("Alignment")
	  << "------------------------------------------------------------------------\n"
	  << " ALIGNABLE: " << setw(6) << naligned
	  << '\n'
	  << "hits: "   << setw(4) << m2_Nhit
	  << " type: "  << setw(4) << m2_Type
	  << " layer: " << setw(4) << m2_Layer
	  << " id: "    << setw(4) << m2_Id
	  << " objId: " << setw(4) << m2_ObjId
	  << '\n'
	  << std::fixed << std::setprecision(5)
	  << "x,y,z: "
	  << setw(12) << m2_Xpos
	  << setw(12) << m2_Ypos 
	  << setw(12) << m2_Zpos
	  << " eta,phi: "
	  << setw(12) << m2_Eta
	  << setw(12) << m2_Phi
	  << '\n'
	  << "params: "
	  << setw(12) << pars[0]
	  << setw(12) << pars[1]
	  << setw(12) << pars[2]
	  << setw(12) << pars[3]
	  << setw(12) << pars[4]
	  << setw(12) << pars[5];
      }
      
      naligned++;
      theTree2->Fill();
    }
  }
}

// ----------------------------------------------------------------------------

bool HIPAlignmentAlgorithm::calcParameters(Alignable* ali)
{
  // Alignment parameters
  AlignmentParameters* par = ali->alignmentParameters();
  // access user variables
  HIPUserVariables* uservar = dynamic_cast<HIPUserVariables*>(par->userVariables());
  int nhit = uservar->nhit;
  // The following variable is needed for the extended 1D/2D hit fix using
  // matrix shrinkage and expansion
  // int hitdim = uservar->hitdim;
  
  if (nhit < theMinimumNumberOfHits) {
    par->setValid(false);
    return false;
  }

  AlgebraicSymMatrix jtvj = uservar->jtvj;
  AlgebraicVector jtve = uservar->jtve;

  // Shrink input in case of 1D hits and 'v' selected
  // in alignment parameters
  //   if (hitdim==1 && selector[1]==true) {
  //     int iremove = 1;
  //     if (selector[0]==false) iremove--;
  
  //     AlgebraicSymMatrix tempjtvj(jtvj.num_row()-1);
  //     int nr = 0, nc = 0;
  //     for (int r=0;r<jtvj.num_row();r++) {
  //       if (r==iremove) continue;
  //       nc = 0;
  //       for (int c=0;c<jtvj.num_col();c++) {
  //  	if (c==iremove) continue;
  //  	tempjtvj[nr][nc] = jtvj[r][c];
  //  	nc++;
  //       }
  //       nr++;
  //     }
  //     jtvj = tempjtvj;
  
  //     AlgebraicVector tempjtve(jtve.num_row()-1);
  //     nr = 0;
  //     for (int r=0;r<jtve.num_row();r++) {
  //       if (r==iremove) continue;
  //       tempjtve[nr] = jtve[r];
  //       nr++;
  //     }
  //     jtve = tempjtve;
  //   }
  
  int ierr;
  AlgebraicSymMatrix jtvjinv = jtvj.inverse(ierr);

  if (ierr !=0) {
    edm::LogError("Alignment") << "Matrix inversion failed!"; 
    return false;
  }
  
  // these are the alignment corrections+covariance (for selected params)
  AlgebraicVector params = - (jtvjinv * jtve);
  AlgebraicSymMatrix cov = jtvjinv;

  edm::LogInfo("Alignment") << "parameters " << params;
	
  // errors of parameters
  int npar = params.num_row();    
  AlgebraicVector paramerr(npar);
  AlgebraicVector relerr(npar);
  for (int i=0;i<npar;i++) {
    if (abs(cov[i][i])>0) paramerr[i] = sqrt(abs(cov[i][i]));
    else paramerr[i] = params[i];
    relerr[i] = abs(paramerr[i]/params[i]);
    if (relerr[i] >= theMaxRelParameterError) { 
      params[i] = 0; 
      paramerr[i]=0; 
    }
  }

  // expand output in case of 1D hits and 'v' selected
  // in alignment parameters
  //   if (hitdim==1 && selector[1]==true) {
  //     int iremove = 1;
  //     if (selector[0]==false) iremove--;
  
  //     AlgebraicSymMatrix tempcov(cov.num_row()+1,0);
  //     int nr = 0, nc = 0;
  //     for (int r=0;r<cov.num_row();r++) {
  //       if (r==iremove) nr++;
  //       nc = 0;
  //       for (int c=0;c<cov.num_col();c++) {
  //  	if (c==iremove) nc++;
  //  	tempcov[nr][nc] = cov[r][c];
  //  	nc++;
  //       }
  //       nr++;
  //     }
  //     cov = tempcov;
  
  //     AlgebraicVector tempparams(params.num_row()+1,0);
  //     nr = 0;
  //     for (int r=0;r<params.num_row();r++) {
  //       if (r==iremove) nr++;
  //       tempparams[nr] = params[r];
  //       nr++;
  //     }
  //     params = tempparams;
  //   }
  
  // store alignment parameters
  AlignmentParameters* parnew = par->cloneFromSelected(params,cov);
  ali->setAlignmentParameters(parnew);
  parnew->setValid(true);

  return true;
}

//-----------------------------------------------------------------------------

void HIPAlignmentAlgorithm::collector(void)
{
  edm::LogWarning("Alignment") << "[HIPAlignmentAlgorithm::collector] called for iteration "
			       << theIteration << std::endl;
	
  HIPUserVariablesIORoot HIPIO;
	
  for (int ijob=1;ijob<=theCollectorNJobs;ijob++) {
		
    edm::LogWarning("Alignment") << "reading uservar for job " << ijob;
    
    std::stringstream ss;
    std::string str;
    ss << ijob;
    ss >> str;
    std::string uvfile = theCollectorPath+"/job"+str+"/IOUserVariables.root";
		
    std::vector<AlignmentUserVariables*> uvarvec = 
      HIPIO.readHIPUserVariables(theAlignables, uvfile.c_str(),
				 theIteration, ioerr);
    
    if (ioerr!=0) { 
      edm::LogWarning("Alignment") << "[HIPAlignmentAlgorithm::collector] could not read user variable files for job " 
				   << ijob;
      continue;
    }
		
    // add
    std::vector<AlignmentUserVariables*> uvarvecadd;
    std::vector<AlignmentUserVariables*>::const_iterator iuvarnew=uvarvec.begin(); 
    for (std::vector<Alignable*>::const_iterator it=theAlignables.begin(); 
	 it!=theAlignables.end();
	 ++it) {
      Alignable* ali = *it;
      AlignmentParameters* ap = ali->alignmentParameters();
			
      HIPUserVariables* uvarold = dynamic_cast<HIPUserVariables*>(ap->userVariables());
      HIPUserVariables* uvarnew = dynamic_cast<HIPUserVariables*>(*iuvarnew);
			
      HIPUserVariables* uvar = uvarold->clone();
      if (uvarnew!=0) {
	uvar->nhit = (uvarold->nhit)+(uvarnew->nhit);
	uvar->jtvj = (uvarold->jtvj)+(uvarnew->jtvj);
	uvar->jtve = (uvarold->jtve)+(uvarnew->jtve);
	uvar->alichi2 = (uvarold->alichi2)+(uvarnew->alichi2);
	uvar->alindof = (uvarold->alindof)+(uvarnew->alindof);
	delete uvarnew;
      }
			
      uvarvecadd.push_back(uvar);
      iuvarnew++;
    }
    
    theAlignmentParameterStore->attachUserVariables(theAlignables, uvarvecadd, ioerr);
		
    //fill Eventwise Tree
    if (theFillTrackMonitoring) {
      uvfile = theCollectorPath+"/job"+str+"/HIPAlignmentEvents.root";
      edm::LogWarning("Alignment") << "Added to the tree "
				   << fillEventwiseTree(uvfile.c_str(), theIteration, ioerr)
				   << "tracks";
    }
		
  }//end loop on jobs
}

//------------------------------------------------------------------------------------

int HIPAlignmentAlgorithm::fillEventwiseTree(const char* filename, int iter, int ierr)
{	
  int totntrk = 0;
  char treeName[64];
  snprintf(treeName, sizeof(treeName), "T1_%d", iter);
  //open the file "HIPAlignmentEvents.root" in the job directory
  TFile *jobfile = new TFile(filename, "READ");
  //grab the tree corresponding to this iteration
  TTree *jobtree = (TTree*)jobfile->Get(treeName);
  //address and read the variables 
  static const int nmaxtrackperevent = 1000;
  int jobNtracks, jobNhitspertrack[nmaxtrackperevent], jobnhPXB[nmaxtrackperevent], jobnhPXF[nmaxtrackperevent],jobnhTIB[nmaxtrackperevent], jobnhTOB[nmaxtrackperevent],jobnhTID[nmaxtrackperevent], jobnhTEC[nmaxtrackperevent];
  float jobP[nmaxtrackperevent], jobPt[nmaxtrackperevent], jobEta[nmaxtrackperevent] , jobPhi[nmaxtrackperevent];
  float jobd0[nmaxtrackperevent], jobdz[nmaxtrackperevent] , jobChi2n[nmaxtrackperevent];
	
  jobtree->SetBranchAddress("Ntracks", &jobNtracks);
  jobtree->SetBranchAddress("Nhits",   jobNhitspertrack);
  jobtree->SetBranchAddress("nhPXB",   jobnhPXB);
  jobtree->SetBranchAddress("nhPXF",   jobnhPXF);
  jobtree->SetBranchAddress("nhTIB",   jobnhTIB);
  jobtree->SetBranchAddress("nhTOB",   jobnhTOB);
  jobtree->SetBranchAddress("nhTID",   jobnhTID);
  jobtree->SetBranchAddress("nhTEC",   jobnhTEC);
  jobtree->SetBranchAddress("Pt",      jobPt);
  jobtree->SetBranchAddress("P",       jobP);
  jobtree->SetBranchAddress("d0",      jobd0);
  jobtree->SetBranchAddress("dz",      jobdz);
  jobtree->SetBranchAddress("Eta",     jobEta);
  jobtree->SetBranchAddress("Phi",     jobPhi);
  jobtree->SetBranchAddress("Chi2n",   jobChi2n);
  int ievent = 0;
  for (ievent=0;ievent<jobtree->GetEntries();++ievent) {
    jobtree->GetEntry(ievent);
		
    //fill the collector tree with them
    
    //  TO BE IMPLEMENTED: a prescale factor like in run()
    m_Ntracks = jobNtracks;
    int ntrk = 0;
    while (ntrk<m_Ntracks) {
      if (ntrk<MAXREC) {
	totntrk = ntrk+1;
	m_Nhits[ntrk] = jobNhitspertrack[ntrk];
	m_Pt[ntrk] = jobPt[ntrk];
	m_P[ntrk] = jobP[ntrk];
	m_nhPXB[ntrk] = jobnhPXB[ntrk];
	m_nhPXF[ntrk] = jobnhPXF[ntrk];
	m_nhTIB[ntrk] = jobnhTIB[ntrk];
	m_nhTOB[ntrk] = jobnhTOB[ntrk];
	m_nhTID[ntrk] = jobnhTID[ntrk];
	m_nhTEC[ntrk] = jobnhTEC[ntrk];
	m_Eta[ntrk] = jobEta[ntrk];
	m_Phi[ntrk] = jobPhi[ntrk];
	m_Chi2n[ntrk] = jobChi2n[ntrk];
	m_d0[ntrk] = jobd0[ntrk];
	m_dz[ntrk] = jobdz[ntrk];
      }//end if j<MAXREC
      else{
	edm::LogWarning("Alignment") << "[HIPAlignmentAlgorithm::fillEventwiseTree] Number of tracks in Eventwise tree exceeds MAXREC: "
				     << m_Ntracks << "  Skipping exceeding tracks.";
	ntrk = m_Ntracks+1;
      }
      ++ntrk;
    }//end while loop
    theTree->Fill();
  }//end loop on i - entries in the job tree

  //clean up
  delete jobtree;
  delete jobfile;
  
  return totntrk;
}//end fillEventwiseTree
