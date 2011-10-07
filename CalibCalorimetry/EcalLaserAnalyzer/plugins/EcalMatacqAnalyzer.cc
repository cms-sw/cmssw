/* 
 *  \class EcalMatacqAnalyzer
 *
 *  $Date: 2010/04/12 14:30:53 $
 *  primary author: Gautier Hamel De Monchenault - CEA/Saclay
 *  author: Julie Malcles - CEA/Saclay
 */


#include <TFile.h>
#include <TTree.h>
#include <TProfile.h>
#include <TChain.h>
#include <TF1.h>
#include <TVirtualFFT.h>
#include <vector>

#include <EcalMatacqAnalyzer.h>

#include <sstream>
#include <iostream>
#include <iomanip>

#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include <FWCore/Utilities/interface/Exception.h>

#include <DataFormats/EcalDetId/interface/EcalDetIdCollections.h>
#include <DataFormats/EcalRawData/interface/EcalRawDataCollections.h>

#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>

#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TMatacq.h>
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TMom.h>
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TMTQ.h>

using namespace std;

//========================================================================
EcalMatacqAnalyzer::EcalMatacqAnalyzer(const edm::ParameterSet& iConfig)
 //========================================================================
  :

iEvent(0),

// framework parameters with default values

_presample(     iConfig.getUntrackedParameter< double       >( "nPresamples",   300. ) ),
_nsamplesaftmax(iConfig.getUntrackedParameter< unsigned int >( "nSamplesAftMax",  80 ) ),
_noiseCut(      iConfig.getUntrackedParameter< unsigned int >( "noiseCut",         7 ) ),
_parabnbefmax(  iConfig.getUntrackedParameter< unsigned int >( "paraBeforeMax",    8 ) ),
_parabnaftmax(  iConfig.getUntrackedParameter< unsigned int >( "paraAfterMax",     7 ) ),
_thres(         iConfig.getUntrackedParameter< unsigned int >( "threshold",       10 ) ),
_lowlev(        iConfig.getUntrackedParameter< unsigned int >( "lowLevel",        20 ) ),
_highlev(       iConfig.getUntrackedParameter< unsigned int >( "highLevel",       80 ) ),
_nevlasers(     iConfig.getUntrackedParameter< unsigned int >( "nEventLaser",    600 ) ),
_timebefmax(    iConfig.getUntrackedParameter< unsigned int >( "timeBefMax",     100 ) ),
_timeaftmax(    iConfig.getUntrackedParameter< unsigned int >( "timeAftMax",     250 ) ),
_cutwindow(     iConfig.getUntrackedParameter< double       >( "cutWindow",      0.001) ),
_nsamplesshape( iConfig.getUntrackedParameter< unsigned int >( "nSamplesShape",  250 ) ),
_presampleshape(iConfig.getUntrackedParameter< unsigned int >( "nPresamplesShape",50 ) ),
_slide(         iConfig.getUntrackedParameter< unsigned int >( "nSlide",         100 ) ),
_fedid(         iConfig.getUntrackedParameter< int          >( "fedID",         -999 ) ),
_debug(         iConfig.getUntrackedParameter< int          >( "debug",           0  ) ),             
nSides(NSIDES), lightside(0), runType(-1), runNum(0), 
event(0), color(-1), maxsamp(0), nsamples(0), tt(0)
  
  //========================================================================
{
  

  //now do what ever initialization is needed
  
  resdir_                  = iConfig.getUntrackedParameter<std::string>("resDir");
  
  digiCollection_          = iConfig.getParameter<std::string>("digiCollection");
  digiProducer_            = iConfig.getParameter<std::string>("digiProducer");
  
  eventHeaderCollection_   = iConfig.getParameter<std::string>("eventHeaderCollection");
  eventHeaderProducer_     = iConfig.getParameter<std::string>("eventHeaderProducer");
  
}

//========================================================================
EcalMatacqAnalyzer::~EcalMatacqAnalyzer(){
//========================================================================

// do anything here that needs to be done at desctruction time
// (e.g. close files, deallocate resources etc.)

}



//========================================================================
void EcalMatacqAnalyzer::beginJob() {
//========================================================================

// Define temporary file name

  sampfile=resdir_;
  sampfile+="/TmpTreeMatacqAnalyzer.root";
  
  sampFile = new TFile(sampfile.c_str(),"RECREATE");


  // declaration of the tree to fill
  
  tree = new TTree("MatacqTree","MatacqTree");


    //List of branches

    tree->Branch( "event",       &event,        "event/I"        );
    tree->Branch( "color",       &color ,       "color/I"        );
    tree->Branch( "matacq",      &matacq ,      "matacq[2560]/D" );
    tree->Branch( "nsamples",    &nsamples ,    "nsamples/I"     );
    tree->Branch( "maxsamp",     &maxsamp ,     "maxsamp/I"      );
    tree->Branch( "tt",          &tt ,          "tt/D"           );
    tree->Branch( "lightside",   &lightside ,   "lightside/I"    );
    
    tree->SetBranchAddress( "event",       &event       );
    tree->SetBranchAddress( "color",       &color       );
    tree->SetBranchAddress( "matacq",      matacq       ); 
    tree->SetBranchAddress( "nsamples",    &nsamples    );
    tree->SetBranchAddress( "maxsamp",     &maxsamp     );
    tree->SetBranchAddress( "tt",          &tt          );
    tree->SetBranchAddress( "lightside",   &lightside   );

    
    // Define output results files' names
    
    stringstream namefile;
    namefile << resdir_ <<"/MATACQ.root";      
    outfile=namefile.str();
     

    // Laser events counte
    laserEvents[0]=0;
    matacqEvents[0]=0;
    laserEvents[1]=0;
    matacqEvents[1]=0;
    isThereMatacq=false;      

}


//========================================================================
void EcalMatacqAnalyzer:: analyze( const edm::Event & e, const  edm::EventSetup& c){
//========================================================================

  ++iEvent;
  
  if (_debug==2 )cout << "-- debug test -- Entering Analyze -- event= "<<iEvent<< endl; 

  // retrieving MATACQ :
  edm::Handle<EcalMatacqDigiCollection> pmatacqDigi;
  const EcalMatacqDigiCollection* matacqDigi=0;
  try {
    e.getByLabel(digiProducer_,digiCollection_, pmatacqDigi); 
    matacqDigi=pmatacqDigi.product();
    if (_debug==2 )cout << "-- debug test -- Matacq Digis Found -- "<< endl; 
    
  }catch ( std::exception& ex ) {
    std::cerr << "Error! can't get the product EcalMatacqDigi producer:" << digiProducer_.c_str()<<" collection:"<<digiCollection_.c_str() << std::endl;
    if (_debug==2 )cout << "-- debug test -- No Matacq Digis Found -- "<< endl; 
    return;
  }
  
  // retrieving DCC header

  edm::Handle<EcalRawDataCollection> pDCCHeader;
  const  EcalRawDataCollection* DCCHeader=0;
  try {
    e.getByLabel(eventHeaderProducer_,eventHeaderCollection_, pDCCHeader);
    DCCHeader=pDCCHeader.product();
  }catch ( std::exception& ex ) {
    std::cerr << "Error! can't get the product EcalRawData producer:" << eventHeaderProducer_.c_str()<<" collection:"<<eventHeaderCollection_.c_str() << std::endl;
    return;
  }

  // ====================================
  // Decode Basic DCCHeader Information 
  // ====================================
  
  if (_debug==2)  cout <<"-- debug test -- Before header -- "<< endl; 
  
  for ( EcalRawDataCollection::const_iterator headerItr= DCCHeader->begin();headerItr != DCCHeader->end(); 
	++headerItr ) {
    
    EcalDCCHeaderBlock::EcalDCCEventSettings settings = headerItr->getEventSettings(); 
    color = (int) settings.wavelength;
    if( color<0 ) return;

    // Get run type and run number 

    int fed = headerItr->fedId();  
    
    if(fed!=_fedid && _fedid!=-999) continue; 
    
    runType=headerItr->getRunType();
    runNum=headerItr->getRunNumber();
    event=headerItr->getLV1();

    if (_debug==2)  cout <<"-- debug test -- runtype:"<<runType<<" event:"<<event <<" runNum:"<<runNum <<endl; 

    dccID=headerItr->getDccInTCCCommand();
    fedID=headerItr->fedId();  
    lightside=headerItr->getRtHalf();

    //assert (lightside<2 && lightside>=0);

    if( lightside!=1 && lightside!=0 ) {
      cout << "Unexpected lightside: "<< lightside<<" for event "<<iEvent << endl;
      return;
    }
    if (_debug==2) {
      cout <<"-- debug test -- Inside header before fed cut -- color="<<color<< ", dcc="<<dccID<<", fed="<< fedID<<",  lightside="<<lightside<<", runType="<<runType<< endl;
    }

    // take event only if the fed corresponds to the DCC in TCC
    if( 600+dccID != fedID ) continue;
    
    if (_debug==2) {
      cout <<"-- debug test -- Inside header after fed cut -- color="<<color<< ", dcc="<<dccID<<", fed="<< fedID<<",  lightside="<<lightside<<", runType="<<runType<< endl;
    }

    // Cut on runType

    if ( runType!=EcalDCCHeaderBlock::LASER_STD && runType!=EcalDCCHeaderBlock::LASER_GAP && runType!=EcalDCCHeaderBlock::LASER_POWER_SCAN && runType!=EcalDCCHeaderBlock::LASER_DELAY_SCAN ) return; 
    
    
    vector<int>::iterator iter= find( colors.begin(), colors.end(), color );
    if( iter==colors.end() ){
      colors.push_back( color );
    }
    vector<int>::iterator iterside = find( sides.begin(), sides.end(), lightside );
    if( iterside==sides.end() ){
      sides.push_back( lightside );
    }
  }
  
  if (_debug==2) cout <<"-- debug test -- Before digis -- Event:"<<iEvent<< endl; 

  // Count laser events
  laserEvents[lightside]++;
  


// ===========================
// Decode Matacq Information
// ===========================

  int iCh=0;
  double max=0., min=0.;

  for(EcalMatacqDigiCollection::const_iterator it = matacqDigi->begin(); it!=matacqDigi->end(); ++it){

    // Loop on matacq channel 
    
    // be carefull here:
    //====================
    // 2 matacq channels corresponding to 2 diodes!
    // first one is more relevant
    // adccounts are >0 for first one and <0 for second one

 
    const EcalMatacqDigi& digis = *it;
    
    if (_debug==2) {
      cout <<"-- debug test -- Inside digis -- channel="<< iCh<<" size="<<digis.size()<< endl;
    }

    if( digis.size()!= N_samples ) continue; 
    else{
      isThereMatacq=true;
    }

    max=0;
    min=0;
    maxsamp=0;
    nsamples=digis.size();
    tt=digis.tTrig();

    // TESTJULIE
    /*
    int is=(2559-440);
    for(int i=0; i<digis.size(); ++i){ // Loop on matacq samples      

      matacq[is]=digis.adcCount(i);
      if(++is>=N_samples)is-=N_samples;
      
      if(matacq[is]>max) {
	max=matacq[is];
	maxsamp=is;
      }
      if(matacq[is]<min) {
	min=matacq[is];
      }
    }
    */
    
    for(int i=0; i<digis.size(); ++i){ // Loop on matacq samples      
      matacq[i]=digis.adcCount(i);
      if(matacq[i]>max) {
	max=matacq[i];
	maxsamp=i;
      }
      if(matacq[i]<min) {
	min=matacq[i];
      }
    } 
    
    if (_debug==1) cout <<"min="<<min<<" max="<<max<<" iCh="<<iCh<< endl;
    
    // cut second channel and low amplitudes

    //if( max<50.0 || iCh==1 ){
    if( iCh==1 ){
      continue;
    }
    
    matacqEvents[lightside]++;
  
    if (_debug==2) {
      cout <<"-- debug test -- Inside digis -- nsamples="<<nsamples<< ", max="<<max<< endl;
    }
    
    tree->Fill();

    iCh++; 
    
  }
  
  
} // analyze


//========================================================================
void EcalMatacqAnalyzer::endJob() 
{
  
  // Don't do anything if there is no events
  if( !isThereMatacq ) {
    
    cout << "\n\t***  No MATACQ Events  ***"<< endl;

    // Remove temporary file    
    FILE *test; 
    test = fopen(sampfile.c_str(),"r");
    if (test){
      fclose(test);
      stringstream del2;
      del2 << "rm " <<sampfile;
      system(del2.str().c_str());
    }
    return;
  }
  
  assert( colors.size()<= nColor );
  unsigned int nCol=colors.size();
  unsigned int nSide=sides.size();
  if (_debug==2) cout<<"Endjob NCOLOR:"<< nCol<<" NSIDES:"<<nSide<< endl;

  TProfile *shapeMatTmp[NCOL][NSIDES];
  TProfile *shapeMarcTmp[NCOL][NSIDES];  


  TH1D *shapeMat[NCOL][NSIDES];
  
  TH1D *shapeBigMarc[NCOL][NSIDES];
  TH1D *shapeBigMarcNorm[NCOL][NSIDES];

  TH1D *shapeMarc[NCOL][NSIDES];

  double fmax=1000.; // Matacq has a digitizing step of 1 ns, So max frequency of FFT is 1 GHz=1000 MHz
  TH1F *hphase1 = new TH1F("phase1","Matacq spectrum phase",FFT2_SIZE-1,0.,fmax); // First bin is for DC content of the signal
  TH1F *hphase2 = new TH1F("phase2","Matacq spectrum phase, filtered",FFT2_SIZE-1,0.,fmax);
  TH1F *hmod1   = new TH1F("mod1","Matacq spectrum modulus",FFT2_SIZE-1,0.,fmax);
  TH1F *hmod2   = new TH1F("mod2","Matacq spectrum modulus, filtered",FFT2_SIZE-1,0.,fmax);
  TH1F *htmp    =new TH1F("ph_tmp","ph_tmp",FFT2_SIZE,0.,fmax);
  TF1 *g1   = new TF1("g1","[0]*exp(-(x-[1])*(x-[1])/2./[2]/[2])",0.,fmax);
  TF1 *p1   = new TF1("p1","pol1",0.,fmax);
  TF1 *exp1 = new TF1("exp1","expo",0.,N_samples);



  
  for(unsigned int iCol=0;iCol<nCol;iCol++){
    for(unsigned int iSide=0;iSide<NSIDES;iSide++){
      
      MTQ[iCol][iSide] = new TMTQ();

      stringstream namea;
      namea<<"LaserCol"<<colors[iCol]<<"Side"<<iSide;
      stringstream nameb;
      nameb<<"LaserTmpCol"<<colors[iCol]<<"Side"<<iSide;

      stringstream namec;
      namec<<"BigMarcCol"<<colors[iCol]<<"Side"<<iSide;

      stringstream namec2;      
      namec2<<"BigMarcNormCol"<<colors[iCol]<<"Side"<<iSide;

      stringstream named;
      named<<"LaserTmpMarcCol"<<colors[iCol]<<"Side"<<iSide;
      stringstream namee;
      
      namee<<"LaserMarcCol"<<colors[iCol]<<"Side"<<iSide;
      
      
      shapeMat[iCol][iSide] = new TH1D(namea.str().c_str(), namea.str().c_str(), _nsamplesshape,-0.5,double(_nsamplesshape)-0.5);
      shapeMatTmp[iCol][iSide] = new TProfile(nameb.str().c_str(),nameb.str().c_str(),_timeaftmax+_timebefmax,-0.5,double(_timeaftmax+_timebefmax)-0.5,-2000.0,2000.0); 

      shapeMarcTmp[iCol][iSide] = new TProfile(named.str().c_str(),named.str().c_str(),N_samples,0.0,double(N_samples),-2000.0,2000.0);     
      
      shapeBigMarc[iCol][iSide] = new TH1D(namec.str().c_str(),namec.str().c_str(),N_samples,-0.5,double(N_samples)-0.5);
      shapeBigMarcNorm[iCol][iSide] = new TH1D(namec2.str().c_str(),namec2.str().c_str(),N_samples,-0.5,double(N_samples)-0.5);
     
      shapeMarc[iCol][iSide] = new TH1D(namee.str().c_str(),namee.str().c_str(), _nsamplesshape,-0.5,double(_nsamplesshape)-0.5);

    }
  }
  
  outFile = new TFile(outfile.c_str(),"RECREATE");
  
  cout << "\n\t+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+" << endl;
  cout <<   "\t+=+     Analyzing MATACQ data     +=+" << endl;
  cout<<    "\t+=+     ........... events: "<<matacqEvents[0]+matacqEvents[1]<<"  +=+" << endl;
  //  cout<<    "\t+=+     ........... laser events: side 0: "<<matacqEvents[0]<<"  +=+" << endl;
  //cout<<    "\t+=+     ........... laser events: side 1: "<<matacqEvents[1]<<"  +=+" << endl;
  //cout<<    "\t+=+     ........... matacq events: side 0: "<<laserEvents[0]<<"  +=+" << endl;
  //cout<<    "\t+=+     ........... matacq events: side 1: "<<laserEvents[1]<<"  +=+" << endl;
  
  
  //
  // create output ntuple
  //

  mtqShape = new TTree("MatacqShape","MatacqShape");

  // list of branches 
  // keep Patrice's notations

  mtqShape->Branch( "event",       &event,       "event/I"       );
  mtqShape->Branch( "color",       &color,       "color/I"       );
  mtqShape->Branch( "side",        &lightside,   "lightside/I"   );
  mtqShape->Branch( "status",      &status,      "status/I"      );
  mtqShape->Branch( "peak",        &peak ,       "peak/D"        );
  mtqShape->Branch( "sigma",       &sigma ,      "sigma/D"       );
  mtqShape->Branch( "fit",         &fit ,        "fit/D"         );
  mtqShape->Branch( "ampl",        &ampl ,       "ampl/D"        );
  mtqShape->Branch( "trise",       &trise ,      "trise/D"       );  
  mtqShape->Branch( "fwhm",        &fwhm ,       "fwhm/D"        );    
  mtqShape->Branch( "fw20",        &fw20 ,       "fw20/D"        );  
  mtqShape->Branch( "fw80",        &fw80 ,       "fw80/D"        );  
  mtqShape->Branch( "ped",         &ped ,        "ped/D"         );   
  mtqShape->Branch( "pedsig",      &pedsig ,     "pedsig/D"      );   
  mtqShape->Branch( "ttrig",       &ttrig ,      "ttrig/D"       );  
  mtqShape->Branch( "sliding",     &sliding ,    "sliding/D"     );  

  mtqShape->SetBranchAddress( "event",       &event       );
  mtqShape->SetBranchAddress( "color",       &color       );
  mtqShape->SetBranchAddress( "side",        &lightside   );
  mtqShape->SetBranchAddress( "status",      &status      ); 
  mtqShape->SetBranchAddress( "peak",        &peak        ); 
  mtqShape->SetBranchAddress( "sigma",       &sigma       ); 
  mtqShape->SetBranchAddress( "fit",         &fit         ); 
  mtqShape->SetBranchAddress( "ampl",        &ampl        ); 
  mtqShape->SetBranchAddress( "fwhm",        &fwhm        ); 
  mtqShape->SetBranchAddress( "fw20",        &fw20        ); 
  mtqShape->SetBranchAddress( "fw80",        &fw80        ); 
  mtqShape->SetBranchAddress( "trise",       &trise       ); 
  mtqShape->SetBranchAddress( "ped",         &ped         ); 
  mtqShape->SetBranchAddress( "pedsig",      &pedsig      ); 
  mtqShape->SetBranchAddress( "ttrig",       &ttrig       ); 
  mtqShape->SetBranchAddress( "sliding",     &sliding     ); 

  
  unsigned int endsample;
  unsigned int presample;
  
  double matacq_noped[2560];
  double *pedcyc=new double[20];
  
  // loop over the entries of the tree
  TChain* fChain = (TChain*)tree;
  Long64_t nentries = fChain->GetEntriesFast();
  Long64_t nbytes = 0, nb = 0;
  
  for( Long64_t jentry=0; jentry<nentries; jentry++ ) 
    {
      // load the event
      Long64_t ientry = fChain->LoadTree(jentry);
      if (ientry < 0) break;
      nb = fChain->GetEntry( jentry );   nbytes += nb;      

      bool isok=true;

      status   = 0;
      peak    = -1;
      sigma   =  0;
      fit     = -1;
      ampl    = -1;
      trise   = -1; 
      ttrig   = tt; 
      fwhm    =  0;
      fw20    =  0;
      fw80    =  0;
      ped     =  0;
      pedsig  =  0;
      sliding =  0;


      if (_debug==2)cout <<"-- debug test -- inside loop 1  -- jentry:"<<jentry<<" over nentries="<<nentries<< endl; 

      // create the object for Matacq data analysis

      endsample = maxsamp+_nsamplesaftmax;
      presample=int(_presample);

      TMatacq* mtq = new TMatacq( nsamples, presample, endsample,
				  _noiseCut, _parabnbefmax, _parabnaftmax,
				  _thres, _lowlev, _highlev,
				  _nevlasers , _slide);

      if (_debug>=1){
	if(jentry<10){
	  cout<<" matacqinloop["<< jentry<<"] ";
	  for(int i=0;i<2560;i++) cout <<matacq[i]<<" ";
	  cout <<endl;
	}
      }
      
      // analyze the Matacq data
      if( mtq->rawPulseAnalysis( nsamples, &matacq[0] )==0 ) 
	{
	  status = 1;
	  ped =  mtq->getBaseLine();
	  pedsig =  mtq->getsigBaseLine();
	  pedcyc = mtq->getPedCyc();

	  if (_debug==2) cout <<"-- debug test -- inside loop 3  -- ped:"<<ped << endl; 
	  if( mtq->findPeak()==0 ) 
	    {
	      peak = mtq->getTimpeak();
	      sigma = mtq->getsigTimpeak();
	    }else{
	    // isok=false;
	    peak=0;
	    sigma=0;
	    if (_debug==2)cout<<" Pulse "<< jentry<<" not OK: Peak Not Found "<< endl;
	  }

	  if (_debug==2)cout <<"-- debug test -- inside loop 4  -- peak:"<<peak<< endl; 
	  if( mtq->doFit2()==0 ) 
	    {
	      fit  = mtq->getTime();
	      ampl = mtq->getAmpl(); 
	      fwhm = mtq->getFwhm();
	      fw20 = mtq->getWidth20();
	      fw80 = mtq->getWidth80();
	      sliding = mtq->getSlide();
	      if( mtq->compute_trise()==0 ) 
		{
		  trise = mtq->getTrise();
		}
	    }else {
	    if (_debug==2)cout<<" Pulse "<< jentry <<" not OK for doFit2"<< endl;
	    isok=false;
	  }
	  if (_debug==2)cout <<"-- debug test -- inside loop 4  -- ampl:"<<ampl<< endl; 
	  if (_debug==2)cout <<"-- debug test -- inside loop 4  -- time:"<<fit<< endl; 
	  if (_debug==2)cout <<"-- debug test -- inside loop 5  -- trise:"<<trise<< endl; 
	}else{
	if (_debug==2) cout<<" Pulse "<< jentry <<" not OK for rawPulseAnalysis"<< endl;
	isok=false;
      }    
      if(ampl<50. ) {
	isok=false;
	if (_debug==2) cout<<" Pulse "<< jentry <<" not OK for low amplitude:"<<ampl<< endl;
      }
      if (_debug==2)cout <<"-- debug test -- inside loop 6  -- status:"<<status<<" isok:"<<isok<< endl; 

      
      // get back color
      //================
      int iCol=0;
      for(unsigned int i=0;i<nCol;i++){
	if(color==colors[i]) {
	  iCol=i;
	  i=nCol;
	}
      }
    
      if(_debug==1){
	if(jentry<10){
	  cout<<" matacqinloop2["<< jentry<<"] ";
	  for(int i=0;i<2560;i++)cout <<matacq[i]<<" ";
	  cout <<endl;
	}
      }
      
      
      if (_debug==2) cout <<"-- debug test -- inside loop 6.5  -- iCol:"<<iCol<<" "<<
	isok<<" "<<jentry<<" "<<lightside<< endl; 
      
      if( isok ){
	
	int firstS=int(fit-double(_timebefmax));
	int lastS=int(fit+double(_timeaftmax));
	
	if(_debug>=1 && jentry<10){
	  cout <<" matacq results["<<jentry<<"] ampl="<<ampl<<" "<< fwhm<<" "<<fit<<" "<<N_samples<<endl;
	  cout<<"ped_cyc ";
	  for(int j=0;j<20;j++){
	    if(j<19) cout <<pedcyc[j]<<" ";
	    else cout <<pedcyc[j]<<endl;
	  }
	  cout<<" matacq_noped["<< jentry <<"] ";
	}
      
	double fdumx, fdumy;
	for (int imtqq=0;imtqq<N_samples;imtqq++){	  
	  if (_debug==1) cout<<" Begin loop 1 : " << imtqq<<"  " <<matacq[imtqq]<<"  " <<pedcyc[imtqq%20] << endl;
	  matacq_noped[imtqq]=matacq[imtqq]-pedcyc[imtqq%20];
	  if (_debug==1) cout<<" Begin loop 2 : " << imtqq<<"  " <<matacq_noped[imtqq] << endl;
	  
	  fdumx=(double)imtqq;
	  fdumy=matacq_noped[imtqq];
	  fdumx+=(1450.-fit);
	  
	  if(fdumx<0.) fdumx+=N_samples; // TESTJULIE
	  //if((double(imtqq)+1450.-fit)>0.0 && ampl>0.0){
	  
	  if (_debug==1) cout<<" Filling " << fdumx<<"  " <<fdumy<<"  "<<matacq_noped[imtqq]<<"  " <<ampl<<"  " <<iCol<<"  " <<lightside<<" ..." << endl;
	  shapeMarcTmp[iCol][lightside]->Fill(fdumx,fdumy);  	     
	
	  if (_debug==1) cout<<" ... Done" << endl;
	  //}
	  
	  if (_debug==1) cout<<" End loop " <<imtqq<< endl;
	}
      
	
	// Fill histo if there are enough samples
	if (_debug==2)cout <<"-- debug test -- inside loop 7  -- firstS:"<<firstS<<", nsamples:"<< nsamples<< endl;
	
	if(firstS>=0 && lastS<=nsamples){
	  
	  for (int i=firstS;i<lastS;i++){
	    shapeMatTmp[iCol][lightside]->Fill(double(i)-firstS,matacq_noped[i]);
	  }
	  
	}else{  // else extrapolate 
	  
	  
	  int firstSBis;
	  
	  if(firstS<0){ // fill first bins with 0
	    
	    double thisped=0.0;
	    //thisped=(matacq[0]+matacq[1]+matacq[2]+matacq[4]+matacq[5])/5.0;
	    
	    for(int i=firstS;i<0;i++){
	      shapeMatTmp[iCol][lightside]->Fill(double(i)-firstS,thisped); 
	    }
	    firstSBis=0; 
	  }else{
	    firstSBis=firstS;
	  }
	  
	  if(lastS>nsamples){
	    
	    for(int i=firstSBis;i<int(nsamples);i++){
	      shapeMatTmp[iCol][lightside]->Fill(double(i)-firstS,matacq_noped[i]);
	    }
	    
	    //extrapolate with expo tail

	    double expb=0.998;
	    double matacqval=expb*matacq_noped[nsamples-1];
	    
	    for(int i=nsamples;i<lastS;i++){
	      shapeMatTmp[iCol][lightside]->Fill(double(i)-firstS,matacqval);
	      matacqval*=expb;
	    }
	    
	  }else{    
	    for (int i=firstSBis;i<lastS;i++){
	      shapeMatTmp[iCol][lightside]->Fill(double(i)-firstS,matacq_noped[i]);
	    }	    
	  }
	}
	
      }

      if (_debug==2)cout <<"-- debug test -- inside loop 8"<< endl;

      if (_debug==2)cout <<"-- debug test -- inside loop 8bis color:"<<color<<" iCol:"<<iCol<<" nCol:"<< nCol<< endl;
      
      // fill TMTQ 

      if( isok )
	MTQ[iCol][lightside]->addEntry(peak, sigma, fit, ampl, trise, fwhm, fw20, fw80, ped, pedsig, sliding);
      
      // fill the output tree
      
      if (_debug==2)cout <<"-- debug test -- inside loop 9"<< endl;
      mtqShape->Fill();
      
      // clean up
      delete mtq;
    }
  
  if (_debug==2){
    cout <<"-- debug test -- after loop "<< endl;
  }
  sampFile->Close();


  // Compute Marc Shape:
  //=====================

  double laser1[N_samples], laser2[FFT2_SIZE];
  double laser3[FFT2_SIZE];
  int  nbin=FFT2_SIZE;
  
  TVirtualFFT *fft_f = TVirtualFFT::FFT(1, &nbin, "C2CF M K");
  TVirtualFFT *fft_b = TVirtualFFT::FFT(1, &nbin, "C2CB M K");
  double rex[FFT2_SIZE],imx[FFT2_SIZE],rey[FFT2_SIZE],imy[FFT2_SIZE];
  double mod[FFT2_SIZE],phase[FFT2_SIZE];
  double pi=asin(1.)*2.; 
  int iMarcfirst, iMarcmax;

  double FW_50[NCOL][NSIDES], FW_10[NCOL][NSIDES], FW_05[NCOL][NSIDES]; // from mean shape

  for (unsigned int icol=0;icol<nCol;icol++){
    for(unsigned int iside=0; iside<nSide; iside++){
      
      for(int ibin=0; ibin<N_samples; ibin++) laser1[ibin] = shapeMarcTmp[icol][iside]->GetBinContent(ibin+1);
      
      // Extract the signal and extend it up "nbin" ns
      iMarcfirst=-1;
      iMarcmax=0;
      double las_max=0., las_ped=0.;
      
      for(int i=0; i<N_samples; i++)
      {
        if(i>100 && i<=300)las_ped+=laser1[i];
        if(fabs(laser1[i])>5. && iMarcfirst==-1)iMarcfirst=i;
        if(fabs(laser1[i])>fabs(las_max))
        {
          las_max=laser1[i];
          iMarcmax=i;
        }
      }
      las_ped/=200.;
      if(_debug == 1) printf("Matacq start at %d, max = %f at %d, ped = %f\n",iMarcfirst,las_max,iMarcmax,las_ped);
      if(las_ped>0.01)printf("Warning : laser pedestal too high (%f) should not happen !\n",las_ped);
      for(int ibin=0; ibin<N_samples; ibin++) laser1[ibin] -= las_ped;
    
      double w50, w10, w05;

      if(fabs(las_max)>1. && iMarcmax>FFT_START)
      {
// Apply filter to remove the oscillations :
        shapeMarcTmp[icol][iside]->Fit(exp1,"q","w",2000.,2500.);

        for(int i=0; i<FFT2_SIZE; i++) 
        {    
          if(i+FFT_START<2300)
            rex[i]=laser1[FFT_START+i];
          else 
            rex[i]=exp1->Eval((double)(i+FFT_START));
          imx[i]=0.;
          rey[i]=0.;
          imy[i]=0.;
        }    
        fft_f->SetPointsComplex(rex, imx);
        fft_f->Transform();
        fft_f->GetPointsComplex(rey, imy);
        for(int i=0; i<FFT2_SIZE; i++) 
        {    
          rey[i]/=FFT2_SIZE;
          imy[i]/=FFT2_SIZE;
          mod[i]=sqrt(rey[i]*rey[i]+imy[i]*imy[i]);
          phase[i]=atan2(imy[i],rey[i]);
          if(_debug == 1) printf("%d %f %f %f %f\n",i,rey[i],imy[i],mod[i],phase[i]);
        }    
        hmod1->Reset();
        hphase1->Reset();
        hmod2->Reset();
        hphase2->Reset();
        htmp->Reset();
        for(int i=1; i<FFT2_SIZE; i++) 
        {    
          int j=i-1; // bin 0 is for DC content of signal
          hmod1->SetBinContent(j+1,mod[i]);
          hphase1->SetBinContent(j+1,phase[i]);
          hmod2->SetBinContent(j+1,mod[i]);
          hphase2->SetBinContent(j+1,phase[i]);
        }    
        g1->SetParameter(0,mod[1]*2.);
        g1->SetParameter(1,-10.);
        g1->SetParameter(2,20.);
        hmod1->Fit(g1,"Q","W",8.,19.); // Fit on a region where we have no/low noise

        double ph_old=phase[0];
        for(int i=1; i<=FFT2_SIZE/2; i++) 
        {    
          int j=i-1;
          double ph=phase[i];
          double dph=ph_old-ph;
          if(dph<0.)dph+=2*pi;
          if(dph>2*pi)dph-=2*pi;
          ph_old=ph;
          htmp->SetBinContent(j+1,dph);
        }
        htmp->Fit(p1,"NQ","W",5.,19.);


        double ph_step=p1->Eval(20.);
	//        printf("Fit phase step : %f rad/MHz + %f rad/MHz2\n",p1->GetParameter(0),p1->GetParameter(1));
	//   printf("Average phase step : %f rad/MHz\n",ph_step);
        double cor_ref=-1.;
        for(int i=1; i<=FFT2_SIZE/2; i++)
        {
          int j=i-1;
          double freq=hmod1->GetBinCenter(j+1);
          if(freq>19.)
          {
            double cor_mod=g1->Eval(freq);
            if(cor_ref<0.)cor_ref=mod[i]/cor_mod;
            mod[i]=cor_mod*cor_ref;
            phase[i]=phase[i-1]-ph_step;
            if(phase[i]<-pi)phase[i]+=2*pi;
            if(phase[i]>+pi)phase[i]-=2*pi;
            rey[i]=mod[i]*cos(phase[i]);
            imy[i]=mod[i]*sin(phase[i]);
            mod[FFT2_SIZE-i]=mod[i];
            phase[FFT2_SIZE-i]=-phase[i];
            rey[FFT2_SIZE-i]=rey[i];
            imy[FFT2_SIZE-i]=-imy[i];
            hmod2->SetBinContent(j+1,mod[i]);
            hphase2->SetBinContent(j+1,phase[i]);
            hmod2->SetBinContent(FFT2_SIZE-j,mod[i]);
            hphase2->SetBinContent(FFT2_SIZE-j,-phase[i]);
         
	  }
        }
        fft_b->SetPointsComplex(rey, imy);
        fft_b->Transform();
        fft_b->GetPointsComplex(rex, imx);

// Overwrite pulse with filtered one :
        for(int i=0; i<FFT2_SIZE && i<N_samples-FFT_START; i++) laser1[FFT_START+i]=rex[i];
	for(int i=0; i<FFT2_SIZE && i<N_samples-FFT_START; i++) shapeBigMarc[icol][iside]->SetBinContent(FFT_START+i+1,rex[i]);
      
//Recompute maximum and max position :

        las_max=0.;
        for(int i=0; i<N_samples; i++)
        {
          if(fabs(laser1[i])>fabs(las_max))
          {
            las_max=laser1[i];
            iMarcmax=i;
          }
        }
        if(fabs(las_max)<1. || iMarcmax<FFT_START)
        {
          las_max=1.;
          iMarcmax=1450;
        }
	
        for(iMarcfirst=iMarcmax; iMarcfirst>FFT_START && laser1[iMarcfirst]>las_max/100.; iMarcfirst--);

        for(unsigned int i=0; i<FFT2_SIZE;i++)
        {
          laser2[i]=rex[i]/las_max;
        }
	
	for(int i=0; i<FFT2_SIZE && i<N_samples-FFT_START; i++) shapeBigMarcNorm[icol][iside]->SetBinContent(FFT_START+i+1,rex[i]/las_max);
	
	if (_debug==2) cout<<" laser2 filled iMarcFirst="<< iMarcfirst<< endl;
	
        // Search for W50, W10, W05, W01 and RT10-80
        int i50=0, i90=0, i10=0, i05=0, i01=0;
        int j50=nbin, j10=nbin, j05=nbin, j01=nbin;
	
      

        for(int i=iMarcmax-FFT_START; i>=0; i--)
	{
	  if(laser2[i]<=0.01 && i01==0)i01=i;
	  if(laser2[i]<=0.05 && i05==0)i05=i;
	  if(laser2[i]<=0.10 && i10==0)i10=i;
	  if(laser2[i]<=0.50 && i50==0)i50=i;
	  if(laser2[i]<=0.90 && i90==0)i90=i;
	}
        double f01=(double)i01;
        double f05=(double)i05;
        double f10=(double)i10;
        double f50=(double)i50;
        double f90=(double)i90;
        if(i01>0)f01=(double)i01-(laser2[i01]-0.01)/(laser2[i01]-laser2[i01-1]);
        if(i05>0)f05=(double)i05-(laser2[i05]-0.05)/(laser2[i05]-laser2[i05-1]);
        if(i10>0)f10=(double)i10-(laser2[i10]-0.10)/(laser2[i10]-laser2[i10-1]);
        if(i50>0)f50=(double)i50-(laser2[i50]-0.50)/(laser2[i50]-laser2[i50-1]);
        if(i90>0)f90=(double)i90-(laser2[i90]-0.90)/(laser2[i90]-laser2[i90-1]);
      
        for(int i=iMarcmax-FFT_START; i<FFT2_SIZE; i++)
        {
          if(laser2[i]<=0.01 && j01==nbin)j01=i;
	  if(laser2[i]<=0.05 && j05==nbin)j05=i;
	  if(laser2[i]<=0.10 && j10==nbin)j10=i;
	  if(laser2[i]<=0.50 && j50==nbin)j50=i;
	  //printf("las %f,  j50 %d, j10 %d, j05 %d, j01 %d\n",laser2[i],j50,j10,j05,j01);
	}
        double g01=(double)j01;
        double g05=(double)j05;
        double g10=(double)j10;
        double g50=(double)j50;
        if(j01<nbin-1)g01=(double)j01+(laser2[j01]-0.01)/(laser2[j01]-laser2[j01+1]);
        if(j05<nbin-1)g05=(double)j05+(laser2[j05]-0.05)/(laser2[j05]-laser2[j05+1]);
        if(j10<nbin-1)g10=(double)j10+(laser2[j10]-0.10)/(laser2[j10]-laser2[j10+1]);
        if(j50<nbin-1)g50=(double)j50+(laser2[j50]-0.50)/(laser2[j50]-laser2[j50+1]);
        w50=g50-f50;
	w10=g10-f10;
	w05=g05-f05;
        //      double w01=g01-f01;
        //      double rt10_90=f90-f10;
      
        FW_50[icol][iside]=w50;
        FW_10[icol][iside]=w10;
        FW_05[icol][iside]=w05;
      }
      else
      {
	//        rt10_90=0.;
        w50=0.;
        w10=0.;
        w05=0.;
        //w01=0.;
        las_max=0.;
        //prepulse=0.;
        iMarcmax=1450;
        iMarcfirst=1400;
        for(int i=0; i<FFT2_SIZE; i++) laser2[i]=0.;
	
      }
      /*
      int ilasfirst=ilasmax;
      for(ilasfirst=ilasmax; ilasfirst>=0 && ashape[iside][ilasfirst]>qlasmax/100.; ilasfirst--);
      ilasfirst-=(int) ((w50[iside]+5.)/10.);
      las_imax[iside]=ilasmax;
      las_ifirst[iside]=ilasfirst;
      */
      iMarcfirst-=(int) ((w50+5.)/10.);// TESTJULIE

      int posmax=iMarcmax-int(FFT_START);
      int posmin;
      for(posmin=posmax; posmin>=0 && laser2[posmin]>0.01; posmin--);
      posmin -= (int)((w50+5.)/10.);
      double nphot=0.;
      for(int i=posmax; i>=0 && i>=posmin; nphot+=laser2[i--]);
      for(int i=posmax+1; i<FFT2_SIZE && i<5.*w50+posmax; nphot+=laser2[i++]);
      
      if(fabs(las_max)<0.1)las_max=1.;

      if(nphot==0.) nphot=1.;
      
      
      //cout<<" posmin:"<< posmin<<" posmax:"<< posmax <<" iMarcfirst:"<< iMarcfirst<< " iMarcmax:"<<iMarcmax<<" nphot:"<<nphot<<" las_max:"<<las_max<<endl;


      for(unsigned int i=0; i<_nsamplesshape; i++)
	{
	  laser3[i]=laser2[i+iMarcfirst-int(FFT_START)]/nphot;	  
	  //cout<<" laser3["<<i<<"]="<<laser3[i]<< endl;	
	  shapeMarc[icol][iside]->SetBinContent(i+1,laser3[i]);
	} 
    
    }
  }
  
  double Peak[6], Sigma[6], Fit[6], Ampl[6], Trise[6], Fwhm[6], Fw20[6], Fw80[6], Ped[6], Pedsig[6], Sliding[6];

  double FWHM, FW10, FW05; // from mean shape
  int Side;
  
  for (unsigned int iColor=0;iColor<nCol;iColor++){
     
    stringstream nametree;
    nametree <<"MatacqCol"<<colors[iColor];
    meanTree[iColor]= new TTree(nametree.str().c_str(),nametree.str().c_str());
    meanTree[iColor]->Branch( "side",        &Side ,       "Side/I"           );
    meanTree[iColor]->Branch( "peak",        &Peak ,       "Peak[6]/D"        );
    meanTree[iColor]->Branch( "sigma",       &Sigma ,      "Sigma[6]/D"       );
    meanTree[iColor]->Branch( "fit",         &Fit ,        "Fit[6]/D"         );
    meanTree[iColor]->Branch( "ampl",        &Ampl ,       "Ampl[6]/D"        );
    meanTree[iColor]->Branch( "trise",       &Trise ,      "Trise[6]/D"       );  
    meanTree[iColor]->Branch( "fwhm",        &Fwhm ,       "Fwhm[6]/D"        );    
    meanTree[iColor]->Branch( "fw20",        &Fw20 ,       "Fw20[6]/D"        );  
    meanTree[iColor]->Branch( "fw80",        &Fw80 ,       "Fw80[6]/D"        );  
    meanTree[iColor]->Branch( "ped",         &Ped ,        "Ped[6]/D"         );   
    meanTree[iColor]->Branch( "pedsig",      &Pedsig ,     "Pedsig[6]/D"      );   
    meanTree[iColor]->Branch( "sliding",     &Sliding ,    "Sliding[6]/D"     );  
    meanTree[iColor]->Branch( "FWHM",        &FWHM ,       "FWHM/D"        );    
    meanTree[iColor]->Branch( "FW10",        &FW10 ,       "FW10/D"        );  
    meanTree[iColor]->Branch( "FW05",        &FW05 ,       "FW05/D"        );  
    
    meanTree[iColor]->SetBranchAddress( "side",        &Side       );
    meanTree[iColor]->SetBranchAddress( "peak",        Peak        ); 
    meanTree[iColor]->SetBranchAddress( "sigma",       Sigma       ); 
    meanTree[iColor]->SetBranchAddress( "fit",         Fit         ); 
    meanTree[iColor]->SetBranchAddress( "ampl",        Ampl        ); 
    meanTree[iColor]->SetBranchAddress( "fwhm",        Fwhm        ); 
    meanTree[iColor]->SetBranchAddress( "fw20",        Fw20        ); 
    meanTree[iColor]->SetBranchAddress( "fw80",        Fw80        ); 
    meanTree[iColor]->SetBranchAddress( "trise",       Trise       ); 
    meanTree[iColor]->SetBranchAddress( "ped",         Ped         ); 
    meanTree[iColor]->SetBranchAddress( "pedsig",      Pedsig      ); 
    meanTree[iColor]->SetBranchAddress( "sliding",     Sliding     );
    meanTree[iColor]->SetBranchAddress( "FWHM",       &FWHM        ); 
    meanTree[iColor]->SetBranchAddress( "FW10",       &FW10        ); 
    meanTree[iColor]->SetBranchAddress( "FW05",       &FW05        ); 
    
  }

  for(unsigned int iCol=0;iCol<nCol;iCol++){
    for(unsigned int iSide=0;iSide<nSide;iSide++){
      
      Side=iSide;
      std::vector<double> val[TMTQ::nOutVar];
      
      for(int iVar=0;iVar<TMTQ::nOutVar;iVar++){
	
	val[iVar] = MTQ[iCol][iSide]->get(iVar);

	for(unsigned int i=0;i<val[iVar].size();i++){
	  
	  switch (iVar){
	    
	  case TMTQ::iPeak: Peak[i]=val[iVar][i];
	  case TMTQ::iSigma: Sigma[i]=val[iVar][i];
	  case TMTQ::iFit: Fit[i]=val[iVar][i];
	  case TMTQ::iAmpl: Ampl[i]=val[iVar][i];
	  case TMTQ::iFwhm: Fwhm[i]=val[iVar][i];
	  case TMTQ::iFw20: Fw20[i]=val[iVar][i];
	  case TMTQ::iFw80: Fw80[i]=val[iVar][i];
	  case TMTQ::iTrise: Trise[i]=val[iVar][i];
	  case TMTQ::iPed: Ped[i]=val[iVar][i];
	  case TMTQ::iPedsig: Pedsig[i]=val[iVar][i];
	  case TMTQ::iSlide: Sliding[i]=val[iVar][i];
	  }
	}
	FWHM=FW_50[iCol][iSide];
	FW10=FW_10[iCol][iSide];
	FW05=FW_05[iCol][iSide];
      }
      meanTree[iCol]->Fill();
      if (_debug==2)cout <<"-- debug test -- inside final loop  "<< endl;
    }
  }

  // Compute laser shape: 
  //=====================

  for(unsigned int iCol=0;iCol<nCol;iCol++){
    for(unsigned int iSide=0;iSide<nSide;iSide++){
      
      if(shapeMatTmp[iCol][iSide]->GetEntries()==0) continue;
      
      // Calculate maximum with pol 2
      
      int im = shapeMatTmp[iCol][iSide]->GetMaximumBin();
      double q1=shapeMatTmp[iCol][iSide]->GetBinContent(im-1);
      double q2=shapeMatTmp[iCol][iSide]->GetBinContent(im);
      double q3=shapeMatTmp[iCol][iSide]->GetBinContent(im+1);
      
      double a2=(q3+q1)/2.0-q2;
      double a1=q2-q1+a2*(1-2*im);
      double a0=q2-a1*im-a2*im*im;
      
      double tm=0;
      if(a2!=0) tm=-a1/(2.0*a2);
      double am=a0-a1*a1/(4*a2);
      
      if (_debug>=1) cout <<"-- debug test -- computing shape max: "<<am<<" time:"<< tm<< endl;
      
      // Compute pedestal 
      
      double bl=0;
      for (unsigned int i=1; i<_presampleshape+1;i++){ 
	bl+=shapeMatTmp[iCol][iSide]->GetBinContent(i);
      }
      bl/=_presampleshape;
      
      // Compute and save laser shape
      
      if (_debug>=1) cout <<"-- debug test -- computing shape ped:"<<bl<<" over "<<
		       _presampleshape<<" samples"<<  endl;
      
      int firstBin=0;
      double height=0.0;
      double norm=0.0;
      
      for (unsigned int i=_timebefmax; i>_presampleshape;i--){ 

	height=shapeMatTmp[iCol][iSide]->GetBinContent(i)-bl;
	if(height<(am-bl)*_cutwindow){
	  firstBin=i;
	  i=_presampleshape;
	}
      }
      
      for(unsigned int i=_presampleshape;i< (unsigned) shapeMatTmp[iCol][iSide]->GetNbinsX();i++){
	height=shapeMatTmp[iCol][iSide]->GetBinContent(i)-bl;
	if(height>norm) norm=height;	
      }
      
      if(norm==0.0) norm=1.0;
      
      unsigned int lastBin=firstBin+_nsamplesshape;
      
      for(unsigned int i=firstBin;i<lastBin;i++){
	shapeMat[iCol][iSide]->SetBinContent(i-firstBin+1,(shapeMatTmp[iCol][iSide]->GetBinContent(i)-bl)/norm);
	//if( shapeMatTmp[iCol][iSide]->GetBinContent(i)-bl < 0.0 ) cout<< "negative stuff "<<shapeMatTmp[iCol][iSide]->GetBinContent(i)-bl<<" "<<bl<<" "<< shapeMatTmp[iCol][iSide]->GetBinContent(i)<< endl;
      }
   

      hphase1->Write();
      hmod1->Write();
      hphase2->Write();
      hmod2->Write();


      shapeMat[iCol][iSide]->Write();
      shapeMarc[iCol][iSide]->Write();

      shapeMatTmp[iCol][iSide]->Write();
      shapeMarcTmp[iCol][iSide]->Write();

      shapeBigMarc[iCol][iSide]->Write();
      shapeBigMarcNorm[iCol][iSide]->Write();
      
    }
  }
  if (_debug==2)cout <<"-- debug test -- writing  "<< endl;  
  mtqShape->Write();
  for (unsigned int iColor=0;iColor<nCol;iColor++){
    meanTree[iColor]->Write();
  }

  // close the output file
  outFile->Close();
  
  // Remove temporary file    
  FILE *test; 
  test = fopen(sampfile.c_str(),"r");
  if (test){
    fclose(test);
    stringstream del2;
    del2 << "rm " <<sampfile;
    system(del2.str().c_str()); 
  }
  
  cout <<   "\t+=+    .................... done  +=+" << endl;
  cout <<   "\t+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+" << endl;
 
}

DEFINE_FWK_MODULE(EcalMatacqAnalyzer);

