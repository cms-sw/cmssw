#include <cstdlib>
#include <ctime>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <map>
#include <string>

#include "TSystem.h"
#include "TStyle.h"
#include "TXMLEngine.h"
#include "TFile.h"
#include "TH1D.h"
#include "TCanvas.h"
#include "TKey.h"


using namespace std;


// macro parameters
const Bool_t createNewData( kFALSE );               // extract information from harvest files (again)?
const Bool_t closeCanvas( kTRUE );                  // close created canvases again at end of processing?
const string drawFormat( "gif" );
const string nameFileIn( "certDqmHarvestFiles.txt" ); // name of file containing harvesting file list
const string pathHarvestFiles( "/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/data/Cosmics__Commissioning08_CRAFT_ALL_V9_225-v2__RECO" );
const string nameFileRR( "certRunRegistry.xml" );     // name of file containing run registry information
const string xmlRRAddress( "http://pccmsdqm04.cern.ch/runregistry/runregisterdata?format=xml&intpl=xml&mime=text/xml&qtype=RUN_NUMBER&sortname=RUN_NUMBER" );
const string nameFileCache( "getData.txt" );      // name of file containing extracted harvesting data
const string nameFileOut( "certRunFlags" );           // base name of files containing final flags
const string nameFileHistos( "certHistos.root" );     // name of RooT file containing history histogramms
const Int_t    minNEvt( 1 );                                  // min. number of events
const Int_t    minNTrk( 1 );                                  // min. number of tracks (each tracking algo)
const Double_t minRate( 0.01 );                               // min. rate of tracks/event (each tracking algo)
const Double_t maxOffTrkCl( 10000. );                         // max. number of off-track clusters in CKF
const Double_t minSToN[] = { 25., 25., 30., 27. };            // min. corr. S/N of clusters in the order TIB, TID, TOB, TEC
// const Double_t minFractSubDet( 0.95 );                        // min. fraction of modules/sub-detector passing quality tests (each sub-detector)
const Double_t minFractSubDet( 0.85 );                        // min. fraction of modules/sub-detector passing quality tests (each sub-detector)
const Bool_t   avForwBackw( kTRUE );                          // use average values for forward/backward sub-detector pairs?
const Bool_t   useTEC( kTRUE );                               // consider TEC quality?
const Bool_t   useTID( kFALSE );                              // consider TID quality?

// derived parameters
const string nameFileOutTxt( nameFileOut + ".txt" );          // name of file containing final flags in ASCII
const string nameFileInTwiki( nameFileOut + "Old.txt" );        // name of file containing old flags in ASCII
const string nameFileCacheTwiki( nameFileOut + "Cache.txt" ); // name of temporary file containing final flags for Twiki in ASCII format
const string nameFileOutTwiki( nameFileOut + ".twiki" );      // name of final file containing old and final flags in Twiki format
const string nameFileOutXml( nameFileOut + ".xml" );          // name of file containing final flags in XML
const string nameFileRRTmp( nameFileRR + ".tmp" );

// global constants  
const string BAD( "Bad" );
const string GOOD( "Good" );
const string EXCL( "Excl" );
const string NO( "No" );
const string YES( "Yes" );


string coloredFlag( const string & flag )
{
  string color;
  if      ( flag == BAD  || flag == YES ) color = "RED";
  else if ( flag == GOOD                ) color = "GREEN";
  else if ( flag == EXCL                ) color = "ORANGE";
  else                                    color = "BLACK";
  string formated( "%" + color + "%" + flag + "%ENDCOLOR%" );
  if ( flag != NO ) {
    const string bold( "*" );
    formated.insert( 0,  bold );
    formated.append(     bold );
  }
  return formated;
}


void certification()
{
  // definitions of constants
  vector< string > namesDet;
  namesDet.push_back( "TIB" );
  namesDet.push_back( "TID" );
  namesDet.push_back( "TOB" );
  namesDet.push_back( "TEC" );
  vector< string > namesSubDet;
  namesSubDet.push_back( "SiStrip_TECB" );
  namesSubDet.push_back( "SiStrip_TECF" );
  namesSubDet.push_back( "SiStrip_TIB" );
  namesSubDet.push_back( "SiStrip_TIDB" );
  namesSubDet.push_back( "SiStrip_TIDF" );
  namesSubDet.push_back( "SiStrip_TOB" );
  vector< string > namesAlgo;
  namesAlgo.push_back( "CKF" );
  namesAlgo.push_back( "Cosmic" );
  namesAlgo.push_back( "RS" );
    
  string                  sRun;
  Int_t                   iRun;
  Int_t                   nEvt;
  map< string, Int_t >    nTrk;
  map< string, Double_t > rate;
  map< string, Double_t > chi2;
  Double_t                offTrkCl;
  map< string, Double_t > sToN;
  map< string, Double_t > fractSubDet;

  gSystem->Exec( string( "ls -1 " + pathHarvestFiles + "/*/*.root > " + nameFileIn ).c_str() );
  ofstream fileInCorrect;
  fileInCorrect.open( nameFileIn.c_str(), ios_base::app );
  fileInCorrect << "EOF";
  fileInCorrect.close();
  gSystem->Exec( string( "wget -q -O " + nameFileRRTmp + " " + xmlRRAddress ).c_str() ); // FIXME correct first line of XML file (spaces, quotes)
  clock_t sleep( 2 * CLOCKS_PER_SEC + clock() ); // minimum 2 seconds delay to have the file completely downloaded (evaluated before first use of the file)

  ifstream fileIn;
  ofstream fileCacheOut;
  fileIn.open( nameFileIn.c_str() );
  if ( ! fileIn ) {
    cout << "  ERROR: no input file list " << nameFileIn << " found" << endl;
    return;
  }
  if ( createNewData ) fileCacheOut.open( nameFileCache.c_str() );
  
  Int_t minRun( 1000000 );
  Int_t maxRun(       0 );
  Int_t nFile( 0 );
  while ( fileIn.good() ) {
  
    string nameFile;
    fileIn >> nameFile;
    if ( nameFile == "EOF" ) break;
    sRun = nameFile.substr( nameFile.find( "_R0000" ) + 6, 5);
    iRun = atoi( sRun.c_str() );
    if ( iRun < minRun ) minRun = iRun;
    if ( iRun > maxRun ) maxRun = iRun;
    Bool_t goodRead( kTRUE );
    nEvt     = 0;
    offTrkCl = 0.;
    nTrk.clear();
    rate.clear();
    chi2.clear();
    sToN.clear();
    fractSubDet.clear();
    
    if ( createNewData ) {
      TFile * fileRoot( TFile::Open( nameFile.c_str() ) );
      if ( fileRoot ) {
        const string nameDir( "/DQMData/Run " + sRun + "/SiStrip/Run summary/" );
        const string nameDirTrk( nameDir + "Tracks" );
        const string nameDirMech( nameDir + "MechanicalView" );
        const string nameDirEvt( nameDir + "EventInfo/reportSummaryContents" );
        TDirectory * dirTrk = (TDirectory*)fileRoot->Get( nameDirTrk.c_str() );
        if ( dirTrk ) {
          for ( size_t iAlgo = 0; iAlgo < namesAlgo.size(); ++iAlgo ) {
            const string nameTrk( "NumberOfTracks_" + namesAlgo.at( iAlgo ) + "Tk" );
            const string nameHits( "NumberOfRecHitsPerTrack_" + namesAlgo.at( iAlgo ) + "Tk" );
            const string nameChi2( "Chi2_" + namesAlgo.at( iAlgo ) + "Tk" );
            TH1 * h1Trk  = (TH1*)dirTrk->Get( nameTrk.c_str() );
            TH1 * h1Hits = (TH1*)dirTrk->Get( nameHits.c_str() );
            TH1 * h1Chi2 = (TH1*)dirTrk->Get( nameChi2.c_str() );
            if ( iAlgo == 0 ) {
              if ( h1Trk ) nEvt = ( Int_t )h1Trk->GetEntries();
              else         nEvt = -1;
            }
            if ( h1Hits ) nTrk[ namesAlgo.at( iAlgo ) ] = ( Int_t )h1Hits->GetEntries();
            else          nTrk[ namesAlgo.at( iAlgo ) ] = -1;
            if ( h1Trk ) rate[ namesAlgo.at( iAlgo ) ] = h1Trk->GetMean();
            else         rate[ namesAlgo.at( iAlgo ) ] = -1.;
            if ( h1Chi2 ) chi2[ namesAlgo.at( iAlgo ) ] = h1Chi2->GetMean();
            else          chi2[ namesAlgo.at( iAlgo ) ] = -1.;
          }
          TH1 * h1Clus = (TH1*)dirTrk->Get( "OffTrack_TotalNumberOfClusters" );
          if (h1Clus  ) offTrkCl = h1Clus->GetMean();
          else          offTrkCl = -1.;
        } else {
          cout << "  ERROR: no track info from run " << iRun << endl;
          goodRead = kFALSE;
        }
        TDirectory * dirMech = (TDirectory*)fileRoot->Get( nameDirMech.c_str() );
        if ( dirMech ) {
          for ( size_t iDet = 0; iDet < namesDet.size(); ++iDet ) {
            const string nameSToN( namesDet.at( iDet ) + "/Summary_ClusterStoNCorr_OnTrack_in_" + namesDet.at( iDet ) );
            TH1 * h1StoN = (TH1*)dirMech->Get( nameSToN.c_str() );
            if ( h1StoN ) sToN[ namesDet.at( iDet ) ] = h1StoN->GetMean();
            else          sToN[ namesDet.at( iDet ) ] = -1.;
          }
        } else {
          cout << "  ERROR: no sub-detector info from run " << iRun << endl;
          goodRead = kFALSE;
        }
        TDirectory * dirEvt = (TDirectory*)fileRoot->Get( nameDirEvt.c_str() );
        if ( dirEvt ) {
          TIter nextKey( dirEvt->GetListOfKeys() );
          TKey * key;
          while ( key = (TKey*)nextKey() ) {
            const string nameKey( key->GetName() );
            const string nameSubDet( nameKey.substr( 1, nameKey.find_first_of( ">" ) - 1 ) );
            Bool_t found( false );
            for ( size_t iSubDet = 0; iSubDet < namesSubDet.size(); ++iSubDet ) {
              if ( nameSubDet == namesSubDet.at( iSubDet ) ) {
                fractSubDet[ nameSubDet ] = atof( ( nameKey.substr( nameKey.find( "f=" ) + 2 ) ).c_str() );
                found = true;
                break;
              }
            }
            if ( ! found ) cout << "  ERROR: did not find SubDet" << nameSubDet << endl;
          }
        } else {
          cout << "  ERROR: no event info from run " << iRun << endl;
          goodRead = kFALSE;
        }
        fileRoot->Close();
        
        if ( nFile > 0 ) fileCacheOut << endl;
        fileCacheOut << iRun << " " << nEvt;
        for ( size_t iAlgo = 0; iAlgo < namesAlgo.size(); ++iAlgo ) fileCacheOut << " " << nTrk[ namesAlgo.at( iAlgo ) ];
        for ( size_t iAlgo = 0; iAlgo < namesAlgo.size(); ++iAlgo ) fileCacheOut << " " << rate[ namesAlgo.at( iAlgo ) ];
        for ( size_t iAlgo = 0; iAlgo < namesAlgo.size(); ++iAlgo ) fileCacheOut << " " << chi2[ namesAlgo.at( iAlgo ) ];
        fileCacheOut << " " << offTrkCl;
        for ( size_t iDet    = 0; iDet    < namesDet.size()   ; ++iDet    ) fileCacheOut << " " << sToN[ namesDet.at( iDet ) ];
        for ( size_t iSubDet = 0; iSubDet < namesSubDet.size(); ++iSubDet ) fileCacheOut << " " << fractSubDet[ namesSubDet.at( iSubDet ) ];
        if ( ! goodRead ) fileCacheOut << " \tERROR in file reading" << endl;
      } else {
        cout << "  ERROR: file " << nameFile << " cannot be opened" << endl;
      }
    }
    ++nFile;
  }
  
  if ( createNewData ) fileCacheOut.close();
  fileIn.close();
  
  TFile * fileHistos = new TFile( nameFileHistos.c_str(), "RECREATE" );
  const Int_t    nBins( maxRun - minRun + 1 );
  const Double_t low( (Double_t)minRun - 0.5 );
  const Double_t high( (Double_t)maxRun + 0.5 );
  TH1D * gFracTrkCKF =    new TH1D( "gFracTrkCKF"   , "Average fraction of tracks in event", nBins, low, high );
  TH1D * gFracTrkCosmic = new TH1D( "gFracTrkCosmic", "Average fraction of tracks in event", nBins, low, high );
  TH1D * gFracTrkRS =     new TH1D( "gFracTrkRS"    , "Average fraction of tracks in event", nBins, low, high );
  TH1D * gChi2CKF =       new TH1D( "gChi2CKF"      , "Mean of #chi^{2}"                   , nBins, low, high );
  TH1D * gChi2Cosmic =    new TH1D( "gChi2Cosmic"   , "Mean of #chi^{2}"                   , nBins, low, high );
  TH1D * gChi2RS =        new TH1D( "gChi2RS"       , "Mean of #chi^{2}"                   , nBins, low, high );
  TH1D * gSnTIB =         new TH1D( "gSnTIB"        , "S/N clusters"                       , nBins, low, high );
  TH1D * gSnTID =         new TH1D( "gSnTID"        , "S/N clusters"                       , nBins, low, high );
  TH1D * gSnTOB =         new TH1D( "gSnTOB"        , "S/N clusters"                       , nBins, low, high );
  TH1D * gSnTEC =         new TH1D( "gSnTEC"        , "S/N clusters"                       , nBins, low, high );
  TH1D * gClstOff =       new TH1D( "gClstOff"      , "Mean of off-track clusters"         , nBins, low, high );
  gFracTrkCKF->SetXTitle( "run number" );
  gFracTrkCKF->SetMarkerStyle( 20 );
  gFracTrkCKF->SetMarkerSize( 1. );
  gFracTrkCKF->SetMarkerColor( kRed );
  gFracTrkCKF->SetLineColor( kRed );
  gFracTrkCosmic->SetXTitle( "run number" );
  gFracTrkCosmic->SetMarkerStyle( 21 );
  gFracTrkCosmic->SetMarkerSize( 1. );
  gFracTrkCosmic->SetMarkerColor( kBlack );
  gFracTrkCosmic->SetLineColor( kBlack );
  gFracTrkRS->SetXTitle( "run number" );
  gFracTrkRS->SetMarkerStyle( 22 );
  gFracTrkRS->SetMarkerSize( 1. );
  gFracTrkRS->SetMarkerColor( kBlue );
  gFracTrkRS->SetLineColor( kBlue );
  gChi2CKF->SetXTitle( "run number" );
  gChi2CKF->SetMarkerStyle( 20 );
  gChi2CKF->SetMarkerSize( 1. );
  gChi2CKF->SetMarkerColor( kRed );
  gChi2CKF->SetLineColor( kRed );
  gChi2Cosmic->SetXTitle( "run number" );
  gChi2Cosmic->SetMarkerStyle( 21 );
  gChi2Cosmic->SetMarkerSize( 1. );
  gChi2Cosmic->SetMarkerColor( kBlack );
  gChi2Cosmic->SetLineColor( kBlack );
  gChi2RS->SetXTitle( "run number" );
  gChi2RS->SetMarkerStyle( 22 );
  gChi2RS->SetMarkerSize( 1. );
  gChi2RS->SetMarkerColor( kBlue );
  gChi2RS->SetLineColor( kBlue );
  gSnTIB->SetXTitle( "run number" );
  gSnTIB->SetMarkerStyle( 20 );
  gSnTIB->SetMarkerSize( 1. );
  gSnTIB->SetMarkerColor( kRed );
  gSnTIB->SetLineColor( kRed );
  gSnTID->SetXTitle( "run number" );
  gSnTID->SetMarkerStyle( 21 );
  gSnTID->SetMarkerSize( 1. );
  gSnTID->SetMarkerColor( kBlack );
  gSnTID->SetLineColor( kBlack );
  gSnTOB->SetXTitle( "run number" );
  gSnTOB->SetMarkerStyle( 22 );
  gSnTOB->SetMarkerSize( 1. );
  gSnTOB->SetMarkerColor( kBlue );
  gSnTOB->SetLineColor( kBlue );
  gSnTEC->SetXTitle( "run number" );
  gSnTEC->SetMarkerStyle( 23 );
  gSnTEC->SetMarkerSize( 1. );
  gSnTEC->SetMarkerColor( kOrange );
  gSnTEC->SetLineColor( kOrange );
  gClstOff->SetXTitle( "run number" );
  gClstOff->SetMarkerStyle( 20 );
  gClstOff->SetMarkerSize( 1. );
  gClstOff->SetMarkerColor( kRed );
  gClstOff->SetLineColor( kRed );
  TH1D * aFracTrkCKF =    new TH1D( *( (TH1D*)gFracTrkCKF->Clone( "aFracTrkCKF" ) ) );
  TH1D * aFracTrkCosmic = new TH1D( *( (TH1D*)gFracTrkCosmic->Clone( "aFracTrkCosmic" ) ) );
  TH1D * aFracTrkRS =     new TH1D( *( (TH1D*)gFracTrkRS->Clone( "aFracTrkRS" ) ) );
  TH1D * aChi2CKF =       new TH1D( *( (TH1D*)gChi2CKF->Clone( "aChi2CKF" ) ) );
  TH1D * aChi2Cosmic =    new TH1D( *( (TH1D*)gChi2Cosmic->Clone( "aChi2Cosmic" ) ) );
  TH1D * aChi2RS =        new TH1D( *( (TH1D*)gChi2RS->Clone( "aChi2RS" ) ) );
  TH1D * aSnTIB =         new TH1D( *( (TH1D*)gSnTIB->Clone( "aSnTIB" ) ) );
  TH1D * aSnTID =         new TH1D( *( (TH1D*)gSnTID->Clone( "aSnTID" ) ) );
  TH1D * aSnTOB =         new TH1D( *( (TH1D*)gSnTOB->Clone( "aSnTOB" ) ) );
  TH1D * aSnTEC =         new TH1D( *( (TH1D*)gSnTEC->Clone( "aSnTEC" ) ) );
  TH1D * aClstOff =       new TH1D( *( (TH1D*)gClstOff->Clone( "aClstOff" ) ) );
  
  ifstream fileCacheIn;
  ofstream fileOut;
  ofstream fileCacheOutTwiki;
  fileCacheIn.open( nameFileCache.c_str() );
  fileOut.open( nameFileOutTxt.c_str() );
  fileCacheOutTwiki.open( nameFileCacheTwiki.c_str() );
  TXMLEngine * xml = new TXMLEngine;
  XMLNodePointer_t nodeMain( xml->NewChild( 0, 0, "CERTIFICATION" ) );
  XMLNodePointer_t nodeCriteria( xml->NewChild( nodeMain, 0, "CRITERIA" ) );
  XMLNodePointer_t nodeCriterion( xml->NewChild( nodeCriteria, 0, "CRITERION", "Minimum number of events" ) );
  xml->NewAttr( nodeCriterion, 0, "name", "minNEvt" );
  ostringstream sMinNEvt;
  sMinNEvt << minNEvt;
  xml->NewAttr( nodeCriterion, 0, "value", sMinNEvt.str().c_str() );
  nodeCriterion = xml->NewChild( nodeCriteria, 0, "CRITERION", "Minimum number of reconstructed tracks" );
  xml->NewAttr( nodeCriterion, 0, "name", "minNTrk" );
  ostringstream sMinNTrk;
  sMinNTrk << minNTrk;
  xml->NewAttr( nodeCriterion, 0, "value", sMinNTrk.str().c_str() );
  nodeCriterion = xml->NewChild( nodeCriteria, 0, "CRITERION", "Minimum average number of reconstructed tracks per event" );
  xml->NewAttr( nodeCriterion, 0, "name", "minRate" );
  ostringstream sMinRate;
  sMinRate << minRate;
  xml->NewAttr( nodeCriterion, 0, "value", sMinRate.str().c_str() );
  nodeCriterion = xml->NewChild( nodeCriteria, 0, "CRITERION", "Maximum average number of off-track clusters" );
  xml->NewAttr( nodeCriterion, 0, "name", "maxOffTrkCl" );
  ostringstream sMaxOffTrkCl;
  sMaxOffTrkCl << maxOffTrkCl;
  xml->NewAttr( nodeCriterion, 0, "value", sMaxOffTrkCl.str().c_str() );
  nodeCriterion = xml->NewChild( nodeCriteria, 0, "CRITERION", "Minimum corr. S/N of clusters per sub-detector" );
  xml->NewAttr( nodeCriterion, 0, "name", "minSToN" );
  XMLNodePointer_t nodeSubCriterion;
  for ( size_t iDet = 0; iDet < namesDet.size(); ++iDet ) {
    nodeSubCriterion = xml->NewChild( nodeCriterion, 0, "SUBCRITERION" );
    xml->NewAttr( nodeSubCriterion, 0, "subdet", namesDet.at( iDet ).c_str() );
    ostringstream sMinSToN;
    sMinSToN << minSToN[ iDet ];
    xml->NewAttr( nodeSubCriterion, 0, "value", sMinSToN.str().c_str() );
  }
  nodeCriterion = xml->NewChild( nodeCriteria, 0, "CRITERION", "Minimum fraction of good modules in sub-detectors" );
  xml->NewAttr( nodeCriterion, 0, "name", "minFractSubDet" );
  ostringstream sMinFractSubDet;
  sMinFractSubDet << minFractSubDet;
  xml->NewAttr( nodeCriterion, 0, "value", sMinFractSubDet.str().c_str() );
  XMLNodePointer_t nodeRuns( xml->NewChild( nodeMain, 0, "RUNS" ) );
  
  Int_t nRuns( 0 );
  Int_t nRunsGood( 0 );
  Int_t nRunsBad( 0 );
  Int_t nRunsNoEvents( 0 );
  Int_t nRunsNoTracks( 0 );
  Int_t nEvents( 0 );
  Int_t nEventsGood( 0 );
  while ( fileCacheIn.good() ) {
  
    string lineTxt( "" );
    string lineTwiki( " " );
    string sFlag( " run is" );
    string flagList( "" );
    nEvt = 0;
    offTrkCl = 0.;
    nTrk.clear();
    rate.clear();
    chi2.clear();
    sToN.clear();
    fractSubDet.clear();
    fileCacheIn >> iRun >> nEvt;
    for ( size_t iAlgo = 0; iAlgo < namesAlgo.size(); ++iAlgo ) fileCacheIn >> nTrk[ namesAlgo.at( iAlgo ) ];
    for ( size_t iAlgo = 0; iAlgo < namesAlgo.size(); ++iAlgo ) fileCacheIn >> rate[ namesAlgo.at( iAlgo ) ];
    for ( size_t iAlgo = 0; iAlgo < namesAlgo.size(); ++iAlgo ) fileCacheIn >> chi2[ namesAlgo.at( iAlgo ) ];
    fileCacheIn >> offTrkCl;
    for ( size_t iDet    = 0; iDet    < namesDet.size()   ; ++iDet    ) fileCacheIn >> sToN[ namesDet.at( iDet ) ];
    for ( size_t iSubDet = 0; iSubDet < namesSubDet.size(); ++iSubDet ) fileCacheIn >> fractSubDet[ namesSubDet.at( iSubDet ) ];
    
    XMLNodePointer_t nodeRun( xml->NewChild( nodeRuns, 0, "RUN" ) );
    ostringstream sRun;
    sRun << iRun;
    xml->NewAttr( nodeRun, 0, "number", sRun.str().c_str() );

    Bool_t goodRun( kTRUE );
    UInt_t bitFlags( 0 );
    UInt_t bitNumber( 1 );
    XMLNodePointer_t nodeFlag( xml->NewChild( nodeRun, 0, "FLAG" ) );
    xml->NewAttr( nodeFlag, 0, "name", "minNEvt" );
    if ( nRuns == 0 ) flagList = "minNEvt, " + flagList;
    if ( nEvt < minNEvt ) {
      const string lineFlag( "no events" );
      lineTxt += " " + lineFlag;
      if ( goodRun ) lineTwiki += lineFlag;
      goodRun = kFALSE;
      ++nRunsNoEvents;
      xml->NewAttr( nodeFlag, 0, "value", "0" );
    } else {
      xml->NewAttr( nodeFlag, 0, "value", "1" );
      bitFlags |= ( 1 << bitNumber );
    }
    bool failedTracks( false );
    for ( size_t iAlgo = 0; iAlgo < namesAlgo.size(); ++iAlgo ) {
      ++bitNumber;
      if ( iAlgo == 0 ) nodeFlag = xml->NewChild( nodeRun, 0, "FLAG" );
      else              nodeFlag = xml->NewChild( nodeRun, 0, "FLAG", "Not included in global flag" );
      xml->NewAttr( nodeFlag, 0, "name", "minNTrk" );
      xml->NewAttr( nodeFlag, 0, "algo", namesAlgo.at( iAlgo ).c_str() );
      if ( nRuns == 0 ) flagList = "minNTrk (" + namesAlgo.at( iAlgo ) + "), " + flagList;
      if ( nTrk[ namesAlgo.at( iAlgo ) ] < minNTrk ) {
        const string lineFlag( "no " + namesAlgo.at( iAlgo ) + " tracks" );
        lineTxt += " " + lineFlag;
        xml->NewAttr( nodeFlag, 0, "value", "0" );
        if ( goodRun ) {
          if ( failedTracks ) lineTwiki += ", ";
          lineTwiki += lineFlag;
          failedTracks = true;
        }
        if ( iAlgo == 0 ) {
          goodRun = kFALSE;
          ++nRunsNoTracks;
        }
      } else {
        xml->NewAttr( nodeFlag, 0, "value", "1" );
        bitFlags |= ( 1 << bitNumber );
      }
    }
    for ( size_t iAlgo = 0; iAlgo < namesAlgo.size(); ++iAlgo ) {
      ++bitNumber;
      if ( iAlgo == 0 ) nodeFlag = xml->NewChild( nodeRun, 0, "FLAG" );
      else              nodeFlag = xml->NewChild( nodeRun, 0, "FLAG", "Not included in global flag" );
      xml->NewAttr( nodeFlag, 0, "name", "minRate" );
      xml->NewAttr( nodeFlag, 0, "algo", namesAlgo.at( iAlgo ).c_str() );
      if ( nRuns == 0 ) flagList = "minRate (" + namesAlgo.at( iAlgo ) + "), " + flagList;
      if ( rate[ namesAlgo.at( iAlgo ) ] < minRate ) {
        const string lineFlag( "too few " + namesAlgo.at( iAlgo ) + " tracks" );
        lineTxt += " " + lineFlag;
        xml->NewAttr( nodeFlag, 0, "value", "0" );
        if ( goodRun ) {
          if ( failedTracks ) lineTwiki += ", ";
          lineTwiki +=  lineFlag;
          failedTracks = true;
        }
        if ( iAlgo == 0 ) {
          goodRun = kFALSE;
        }
      } else {
        xml->NewAttr( nodeFlag, 0, "value", "1" );
        bitFlags |= ( 1 << bitNumber );
      }
    }
    nodeFlag = xml->NewChild( nodeRun, 0, "FLAG" );
    xml->NewAttr( nodeFlag, 0, "name", "maxOffTrkCl" );
    if ( nRuns == 0 ) flagList = "maxOffTrkCl, " + flagList;
    ++bitNumber;
    if ( offTrkCl > maxOffTrkCl ) {
      const string lineFlag( "too many offTrk clusters" );
      lineTxt += " " + lineFlag;
      if ( goodRun ) lineTwiki += lineFlag;
      goodRun = kFALSE;
      xml->NewAttr( nodeFlag, 0, "value", "0" );
    } else {
      xml->NewAttr( nodeFlag, 0, "value", "1" );
      bitFlags |= ( 1 << bitNumber );
    }
    for ( size_t iDet = 0; iDet < namesDet.size(); ++iDet ) {
      ++bitNumber;
      nodeFlag = xml->NewChild( nodeRun, 0, "FLAG" );
      xml->NewAttr( nodeFlag, 0, "name", "minSToN" );
      xml->NewAttr( nodeFlag, 0, "subdet", namesDet.at( iDet ).c_str() );
      if ( nRuns == 0 ) flagList = "minSToN (" + namesDet.at( iDet ) + "), " + flagList;
      if ( sToN[ namesDet.at( iDet ) ] < minSToN[ iDet ] ) {
        const string lineFlag( "too low S/N in " + namesDet.at( iDet ) );
        lineTxt += " " + lineFlag;
        if ( goodRun ) lineTwiki += lineFlag;
        goodRun = kFALSE;
        xml->NewAttr( nodeFlag, 0, "value", "0" );
      } else {
        xml->NewAttr( nodeFlag, 0, "value", "1" );
        bitFlags |= ( 1 << bitNumber );
      }
    }
    nodeFlag = xml->NewChild( nodeRun, 0, "FLAG" );
    xml->NewAttr( nodeFlag, 0, "name", "minFractSubDet" );
    xml->NewAttr( nodeFlag, 0, "subdet", "TIB" );
    if ( nRuns == 0 ) flagList = "minFractSubDet (TIB), " + flagList;
    ++bitNumber;
    if ( fractSubDet[ "SiStrip_TIB" ] < minFractSubDet && fractSubDet[ "SiStrip_TIB" ] != -1. ) {
      const string lineFlag( "too few modules good in TIB" );
      lineTxt += " " + lineFlag;
      if ( goodRun ) lineTwiki += lineFlag;
      goodRun = kFALSE;
      xml->NewAttr( nodeFlag, 0, "value", "0" );
    } else {
      xml->NewAttr( nodeFlag, 0, "value", "1" );
      bitFlags |= ( 1 << bitNumber );
    }
    nodeFlag = xml->NewChild( nodeRun, 0, "FLAG" );
    xml->NewAttr( nodeFlag, 0, "name", "minFractSubDet" );
    xml->NewAttr( nodeFlag, 0, "subdet", "TOB" );
    if ( nRuns == 0 ) flagList = "minFractSubDet (TOB), " + flagList;
    ++bitNumber;
    if ( fractSubDet[ "SiStrip_TOB" ] < minFractSubDet && fractSubDet[ "SiStrip_TOB" ] != -1. ) {
      const string lineFlag( "too few modules good in TOB" );
      lineTxt += " " + lineFlag;
      if ( goodRun ) lineTwiki += lineFlag;
      goodRun = kFALSE;
      xml->NewAttr( nodeFlag, 0, "value", "0" );
    } else {
      xml->NewAttr( nodeFlag, 0, "value", "1" );
      bitFlags |= ( 1 << bitNumber );
    }
    if ( avForwBackw ) { // FIXME Remove all this hardcoding
      ++bitNumber;
      if ( useTEC ) {
        nodeFlag = xml->NewChild( nodeRun, 0, "FLAG" );
        xml->NewAttr( nodeFlag, 0, "name", "minFractSubDet" );
        xml->NewAttr( nodeFlag, 0, "subdet", "TEC" );
        if ( nRuns == 0 ) flagList = "minFractSubDet (TEC), " + flagList;
        if ( fractSubDet[ "SiStrip_TECF" ] == -1. && fractSubDet[ "SiStrip_TECB" ] >= 0. ) {
          if ( fractSubDet[ "SiStrip_TECB" ] < minFractSubDet ) {
            const string lineFlag( "too few modules good in TECB (TECF off)" );
            lineTxt += " " + lineFlag;
            if ( goodRun ) lineTwiki += lineFlag;
            goodRun = kFALSE;
            xml->NewAttr( nodeFlag, 0, "value", "0" );
          } else {
            xml->NewAttr( nodeFlag, 0, "value", "1" );
            bitFlags |= ( 1 << bitNumber );
          }
        } else if ( fractSubDet[ "SiStrip_TECF" ] >= 0. && fractSubDet[ "SiStrip_TECB" ] == -1. ) {
          if ( fractSubDet[ "SiStrip_TECF" ] < minFractSubDet ) {
            const string lineFlag( "too few modules good in TECF (TECB off)" );
            lineTxt += " " + lineFlag;
            if ( goodRun ) lineTwiki += lineFlag;
            goodRun = kFALSE;
            xml->NewAttr( nodeFlag, 0, "value", "0" );
          } else {
            xml->NewAttr( nodeFlag, 0, "value", "1" );
            bitFlags |= ( 1 << bitNumber );
          }
        } else {
          if ( ( fractSubDet[ "SiStrip_TECF" ] + fractSubDet[ "SiStrip_TECB" ] ) / 2. < minFractSubDet && ( fractSubDet[ "SiStrip_TECF" ] + fractSubDet[ "SiStrip_TECB" ] ) / 2. != -1. ) {
            const string lineFlag( "too few modules good in TEC" );
            lineTxt += " " + lineFlag;
            if ( goodRun ) lineTwiki += lineFlag;
            goodRun = kFALSE;
            xml->NewAttr( nodeFlag, 0, "value", "0" );
          } else {
            xml->NewAttr( nodeFlag, 0, "value", "1" );
            bitFlags |= ( 1 << bitNumber );
          }
        }
      } else {
        nodeFlag = xml->NewChild( nodeRun, 0, "FLAG", "Not evaluated, bit set to \"true\"" );
        xml->NewAttr( nodeFlag, 0, "name", "minFractSubDet" );
        xml->NewAttr( nodeFlag, 0, "subdet", "TEC" );
        xml->NewAttr( nodeFlag, 0, "value", "-1" );
        if ( nRuns == 0 ) flagList = "minFractSubDet (TEC), " + flagList;
        bitFlags |= ( 1 << bitNumber );
      }
      ++bitNumber;
      if ( useTID ) {
        nodeFlag = xml->NewChild( nodeRun, 0, "FLAG" );
        xml->NewAttr( nodeFlag, 0, "name", "minFractSubDet" );
        xml->NewAttr( nodeFlag, 0, "subdet", "TID" );
        if ( nRuns == 0 ) flagList = "minFractSubDet (TID), " + flagList;
        if ( fractSubDet[ "SiStrip_TIDF" ] == -1. && fractSubDet[ "SiStrip_TIDB" ] >= 0. ) {
          if ( fractSubDet[ "SiStrip_TIDB" ] < minFractSubDet ) {
            const string lineFlag( "too few modules good in TIDB (TIDF off)" );
            lineTxt += " " + lineFlag;
            if ( goodRun ) lineTwiki += lineFlag;
            goodRun = kFALSE;
            xml->NewAttr( nodeFlag, 0, "value", "0" );
          } else {
            xml->NewAttr( nodeFlag, 0, "value", "1" );
            bitFlags |= ( 1 << bitNumber );
          }
        } else if ( fractSubDet[ "SiStrip_TIDF" ] >= 0. && fractSubDet[ "SiStrip_TIDB" ] == -1. ) {
          if ( fractSubDet[ "SiStrip_TIDF" ] < minFractSubDet ) {
            const string lineFlag( "too few modules good in TIDF (TIDB off)" );
            lineTxt += " " + lineFlag;
            if ( goodRun ) lineTwiki += lineFlag;
            goodRun = kFALSE;
            xml->NewAttr( nodeFlag, 0, "value", "0" );
          } else {
            xml->NewAttr( nodeFlag, 0, "value", "1" );
            bitFlags |= ( 1 << bitNumber );
          }
        } else {
          if ( ( fractSubDet[ "SiStrip_TIDF" ] + fractSubDet[ "SiStrip_TIDB" ] ) / 2. < minFractSubDet && ( fractSubDet[ "TIDF" ] + fractSubDet[ "TIDB" ] ) / 2. != -1. ) {
            const string lineFlag( "too few modules good in TID" );
            lineTxt += " " + lineFlag;
            if ( goodRun ) lineTwiki += lineFlag;
            goodRun = kFALSE;
            xml->NewAttr( nodeFlag, 0, "value", "0" );
          } else {
            xml->NewAttr( nodeFlag, 0, "value", "1" );
            bitFlags |= ( 1 << bitNumber );
          }
        }
      } else {
        nodeFlag = xml->NewChild( nodeRun, 0, "FLAG", "Not evaluated, bit set to \"true\"" );
        xml->NewAttr( nodeFlag, 0, "name", "minFractSubDet" );
        xml->NewAttr( nodeFlag, 0, "subdet", "TID" );
        xml->NewAttr( nodeFlag, 0, "value", "-1" );
        if ( nRuns == 0 ) flagList = "minFractSubDet (TID), " + flagList;
        bitFlags |= ( 1 << bitNumber );
      }
//     } else {
//       for ( size_t iSubDet = 0; iSubDet < namesSubDet.size(); ++iSubDet ) { // FIXME add 'useTEC' and 'useTID' here
//         if ( fractSubDet[ namesSubDet.at( iSubDet ) ] < minFractSubDet && fractSubDet[ namesSubDet.at( iSubDet ) ] != -1. ) {
//           if ( namesSubDet.at( iSubDet ) != "SiStrip_TECF" && namesSubDet.at( iSubDet ) != "SiStrip_TECB" && namesSubDet.at( iSubDet ) != "SiStrip_TIDF" && namesSubDet.at( iSubDet ) != "SiStrip_TIDB" ) { // don't care for TEC and TID
//             lineTxt += " too few modules good in " + namesSubDet.at( iSubDet );
//             goodRun = kFALSE;
//             break;
//           }
//         }
//       }
    }

    nodeFlag = xml->NewChild( nodeRun, 0, "GLOBAL_FLAG" );
    if ( goodRun ) {
      xml->NewAttr( nodeFlag, 0, "value", "1" );
      bitFlags |= ( 1 << 0 );
    } else {
      xml->NewAttr( nodeFlag, 0, "value", "0" );
    }
    if ( nRuns == 0 ) {
      flagList += "global";
      XMLNodePointer_t nodeFlagList( xml->NewChild( nodeCriteria, 0, "FLAG_LIST", flagList.c_str() ) );
    }
    ostringstream sBitFlags;
    sBitFlags << "0x" << hex << setfill( '0' ) << setw(8) << bitFlags<< dec << setfill( ' ' );
    nodeFlag = xml->NewChild( nodeRun, 0, "FLAG_BITS" );
    xml->NewAttr( nodeFlag, 0, "value", sBitFlags.str().c_str() );
    
    string flag( " " );
    if ( goodRun ) {
      flag += GOOD;    
      ++nRunsGood;
      nEventsGood += nEvt;
      gFracTrkCKF->Fill( iRun, rate[ namesAlgo.at( 0 ) ] );
      gFracTrkCosmic->Fill( iRun, rate[ namesAlgo.at( 1 ) ] );
      gFracTrkRS->Fill( iRun, rate[ namesAlgo.at( 2 ) ] );
      gChi2CKF->Fill( iRun, chi2[ namesAlgo.at( 0 ) ] );
      gChi2Cosmic->Fill( iRun, chi2[ namesAlgo.at( 1 ) ] );
      gChi2RS->Fill( iRun, chi2[ namesAlgo.at( 2 ) ] );
      gSnTIB->Fill( iRun, sToN[ namesDet.at( 0 ) ] );
      gSnTID->Fill( iRun, sToN[ namesDet.at( 1 ) ] );
      gSnTOB->Fill( iRun, sToN[ namesDet.at( 2 ) ] );
      gSnTEC->Fill( iRun, sToN[ namesDet.at( 3 ) ] );
      gClstOff->Fill( iRun, offTrkCl );   
    } else {
      flag += BAD;    
      ++nRunsBad;
    }
    sFlag += flag + " ";
    aFracTrkCKF->Fill( iRun, rate[ namesAlgo.at( 0 ) ] );
    aFracTrkCosmic->Fill( iRun, rate[ namesAlgo.at( 1 ) ] );
    aFracTrkRS->Fill( iRun, rate[ namesAlgo.at( 2 ) ] );
    aChi2CKF->Fill( iRun, chi2[ namesAlgo.at( 0 ) ] );
    aChi2Cosmic->Fill( iRun, chi2[ namesAlgo.at( 1 ) ] );
    aChi2RS->Fill( iRun, chi2[ namesAlgo.at( 2 ) ] );
    aSnTIB->Fill( iRun, sToN[ namesDet.at( 0 ) ] );
    aSnTID->Fill( iRun, sToN[ namesDet.at( 1 ) ] );
    aSnTOB->Fill( iRun, sToN[ namesDet.at( 2 ) ] );
    aSnTEC->Fill( iRun, sToN[ namesDet.at( 3 ) ] );
    aClstOff->Fill( iRun, offTrkCl );   
    
    fileOut           << iRun << sFlag << lineTxt   << endl;
    if ( nRuns > 0 ) fileCacheOutTwiki << endl;
    fileCacheOutTwiki << iRun << flag  << lineTwiki;

    ++nRuns;
    nEvents += nEvt;
  }
  nRunsNoTracks -= nRunsNoEvents;
  cout << "Runs processed                     : " << nRuns                                    << " (" << nEvents     << " ev.)" << endl;
  cout << "Runs good                          : " << nRunsGood                                << " (" << nEventsGood << " ev.)" << endl;
  cout << "Runs bad                           : " << nRunsBad                                                                   << endl;
  cout << "--> w/o events                     : " << nRunsNoEvents                                                              << endl;
  cout << "--> w/  events, but no (CKF) tracks: " << nRunsNoTracks                                                              << endl;
  cout << "--> other reason                   : " << nRunsBad - nRunsNoTracks - nRunsNoEvents                                   << endl;
  
  XMLDocPointer_t xmlDoc( xml->NewDoc() );
  xml->DocSetRootElement( xmlDoc, nodeMain );
  xml->SaveDoc( xmlDoc, nameFileOutXml.c_str() );
  xml->FreeDoc( xmlDoc );
  
  fileCacheOutTwiki.close();
  fileOut.close();
  fileCacheIn.close();
  
  while ( sleep > clock() ); // here the delay is needed
  TXMLEngine * xmlRROut = new TXMLEngine;
  XMLDocPointer_t xmlRROutDoc( xmlRROut->NewDoc() );
  xmlRROut->SaveDoc( xmlRROutDoc, string( nameFileRR  ).c_str() );
  xmlRROut->FreeDoc( xmlRROutDoc );
  ifstream fileRRIn;
  fileRRIn.open( nameFileRRTmp.c_str() );
  ofstream fileRRCorrect;
  fileRRCorrect.open( nameFileRR.c_str(), ios_base::app );
  const UInt_t maxLength( 131071 );
  char xmlLine[ maxLength ];
  while ( fileRRIn.getline( xmlLine, maxLength ) ) fileRRCorrect << xmlLine << endl;
  fileRRCorrect.close();
  fileRRIn.close();
  gSystem->Exec( string( "rm " + nameFileRRTmp ).c_str() );
  
  TXMLEngine * xmlRR = new TXMLEngine;
  XMLDocPointer_t xmlRRDoc( xmlRR->ParseFile( nameFileRR.c_str() ) );
  nodeMain = xmlRR->DocGetRootElement( xmlRRDoc );
  ifstream fileCacheInTwikiOld;
  ifstream fileCacheInTwiki;
  ofstream fileOutTwiki;
  fileCacheInTwikiOld.open( string( string( gSystem->Getenv( "CMSSW_BASE" ) ) + "/src/DQM/SiStripMonitorClient/data/" + nameFileInTwiki ).c_str() );
  fileCacheInTwiki.open( nameFileCacheTwiki.c_str() );
  fileOutTwiki.open( nameFileOutTwiki.c_str() );
  fileOutTwiki << "%TABLE{ sort=\"on\" initsort=\"1\" initdirection=\"down\" tableborder=\"0\" cellpadding=\"4\" cellspacing=\"3\" cellborder=\"0\" headerbg=\"#D5CCB1\"  headercolor=\"#666666\" databg=\"#FAF0D4, #F3DFA8\" headerrows=\"1\"}%" << endl;
  fileOutTwiki << "%EDITTABLE{ format=\"| text, -1| date, -1, %SERVERTIME{\"$day-$mon-$year\"}%, %e-%b-%Y| select, -1, ,*%GREEN%Good%ENDCOLOR%*, *%RED%Bad%ENDCOLOR%*, Waiting | text, -1| select, -1, ,%GREEN%*Good*%ENDCOLOR%, %RED%*Bad*%ENDCOLOR%, %ORANGE%*Excl*%ENDCOLOR% | select, -1, ,%GREEN%*Good*%ENDCOLOR%, %RED%*Bad*%ENDCOLOR% | select, -1, ,No, %RED%*Yes*%ENDCOLOR% | text, -1|\"changerows=\"on\" }%"                                                                                                                                              << endl;
  fileOutTwiki << "| *Run* | *Date* | *Flag (prompt reco)* | *Comment (prompt reco)* | *Flag (CMS)* | *Flag (re-reco)* | *changed (CMS &rarr; re-reco)* | *Comment (re-reco)* |"                                                                                                                                                                                  << endl;

  Int_t  iRunOld( 0 ), iRunNew( 0 ) ;
  string sFlagOld    , sFlagNew   , sFlagRR;
  string sCommentOld , sCommentNew, sFlagDiff;
  string sDateOld;
  Bool_t fileGood( true );
  Bool_t newWasBigger( iRunNew > iRunOld );
  while ( fileCacheInTwikiOld.good() ) {
    const size_t maxLength( 128 );
    if ( iRunNew >= iRunOld || ! fileGood ) {
      fileCacheInTwikiOld >> iRunOld >> sDateOld >> sFlagOld;
      sFlagOld = coloredFlag( sFlagOld );
      char cLineInOld[ maxLength ];
      fileCacheInTwikiOld.getline( cLineInOld, maxLength );
      sCommentOld = string( cLineInOld );
      const size_t firstCharOld( sCommentOld.find_first_not_of( " " ) );
      sCommentOld.erase( 0, firstCharOld );
    }
    while ( ( fileGood = fileCacheInTwiki.good() ) && iRunNew < iRunOld ) {
      if ( ! newWasBigger ) {
        char cLineInNew[ maxLength ];
        fileCacheInTwiki >> iRunNew >> sFlagNew;
        sFlagNew = coloredFlag( sFlagNew );
        fileCacheInTwiki.getline( cLineInNew, maxLength );
        sCommentNew = string( cLineInNew );
        const size_t firstCharNew( sCommentNew.find_first_not_of( " " ) );
        sCommentNew.erase( 0, firstCharNew );
        sFlagRR   = "";
        sFlagDiff = "";
        XMLNodePointer_t nodeRun( xmlRR->GetChild( nodeMain ) );
        while ( nodeRun ) {
          XMLNodePointer_t nodeRunChild = xmlRR->GetChild( nodeRun );
          while ( nodeRunChild && string( xmlRR->GetNodeName( nodeRunChild ) ) != "RUN_NUMBER" ) {
            nodeRunChild = xmlRR->GetNext( nodeRunChild );
          }
          if ( nodeRunChild ) {
            if ( atoi( xmlRR->GetNodeContent( nodeRunChild ) ) == iRunNew ) {
              nodeRunChild = xmlRR->GetChild( nodeRun );
              while ( nodeRunChild ) {
                if ( string( xmlRR->GetNodeName( nodeRunChild ) ) == "SIST" ) {
                  sFlagRR = xmlRR->GetNodeContent( nodeRunChild );
                  break;
                }
                nodeRunChild = xmlRR->GetNext( nodeRunChild );
              }
              break;
            }
          }
          nodeRun = xmlRR->GetNext( nodeRun );
        }
        if      ( sFlagRR == "GOOD" ) sFlagRR = coloredFlag( GOOD );
        else if ( sFlagRR == "BAD"  ) sFlagRR = coloredFlag( BAD );
        else if ( sFlagRR == "EXCL" ) sFlagRR = coloredFlag( EXCL );
        if ( sFlagRR == sFlagNew || ( sFlagRR == coloredFlag( EXCL ) && sFlagNew == coloredFlag( BAD ) ) || sFlagRR.empty() )
          sFlagDiff = coloredFlag( NO );
        else
          sFlagDiff = coloredFlag( YES );
      }
      if ( iRunNew < iRunOld ) {
        fileOutTwiki << "|  " << iRunNew << " | | | |  " << sFlagRR << " |  " << sFlagNew << " |  " << sFlagDiff << " | " << sCommentNew << "  |" << endl;
        newWasBigger = false;
      }
    }
    if ( iRunNew == iRunOld ) {
      fileOutTwiki << "|  " << iRunNew << " |  " << sDateOld << " |  " << sFlagOld << " | " << sCommentOld << "  |  " << sFlagRR << " |  " << sFlagNew << " |  " << sFlagDiff << " | " << sCommentNew << "  |" << endl;
      newWasBigger = false;
    } else if ( iRunNew > iRunOld || ! fileGood ) {
      fileOutTwiki << "|  " << iRunOld << " |  " << sDateOld << " |  " << sFlagOld << " | " << sCommentOld << "  | | | | |" << endl;
      newWasBigger = true;
    }
  }
  
  xmlRR->FreeDoc( xmlRRDoc );
  fileCacheInTwiki.close();
  fileOutTwiki.close();
  gSystem->Exec( string( "rm " + nameFileRR ).c_str() );
  gSystem->Exec( string( "rm " + nameFileCacheTwiki ).c_str() );
  
  gStyle->SetOptStat( 0 );
  TCanvas * gCanvas = new TCanvas( "gCanvas", "SiStrip Offline Run Certification - CRAFT (good runs)", 1392, 1000 );
  gCanvas->Divide( 2, 2 );
  TVirtualPad * currentPad = gCanvas->cd( 1 );
  gFracTrkCosmic->DrawCopy( "PL" ); // has to be the first do be drawn due to highest values
  gFracTrkCKF->DrawCopy( "SamePL" );
  gFracTrkRS->DrawCopy( "SamePL" );
  currentPad->SaveAs( string( "gFracTrk." + drawFormat).c_str() );
  currentPad = gCanvas->cd( 2 );
  gChi2CKF->DrawCopy( "PL" );
  gChi2Cosmic->DrawCopy( "SamePL" );
  gChi2RS->DrawCopy( "SamePL" );
  currentPad->SaveAs( string( "gChi2." + drawFormat ).c_str() );
  currentPad = gCanvas->cd( 3 );
  gClstOff->DrawCopy( "PL" );
  currentPad->SaveAs( string( "gClstOff." + drawFormat ).c_str() );
  currentPad = gCanvas->cd( 4 );
  gSnTOB->DrawCopy( "PL" ); // has to be the first do be drawn due to highest values
  gSnTIB->DrawCopy( "SamePL" );
  gSnTID->DrawCopy( "SamePL" );
  gSnTEC->DrawCopy( "SamePL" );
  currentPad->SaveAs( string( "gSToN." + drawFormat ).c_str() );
  gCanvas->SaveAs( string( "gPlots." + drawFormat ).c_str() );
  if ( closeCanvas ) {
    gCanvas->Close();
    delete gCanvas;
  }
  TCanvas * aCanvas = new TCanvas( "aCanvas", "SiStrip Offline Run Certification - CRAFT (all runs)", 1392, 1000 );
  aCanvas->Divide( 2, 2 );
  currentPad = aCanvas->cd( 1 );
  aFracTrkRS->DrawCopy( "PL" ); // has to be the first do be drawn due to highest values
  aFracTrkCKF->DrawCopy( "SamePL" );
  aFracTrkCosmic->DrawCopy( "SamePL" );
  currentPad->SaveAs( string( "aFracTrk." + drawFormat ).c_str() );
  currentPad = aCanvas->cd( 2 );
  aChi2CKF->DrawCopy( "PL" );
  aChi2Cosmic->DrawCopy( "SamePL" );
  aChi2RS->DrawCopy( "SamePL" );
  currentPad->SaveAs( string( "aChi2." + drawFormat ).c_str() );
  currentPad = aCanvas->cd( 3 );
  aClstOff->DrawCopy( "PL" );
  currentPad->SaveAs( string( "aClstOff." + drawFormat ).c_str() );
  currentPad = aCanvas->cd( 4 );
  aSnTOB->DrawCopy( "PL" ); // has to be the first do be drawn due to highest values
  aSnTIB->DrawCopy( "SamePL" );
  aSnTID->DrawCopy( "SamePL" );
  aSnTEC->DrawCopy( "SamePL" );
  currentPad->SaveAs( string( "aSToN." + drawFormat ).c_str() );
  aCanvas->SaveAs( string( "aPlots." + drawFormat ).c_str() );
  if ( closeCanvas ) {
    aCanvas->Close();
    delete aCanvas;
  }
  
  gFracTrkCKF->Write();
  gFracTrkCosmic->Write();
  gFracTrkRS->Write();
  gChi2CKF->Write();
  gChi2Cosmic->Write();
  gChi2RS->Write();
  gSnTIB->Write();
  gSnTID->Write();
  gSnTOB->Write();
  gSnTEC->Write();
  gClstOff->Write();
  fileHistos->Close();
}
