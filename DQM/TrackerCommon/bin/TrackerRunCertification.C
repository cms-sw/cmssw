// -*- C++ -*-
//
// Package: DQM/TrackerCommon
//
// $Id: TrackerRunCertification.C,v 1.5 2010/09/09 15:52:51 vadler Exp $
//
/**
  \brief    Performs DQM offline data certification for SiStrip, Pixel and Tracking

   Purpose:

   The procedure of certifying data of a given run range is automated in order to speed up the procedure and to reduce the Tracker Offline Shift Leader's workload.

   Input:

   - RunRegistry
   - DQM output files available in AFS

   Output:

   Text file
   - [as explained for command line option '-o']
   to be sent directly to the CMS DQM team as reply to the weekly certification request.
   It contains a list of all flags changed with respect to the RunRegistry, including the reason(s) in case the flag is changed to BAD.

   The verbose ('-v') stdout can provide a complete list of all in-/output flags of all analyzed runs and at its end a summary only with the output flags.
   It makes sense to pipe the stdout to another text file.

   Usage:

   $ cmsrel CMSSW_RELEASE
   $ cd CMSSW_RELEASE/src
   $ cmsenv
   $ cvs co -r Vxx-yy-zz DQM/TrackerCommon
   $ scram b -j 5
   $ rehash
   $ cd WORKING_DIRECTORY
   $ [create input files]
   $ TrackerRunCertification [ARGUMENTOPTION1] [ARGUMENT1] ... [OPTION2] ...

   Valid argument options are:
     -d
       MANDATORY: dataset as in RunRegistry
       no default
     -g
       MANDATORY: group name as in RunRegistry
       no default
     -p
       path to DQM files
       default: /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/data/OfflineData/Run2010/StreamExpress
     -P
       pattern of DQM file names in the DQM file path
       default: *[DATASET from '-d' option with '/' --> '__'].root
     -o
       path to output log file
       default: ./trackerRunCertification[DATASET from '-d' option with '/' --> '__']-[GROUP from '-g'].txt
     -L
       path to file with DQM input file list
       default: ./fileList[DATASET from '-d' option with '/' --> '__'].txt
     -l
       lower bound of run numbers to consider
       default: 0
     -u
       upper bound of run numbers to consider
       default: 1073741824 (2^30)
     -R
       web address of the RunRegistry
       default: http://pccmsdqm04.cern.ch/runregistry
     The default is used for any option not explicitely given in the command line.

   Valid options are:
     -rr
       switch on creation of new RR file
     -rronly
       only create new RR file, do not run certification
     -a
       certify all runs, not only those in "SIGNOFF" status
     -v
       switch on verbose logging to stdout
     -h
       display this help and exit

  \author   Volker Adler
  \version  $Id: TrackerRunCertification.C,v 1.5 2010/09/09 15:52:51 vadler Exp $
*/

#include <algorithm>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <iostream>
#include <sstream>

// RooT, needs '<use name="root">' in the BuildFile
#include "TROOT.h"
#include "TSystem.h"
#include "TString.h"
#include "TFile.h"
#include "TKey.h"
#include "TXMLEngine.h" // needs '<flags LDFLAGS="-lXMLIO">' in the BuildFile


using namespace std;


// Functions
Bool_t  createInputFileList();
Bool_t  createRRFile();
Bool_t  readData( const TString & pathFile );
Bool_t  readRR( const TString & pathFile );
Bool_t  readRRLumis( const TString & pathFile );
Bool_t  readRRTracker( const TString & pathFile );
Bool_t  readDQM( const TString & pathFile );
void    readCertificates( TDirectory * dir );
void    certifyRunRange();
void    certifyRun();
void    writeOutput();
void    displayHelp();
TString RunNumber( const TString & pathFile );
Int_t   FlagConvert( const TString & flag );
TString FlagConvert( const Int_t flag );

// Configurables
map< TString, TString > sArguments;
map< TString, Bool_t >  sOptions;
TString convertDataset_;
Int_t   minRange_;
Int_t   maxRange_;
Int_t   minRun_;
Int_t   maxRun_;

// Global constants
const TString nameFileRR_( "RunRegistry.xml" );
const TString nameFileRunsRR_( TString( "runs" ).Append( nameFileRR_ ) );
const TString nameFileLumisRR_( TString( "lumis" ).Append( nameFileRR_ ) );
const TString nameFileTrackerRR_( TString( "tracker" ).Append( nameFileRR_ ) );
const TString nameDirHead_( "DQMData" );
const TString nameDirBase_( "EventInfo" );
const TString nameDirCert_( "CertificationContents" );
const TString nameDirReport_( "reportSummaryContents" );
const TString nameDirDAQ_( "DAQContents" );
const TString nameDirDCS_( "DCSContents" );
const TString pathRunFragment_( "/Run /" );
const UInt_t  nSubSys_( 3 );
const TString sSubSys_[ nSubSys_ ] = { // sub-system directory names in DQM files
  "SiStrip",
  "Pixel",
  "Tracking",
};
enum SubSystems { // according enumeration
  SiStrip,
  Pixel,
  Tracking
};
enum Flags { // flags' enumeration
  MISSING = -100,
  NOTSET  =  -99,
  EXCL    =   -1,
  BAD     =    0,
  GOOD    =    1
};
const Double_t minGood_( 0.95 );
const Double_t maxBad_( 0.85 );
const Int_t    iRunStartDecon_( 110213 ); // first run in deconvolution mode

// Certificates and flags
vector< TString > sRunNumbers_;
UInt_t nRuns_( 0 );
UInt_t nRunsNotRR_( 0 );
UInt_t nRunsNotGroup_( 0 );
UInt_t nRunsNotDataset_( 0 );
UInt_t nRunsNotSignoff_( 0 );
UInt_t nRunsNotRRLumis_( 0 );
UInt_t nRunsNotDatasetLumis_( 0 );
UInt_t nRunsSiStripOff_( 0 );
UInt_t nRunsPixelOff_( 0 );
UInt_t nRunsNotRRTracker_( 0 );
UInt_t nRunsNotGroupTracker_( 0 );
UInt_t nRunsNotDatasetTracker_( 0 );
UInt_t nRunsExclSiStrip_( 0 );
UInt_t nRunsMissSiStrip_( 0 );
UInt_t nRunsBadSiStrip_( 0 );
UInt_t nRunsChangedSiStrip_( 0 );
map< TString, TString > sSiStrip_;
map< TString, TString > sRRSiStrip_;
map< TString, TString > sRRTrackerSiStrip_;
map< TString, TString > sDQMSiStrip_;
map< TString, vector< TString > > sRunCommentsSiStrip_;
map< TString, TString > sRRCommentsSiStrip_;
map< TString, TString > sRRTrackerCommentsSiStrip_;
UInt_t nRunsExclPixel_( 0 );
UInt_t nRunsMissPixel_( 0 );
UInt_t nRunsBadPixel_( 0 );
UInt_t nRunsChangedPixel_( 0 );
map< TString, TString > sPixel_;
map< TString, TString > sRRPixel_;
map< TString, TString > sRRTrackerPixel_;
map< TString, TString > sDQMPixel_;
map< TString, vector< TString > > sRunCommentsPixel_;
map< TString, TString > sRRCommentsPixel_;
map< TString, TString > sRRTrackerCommentsPixel_;
UInt_t nRunsNoTracking_( 0 );
UInt_t nRunsBadTracking_( 0 );
UInt_t nRunsChangedTracking_( 0 );
map< TString, TString > sTracking_;
map< TString, TString > sRRTracking_;
map< TString, TString > sRRTrackerTracking_;
map< TString, TString > sDQMTracking_;
map< TString, vector< TString > > sRunCommentsTracking_;
map< TString, TString > sRRCommentsTracking_;
map< TString, TString > sRRTrackerCommentsTracking_;
// Certificates and flags (run-by-run)
TString sRunNumber_;
TString sVersion_;
map< TString, Double_t > fCertificates_;
map< TString, Int_t >    iFlagsRR_;
map< TString, Int_t >    iFlagsRRTracker_;
map< TString, Bool_t >   bAvailable_;
Bool_t                   bSiStripOn_;
Bool_t                   bPixelOn_;



/// Checks arguments and runs input check/creation and run certification incl. output
int main( int argc, char * argv[] )
{

  cout << endl;

  // Initialize defaults
  sArguments[ "-d" ] = "";                                                   // dataset
  sArguments[ "-g" ] = "";                                                   // group
  sArguments[ "-p" ] = "/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/data/OfflineData/Run2010/StreamExpress"; // path to DQM files
  sArguments[ "-P" ] = "";                                                   // pattern of DQM file names in the DQM file path
  sArguments[ "-l" ] = "0";                                                  // lower bound of run numbers to consider
  sArguments[ "-u" ] = "1073741824"; // 2^30                                 // upper bound of run numbers to consider
  sArguments[ "-o" ] = "";                                                   // path to main output file
  sArguments[ "-L" ] = "";                                                   // path to file with DQM input file list
  sArguments[ "-R" ] = "http://pccmsdqm04.cern.ch/runregistry";              // web address of the RunRegistry
  minRun_ = sArguments[ "-u" ].Atoi();
  maxRun_ = sArguments[ "-l" ].Atoi();
  sOptions[ "-rr" ]     = kFALSE;
  sOptions[ "-rronly" ] = kFALSE;
  sOptions[ "-a" ]      = kFALSE;
  sOptions[ "-v" ]      = kFALSE;
  sOptions[ "-h" ]      = kFALSE;

  // Input arguments (very simple)
  if ( argc == 1 ) {
    displayHelp();
    return 0;
  }
  for ( int iArgument = 1; iArgument < argc; ++iArgument ) {
    if ( sArguments.find( argv[ iArgument ] ) != sArguments.end() ) {
      if ( sArguments.find( argv[ iArgument + 1 ] ) == sArguments.end() && sOptions.find( argv[ iArgument + 1 ] ) == sOptions.end() ) {
        sArguments[ argv[ iArgument ] ] = argv[ iArgument + 1 ];
      }
    } else if ( sOptions.find( argv[ iArgument ] ) != sOptions.end() ) {
      sOptions[ argv[ iArgument ] ] = kTRUE;
    }
  }
  if ( sOptions[ "-h" ] ) {
    displayHelp();
    return 0;
  }
  if ( sArguments[ "-d" ] == "" ) {
    cerr << "    ERROR: no dataset given with '-d' option" << endl;
    return 1;
  }
  if ( sArguments[ "-g" ] == "" && ! sOptions[ "-rronly" ] ) {
    cerr << "    ERROR: no group name given with '-g' option" << endl;
    return 1;
  }
  convertDataset_ = sArguments[ "-d" ];
  convertDataset_.ReplaceAll( "/", "__" );
  if ( sArguments[ "-o" ] == "" ) {
    sArguments[ "-o" ] = TString( "trackerRunCertification" + convertDataset_ + "-" + sArguments[ "-g" ] + ".txt" );
  }
  if ( sArguments[ "-L" ] == "" ) {
    sArguments[ "-L" ] = TString( "fileList" + convertDataset_ + ".txt" );
  }
  if ( sArguments[ "-P" ] == "" ) {
    sArguments[ "-P" ] = TString( "*" + convertDataset_ + ".root" );
  }
  minRange_ = sArguments[ "-l" ].Atoi();
  maxRange_ = sArguments[ "-u" ].Atoi();

  // Run
  if ( ! createInputFileList() ) return 12;
  if ( sOptions[ "-rronly" ] ) {
    if ( ! createRRFile() ) return 13;
    return 0;
  }
  if ( sOptions[ "-rr" ] && ! createRRFile() ) return 13;
  certifyRunRange();

  return 0;

}


/// Checks for DQM RooT files in pre-defined directory, compares to optinally given run range and writes the resulting file list to a file
/// Returns 'kTRUE', if DQM files for the given run range and path have been found, 'kFALSE' otherwise.
Bool_t createInputFileList()
{

  // Create input file list on the fly
  gSystem->Exec( TString( "rm -f " + sArguments[ "-L" ] ).Data() );
  gSystem->Exec( TString( "ls -1 " + sArguments[ "-p" ] + "/*/" + sArguments[ "-P" ] + " > " + sArguments[ "-L" ] ).Data() );
  ofstream fileListWrite;
  fileListWrite.open( sArguments[ "-L" ].Data(), ios_base::app );
  fileListWrite << "EOF";
  fileListWrite.close();

  // Loop over input file list and recreate it according to run range
  ifstream fileListRead;
  fileListRead.open( sArguments[ "-L" ].Data() );
  ofstream fileListNewWrite;
  const TString nameFileListNew( sArguments[ "-L" ] + ".new" );
  fileListNewWrite.open( nameFileListNew, ios_base::app );
  UInt_t nFiles( 0 );
  while ( fileListRead.good() ) {
    TString pathFile;
    fileListRead >> pathFile;
    if ( pathFile.Length() == 0 ) continue;
    sRunNumber_ = RunNumber( pathFile );
    if ( ! RunNumber( pathFile ).IsDigit() ) continue;
    ++nFiles;
    const Int_t iRun( RunNumber( pathFile ).Atoi() );
    if ( minRange_ > iRun || iRun > maxRange_ ) continue;
    fileListNewWrite << pathFile << endl;
    if ( iRun < minRun_ ) minRun_ = iRun;
    if ( iRun > maxRun_ ) maxRun_ = iRun;
  }

  fileListRead.close();
  fileListNewWrite.close();
  gSystem->Exec( TString( "mv " ).Append( nameFileListNew ).Append( " " ).Append( sArguments[ "-L" ] ) );

  if ( nFiles == 0 || maxRun_ < minRun_ ) {
    cerr << "  ERROR: no files to certify" << endl;
    cerr << "  no files found in " << sArguments[ "-p" ] << " between the run numbers " << minRange_ << " and " << maxRange_ << endl;
    return kFALSE;
  }
  return kTRUE;

}


/// Gets XML file with complete RunRegistry information from the web server
/// Returns 'kTRUE', if XML file is present and not empty, 'kFALSE' otherwise.
Bool_t createRRFile()
{

  ostringstream minRun; minRun << minRun_;
  ostringstream maxRun; maxRun << maxRun_;
  cerr << "  Extracting RunRegistry output for runs " << minRun.str() << " - " << maxRun.str() << " ...";
  TString commandBase( TString( gSystem->Getenv( "CMSSW_BASE" ) ).Append( "/src/DQM/TrackerCommon/bin/getRunRegistry.py" ).Append( " -s " ).Append( sArguments[ "-R" ] ).Append( "/xmlrpc" ).Append( " ").Append( " -l " ).Append( minRun.str() ).Append( " -u " ).Append( maxRun.str() ) );
  TString commandRuns( commandBase );
  commandRuns.Append( " -f " ).Append( nameFileRunsRR_ ).Append( " -T RUN -t xml_all" );
  if ( sOptions[ "-v" ] ) cerr << endl << endl << "    " << commandRuns.Data() << endl;
  gSystem->Exec( commandRuns );
  TString commandLumis( commandBase );
  commandLumis.Append( " -f " ).Append( nameFileLumisRR_ ).Append( " -T RUNLUMISECTION -t xml" );
  if ( sOptions[ "-v" ] ) cerr << "    " << commandLumis.Data() << endl;
  gSystem->Exec( commandLumis );
  TString commandTracker( commandBase );
  commandTracker.Append( " -f " ).Append( nameFileTrackerRR_ ).Append( " -T RUN -t xml_all -w TRACKER" );
  if ( sOptions[ "-v" ] ) cerr << "    " << commandTracker.Data() << endl << endl << "  ...";
  gSystem->Exec( commandTracker );
  cerr << " done!" << endl
       << endl;

  const UInt_t maxLength( 131071 ); // FIXME hard-coding for what?
  char xmlLine[ maxLength ];
  UInt_t lines( 0 );
  ifstream fileRunsRR;
  fileRunsRR.open( nameFileRunsRR_.Data() );
  if ( ! fileRunsRR ) {
    cerr << "  ERROR: RR file " << nameFileRunsRR_.Data() << " does not exist" << endl;
    cerr << "  Please, check access to RR" << endl;
    return kFALSE;
  }
  while ( lines <= 1 && fileRunsRR.getline( xmlLine, maxLength ) ) ++lines;
  if ( lines <= 1 ) {
    cerr << "  ERROR: empty RR file " << nameFileRunsRR_.Data() << endl;
    cerr << "  Please, check access to RR:" << endl;
    cerr << "  - DQM/TrackerCommon/bin/getRunRegistry.py" << endl;
    cerr << "  - https://twiki.cern.ch/twiki/bin/view/CMS/DqmRrApi" << endl;
    return kFALSE;
  }
  fileRunsRR.close();
  ifstream fileLumisRR;
  fileLumisRR.open( nameFileLumisRR_.Data() );
  if ( ! fileLumisRR ) {
    cerr << "  ERROR: RR file " << nameFileLumisRR_.Data() << " does not exist" << endl;
    cerr << "  Please, check access to RR" << endl;
    return kFALSE;
  }
  while ( lines <= 1 && fileLumisRR.getline( xmlLine, maxLength ) ) ++lines;
  if ( lines <= 1 ) {
    cerr << "  ERROR: empty RR file " << nameFileLumisRR_.Data() << endl;
    cerr << "  Please, check access to RR:" << endl;
    cerr << "  - DQM/TrackerCommon/bin/getRunRegistry.py" << endl;
    cerr << "  - https://twiki.cern.ch/twiki/bin/view/CMS/DqmRrApi" << endl;
    return kFALSE;
  }
  fileLumisRR.close();
  ifstream fileTrackerRR;
  fileTrackerRR.open( nameFileTrackerRR_.Data() );
  if ( ! fileTrackerRR ) {
    cerr << "  ERROR: RR file " << nameFileTrackerRR_.Data() << " does not exist" << endl;
    cerr << "  Please, check access to RR" << endl;
    return kFALSE;
  }
  while ( lines <= 1 && fileTrackerRR.getline( xmlLine, maxLength ) ) ++lines;
  if ( lines <= 1 ) {
    cerr << "  ERROR: empty RR file " << nameFileTrackerRR_.Data() << endl;
    cerr << "  Please, check access to RR:" << endl;
    cerr << "  - DQM/TrackerCommon/bin/getRunRegistry.py" << endl;
    cerr << "  - https://twiki.cern.ch/twiki/bin/view/CMS/DqmRrApi" << endl;
    return kFALSE;
  }
  fileTrackerRR.close();

  return kTRUE;

}


/// Loops over runs
void certifyRunRange()
{

  // Loop over runs
  ifstream fileListRead;
  fileListRead.open( sArguments[ "-L" ].Data() );
  while ( fileListRead.good() ) {
    TString pathFile;
    fileListRead >> pathFile;
    if ( pathFile.Length() == 0 ) continue;
    ++nRuns_;
    sRunNumber_ = RunNumber( pathFile );
    cout << "  Processing RUN " << sRunNumber_.Data() << endl;
    if ( readData( pathFile ) ) {
      sRunNumbers_.push_back( sRunNumber_ );
      certifyRun();
    }
  }
  fileListRead.close();
  writeOutput();

  return;

}


/// Reads input data for a given run
/// Returns 'kTRUE', if RR and DQM information have been read successfully, 'kFALSE' otherwise.
Bool_t readData( const TString & pathFile )
{

  if ( ! readRR( pathFile ) )        return kFALSE;
//   if ( ! readRRLumis( pathFile ) )   return kFALSE; // LS currently not used
  if ( ! readRRTracker( pathFile ) ) return kFALSE;
  if ( ! readDQM( pathFile ) )       return kFALSE;
  return kTRUE;

}


/// Reads manually set RR certification flags for a given run
/// Returns 'kTRUE', if a given run is present in RR, 'kFALSE' otherwise.
Bool_t readRR( const TString & pathFile )
{

  // Initialize
  map< TString, TString > sFlagsRR;
  map< TString, TString > sCommentsRR;
  iFlagsRR_.clear();
  Bool_t foundRun( kFALSE );
  Bool_t foundGroup( kFALSE );
  Bool_t foundDataset( kFALSE );
  Bool_t foundSignoff( kFALSE );
  vector< TString > nameCmpNode;
  nameCmpNode.push_back( "STRIP" );
  nameCmpNode.push_back( "PIX" );
  nameCmpNode.push_back( "TRACK" );

  // Read RR file corresponding to output format type 'xml_all'
  TXMLEngine * xmlRR( new TXMLEngine );
  XMLDocPointer_t  xmlRRDoc( xmlRR->ParseFile( nameFileRunsRR_.Data() ) );
  XMLNodePointer_t nodeMain( xmlRR->DocGetRootElement( xmlRRDoc ) );
  XMLNodePointer_t nodeRun( xmlRR->GetChild( nodeMain ) );
  while ( nodeRun ) {
    XMLNodePointer_t nodeRunChild( xmlRR->GetChild( nodeRun ) );
    while ( nodeRunChild && TString( xmlRR->GetNodeName( nodeRunChild ) ) != "NUMBER" )
      nodeRunChild = xmlRR->GetNext( nodeRunChild );
    if ( nodeRunChild && xmlRR->GetNodeContent( nodeRunChild ) == sRunNumber_ ) {
      foundRun = kTRUE;
      nodeRunChild = xmlRR->GetChild( nodeRun );
      while ( nodeRunChild && TString( xmlRR->GetNodeName( nodeRunChild ) ) != "GROUP_NAME" )
        nodeRunChild = xmlRR->GetNext( nodeRunChild );
      if ( nodeRunChild && xmlRR->GetNodeContent( nodeRunChild ) == sArguments[ "-g" ] ) {
        foundGroup = kTRUE;
        nodeRunChild = xmlRR->GetChild( nodeRun );
        while ( nodeRunChild && TString( xmlRR->GetNodeName( nodeRunChild ) ) != "DATASETS" )
          nodeRunChild = xmlRR->GetNext( nodeRunChild );
        if ( nodeRunChild ) {
          XMLNodePointer_t nodeDataset( xmlRR->GetChild( nodeRunChild ) );
          while ( nodeDataset ) {
            XMLNodePointer_t nodeDatasetChild( xmlRR->GetChild( nodeDataset ) );
            while ( nodeDatasetChild && TString( xmlRR->GetNodeName( nodeDatasetChild ) ) != "NAME" )
              nodeDatasetChild = xmlRR->GetNext( nodeDatasetChild );
            if ( nodeDatasetChild && TString( xmlRR->GetNodeContent( nodeDatasetChild ) ) == sArguments[ "-d" ] ) {
              foundDataset = kTRUE;
              nodeDatasetChild = xmlRR->GetChild( nodeDataset );
              while ( nodeDatasetChild && xmlRR->GetNodeName( nodeDatasetChild ) != TString( "STATE" ) )
                nodeDatasetChild = xmlRR->GetNext( nodeDatasetChild );
              if ( sOptions[ "-a" ] || ( nodeDatasetChild && TString( xmlRR->GetNodeContent( nodeDatasetChild ) ) == "SIGNOFF" ) ) {
                foundSignoff = kTRUE;
                nodeDatasetChild = xmlRR->GetChild( nodeDataset );
                while ( nodeDatasetChild && TString( xmlRR->GetNodeName( nodeDatasetChild ) ) != "CMPS" )
                  nodeDatasetChild = xmlRR->GetNext( nodeDatasetChild );
                if ( nodeDatasetChild ) {
                  XMLNodePointer_t nodeCmp( xmlRR->GetChild( nodeDatasetChild ) );
                  while ( nodeCmp ) {
                    XMLNodePointer_t nodeCmpChild( xmlRR->GetChild( nodeCmp ) );
                    while ( nodeCmpChild && TString( xmlRR->GetNodeName( nodeCmpChild ) ) != "NAME" )
                      nodeCmpChild = xmlRR->GetNext( nodeCmpChild );
                    if ( nodeCmpChild ) {
                      for ( UInt_t iNameNode = 0; iNameNode < nameCmpNode.size(); ++iNameNode ) {
                        if ( xmlRR->GetNodeContent( nodeCmpChild ) == nameCmpNode.at( iNameNode ) ) {
                          TString nameNode( "RR_" + nameCmpNode.at( iNameNode ) );
                          XMLNodePointer_t nodeCmpChildNew = xmlRR->GetChild( nodeCmp );
                          while ( nodeCmpChildNew && TString( xmlRR->GetNodeName( nodeCmpChildNew ) ) != "VALUE" )
                            nodeCmpChildNew = xmlRR->GetNext( nodeCmpChildNew );
                          if ( nodeCmpChildNew ) {
                            sFlagsRR[ nameNode ] = TString( xmlRR->GetNodeContent( nodeCmpChildNew ) );
                            if ( sFlagsRR[ nameNode ] == "BAD" ) {
                              nodeCmpChildNew = xmlRR->GetChild( nodeCmp );
                              while ( nodeCmpChildNew && TString( xmlRR->GetNodeName( nodeCmpChildNew ) ) != "COMMENT" )
                                nodeCmpChildNew = xmlRR->GetNext( nodeCmpChildNew );
                              if ( nodeCmpChildNew ) {
                                sCommentsRR[ nameNode ] = TString( xmlRR->GetNodeContent( nodeCmpChildNew ) );
                              }
                            }
                          }
                        }
                      }
                    }
                    nodeCmp = xmlRR->GetNext( nodeCmp );
                  }
                }
                break;
              }
              break;
            }
            nodeDataset = xmlRR->GetNext( nodeDataset );
          }
        }
      }
      break;
    }
    nodeRun = xmlRR->GetNext( nodeRun );
  }

  if ( ! foundRun ) {
    ++nRunsNotRR_;
    cout << "    Run not found in RR" << endl;
    return kFALSE;
  }
  if ( ! foundGroup ) {
    ++nRunsNotGroup_;
    cout << "    Group " << sArguments[ "-g" ] << " not found in RR" << endl;
    return kFALSE;
  }
  if ( ! foundDataset ) {
    ++nRunsNotDataset_;
    cout << "    Dataset " << sArguments[ "-d" ] << " not found in RR" << endl;
    return kFALSE;
  }
  if ( ! foundSignoff ) {
    ++nRunsNotSignoff_;
    cout << "    Dataset not in SIGNOFF in RR" << endl;
    return kFALSE;
  }

  if ( sOptions[ "-v" ] ) for ( map< TString, TString >::const_iterator flag = sFlagsRR.begin(); flag != sFlagsRR.end(); ++flag ) cout << "    " << flag->first << ": " << flag->second << endl;
  for ( UInt_t iNameNode = 0; iNameNode < nameCmpNode.size(); ++iNameNode ) {
    TString nameNode( "RR_" + nameCmpNode.at( iNameNode ) );
    if ( sFlagsRR.find( nameNode ) == sFlagsRR.end() ) {
      cout << "    WARNING: component " << nameCmpNode.at( iNameNode ).Data() << " not found in RR" << endl;
      cout << "    Automatically set to MISSING" << endl;
      sFlagsRR[ nameNode ] = "MISSING";
    }
  }

  sRRCommentsSiStrip_[ sRunNumber_ ]  = sCommentsRR[ "RR_STRIP" ];
  sRRCommentsPixel_[ sRunNumber_ ]    = sCommentsRR[ "RR_PIX" ];
  sRRCommentsTracking_[ sRunNumber_ ] = sCommentsRR[ "RR_TRACK" ];
  iFlagsRR_[ sSubSys_[ SiStrip ] ]  = FlagConvert( sFlagsRR[ "RR_STRIP" ] );
  iFlagsRR_[ sSubSys_[ Pixel ] ]    = FlagConvert( sFlagsRR[ "RR_PIX" ] );
  iFlagsRR_[ sSubSys_[ Tracking ] ] = FlagConvert( sFlagsRR[ "RR_TRACK" ] );
  if ( iFlagsRR_[ sSubSys_[ SiStrip ] ] == EXCL ) ++nRunsExclSiStrip_;
  if ( iFlagsRR_[ sSubSys_[ Pixel ] ]   == EXCL ) ++nRunsExclPixel_;
  if ( iFlagsRR_[ sSubSys_[ SiStrip ] ] == MISSING ) ++nRunsMissSiStrip_;
  if ( iFlagsRR_[ sSubSys_[ Pixel ] ]   == MISSING ) ++nRunsMissPixel_;
  if ( ( iFlagsRR_[ sSubSys_[ SiStrip ] ] == EXCL || iFlagsRR_[ sSubSys_[ SiStrip ] ] == MISSING ) &&
       ( iFlagsRR_[ sSubSys_[ Pixel ] ]   == EXCL || iFlagsRR_[ sSubSys_[ Pixel ] ]   == MISSING ) ) ++nRunsNoTracking_;

  return kTRUE;

}


/// Reads RR HV states for lumi ranges in a given run
/// Returns 'kFALSE', if Tracker was in STANDBY during the run, 'kTRUE' otherwise.
Bool_t readRRLumis( const TString & pathFile )
{

  bSiStripOn_ = false ;
  bPixelOn_   = false ;
  map< TString, Bool_t > bLumiSiStripOn_;
  map< TString, Bool_t > bLumiPixelOn_;
  Bool_t foundRun( kFALSE );
  Bool_t foundDataset( kFALSE );

  // Read RR file corresponding to output format type 'xml'
  TXMLEngine * xmlRR( new TXMLEngine );
  XMLDocPointer_t  xmlRRDoc( xmlRR->ParseFile( nameFileLumisRR_.Data() ) );
  XMLNodePointer_t nodeMain( xmlRR->DocGetRootElement( xmlRRDoc ) );
  XMLNodePointer_t nodeRun( xmlRR->GetChild( nodeMain ) );
  while ( nodeRun ) {
    XMLNodePointer_t nodeRunChild( xmlRR->GetChild( nodeRun ) );
    while ( nodeRunChild && TString( xmlRR->GetNodeName( nodeRunChild ) ) != "NUMBER" )
      nodeRunChild = xmlRR->GetNext( nodeRunChild );
    if ( nodeRunChild && xmlRR->GetNodeContent( nodeRunChild ) == sRunNumber_ ) {
      foundRun = kTRUE;
      nodeRunChild = xmlRR->GetChild( nodeRun );
      while ( nodeRunChild && TString( xmlRR->GetNodeName( nodeRunChild ) ) != "DATASET" )
        nodeRunChild = xmlRR->GetNext( nodeRunChild );
      if ( nodeRunChild ) {
        XMLNodePointer_t nodeDatasetChild( xmlRR->GetChild( nodeRunChild ) );
        while ( nodeDatasetChild && TString( xmlRR->GetNodeName( nodeDatasetChild ) ) != "NAME" )
          nodeDatasetChild = xmlRR->GetNext( nodeDatasetChild );
        if ( nodeDatasetChild && xmlRR->GetNodeContent( nodeDatasetChild ) == sArguments[ "-d" ] ) {
          foundDataset = kTRUE;
          XMLNodePointer_t nodeLumiRange( xmlRR->GetChild( nodeRunChild ) );
          while ( nodeLumiRange ) {
            bLumiSiStripOn_[ "DCSTIBTID" ] = kFALSE;
            bLumiSiStripOn_[ "DCSTOB" ]    = kFALSE;
            bLumiSiStripOn_[ "DCSTECM" ]   = kFALSE;
            bLumiSiStripOn_[ "DCSTECP" ]   = kFALSE;
            bLumiPixelOn_[ "DCSFPIX" ] = kFALSE;
            bLumiPixelOn_[ "DCSBPIX" ] = kFALSE;
            if ( TString( xmlRR->GetNodeName( nodeLumiRange ) ) == "LUMI_SECTION_RANGE" ) {
              XMLNodePointer_t nodeLumiRangeChild( xmlRR->GetChild( nodeLumiRange ) );
              while ( nodeLumiRangeChild &&  TString( xmlRR->GetNodeName( nodeLumiRangeChild ) ) != "PARAMETERS")
                nodeLumiRangeChild = xmlRR->GetNext( nodeLumiRangeChild );
              if ( nodeLumiRangeChild ) {
                XMLNodePointer_t nodeParameter( xmlRR->GetChild( nodeLumiRangeChild ) );
                while ( nodeParameter ) {
                  if ( xmlRR->GetNodeContent( nodeParameter ) && xmlRR->GetNodeContent( nodeParameter ) == TString( "true" ) ) {
                    const TString nodeName( xmlRR->GetNodeName( nodeParameter ) );
                    if ( bLumiSiStripOn_.find( nodeName ) != bLumiSiStripOn_.end() ) {
                      bLumiSiStripOn_[ nodeName ] = kTRUE;
                    } else if ( bLumiPixelOn_.find( nodeName ) != bLumiPixelOn_.end() ) {
                      bLumiPixelOn_[ nodeName ] = kTRUE;
                    }
                  }
                  nodeParameter = xmlRR->GetNext( nodeParameter );
                }
              }
            }
            Bool_t siStripOn( kTRUE );
            Bool_t pixelOn( kTRUE );
            for ( map< TString, Bool_t >::const_iterator iMap = bLumiSiStripOn_.begin(); iMap != bLumiSiStripOn_.end(); ++iMap ) {
              if ( ! iMap->second ) siStripOn = kFALSE;
              break;
            }
            for ( map< TString, Bool_t >::const_iterator iMap = bLumiPixelOn_.begin(); iMap != bLumiPixelOn_.end(); ++iMap ) {
              if ( ! iMap->second ) pixelOn = kFALSE;
              break;
            }
            if ( siStripOn ) bSiStripOn_ = kTRUE;
            if ( pixelOn )   bPixelOn_   = kTRUE;
            if ( bSiStripOn_ && bPixelOn_ ) break;
            nodeLumiRange = xmlRR->GetNext( nodeLumiRange );
          }
          break;
        }
      }
      break;
    }
    nodeRun = xmlRR->GetNext( nodeRun );
  }

  if ( ! foundRun ) {
    ++nRunsNotRRLumis_;
    cout << "    Run " << sRunNumber_ << " not found in RR lumis" << endl;
    return kFALSE;
  }
  if ( ! foundDataset ) {
    ++nRunsNotDatasetLumis_;
    cout << "    Dataset " << sArguments[ "-d" ] << " not found in RR lumis" << endl;
    return kFALSE;
  }
  if ( ! bSiStripOn_ ) {
    ++nRunsSiStripOff_;
    cout << "    SiStrip (partially) OFF during the whole run" << endl;
  }
  if ( ! bPixelOn_ ) {
    ++nRunsPixelOff_;
    cout << "    Pixel (partially) OFF during the whole run" << endl;
  }

  return kTRUE;

}


/// Reads RR certification flags for lumi ranges in a given run
/// Returns 'kTRUE', if a given run is present in RR, 'kFALSE' otherwise.
Bool_t readRRTracker( const TString & pathFile )
{

  map< TString, TString > sFlagsRRTracker;
  map< TString, TString > sCommentsRRTracker;
  iFlagsRRTracker_.clear();
  Bool_t foundRun( kFALSE );
  Bool_t foundGroup( kFALSE );
  Bool_t foundDataset( kFALSE );
  vector< TString > nameCmpNode;
  nameCmpNode.push_back( "STRIP" );
  nameCmpNode.push_back( "PIXEL" );
  nameCmpNode.push_back( "TRACKING" );

  // Read RR file corresponding to output format type 'xml'
  TXMLEngine * xmlRR( new TXMLEngine );
  XMLDocPointer_t  xmlRRDoc( xmlRR->ParseFile( nameFileTrackerRR_.Data() ) );
  XMLNodePointer_t nodeMain( xmlRR->DocGetRootElement( xmlRRDoc ) );
  XMLNodePointer_t nodeRun( xmlRR->GetChild( nodeMain ) );
  while ( nodeRun ) {
    XMLNodePointer_t nodeRunChild( xmlRR->GetChild( nodeRun ) );
    while ( nodeRunChild && TString( xmlRR->GetNodeName( nodeRunChild ) ) != "NUMBER" )
      nodeRunChild = xmlRR->GetNext( nodeRunChild );
    if ( nodeRunChild && xmlRR->GetNodeContent( nodeRunChild ) == sRunNumber_ ) {
      foundRun = kTRUE;
      nodeRunChild = xmlRR->GetChild( nodeRun );
      while ( nodeRunChild && TString( xmlRR->GetNodeName( nodeRunChild ) ) != "GROUP_NAME" )
        nodeRunChild = xmlRR->GetNext( nodeRunChild );
      if ( nodeRunChild && xmlRR->GetNodeContent( nodeRunChild ) == sArguments[ "-g" ] ) {
        foundGroup = kTRUE;
        nodeRunChild = xmlRR->GetChild( nodeRun );
        while ( nodeRunChild && TString( xmlRR->GetNodeName( nodeRunChild ) ) != "DATASETS" )
          nodeRunChild = xmlRR->GetNext( nodeRunChild );
        if ( nodeRunChild ) {
          XMLNodePointer_t nodeDataset( xmlRR->GetChild( nodeRunChild ) );
          while ( nodeDataset ) {
            XMLNodePointer_t nodeDatasetChild( xmlRR->GetChild( nodeDataset ) );
            while ( nodeDatasetChild && TString( xmlRR->GetNodeName( nodeDatasetChild ) ) != "NAME" )
              nodeDatasetChild = xmlRR->GetNext( nodeDatasetChild );
//             if ( nodeDatasetChild && TString( xmlRR->GetNodeContent( nodeDatasetChild ) ) == sArguments[ "-d" ] ) {
            if ( nodeDatasetChild && TString( xmlRR->GetNodeContent( nodeDatasetChild ) ) == TString( "/Global/Online/ALL" ) ) { // currently cretaed under this dataset name in RR TRACKER
              foundDataset = kTRUE;
              nodeDatasetChild = xmlRR->GetChild( nodeDataset );
              while ( nodeDatasetChild && TString( xmlRR->GetNodeName( nodeDatasetChild ) ) != "CMPS" )
                nodeDatasetChild = xmlRR->GetNext( nodeDatasetChild );
              if ( nodeDatasetChild ) {
                XMLNodePointer_t nodeCmp( xmlRR->GetChild( nodeDatasetChild ) );
                while ( nodeCmp ) {
                  XMLNodePointer_t nodeCmpChild( xmlRR->GetChild( nodeCmp ) );
                  while ( nodeCmpChild && TString( xmlRR->GetNodeName( nodeCmpChild ) ) != "NAME" )
                    nodeCmpChild = xmlRR->GetNext( nodeCmpChild );
                  if ( nodeCmpChild ) {
                    for ( UInt_t iNameNode = 0; iNameNode < nameCmpNode.size(); ++iNameNode ) {
                      if ( xmlRR->GetNodeContent( nodeCmpChild ) == nameCmpNode.at( iNameNode ) ) {
                        TString nameNode( "RRTracker_" + nameCmpNode.at( iNameNode ) );
                        XMLNodePointer_t nodeCmpChildNew( xmlRR->GetChild( nodeCmp ) );
                        while ( nodeCmpChildNew && TString( xmlRR->GetNodeName( nodeCmpChildNew ) ) != "VALUE" )
                          nodeCmpChildNew = xmlRR->GetNext( nodeCmpChildNew );
                        if ( nodeCmpChildNew ) {
                          sFlagsRRTracker[ nameNode ] = TString( xmlRR->GetNodeContent( nodeCmpChildNew ) );
                          if ( sFlagsRRTracker[ nameNode ] == "BAD" ) {
                            nodeCmpChildNew = xmlRR->GetChild( nodeCmp );
                            while ( nodeCmpChildNew && TString( xmlRR->GetNodeName( nodeCmpChildNew ) ) != "COMMENT" )
                              nodeCmpChildNew = xmlRR->GetNext( nodeCmpChildNew ); // FIXME Segmentation violation???
                            if ( nodeCmpChildNew ) {
                              sCommentsRRTracker[ nameNode ] = TString( xmlRR->GetNodeContent( nodeCmpChildNew ) );
                            }
                          }
                        }
                      }
                    }
                  }
                  nodeCmp = xmlRR->GetNext( nodeCmp );
                }
              }
              break;
            }
            nodeDataset = xmlRR->GetNext( nodeDataset );
          }
        }
      }
      break;
    }
    nodeRun = xmlRR->GetNext( nodeRun );
  }

  if ( ! foundRun ) {
    ++nRunsNotRRTracker_;
    cout << "    Run " << sRunNumber_ << " not found in RR Tracker" << endl;
    return kFALSE;
  }
  if ( ! foundGroup ) {
    ++nRunsNotGroupTracker_;
    cout << "    Group " << sArguments[ "-g" ] << " not found in RR" << endl;
    return kFALSE;
  }
  if ( ! foundDataset ) {
    ++nRunsNotDatasetTracker_;
    cout << "    Dataset " << sArguments[ "-d" ] << " not found in RR Tracker" << endl;
    return kFALSE;
  }

  sRRTrackerCommentsSiStrip_[ sRunNumber_ ]  = sCommentsRRTracker[ "RRTracker_STRIP" ];
  sRRTrackerCommentsPixel_[ sRunNumber_ ]    = sCommentsRRTracker[ "RRTracker_PIXEL" ];
  sRRTrackerCommentsTracking_[ sRunNumber_ ] = sCommentsRRTracker[ "RRTracker_TRACKING" ];
  iFlagsRRTracker_[ sSubSys_[ SiStrip ] ]  = FlagConvert( sFlagsRRTracker[ "RRTracker_STRIP" ] );
  iFlagsRRTracker_[ sSubSys_[ Pixel ] ]    = FlagConvert( sFlagsRRTracker[ "RRTracker_PIXEL" ] );
  iFlagsRRTracker_[ sSubSys_[ Tracking ] ] = FlagConvert( sFlagsRRTracker[ "RRTracker_TRACKING" ] );

  return kTRUE;

}


/// Reads automatically created certification flags/values from the DQM file for a given run
/// Returns 'kTRUE', if the DQM file is readable, 'kFALSE' otherwise.
Bool_t readDQM( const TString & pathFile )
{

  // Initialize
  fCertificates_.clear();
  bAvailable_.clear();

  // Open DQM file
  TFile * fileDQM( TFile::Open( pathFile.Data() ) );
  if ( ! fileDQM ) {
    cerr << "    ERROR: DQM file not found" << endl;
    cerr << "    Please, check path to DQM files" << endl;
    return kFALSE;
  }

  // Browse certification folders
  vector< TString > nameCertDir;
  nameCertDir.push_back( nameDirHead_ );
  for ( UInt_t iSys = 0; iSys < nSubSys_; ++iSys ) {
    bAvailable_[ sSubSys_[ iSys ] ] = ( iFlagsRR_[ sSubSys_[ iSys ] ] != EXCL );
    if ( bAvailable_[ sSubSys_[ iSys ] ] ) {
      const TString baseDir( nameDirHead_ + pathRunFragment_ + sSubSys_[ iSys ] + "/Run summary/" + nameDirBase_ );
      nameCertDir.push_back( baseDir );
      nameCertDir.push_back( baseDir + "/" + nameDirCert_ );
      nameCertDir.push_back( baseDir + "/" + nameDirReport_ );
      if ( iSys != Tracking ) {
        nameCertDir.push_back( baseDir + "/" + nameDirDAQ_ );
        nameCertDir.push_back( baseDir + "/" + nameDirDCS_ );
      }
    }
  }
  for ( UInt_t iDir = 0; iDir < nameCertDir.size(); ++iDir ) {
    const TString nameCurDir( nameCertDir.at( iDir ).Contains( pathRunFragment_ ) ? nameCertDir.at( iDir ).Insert( nameCertDir.at( iDir ).Index( "Run " ) + 4, sRunNumber_ ) : nameCertDir.at( iDir ) );
    TDirectory * dirSub( ( TDirectory * )fileDQM->Get( nameCurDir.Data() ) );
    if ( ! dirSub ) {
      cout << "    WARNING: " << nameCurDir.Data() << " does not exist" << endl;
      continue;
    }
    readCertificates( dirSub );
  }

  fileDQM->Close();

  if ( sOptions[ "-v" ] ) {
    cout << "    " << sVersion_ << endl;
    for ( map< TString, Double_t >::const_iterator cert = fCertificates_.begin(); cert != fCertificates_.end(); ++cert ) cout << "    " << cert->first << ": " << cert->second << endl;
  }

  return kTRUE;

}


/// Extract run certificates from DQM file
void readCertificates( TDirectory * dir )
{

  TIter nextKey( dir->GetListOfKeys() );
  TKey * key;
  while ( ( key = ( TKey * )nextKey() ) ) {
    const TString nameKey( key->GetName() );
    const Int_t index1( nameKey.Index( ">" ) );
    if ( index1 == kNPOS ) continue;
    TString nameCert( nameKey( 1, index1 - 1 ) );
    if ( TString( dir->GetName() ) == nameDirHead_ ) {
      if ( nameCert.CompareTo( "ReleaseTag" ) == 0 ) {
        const Ssiz_t indexKey( nameKey.Index( "s=" ) + 2 );
        const TString nameKeyBrake( nameKey( indexKey, nameKey.Length() - indexKey ) );
        sVersion_ = nameKeyBrake( 0, nameKeyBrake.Index( "<" ) );
      }
      continue;
    }
    TString nameCertFirst( nameCert( 0, 1 ) );
    nameCertFirst.ToUpper();
    nameCert.Replace( 0, 1, nameCertFirst );
    if ( TString( dir->GetName() ) == nameDirBase_ ) { // indicates summaries
      if ( ! nameCert.Contains( "Summary" ) ) continue;
      const TString nameDir( dir->GetPath() );
      const UInt_t index2( nameDir.Index( "/", nameDir.Index( ":" ) + 10 ) );
      const TString nameSub( nameDir( index2 + 1, nameDir.Index( "/", index2 + 1 ) - index2 - 1 ) );
      nameCert.Prepend( nameSub );
    } else if ( TString( dir->GetName() ) == nameDirCert_ ) {
      nameCert.Prepend( "Cert" );
    } else if ( TString( dir->GetName() ) == nameDirDAQ_ ) {
      nameCert.Prepend( "DAQ" );
    } else if ( TString( dir->GetName() ) == nameDirDCS_ ) {
      nameCert.Prepend( "DCS" );
    } else {
      nameCert.Prepend( "Report" );
    }
    const Ssiz_t  indexKey( nameKey.Index( "f=" ) + 2 );
    const TString nameKeyBrake( nameKey( indexKey, nameKey.Length() - indexKey ) );
    const TString nameKeyBrakeAll( nameKeyBrake( 0, nameKeyBrake.Index( "<" ) ) );
    fCertificates_[ nameCert ] = atof( nameKeyBrakeAll.Data() );
  }

  return;

}


/// Determine actual certification flags per run and sub-system
void certifyRun()
{

  // FIXME Currently, LS-wise HV information from the RR is not determined correctly
  //       So, it is not used for the certification yet.

  // Initialize
  map< TString, Int_t > iFlags;

  // SiStrip
  sRRSiStrip_[ sRunNumber_ ]        = FlagConvert( iFlagsRR_[ sSubSys_[ SiStrip ] ] );
  sRRTrackerSiStrip_[ sRunNumber_ ] = FlagConvert( iFlagsRRTracker_[ sSubSys_[ SiStrip ] ] );
  if ( bAvailable_[ sSubSys_[ SiStrip ] ] ) {
    Bool_t flagDet( fCertificates_[ "SiStripReportSummary" ] > minGood_ );
    Bool_t flagDAQ( fCertificates_[ "SiStripDAQSummary" ] == ( Double_t )EXCL || fCertificates_[ "SiStripDAQSummary" ] > minGood_ );
    Bool_t flagDCS( fCertificates_[ "SiStripDCSSummary" ] == ( Double_t )EXCL || fCertificates_[ "SiStripDCSSummary" ] == ( Double_t )GOOD );
    Bool_t flagDQM( flagDet * flagDAQ * flagDCS );
    Bool_t flagCert( iFlagsRRTracker_[ sSubSys_[ SiStrip ] ] );
//     iFlags[ sSubSys_[ SiStrip ] ] = ( Int_t )( flagDQM * bSiStripOn_ * flagCert );
    iFlags[ sSubSys_[ SiStrip ] ] = ( Int_t )( flagDQM * flagCert );
    sDQMSiStrip_[ sRunNumber_ ] = FlagConvert( ( Int_t )( flagDQM ) );
    sSiStrip_[ sRunNumber_ ]    = FlagConvert( iFlags[ sSubSys_[ SiStrip ] ] );
    vector< TString > comments;
    if ( ! flagDet )     comments.push_back( "too low overall fraction of good modules" );
    if ( ! flagDAQ )     comments.push_back( "DAQSummary BAD" );
    if ( ! flagDCS )     comments.push_back( "DCSSummary BAD" );
//     if ( ! bSiStripOn_ ) comments.push_back( "HV off" );
    if ( ! flagCert )    comments.push_back( TString( "Tracker shifter: " + sRRTrackerCommentsSiStrip_[ sRunNumber_ ] ) );
    if ( iFlags[ sSubSys_[ SiStrip ] ] == BAD ) {
      ++nRunsBadSiStrip_;
      if ( flagCert ) comments.push_back( TString( "Tracker shifter differs (GOOD): " + sRRTrackerCommentsSiStrip_[ sRunNumber_ ] ) );
      sRunCommentsSiStrip_[ sRunNumber_ ] = comments;
    }
  } else {
    sDQMSiStrip_[ sRunNumber_ ] = sRRSiStrip_[ sRunNumber_ ];
    sSiStrip_[ sRunNumber_ ]    = sRRSiStrip_[ sRunNumber_ ];
  }

  // Pixel
  sRRPixel_[ sRunNumber_ ]        = FlagConvert( iFlagsRR_[ sSubSys_[ Pixel ] ] );
  sRRTrackerPixel_[ sRunNumber_ ] = FlagConvert( iFlagsRRTracker_[ sSubSys_[ Pixel ] ] );
  if ( bAvailable_[ sSubSys_[ Pixel ] ] ) {
    Bool_t flagReportSummary( fCertificates_[ "PixelReportSummary" ] > maxBad_ );
    Bool_t flagDAQ( ( fCertificates_[ "DAQPixelBarrelFraction" ] == ( Double_t )EXCL || fCertificates_[ "DAQPixelBarrelFraction" ] > 0. ) &&  ( fCertificates_[ "DAQPixelEndcapFraction" ] == ( Double_t )EXCL || fCertificates_[ "DAQPixelEndcapFraction" ] > 0. ) ); // unidentified bug in Pixel DAQ fraction determination
    Bool_t flagDCS( fCertificates_[ "PixelDCSSummary" ] == ( Double_t )EXCL || fCertificates_[ "PixelDCSSummary" ] > maxBad_ );
//     Bool_t flagDQM( flagReportSummary * flagDAQ * flagDCS );
    Bool_t flagDQM( flagDCS ); // bugs in DAQ fraction and report summary
    Bool_t flagCert( iFlagsRRTracker_[ sSubSys_[ Pixel ] ] );
//     iFlags[ sSubSys_[ Pixel ] ] = ( Int_t )( flagDQM * bPixelOn_ * flagCert );
    iFlags[ sSubSys_[ Pixel ] ] = ( Int_t )( flagDQM * flagCert );
    sDQMPixel_[ sRunNumber_ ] = FlagConvert( ( Int_t )( flagDQM ) );
    sPixel_[ sRunNumber_ ]    = FlagConvert( iFlags[ sSubSys_[ Pixel ] ] );
    vector< TString > comments;
    if ( ! flagReportSummary ) comments.push_back( "ReportSummary BAD" );
    if ( ! flagDAQ )           comments.push_back( "DAQSummary BAD" );
    if ( ! flagDCS )           comments.push_back( "DCSSummary BAD" );
//     if ( ! bPixelOn_ )         comments.push_back( "HV off" );
    if ( ! flagCert )          comments.push_back( TString( "Tracker shifter: " + sRRTrackerCommentsPixel_[ sRunNumber_ ] ) );
    if ( iFlags[ sSubSys_[ Pixel ] ] == BAD ) {
      ++nRunsBadPixel_;
      if ( flagCert ) comments.push_back( TString( "Tracker shifter differs (GOOD): " + sRRTrackerCommentsPixel_[ sRunNumber_ ] ) );
      sRunCommentsPixel_[ sRunNumber_ ] = comments;
    }
  } else {
    sDQMPixel_[ sRunNumber_ ] = sRRPixel_[ sRunNumber_ ];
    sPixel_[ sRunNumber_ ]    = sRRPixel_[ sRunNumber_ ];
  }

  // Tracking
  sRRTracking_[ sRunNumber_ ]        = FlagConvert( iFlagsRR_[ sSubSys_[ Tracking ] ] );
  sRRTrackerTracking_[ sRunNumber_ ] = FlagConvert( iFlagsRRTracker_[ sSubSys_[ Tracking ] ] );
  if ( bAvailable_[ sSubSys_[ Tracking ] ] ) {
    Bool_t flagCert( iFlagsRRTracker_[ sSubSys_[ Pixel ] ] );
    Bool_t flagDQM( kFALSE );
    vector< TString > comments;
    if ( iFlagsRR_[ sSubSys_[ SiStrip ] ] == EXCL && iFlagsRR_[ sSubSys_[ Pixel ] ] == EXCL ) {
      comments.push_back( "SiStrip and Pixel EXCL: no reasonable Tracking" );
    } else {
      Bool_t flagChi2( fCertificates_[ "ReportTrackChi2" ] > maxBad_ );
      Bool_t flagRate( fCertificates_[ "ReportTrackRate" ] > maxBad_ );
      Bool_t flagRecHits( fCertificates_[ "ReportTrackRecHits" ] > maxBad_ );
      flagDQM  = flagChi2 * flagRate * flagRecHits;
      if ( ! flagChi2 )    comments.push_back( "Chi2/DoF too low" );
      if ( ! flagRate )    comments.push_back( "Track rate too low" );
      if ( ! flagRecHits ) comments.push_back( "Too few RecHits" );
//       if ( ! bSiStripOn_ ) comments.push_back( "HV SiStrip off" );
      if ( ! flagCert ) comments.push_back( TString( "Tracker shifter: " + sRRTrackerCommentsTracking_[ sRunNumber_ ] ) );
    }
//     iFlags[ sSubSys_[ Tracking ] ] = ( Int_t )( flagDQM * bSiStripOn_ * flagCert );
    iFlags[ sSubSys_[ Tracking ] ] = ( Int_t )( flagDQM * flagCert );
    sDQMTracking_[ sRunNumber_ ] = FlagConvert( ( Int_t )( flagDQM ) );
    sTracking_[ sRunNumber_ ]    = FlagConvert( iFlags[ sSubSys_[ Tracking ] ] );
    if ( iFlags[ sSubSys_[ Tracking ] ] == BAD ) {
      ++nRunsBadTracking_;
      if ( flagCert ) comments.push_back( TString( "Tracker shifter differs (GOOD): " + sRRTrackerCommentsTracking_[ sRunNumber_ ] ) );
      sRunCommentsTracking_[ sRunNumber_ ] = comments;
    }
  } else {
    sDQMTracking_[ sRunNumber_ ] = sRRTracking_[ sRunNumber_ ];
    sTracking_[ sRunNumber_ ]    = sRRTracking_[ sRunNumber_ ];
  }

  for ( map< TString, Int_t >::const_iterator iSys = iFlags.begin(); iSys != iFlags.end(); ++iSys ) {
    cout << "    " << iSys->first << ": ";
    if ( iSys->second != iFlagsRR_[ iSys->first ] ) {
      if ( iSys->first == sSubSys_[ SiStrip ] )  ++nRunsChangedSiStrip_;
      if ( iSys->first == sSubSys_[ Pixel ] )    ++nRunsChangedPixel_;
      if ( iSys->first == sSubSys_[ Tracking ] ) ++nRunsChangedTracking_;
      cout << FlagConvert( iFlagsRR_[ iSys->first ] ) << " --> ";
    }
    cout << FlagConvert( iSys->second ) << endl;
    if ( sOptions[ "-v" ] ) {
      if ( iSys->first == sSubSys_[ SiStrip ] ) {
        for ( UInt_t iCom = 0; iCom < sRunCommentsSiStrip_[ sRunNumber_ ].size(); ++iCom ) {
          cout << "      " << sRunCommentsSiStrip_[ sRunNumber_ ].at( iCom ).Data() << endl;
        }
      }
      if ( iSys->first == sSubSys_[ Pixel ] ) {
        for ( UInt_t iCom = 0; iCom < sRunCommentsPixel_[ sRunNumber_ ].size(); ++iCom ) {
          cout << "      " << sRunCommentsPixel_[ sRunNumber_ ].at( iCom ).Data() << endl;
        }
      }
      if ( iSys->first == sSubSys_[ Tracking ] ) {
        for ( UInt_t iCom = 0; iCom < sRunCommentsTracking_[ sRunNumber_ ].size(); ++iCom ) {
          cout << "      " << sRunCommentsTracking_[ sRunNumber_ ].at( iCom ).Data() << endl;
        }
      }
    }
  }

  return;

}


/// Print summary
void writeOutput()
{

  // Initialize
  ofstream fileLog;
  fileLog.open( sArguments[ "-o" ].Data() );
  fileLog << "Tracker Certification runs " << minRun_ << " - " << maxRun_ << endl
          << "==========================================" << endl
          << endl
          << "Used DQM files found in " << sArguments[ "-p" ] << endl
          << "for dataset             " << sArguments[ "-d" ] << endl
          << "and group name          " << sArguments[ "-g" ] << endl
          << endl
          << "# of runs total                          : " << nRuns_                  << endl
          << "------------------------------------------ "                            << endl
          << "# of runs certified                      : " << sRunNumbers_.size()     << endl
          << "# of runs not found in RR                : " << nRunsNotRR_             << endl
          << "# of runs group not found in RR          : " << nRunsNotGroup_          << endl
          << "# of runs dataset not found in RR        : " << nRunsNotDataset_        << endl;
  if ( ! sOptions[ "-a" ] ) fileLog << "# of runs not in SIGNOFF in RR           : " << nRunsNotSignoff_        << endl;
  fileLog << "# of runs not found in RR Tracker        : " << nRunsNotRRTracker_      << endl
          << "# of runs group not found in RR Tracker  : " << nRunsNotGroupTracker_   << endl
//           << "# of runs dataset not found in RR Tracker: " << nRunsNotDatasetTracker_ << endl
          << "# of runs not found in RR lumis          : " << nRunsNotRRLumis_        << endl
          << "# of runs dataset not found in RR lumis  : " << nRunsNotDatasetLumis_   << endl
          << endl
          << "# of runs w/o SiStrip       : " << nRunsExclSiStrip_     << endl
          << "# of bad runs SiStrip       : " << nRunsBadSiStrip_      << endl
          << "# of changed runs SiStrip   : " << nRunsChangedSiStrip_  << endl
          << "# of runs w/o Pixel         : " << nRunsExclPixel_       << endl
          << "# of bad runs Pixel         : " << nRunsBadPixel_        << endl
          << "# of changed runs Pixel     : " << nRunsChangedPixel_    << endl
          << "# of runs w/o Tracking (BAD): " << nRunsNoTracking_      << endl
          << "# of bad runs Tracking      : " << nRunsBadTracking_     << endl
          << "# of changed runs Tracking  : " << nRunsChangedTracking_ << endl;

  // SiStrip
  fileLog << endl
          << sSubSys_[ 0 ] << ":" << endl
          << endl;
  for ( UInt_t iRun = 0; iRun < sRunNumbers_.size(); ++iRun ) {
    if ( sRRSiStrip_[ sRunNumbers_.at( iRun ) ] != sSiStrip_[ sRunNumbers_.at( iRun ) ] ) {
      fileLog << "  " << sRunNumbers_.at( iRun ) << ": " << sRRSiStrip_[ sRunNumbers_.at( iRun ) ] << " --> " << sSiStrip_[ sRunNumbers_.at( iRun ) ] << endl;
      if ( sRRSiStrip_[ sRunNumbers_.at( iRun ) ] == TString( "BAD" ) ) {
        fileLog << "    RR was: " << sRRCommentsSiStrip_[ sRunNumbers_.at( iRun ) ] << endl;
      }
      for ( UInt_t iCom = 0; iCom < sRunCommentsSiStrip_[ sRunNumbers_.at( iRun ) ].size(); ++iCom ) {
        fileLog << "    " << sRunCommentsSiStrip_[ sRunNumbers_.at( iRun ) ].at( iCom ).Data() << endl;
      }
    }
  }

  // Pixel
  fileLog << endl
          << sSubSys_[ 1 ] << ":" << endl
          << endl;
  for ( UInt_t iRun = 0; iRun < sRunNumbers_.size(); ++iRun ) {
    if ( sRRPixel_[ sRunNumbers_.at( iRun ) ] != sPixel_[ sRunNumbers_.at( iRun ) ] ) {
      fileLog << "  " << sRunNumbers_.at( iRun ) << ": " << sRRPixel_[ sRunNumbers_.at( iRun ) ] << " --> " << sPixel_[ sRunNumbers_.at( iRun ) ] << endl;
      if ( sRRPixel_[ sRunNumbers_.at( iRun ) ] == TString( "BAD" ) ) {
        fileLog << "    RR was: " << sRRCommentsPixel_[ sRunNumbers_.at( iRun ) ] << endl;
      }
      for ( UInt_t iCom = 0; iCom < sRunCommentsPixel_[ sRunNumbers_.at( iRun ) ].size(); ++iCom ) {
        fileLog << "    " << sRunCommentsPixel_[ sRunNumbers_.at( iRun ) ].at( iCom ).Data() << endl;
      }
    }
  }

  // Tracking
  fileLog << endl
          << sSubSys_[ 2 ] << ":" << endl
          << endl;
  for ( UInt_t iRun = 0; iRun < sRunNumbers_.size(); ++iRun ) {
    if ( sRRTracking_[ sRunNumbers_.at( iRun ) ] != sTracking_[ sRunNumbers_.at( iRun ) ] ) {
      fileLog << "  " << sRunNumbers_.at( iRun ) << ": " << sRRTracking_[ sRunNumbers_.at( iRun ) ] << " --> " << sTracking_[ sRunNumbers_.at( iRun ) ] << endl;
      if ( sRRTracking_[ sRunNumbers_.at( iRun ) ] == TString( "BAD" ) ) {
        fileLog << "    RR was: " << sRRCommentsTracking_[ sRunNumbers_.at( iRun ) ] << endl;
      }
      for ( UInt_t iCom = 0; iCom < sRunCommentsTracking_[ sRunNumbers_.at( iRun ) ].size(); ++iCom ) {
        fileLog << "    " << sRunCommentsTracking_[ sRunNumbers_.at( iRun ) ].at( iCom ).Data() << endl;
      }
    }
  }

  fileLog.close();

  cout << endl
       << "SUMMARY:" << endl
       << endl;
  for ( UInt_t iRun = 0; iRun < sRunNumbers_.size(); ++iRun ) {
    cout << "  " << sRunNumbers_.at( iRun ) << ":" << endl;
    cout << "    " << sSubSys_[ 0 ] << ": " << sSiStrip_[ sRunNumbers_.at( iRun ) ] << endl;
    for ( UInt_t iCom = 0; iCom < sRunCommentsSiStrip_[ sRunNumbers_.at( iRun ) ].size(); ++iCom ) {
      cout << "      " << sRunCommentsSiStrip_[ sRunNumbers_.at( iRun ) ].at( iCom ).Data() << endl;
    }
    cout << "    " << sSubSys_[ 1 ] << ": " << sPixel_[ sRunNumbers_.at( iRun ) ] << endl;
    for ( UInt_t iCom = 0; iCom < sRunCommentsPixel_[ sRunNumbers_.at( iRun ) ].size(); ++iCom ) {
      cout << "      " << sRunCommentsPixel_[ sRunNumbers_.at( iRun ) ].at( iCom ).Data() << endl;
    }
    cout << "    " << sSubSys_[ 2 ] << ": " << sTracking_[ sRunNumbers_.at( iRun ) ] << endl;
    for ( UInt_t iCom = 0; iCom < sRunCommentsTracking_[ sRunNumbers_.at( iRun ) ].size(); ++iCom ) {
      cout << "      " << sRunCommentsTracking_[ sRunNumbers_.at( iRun ) ].at( iCom ).Data() << endl;
    }
  }

  cout << endl
       << "Certification SUMMARY to be sent to CMS DQM team available in ./" << sArguments[ "-o" ].Data() << endl
       << endl;

  return;

}


/// Print help
void displayHelp()
{

  cerr << "  TrackerRunCertification" << endl
       << endl
       << "  CMSSW package: DQM/TrackerCommon" << endl
       << endl
       << "  Purpose:" << endl
       << endl
       << "  The procedure of certifying data of a given run range is automated in order to speed up the procedure and to reduce the Tracker Offline Shift Leader's workload." << endl
       << endl
       << "  Input:" << endl
       << endl
       << "  - RunRegistry" << endl
       << "  - DQM output files available in AFS" << endl
       << endl
       << "  Output:" << endl
       << endl
       << "  Text file" << endl
       << "  - [as explained for command line option '-o']" << endl
       << "  to be sent directly to the CMS DQM team as reply to the weekly certification request." << endl
       << "  It contains a list of all flags changed with respect to the RunRegistry, including the reason(s) in case the flag is changed to BAD." << endl
       << endl
       << "  The verbose ('-v') stdout can provide a complete list of all in-/output flags of all analyzed runs and at its end a summary only with the output flags." << endl
       << "  It makes sense to pipe the stdout to another text file." << endl
       << endl
       << "  Usage:" << endl
       << endl
       << "  $ cmsrel CMSSW_RELEASE" << endl
       << "  $ cd CMSSW_RELEASE/src" << endl
       << "  $ cmsenv" << endl
       << "  $ cvs co -r Vxx-yy-zz DQM/TrackerCommon" << endl
       << "  $ scram b -j 5" << endl
       << "  $ rehash" << endl
       << "  $ cd WORKING_DIRECTORY" << endl
       << "  $ [create input files]" << endl
       << "  $ TrackerRunCertification [ARGUMENTOPTION1] [ARGUMENT1] ... [OPTION2] ..." << endl
       << endl
       << "  Valid argument options are:" << endl
       << "    -d" << endl
       << "      MANDATORY: dataset as in RunRegistry" << endl
       << "      no default" << endl
       << "    -g" << endl
       << "      MANDATORY: group name as in RunRegistry" << endl
       << "      no default" << endl
       << "    -p" << endl
       << "      path to DQM files" << endl
       << "      default: /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/data/OfflineData/Run2010/StreamExpress" << endl
       << "    -P" << endl
       << "      pattern of DQM file names in the DQM file path" << endl
       << "      default: *[DATASET from '-d' option with '/' --> '__'].root" << endl
       << "    -o" << endl
       << "      path to output log file" << endl
       << "      default: trackerRunCertification[DATASET from '-d' option with '/' --> '__']-[GROUP from '-g'].txt" << endl
       << "    -l" << endl
       << "      lower bound of run numbers to consider" << endl
       << "      default: 0" << endl
       << "    -u" << endl
       << "      upper bound of run numbers to consider" << endl
       << "      default: 1073741824 (2^30)" << endl
       << "    -R" << endl
       << "      web address of the RunRegistry" << endl
       << "      default: http://pccmsdqm04.cern.ch/runregistry" << endl
       << "    The default is used for any option not explicitely given in the command line." << endl
       << endl
       << "  Valid options are:" << endl
       << "    -rr" << endl
       << "      switch on creation of new RR file" << endl
       << "    -rronly" << endl
       << "      only create new RR file, do not run certification" << endl
       << "    -a" << endl
       << "      certify all runs, not only those in \"SIGNOFF\" status" << endl
       << "    -v" << endl
       << "      switch on verbose logging to stdout" << endl
       << "    -h" << endl
       << "      display this help and exit" << endl
       << endl;
  return;
}


/// Little helper to determine run number (TString) from file name/path
TString RunNumber( const TString & pathFile )
{

  const TString sPrefix( "DQM_V" );
  const TString sNumber( pathFile( pathFile.Index( sPrefix ) + sPrefix.Length() + 6, 9 ) );
  UInt_t index( ( string( sNumber.Data() ) ).find_first_not_of( '0' ) );
  return sNumber( index, sNumber.Length() - index );

}


/// Little helper to convert RR flags from TString into Int_t
Int_t FlagConvert( const TString & flag )
{

  map< TString, Int_t > flagSToI;
  flagSToI[ "MISSING" ] = MISSING;
  flagSToI[ "NOTSET" ]  = NOTSET;
  flagSToI[ "EXCL" ]    = EXCL;
  flagSToI[ "BAD" ]     = BAD;
  flagSToI[ "GOOD" ]    = GOOD;
  return flagSToI[ flag ];

}
/// Little helper to convert RR flags from Int_t into TString
TString FlagConvert( const Int_t flag )
{

  map< Int_t, TString > flagIToS;
  flagIToS[ MISSING ] = "MISSING";
  flagIToS[ NOTSET ]  = "NOTSET";
  flagIToS[ EXCL ]    = "EXCL";
  flagIToS[ BAD ]     = "BAD";
  flagIToS[ GOOD ]    = "GOOD";
  return flagIToS[ flag ];

}
