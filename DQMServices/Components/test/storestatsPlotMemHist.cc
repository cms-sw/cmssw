

#include <utility>
#include <vector>
#include <iostream>
#include <fstream>

#include "TGraph.h"
#include "TFile.h"
#include "TTree.h"
#include "TH1.h"
#include "TCanvas.h"


///
///
///
void plot( std::string inputFileName ) {

  TFile* inputFile = new TFile( inputFileName.c_str(), "READ" );
  TTree* inputTree = (TTree*)inputFile->Get( "dqmstorestats_memhistory" );
  
  const unsigned int nSteps = inputTree->GetEntries();

  // tgraphs want arrays..
  int time;
  float mb;
  float timeA[nSteps];
  float mbA[nSteps];

  inputTree->SetBranchAddress( "seconds", &time );
  inputTree->SetBranchAddress( "megabytes", &mb );
  
  for( unsigned int step = 0; step < nSteps; ++ step ) {
    inputTree->GetEntry( step );
    timeA[step] = time;
    mbA[step] = mb;
  }
  
  TCanvas* canvas = new TCanvas;
  TGraph* memVsTime = new TGraph( nSteps, timeA, mbA );
  memVsTime->SetMarkerStyle( 20 );
  memVsTime->SetMarkerColor( 4 );
  memVsTime->Draw( "AP" );
  memVsTime->SetTitle( "total virtual process memory" );   
  memVsTime->GetHistogram()->SetMinimum( 0. );
  memVsTime->GetHistogram()->SetMaximum( memVsTime->GetHistogram()->GetMaximum() * 1.5 );
  memVsTime->GetHistogram()->GetXaxis()->SetTitle( "time after ctor called [sec]" );
  memVsTime->GetHistogram()->GetYaxis()->SetTitle( "virtual memory [MB]" );
  canvas->Update();
  

}
  
  

// ATTIC

//   // first read in the numbers from the plain CMSSW log file
//   std::vector<std::pair<unsigned int, float> > inputNumbers;
//   std::ifstream inputFile( inputFileName.c_str(), ios::in );
//   if( inputFile.bad() ) {
//     std::cerr << " Cannot read input file: " << inputFileName << std::endl;
//     return;
//   }
//   std::string readBuffer;
//   unsigned int nSteps, time;
//   float mem;
//   bool isSuccess = false;
//   while( !inputFile.eof() ) {
//     inputFile >> readBuffer;
//     if( std::string( "<storestats_mem_hist>" ) == readBuffer ) {
//       isSuccess = true;
//       inputFile >> nSteps;
//       for( unsigned int stepCounter = 0; stepCounter < nSteps; ++stepCounter ) {
// 	inputFile >> time >> mem;
// 	inputNumbers.push_back( std::pair<unsigned int, float>( time, mem ) );
//       }
//       break;
//     }
//   }
//   if( !isSuccess ) {
//     std::cerr << " No memory history in input file." << std::endl;
//     return;
//   }
//   inputFile.close();
