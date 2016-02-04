
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentDataCollector.h"

#include "TGraph.h"
#include "TNtuple.h"
#include "TFile.h"
#include "TH1F.h"

using namespace std;


KalmanAlignmentDataCollector* KalmanAlignmentDataCollector::theDataCollector = new KalmanAlignmentDataCollector();


KalmanAlignmentDataCollector::KalmanAlignmentDataCollector( void ) {}


KalmanAlignmentDataCollector::KalmanAlignmentDataCollector( const edm::ParameterSet & config ) : theConfiguration( config ) {}


KalmanAlignmentDataCollector::~KalmanAlignmentDataCollector() {}


KalmanAlignmentDataCollector* KalmanAlignmentDataCollector::get( void )
{
  //if ( !theDataCollector ) theDataCollector = new KalmanAlignmentDataCollector();
  return theDataCollector;
}


void KalmanAlignmentDataCollector::configure( const edm::ParameterSet & config )
{
  //if ( !theDataCollector ) theDataCollector = new KalmanAlignmentDataCollector();
  theDataCollector->config( config );
}


void KalmanAlignmentDataCollector::fillHistogram( string histo_name, float data )
{
  //if ( !theDataCollector ) theDataCollector = new KalmanAlignmentDataCollector();
  theDataCollector->fillTH1F( histo_name, data );
}


void KalmanAlignmentDataCollector::fillHistogram( string histo_name, int histo_number, float data )
{
  //if ( !theDataCollector ) theDataCollector = new KalmanAlignmentDataCollector();
  theDataCollector->fillTH1F( histo_name, histo_number, data );
}


void KalmanAlignmentDataCollector::fillGraph( string graph_name, float x_data, float y_data )
{
  //if ( !theDataCollector ) theDataCollector = new KalmanAlignmentDataCollector();
  theDataCollector->fillTGraph( graph_name, x_data, y_data );
}


void KalmanAlignmentDataCollector::fillGraph( string graph_name, int graph_number, float x_data, float y_data )
{
  //if ( !theDataCollector ) theDataCollector = new KalmanAlignmentDataCollector();
  theDataCollector->fillTGraph( graph_name, graph_number, x_data, y_data );
}


void KalmanAlignmentDataCollector::fillNtuple( std::string ntuple_name, float data )
{
  //if ( !theDataCollector ) theDataCollector = new KalmanAlignmentDataCollector();
  theDataCollector->fillTNtuple( ntuple_name, data );
}


void KalmanAlignmentDataCollector::write( void )
{
  //if ( !theDataCollector ) theDataCollector = new KalmanAlignmentDataCollector();
  theDataCollector->writeToTFile();
}


void KalmanAlignmentDataCollector::write( string file_name, string mode )
{
  //if ( !theDataCollector ) theDataCollector = new KalmanAlignmentDataCollector();
  theDataCollector->writeToTFile( file_name, mode );
}


void KalmanAlignmentDataCollector::clear( void )
{
  if ( theDataCollector ) theDataCollector->clearData();
}


void KalmanAlignmentDataCollector::config( const edm::ParameterSet & config )
{
  theConfiguration = config;
}


void KalmanAlignmentDataCollector::fillTH1F( string histo_name, float data )
{
  if ( theHistoData.find( histo_name ) == theHistoData.end() )
  {
    theHistoData[histo_name] = vector< float > ( 1, data );
  }
  else
  {
    theHistoData[histo_name].push_back( data );
  }

  return;
}


void KalmanAlignmentDataCollector::fillTH1F( string histo_name, int histo_number, float data )
{
  string full_histo_name = histo_name + toString( histo_number );
  fillTH1F( full_histo_name, data );

  return;
}


void KalmanAlignmentDataCollector::fillTGraph( string graph_name, float x_data, float y_data )
{
  if ( theXGraphData.find( graph_name ) == theXGraphData.end() )
  {
    theXGraphData[graph_name] = vector< float > ( 1, x_data );
    theYGraphData[graph_name] = vector< float > ( 1, y_data );
  }
  else
  {
    theXGraphData[graph_name].push_back( x_data );
    theYGraphData[graph_name].push_back( y_data );
  }

  return;
}


void KalmanAlignmentDataCollector::fillTGraph( string graph_name, int graph_number, float x_data, float y_data )
{
  string full_graph_name = graph_name + toString( graph_number );
  fillTGraph( full_graph_name, x_data, y_data );

  return;
}


void KalmanAlignmentDataCollector::fillTNtuple( std::string ntuple_name, float data )
{
  if ( theNtupleData.find( ntuple_name ) == theNtupleData.end() )
  {
    theNtupleData[ntuple_name] = vector< float > ( 1, data );
   }
  else
  {
    theNtupleData[ntuple_name].push_back( data );
  }
}


void KalmanAlignmentDataCollector::writeToTFile( void )
{
  string fileName = theConfiguration.getUntrackedParameter< string >( "FileName", "KalmanAlignmentData.root" );
  string fileMode = theConfiguration.getUntrackedParameter< string >( "Mode", "RECREATE" );
  writeToTFile( fileName, fileMode );
}


void KalmanAlignmentDataCollector::writeToTFile( string file_name, string mode )
{
  int nBins = theConfiguration.getUntrackedParameter< int >( "NBins", 200 );
  double xMin = theConfiguration.getUntrackedParameter< double >( "XMin", -10. );
  double xMax = theConfiguration.getUntrackedParameter< double >( "XMax", 10. );

  TFile* file = new TFile( file_name.c_str(), mode.c_str() );

  if ( !theHistoData.empty() )
  {
    map< string, vector< float > >::iterator itH = theHistoData.begin();
    vector< float >::iterator itV;
    TH1F* tempHisto;

    while ( itH != theHistoData.end() )
    {
      tempHisto = new TH1F( itH->first.c_str(), itH->first.c_str(), nBins, xMin, xMax );

      itV = itH->second.begin();
      while ( itV != itH->second.end() ) {
	tempHisto->Fill( *itV );
	itV++;
      }

      tempHisto->Write();
      delete tempHisto;

      ++itH;
    }
  }

  if ( !theXGraphData.empty() )
  {
    map< string, vector< float > >::iterator itXG = theXGraphData.begin();
    map< string, vector< float > >::iterator itYG = theYGraphData.begin();

    float* xData;
    float* yData;

    TGraph* tempGraph;

    while ( itXG != theXGraphData.end() )
    {
      int nData = itXG->second.size();

      xData = new float[nData];
      yData = new float[nData];

      for ( int iData = 0; iData < nData; iData++ )
      {
	xData[iData] = itXG->second[iData];
	yData[iData] = itYG->second[iData];
      }

      tempGraph = new TGraph( nData, xData, yData );
      tempGraph->SetName( itXG->first.c_str() );
      tempGraph->SetTitle( itXG->first.c_str() );

      tempGraph->Write();
      delete tempGraph;

      delete[] xData;
      delete[] yData;

      ++itXG;
      ++itYG;
    }
  }


  if ( !theNtupleData.empty() )
  {
    map< string, vector< float > >::iterator itN = theNtupleData.begin();

    TNtuple* ntuple;

    while ( itN != theNtupleData.end() )
    {
      ntuple = new TNtuple( itN->first.c_str(), itN->first.c_str(), itN->first.c_str() );

      vector< float >::iterator itD = itN->second.begin(), itDEnd = itN->second.end();
      while ( itD != itDEnd )
      {
	ntuple->Fill( *itD );
	++itD;
      }

      ntuple->Write();
      delete ntuple;

      ++itN;
    }
  }


  file->Write();
  file->Close();
  delete file;

  return;
}


void KalmanAlignmentDataCollector::clearData( void )
{
  theHistoData.clear();
  theXGraphData.clear();
  theYGraphData.clear();
}


string KalmanAlignmentDataCollector::toString( int i )
{
  char temp[10];
  sprintf( temp, "%u", i );

  return string( temp );
}
